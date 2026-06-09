import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from flowmatching_lthc.checkpoint import load_model_state
from flowmatching_lthc.models import MODEL_NAMES, build_model
from flowmatching_lthc.sampling import sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--state_key', default='ema', choices=['ema', 'model'])
    p.add_argument('--model', default='auto', choices=('auto',) + MODEL_NAMES)
    p.add_argument('--num_samples', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=256, help='Per-rank generation batch size.')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=1.0)
    p.add_argument('--interval_min', type=float, default=0.0)
    p.add_argument('--interval_max', type=float, default=1.0)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--prediction', default='auto', choices=['auto', 'clean', 'velocity'])
    p.add_argument('--num_classes', type=int, default=1000)
    p.add_argument('--fid_stats', default='fid_stats/imagenet256_stats.npz')
    p.add_argument('--device', default='cuda')
    p.add_argument('--attn_backend', default='flash', choices=['flash', 'efficient', 'math', 'default'])
    p.add_argument('--compile', action='store_true')
    p.add_argument(
        '--compile_mode',
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
    )
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--csv_file', default='metrics_history.csv')
    p.add_argument('--sample_dir', default='', help='Optional explicit sample directory. Defaults to output_dir/eval_samples_<step>_<timestamp>.')
    p.add_argument('--keep_samples', action='store_true', help='Keep generated PNGs instead of deleting them after metrics.')
    p.add_argument('--sample_only', action='store_true', help='Only generate PNG samples; do not compute FID/IS or write the metric CSV.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='jit-imagenet256')
    p.add_argument('--wandb_entity', default='tol011-uc-san-diego')
    p.add_argument('--wandb_run_id', default='')
    p.add_argument('--wandb_run_name', default='')
    return p.parse_args()


def apply_eval_overrides(args):
    """Allow a running chunk driver to pick up corrected eval settings safely.

    The train/eval chunk driver is a long-running shell process. Editing its
    bash defaults does not affect already-exported shell variables, but the
    driver loads this Python file fresh for every eval boundary. A small JSON
    file in the eval output directory lets us correct non-training settings
    without interrupting the active training process.
    """
    override_path = Path(args.output_dir) / 'eval_overrides.json'
    if not override_path.exists():
        return args

    with override_path.open() as f:
        overrides = json.load(f)

    allowed = {'cfg', 'interval_min', 'interval_max'}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f'Unsupported eval override key: {key}')
        setattr(args, key, value)
    print(f'[eval] applied overrides from {override_path}: {overrides}', flush=True)
    return args


def init_dist():
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def infer_model_name(args, ckpt):
    if args.model != 'auto':
        return args.model
    ckpt_args = ckpt.get('args') or {}
    if isinstance(ckpt_args, dict) and ckpt_args.get('model') in MODEL_NAMES:
        return ckpt_args['model']
    meta = ckpt.get('meta') or {}
    if isinstance(meta, dict) and meta.get('model') in MODEL_NAMES:
        return meta['model']
    return 'lthc_b4_velocity'


def infer_prediction(args, ckpt):
    if args.prediction != 'auto':
        return args.prediction
    ckpt_args = ckpt.get('args') or {}
    if isinstance(ckpt_args, dict):
        pred = ckpt_args.get('prediction')
        if pred in {'clean', 'velocity'}:
            return pred
    meta = ckpt.get('meta') or {}
    if isinstance(meta, dict):
        pred = meta.get('prediction')
        if pred in {'clean', 'velocity'}:
            return pred
    return 'clean'


def save_grid(images, path, nrow=10):
    images = images[: nrow * nrow].cpu().numpy().transpose(0, 2, 3, 1)
    h, w = images.shape[1], images.shape[2]
    canvas = Image.new('RGB', (nrow * w, nrow * h))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), ((i % nrow) * w, (i // nrow) * h))
    canvas.save(path)


def balanced_labels(num_samples, num_classes):
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(num_samples // num_classes)
    rem = num_samples - labels.numel()
    if rem > 0:
        labels = torch.cat([labels, torch.arange(rem, dtype=torch.long)])
    return labels


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_wandb(args, row):
    if not args.wandb:
        return
    if not args.wandb_run_id:
        raise ValueError('--wandb requires --wandb_run_id so eval logs attach to the training run')
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        name=args.wandb_run_name or None,
        resume='allow',
        config={'eval_script': 'fid_eval/evaluate.py'},
    )
    wandb.define_metric('eval/checkpoint_step')
    wandb.define_metric('eval/*', step_metric='eval/checkpoint_step')
    wandb.define_metric('eval_50k/checkpoint_step')
    wandb.define_metric('eval_50k/*', step_metric='eval_50k/checkpoint_step')
    payload = {
        'eval/checkpoint_step': int(row['checkpoint_step']),
        'eval/fid': row['fid'],
        'eval/inception_score_mean': row['inception_score_mean'],
        'eval/inception_score_std': row['inception_score_std'],
        'eval/num_samples': row['num_samples'],
        'eval/sample_steps': row['sample_steps'],
        'eval/cfg': row['cfg'],
        'eval/sampling_sec': row['sampling_sec'],
        'eval/metric_sec': row['metric_sec'],
        'eval/total_sec': row['total_sec'],
    }
    if int(row['num_samples']) >= 50000:
        payload.update({
            'eval_50k/checkpoint_step': int(row['checkpoint_step']),
            'eval_50k/fid': row['fid'],
            'eval_50k/inception_score_mean': row['inception_score_mean'],
            'eval_50k/inception_score_std': row['inception_score_std'],
            'eval_50k/cfg': row['cfg'],
            'eval_50k/sample_steps': row['sample_steps'],
            'eval_50k/total_sec': row['total_sec'],
        })
    wandb.log(payload)
    run.finish()


def compute_fid_is(sample_dir, fid_stats, cuda):
    torch_home = Path(os.environ.get('TORCH_HOME', '/data/pengrun/tongtong/dataset_imagenet_256/metrics_cache/torch_home'))
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(torch_home))
    # The reference JiT evaluation uses the vendored torch-fidelity fork,
    # which supports precomputed FID statistics through fid_statistics_file.
    # The PyPI/conda API in some environments requires input2 and will reject
    # input2=None, so prefer the vendored implementation when it is available.
    vendor_tf = Path('/data/pengrun/tongtong/Modified_DiT/modified_JiT/src/torch-fidelity')
    if vendor_tf.is_dir() and str(vendor_tf) not in sys.path:
        sys.path.insert(0, str(vendor_tf))
    import torch_fidelity

    return torch_fidelity.calculate_metrics(
        input1=str(sample_dir),
        input2=None,
        fid_statistics_file=str(fid_stats),
        cuda=cuda,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )


def main():
    args = parse_args()
    args = apply_eval_overrides(args)
    distributed, rank, world, local_rank = init_dist()
    device = torch.device(f'cuda:{local_rank}' if distributed and torch.cuda.is_available() else args.device)
    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    step = int(ckpt.get('step', -1))
    sample_dir = Path(args.sample_dir) if args.sample_dir else out / f'eval_samples_step_{step:08d}_tmp'
    if rank == 0:
        if not args.sample_dir and sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
    if rank != 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    model_name = infer_model_name(args, ckpt)
    prediction = infer_prediction(args, ckpt)
    model = build_model(model_name, attn_backend=args.attn_backend).to(device).eval()
    load_model_state(model, ckpt, args.state_key)
    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        print(f'torch_compile_mode={args.compile_mode}', flush=True)
        model = torch.compile(model, mode=args.compile_mode)

    labels_all = balanced_labels(args.num_samples, args.num_classes)
    torch.manual_seed(args.seed + rank * 1000003)

    generated_local = 0
    first_grid = None
    sample_start = time.time()
    write_sec = 0.0
    global_batch = args.batch_size * world

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for start in range(rank * args.batch_size, args.num_samples, global_batch):
            end = min(start + args.batch_size, args.num_samples)
            if start >= end:
                continue
            labels = labels_all[start:end].to(device, non_blocking=True)
            x = sample_heun(
                model,
                labels,
                image_size=256,
                noise_scale=args.noise_scale,
                steps=args.steps,
                cfg_scale=args.cfg,
                interval=(args.interval_min, args.interval_max),
                num_classes=args.num_classes,
                prediction=prediction,
            )
            u8 = to_uint8(x)
            if rank == 0 and first_grid is None:
                first_grid = u8[:100].cpu()

            t_write = time.time()
            arr = u8.cpu().numpy().transpose(0, 2, 3, 1)
            for local_i, img in enumerate(arr):
                Image.fromarray(img).save(sample_dir / f'{start + local_i:08d}.png')
            write_sec += time.time() - t_write
            generated_local += end - start
            print(f'[rank {rank}] sampled {generated_local} local images; last_global={end}/{args.num_samples}', flush=True)

    sampling_sec = time.time() - sample_start
    if rank == 0 and first_grid is not None:
        save_grid(first_grid, out / f'sample_grid_step_{step:08d}.png')

    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    if args.sample_only:
        if rank == 0:
            print('SAMPLE_ONLY_DONE=' + json.dumps({
                'checkpoint': str(args.checkpoint),
                'checkpoint_step': step,
                'sample_dir': str(sample_dir),
                'num_samples': args.num_samples,
                'sampling_sec': sampling_sec,
            }), flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    row = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_step': step,
        'state_key': args.state_key,
        'prediction': prediction,
        'num_samples': args.num_samples,
        'sampler': 'heun',
        'sample_steps': args.steps,
        'cfg': args.cfg,
        'interval_min': args.interval_min,
        'interval_max': args.interval_max,
        'noise_scale': args.noise_scale,
        'batch_size_per_rank': args.batch_size,
        'world_size': world,
        'sampling_sec': sampling_sec,
        'png_write_sec_rank0': write_sec if rank == 0 else '',
        'metric_sec': '',
        'total_sec': '',
        'fid': '',
        'inception_score_mean': '',
        'inception_score_std': '',
        'sample_dir': '' if not args.keep_samples else str(sample_dir),
        'status': 'ok',
        'error': '',
    }

    total_start = sample_start
    if rank == 0:
        metric_start = time.time()
        try:
            actual_samples = sum(1 for _ in sample_dir.glob('*.png'))
            if actual_samples != args.num_samples:
                raise RuntimeError(
                    f'generated sample count mismatch: expected {args.num_samples}, found {actual_samples} in {sample_dir}'
                )
            metrics = compute_fid_is(sample_dir, args.fid_stats, cuda=(device.type == 'cuda'))
            row['metric_sec'] = time.time() - metric_start
            row['total_sec'] = time.time() - total_start
            row['fid'] = float(metrics['frechet_inception_distance'])
            row['inception_score_mean'] = float(metrics['inception_score_mean'])
            row['inception_score_std'] = float(metrics.get('inception_score_std', 0.0))
        except Exception as exc:
            row['metric_sec'] = time.time() - metric_start
            row['total_sec'] = time.time() - total_start
            row['status'] = 'metric_failed'
            row['error'] = repr(exc)
            print(f'[eval] metric failed: {exc!r}', flush=True)

        if row['status'] == 'ok':
            log_wandb(args, row)
        append_csv(out / args.csv_file, row)
        print('EVAL_RESULT_JSON=' + json.dumps(row, ensure_ascii=True), flush=True)
        if not args.keep_samples:
            shutil.rmtree(sample_dir, ignore_errors=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
