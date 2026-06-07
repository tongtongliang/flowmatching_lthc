import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from imaget_lthc.imagenet import build_dataset, build_loader
from imaget_lthc.models import MODEL_NAMES, build_model
from imaget_lthc.optim.muon import Muon, split_muon_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='../dataset_imagenet_256/imagenet256')
    p.add_argument('--run_dir', default='runs/lthc_experiment')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--max_steps', type=int, default=200000)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_steps', type=int, default=0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'])
    p.add_argument('--muon_aux_lr', type=float, default=0.0)
    p.add_argument('--muon_momentum', type=float, default=0.95)
    p.add_argument('--muon_ns_steps', type=int, default=5)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--P_mean', type=float, default=-0.8)
    p.add_argument('--P_std', type=float, default=0.8)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--t_eps', type=float, default=5e-2)
    p.add_argument('--label_drop_prob', type=float, default=0.1)
    p.add_argument('--prediction', default='clean', choices=['clean', 'velocity'])
    p.add_argument('--compile', action='store_true')
    p.add_argument(
        '--compile_mode',
        default='auto',
        choices=['auto', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
        help='torch.compile mode. auto keeps the historical train.py behavior.',
    )
    p.add_argument('--attn_backend', default='flash', choices=['flash','efficient','math','default'])
    p.add_argument('--model', default='jit_b16_shared_time', choices=MODEL_NAMES)
    p.add_argument('--resume', default='')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='jit-imagenet256')
    p.add_argument('--wandb_entity', default='tol011-uc-san-diego')
    p.add_argument('--wandb_group', default='baseline')
    p.add_argument('--wandb_id', default='')
    p.add_argument('--run_name', default='jit_b16_shared_time_adaln_cls_tokens')
    return p.parse_args()


def init_dist():
    if 'RANK' not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def update_ema(ema_params, model_params, decay):
    with torch.no_grad():
        for e, p in zip(ema_params, model_params):
            e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def save_ckpt(path, model, optimizer, ema_params, step, args):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': {name: ema.detach().cpu() for (name, _), ema in zip(model.named_parameters(), ema_params)},
        'step': step,
        'args': vars(args),
        'meta': {
            'model': args.model,
            'prediction': args.prediction,
            'num_classes': 1000,
            'image_size': getattr(model, 'input_size', 256),
            'patch_size': getattr(model, 'patch_size', 16),
        },
    }
    torch.save(state, path)


def read_max_logged_step(csv_path):
    if not csv_path.exists():
        return -1
    max_step = -1
    with csv_path.open(newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == 'step':
                continue
            try:
                max_step = max(max_step, int(row[0]))
            except ValueError:
                continue
    return max_step


def lr_for_step(base_lr, step, warmup_steps):
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, step / warmup_steps)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * group.get('lr_scale', 1.0)


def mark_compile_step():
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
        torch.compiler.cudagraph_mark_step_begin()


def main():
    args = parse_args()
    distributed, rank, world, local_rank = init_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.run_dir)
    if rank == 0:
        (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    torch.manual_seed(0 + rank)
    torch.backends.cudnn.benchmark = True

    dataset = build_dataset(args.data_path, split='train', image_size=256)
    loader = build_loader(dataset, args.batch_size, args.num_workers, distributed, rank, world, seed=0)
    iterator = iter(loader)

    raw_model = build_model(args.model, attn_backend=args.attn_backend).to(device)
    named_params = list(raw_model.named_parameters())
    model_params = [p for _, p in named_params]
    ema_params = [p.detach().clone() for p in model_params]
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
    else:
        muon_params, aux_params, muon_names, aux_names = split_muon_params(named_params)
        if rank == 0:
            print(f"optimizer=muon muon_params={len(muon_params)} aux_params={len(aux_params)}", flush=True)
            print(f"muon_examples={muon_names[:8]}", flush=True)
            print(f"aux_examples={aux_names[:8]}", flush=True)
        aux_lr = args.muon_aux_lr if args.muon_aux_lr > 0 else args.lr
        optimizer = torch.optim.Optimizer(model_params, defaults={})
        optimizer.param_groups = []
        optimizer.state = {}
        # Wrap two optimizers behind a minimal composite object while preserving
        # train.py's state_dict/load_state_dict/save_ckpt expectations.
        class _CompositeOptimizer:
            def __init__(self):
                self.muon = Muon(
                    muon_params,
                    lr=args.lr,
                    momentum=args.muon_momentum,
                    nesterov=True,
                    ns_steps=args.muon_ns_steps,
                    weight_decay=args.weight_decay,
                )
                self.adamw = torch.optim.AdamW(aux_params, lr=aux_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
                for group in self.muon.param_groups:
                    group['lr_scale'] = 1.0
                aux_lr_scale = aux_lr / args.lr if args.lr > 0 else 1.0
                for group in self.adamw.param_groups:
                    group['lr_scale'] = aux_lr_scale
                self.param_groups = self.muon.param_groups + self.adamw.param_groups

            def zero_grad(self, set_to_none=True):
                self.muon.zero_grad(set_to_none=set_to_none)
                self.adamw.zero_grad(set_to_none=set_to_none)

            def step(self):
                self.muon.step()
                self.adamw.step()

            def state_dict(self):
                return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

            def load_state_dict(self, state):
                if 'muon' in state and 'adamw' in state:
                    self.muon.load_state_dict(state['muon'])
                    self.adamw.load_state_dict(state['adamw'])
                else:
                    raise ValueError('cannot load non-Muon optimizer state into composite Muon optimizer')

        optimizer = _CompositeOptimizer()

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = int(ckpt['step'])
        if 'ema' in ckpt:
            ema_state = ckpt['ema']
            for name, e in zip([n for n, _ in named_params], ema_params):
                if name in ema_state:
                    e.copy_(ema_state[name].to(device))

    model = raw_model
    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        compile_mode = args.compile_mode
        if compile_mode == 'auto':
            compile_mode = 'default' if args.grad_accum > 1 else 'reduce-overhead'
        if rank == 0:
            print(f"torch_compile_mode={compile_mode}", flush=True)
        model = torch.compile(raw_model, mode=compile_mode)
    ddp_model = DDP(model, device_ids=[local_rank]) if distributed else model

    wb = None
    if rank == 0 and args.wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            id=args.wandb_id or None,
            name=args.run_name,
            config=vars(args),
            resume='allow',
        )

    csv_path = run_dir / 'logs' / 'train_metrics.csv'
    if rank == 0 and not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['step','loss','lr','samples_per_sec','data_time','step_time'])
    last_logged_step = read_max_logged_step(csv_path) if rank == 0 else -1

    last = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        current_lr = lr_for_step(args.lr, step, args.warmup_steps)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        data_time = 0.0
        loss_for_log = 0.0
        for accum_idx in range(args.grad_accum):
            data_start = time.time()
            try:
                x, y = next(iterator)
            except StopIteration:
                if distributed and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(step)
                iterator = iter(loader)
                x, y = next(iterator)
            data_time += time.time() - data_start
            x = x.to(device, non_blocking=True).float().div_(255).mul_(2).sub_(1)
            y = y.to(device, non_blocking=True).long()
            drop = torch.rand(y.shape[0], device=device) < args.label_drop_prob
            y_in = torch.where(drop, torch.full_like(y, 1000), y)
            t = torch.sigmoid(torch.randn(y.shape[0], device=device) * args.P_std + args.P_mean).view(-1,1,1,1)
            e = torch.randn_like(x) * args.noise_scale
            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(args.t_eps)
            sync_context = ddp_model.no_sync() if distributed and accum_idx < args.grad_accum - 1 else nullcontext()
            with sync_context:
                if args.compile:
                    mark_compile_step()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = ddp_model(z, t.flatten(), y_in)
                    if args.prediction == 'clean':
                        v_pred = (pred - z) / (1 - t).clamp_min(args.t_eps)
                    else:
                        v_pred = pred
                    loss = (v - v_pred).square().mean()
                loss_for_log += float(loss.detach().cpu())
                (loss / args.grad_accum).backward()
        optimizer.step()
        update_ema(ema_params, model_params, args.ema_decay)
        torch.cuda.synchronize(device)
        now = time.time()
        step_time = now - last
        last = now
        if rank == 0 and step % args.log_every == 0 and step > last_logged_step:
            sps = args.batch_size * world * args.grad_accum / max(step_time, 1e-9)
            row = [step, loss_for_log / args.grad_accum, optimizer.param_groups[0]['lr'], sps, data_time, step_time]
            with csv_path.open('a', newline='') as f:
                csv.writer(f).writerow(row)
            if wb:
                wb.log({'train/loss': row[1], 'train/lr': row[2], 'train/samples_per_sec': row[3], 'train/data_time': row[4], 'train/step_time': row[5]}, step=step)
            print(f"step={step} loss={row[1]:.5f} sps={sps:.1f} data={data_time:.3f}s step={step_time:.3f}s", flush=True)
            last_logged_step = step
        if rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
            save_ckpt(run_dir / 'checkpoints' / f'step_{step:08d}.pt', raw_model, optimizer, ema_params, step, args)
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
