#!/usr/bin/env python3
"""Multi-GPU feature-space FD for generated ImageNet samples.

The parent process launches one worker per shard. Each worker loads the feature
model on one GPU and extracts features for a disjoint image slice. The parent
concatenates shard features and computes Fréchet distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_feature_fd import setup_feature_imports, list_images, select_images, stable_tag, dtype_from_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--real_dir', required=True)
    p.add_argument('--fake_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--feature_root', default='/data/pengrun/tongtong/vision_feature_extract')
    p.add_argument('--model_alias', required=True)
    p.add_argument('--checkpoint_step', type=int, required=True)
    p.add_argument('--max_images', type=int, default=50000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--dtype', default='bf16', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    p.add_argument('--csv_file', default='feature_fd.csv')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--split', choices=['real', 'fake'], default='real')
    p.add_argument('--shard_index', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    return p.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from feature_models import load_rgb
        return self.transform(load_rgb(self.paths[idx]))


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def selected_shard_paths(root: Path, max_images: int, seed: int, shard_index: int, num_shards: int) -> list[Path]:
    paths = select_images(list_images(root), max_images, seed)
    return paths[shard_index::num_shards]


def shard_paths(args: argparse.Namespace, split: str, shard_index: int, num_shards: int) -> tuple[Path, Path]:
    root = Path(args.real_dir if split == 'real' else args.fake_dir)
    paths = selected_shard_paths(root, args.max_images, args.seed, shard_index, num_shards)
    tag = stable_tag(paths, root, args.max_images, args.seed)
    step_part = 'real' if split == 'real' else f'fake_step_{args.checkpoint_step:08d}'
    feat_dir = Path(args.output_dir) / 'features'
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f'{step_part}_{args.model_alias}_shard{shard_index:02d}of{num_shards:02d}_n{len(paths)}_{tag}.npy'
    meta_path = feat_path.with_suffix('.json')
    return feat_path, meta_path


@torch.no_grad()
def worker_main(args: argparse.Namespace) -> None:
    setup_feature_imports(Path(args.feature_root))
    from feature_models import MODEL_SPECS, FeatureExtractor, build_transform

    split_root = Path(args.real_dir if args.split == 'real' else args.fake_dir)
    paths = selected_shard_paths(split_root, args.max_images, args.seed, args.shard_index, args.num_shards)
    if not paths:
        raise RuntimeError(f'empty shard split={args.split} shard={args.shard_index}/{args.num_shards}')
    feat_path, meta_path = shard_paths(args, args.split, args.shard_index, args.num_shards)
    if feat_path.exists() and meta_path.exists() and not args.overwrite:
        print(f'[worker] cache hit {feat_path}', flush=True)
        return

    spec = MODEL_SPECS[args.model_alias]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = dtype_from_arg(args.dtype)
    if device == 'cpu':
        torch_dtype = torch.float32
    extractor = FeatureExtractor(spec, dtype=torch_dtype).to(device).eval()
    transform = build_transform(spec)
    loader = DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    feats = []
    start = time.time()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch_dtype, enabled=device == 'cuda' and torch_dtype != torch.float32):
            out = extractor(batch)
        feats.append(out.cpu().numpy().astype(np.float32))
    arr = np.concatenate(feats, axis=0)
    np.save(feat_path, arr)
    meta = {
        'split': args.split,
        'image_dir': str(split_root),
        'model_alias': args.model_alias,
        'checkpoint_step': args.checkpoint_step,
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'num_images_used': len(paths),
        'feature_shape': list(arr.shape),
        'elapsed_sec': time.time() - start,
        'cache_file': str(feat_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    print(f'[worker] wrote {feat_path} shape={arr.shape}', flush=True)


def launch_workers(args: argparse.Namespace, split: str, num_shards: int) -> None:
    procs = []
    logs = Path(args.output_dir) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for shard_index, gpu_id in enumerate(args.gpu_ids):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--worker',
            '--split', split,
            '--shard_index', str(shard_index),
            '--num_shards', str(num_shards),
            '--real_dir', args.real_dir,
            '--fake_dir', args.fake_dir,
            '--output_dir', args.output_dir,
            '--feature_root', args.feature_root,
            '--model_alias', args.model_alias,
            '--checkpoint_step', str(args.checkpoint_step),
            '--max_images', str(args.max_images),
            '--seed', str(args.seed),
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
            '--dtype', args.dtype,
        ]
        if args.overwrite:
            cmd.append('--overwrite')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        log_path = logs / f'feature_{args.model_alias}_{split}_step_{args.checkpoint_step:08d}_shard{shard_index:02d}.log'
        log = log_path.open('w')
        log.write('COMMAND: ' + ' '.join(cmd) + f' CUDA_VISIBLE_DEVICES={gpu_id}\n')
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log, log_path, cmd))
    errors = []
    for proc, log, log_path, cmd in procs:
        ret = proc.wait()
        log.close()
        if ret != 0:
            errors.append(f'worker failed code={ret} log={log_path} cmd={" ".join(cmd)}')
    if errors:
        raise RuntimeError('\n'.join(errors))


def load_split(args: argparse.Namespace, split: str, num_shards: int) -> np.ndarray:
    arrays = []
    for shard_index in range(num_shards):
        feat_path, _ = shard_paths(args, split, shard_index, num_shards)
        if not feat_path.exists():
            raise FileNotFoundError(feat_path)
        arrays.append(np.load(feat_path))
    return np.concatenate(arrays, axis=0)


def parent_main(args: argparse.Namespace) -> None:
    setup_feature_imports(Path(args.feature_root))
    from feature_metrics import frechet_distance

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_shards = len(args.gpu_ids)
    start = time.time()
    for split in ('real', 'fake'):
        launch_workers(args, split, num_shards)
    real = load_split(args, 'real', num_shards)
    fake = load_split(args, 'fake', num_shards)
    metric_start = time.time()
    fd = frechet_distance(real, fake)
    row = {
        'timestamp_unix': time.time(),
        'checkpoint_step': args.checkpoint_step,
        'model_alias': args.model_alias,
        'fd': fd,
        'real_count': int(real.shape[0]),
        'fake_count': int(fake.shape[0]),
        'resolution': '',
        'resize_mode': '',
        'real_dir': args.real_dir,
        'fake_dir': args.fake_dir,
        'real_cache': str(out / 'features'),
        'fake_cache': str(out / 'features'),
        'metric_sec': time.time() - metric_start,
        'total_sec': time.time() - start,
        'num_shards': num_shards,
        'status': 'ok',
        'error': '',
    }
    append_csv(out / args.csv_file, row)
    print('FEATURE_FD_JSON=' + json.dumps(row, ensure_ascii=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == '__main__':
    main()
