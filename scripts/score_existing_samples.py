#!/usr/bin/env python3
"""Score an existing generated sample directory with FID/IS.

Use this when sampling succeeded but metric computation needs to be rerun. It
uses the same compute_fid_is implementation as scripts/evaluate_fid.py.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch

from evaluate_fid import append_csv, compute_fid_is


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--state_key', default='ema')
    p.add_argument('--prediction', default='velocity')
    p.add_argument('--num_samples', type=int, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--world_size', type=int, default=8)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=2.9)
    p.add_argument('--interval_min', type=float, default=0.1)
    p.add_argument('--interval_max', type=float, default=1.0)
    p.add_argument('--noise_scale', type=float, default=1.0)
    p.add_argument('--fid_stats', default='fid_stats/imagenet256_stats.npz')
    p.add_argument('--csv_file', default='fid_is.csv')
    p.add_argument('--sampling_sec', type=float, default=float('nan'))
    p.add_argument('--png_write_sec_rank0', type=float, default=float('nan'))
    p.add_argument('--keep_samples', action='store_true')
    return p.parse_args()


def infer_step(checkpoint: str) -> int:
    path = Path(checkpoint)
    try:
        ckpt = torch.load(path, map_location='cpu')
        if 'step' in ckpt:
            return int(ckpt['step'])
    except Exception:
        pass
    m = re.search(r'step_(\d+)\.pt$', path.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actual = sum(1 for _ in sample_dir.glob('*.png'))
    if actual != args.num_samples:
        raise RuntimeError(f'sample count mismatch: expected {args.num_samples}, found {actual} in {sample_dir}')
    step = infer_step(args.checkpoint)
    metric_start = time.time()
    metrics = compute_fid_is(sample_dir, args.fid_stats, cuda=torch.cuda.is_available())
    metric_sec = time.time() - metric_start
    row = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'checkpoint': str(args.checkpoint),
        'checkpoint_step': step,
        'state_key': args.state_key,
        'prediction': args.prediction,
        'num_samples': args.num_samples,
        'sampler': 'heun',
        'sample_steps': args.steps,
        'cfg': args.cfg,
        'interval_min': args.interval_min,
        'interval_max': args.interval_max,
        'noise_scale': args.noise_scale,
        'batch_size_per_rank': args.batch_size,
        'world_size': args.world_size,
        'sampling_sec': args.sampling_sec,
        'png_write_sec_rank0': args.png_write_sec_rank0,
        'metric_sec': metric_sec,
        'total_sec': args.sampling_sec + metric_sec if args.sampling_sec == args.sampling_sec else metric_sec,
        'fid': float(metrics['frechet_inception_distance']),
        'inception_score_mean': float(metrics['inception_score_mean']),
        'inception_score_std': float(metrics.get('inception_score_std', 0.0)),
        'sample_dir': str(sample_dir) if args.keep_samples else '',
        'status': 'ok',
        'error': '',
    }
    append_csv(out / args.csv_file, row)
    print('EVAL_RESULT_JSON=' + json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
