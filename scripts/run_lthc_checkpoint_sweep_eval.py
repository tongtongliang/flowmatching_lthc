#!/usr/bin/env python3
"""Run a full LTHC checkpoint sweep with release-repo inference code.

The driver is intentionally conservative about disk use: generated 50k PNGs are
kept only long enough to compute FID/IS and feature-space FD, then deleted by
default. Checkpoints are copied into the release repo so the sweep is decoupled
from the original research run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


DEFAULT_SOURCE_RUN = "/data/pengrun/tongtong/ImageNet-256-JiT/runs/im256_local_thc_shared_write_fused_final12_b4_velocity_gpus4567_bs128_accum2_20260531_055810"
DEFAULT_FID_STATS = "/data/pengrun/tongtong/Modified_DiT/modified_JiT/fid_stats/jit_in256_stats.npz"
DEFAULT_REAL_ZIP = "/data/pengrun/tongtong/dataset_imagenet_256/imagenet1k_val/images.zip"
DEFAULT_FEATURE_ROOT = "/data/pengrun/tongtong/vision_feature_extract"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source_run", default=DEFAULT_SOURCE_RUN)
    p.add_argument("--output_dir", default="eval_runs/lthc_patch4_50k_recheck")
    p.add_argument("--steps", nargs="+", type=int, default=[50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000])
    p.add_argument("--state_key", default="ema", choices=["ema", "model"])
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=2.9)
    p.add_argument("--interval_min", type=float, default=0.1)
    p.add_argument("--interval_max", type=float, default=1.0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--fid_stats", default=DEFAULT_FID_STATS)
    p.add_argument("--real_zip", default=DEFAULT_REAL_ZIP)
    p.add_argument("--real_dir", default="")
    p.add_argument("--feature_root", default=DEFAULT_FEATURE_ROOT)
    p.add_argument("--feature_models", nargs="+", default=["dinov2_giant_reg", "siglip2_giant_opt"])
    p.add_argument("--nproc_per_node", type=int, default=8)
    p.add_argument("--sample_batch_per_rank", type=int, default=256)
    p.add_argument("--feature_batch_size", type=int, default=16)
    p.add_argument("--feature_num_workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--compile_mode", default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--keep_samples", action="store_true")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--rerun", dest="skip_existing", action="store_false")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"command failed with code {ret}: {' '.join(cmd)}; see {log_path}")


def run_parallel(jobs: list[tuple[str, list[str], Path, dict[str, str]]]) -> None:
    """Run independent metric jobs concurrently.

    Each job gets its own log file and environment. This is used after sampling
    so Inception, DINO, and SigLIP can occupy separate GPUs.
    """
    procs = []
    for name, cmd, log_path, job_env in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=job_env)
        procs.append((name, cmd, log_path, log, proc))
    errors = []
    for name, cmd, log_path, log, proc in procs:
        ret = proc.wait()
        log.close()
        if ret != 0:
            errors.append(f"{name} failed with code {ret}; see {log_path}; cmd={' '.join(cmd)}")
    if errors:
        raise RuntimeError("\n".join(errors))


def env_for_gpu(base_env: dict[str, str], gpu: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def visible_gpu_ids(env: dict[str, str], fallback_count: int) -> list[int]:
    """Return physical GPU ids available to child metric workers.

    Sampling uses torchrun under the parent's CUDA_VISIBLE_DEVICES list. Metric
    subprocesses, however, explicitly set CUDA_VISIBLE_DEVICES per worker, so
    they need physical ids from the parent visibility list rather than logical
    0..N ids.
    """

    raw = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        ids = [int(x) for x in raw.split(",") if x.strip()]
    else:
        ids = list(range(max(1, fallback_count)))
    if not ids:
        raise RuntimeError("No visible GPU ids available for evaluation")
    return ids


def split_feature_gpus(gpu_ids: list[int], fid_done: bool) -> dict[str, list[int]]:
    """Allocate feature extractors across the currently visible physical GPUs."""

    feature_pool = gpu_ids if fid_done else gpu_ids[1:]
    if not feature_pool:
        feature_pool = gpu_ids
    if len(feature_pool) == 1:
        return {
            "dinov2_giant_reg": feature_pool,
            "siglip2_giant_opt": feature_pool,
        }
    split = max(1, (len(feature_pool) + 1) // 2)
    return {
        "dinov2_giant_reg": feature_pool[:split],
        "siglip2_giant_opt": feature_pool[split:] or feature_pool[-1:],
    }


def feature_done_for(out: Path, step: int, model_alias: str) -> bool:
    for csv_name in ("feature_fd.csv", f"feature_fd_{model_alias}.csv"):
        for row in load_rows(out / csv_name):
            if str(row.get("checkpoint_step")) == str(step) and row.get("model_alias") == model_alias and row.get("status") == "ok":
                return True
    return False


def sync_feature_csvs(out: Path, models: list[str]) -> None:
    """Merge per-model feature CSVs into feature_fd.csv without duplicates."""
    master = out / "feature_fd.csv"
    rows = load_rows(master)
    seen = {(r.get("checkpoint_step"), r.get("model_alias")) for r in rows if r.get("status") == "ok"}
    for model_alias in models:
        for row in load_rows(out / f"feature_fd_{model_alias}.csv"):
            key = (row.get("checkpoint_step"), row.get("model_alias"))
            if row.get("status") == "ok" and key not in seen:
                rows.append(row)
                seen.add(key)
    if not rows:
        return
    fields = list(rows[0].keys())
    with master.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def copy_checkpoint(src_run: Path, out_ckpt_dir: Path, step: int) -> Path:
    src = src_run / "checkpoints" / f"step_{step:08d}.pt"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = out_ckpt_dir / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".pt.copying")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(src, tmp)
        tmp.rename(dst)
    return dst


def ensure_real_dir(args: argparse.Namespace, out: Path) -> Path:
    if args.real_dir:
        real = Path(args.real_dir)
        if not real.exists():
            raise FileNotFoundError(real)
        return real
    real = out / "real_imagenet1k_val"
    marker = real / ".extract_complete"
    if marker.exists():
        return real
    zip_path = Path(args.real_zip)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    real.mkdir(parents=True, exist_ok=True)
    print(f"[real] extracting {zip_path} -> {real}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(real)
    marker.write_text("ok\n")
    return real


def read_last_csv_row(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_combined(out: Path) -> None:
    fid_rows = load_rows(out / "fid_is.csv")
    feat_rows = load_rows(out / "feature_fd.csv")
    by_step: dict[int, dict] = {}
    for row in fid_rows:
        if row.get("status") != "ok":
            continue
        step = int(row["checkpoint_step"])
        by_step.setdefault(step, {"step": step})
        by_step[step].update({
            "fid50k": row.get("fid", ""),
            "is_mean": row.get("inception_score_mean", ""),
            "is_std": row.get("inception_score_std", ""),
            "sample_steps": row.get("sample_steps", ""),
            "cfg": row.get("cfg", ""),
            "state_key": row.get("state_key", ""),
        })
    for row in feat_rows:
        if row.get("status") != "ok":
            continue
        step = int(row["checkpoint_step"])
        alias = row["model_alias"]
        by_step.setdefault(step, {"step": step})
        by_step[step][f"{alias}_fd"] = row.get("fd", "")
    keys = ["step", "fid50k", "is_mean", "is_std", "dinov2_giant_reg_fd", "siglip2_giant_opt_fd", "sample_steps", "cfg", "state_key"]
    rows = [by_step[k] for k in sorted(by_step)]
    with (out / "combined_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def plot_curves(out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(out / "combined_metrics.csv")
    if not rows:
        return
    steps = [int(r["step"]) / 1000 for r in rows]

    def vals(name: str):
        return [float(r[name]) if r.get(name) not in {None, ""} else float("nan") for r in rows]

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)
    axes[0].plot(steps, vals("fid50k"), marker="o", color="#1f4e79")
    axes[0].set_ylabel("FID50k")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, vals("is_mean"), marker="o", color="#7f3b08")
    axes[1].set_ylabel("Inception Score")
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, vals("dinov2_giant_reg_fd"), marker="o", label="DINOv2 FD", color="#2f6f3e")
    axes[2].plot(steps, vals("siglip2_giant_opt_fd"), marker="s", label="SigLIP2 FD", color="#8b1e3f")
    axes[2].set_ylabel("Feature FD")
    axes[2].set_xlabel("training step (k)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle("LTHC-B/4 velocity checkpoint sweep, Heun50 CFG=2.9")
    fig.tight_layout()
    fig.savefig(plots / "lthc_patch4_fid_is_feature_fd.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = (repo / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    real_dir = ensure_real_dir(args, out)
    ckpt_dir = out / "checkpoints"
    logs = out / "logs"
    samples_root = out / "samples"

    env = os.environ.copy()
    env.setdefault("TMPDIR", "/data/pengrun/tongtong/.tmp")
    env.setdefault("TORCH_HOME", "/data/pengrun/tongtong/dataset_imagenet_256/metrics_cache/torch_home")
    env.setdefault("XDG_CACHE_HOME", str(out / "cache" / "xdg"))
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    metric_gpu_ids = visible_gpu_ids(env, args.nproc_per_node)

    for step in args.steps:
        ckpt = copy_checkpoint(Path(args.source_run), ckpt_dir, step)
        sample_dir = samples_root / f"step_{step:08d}"
        fid_done = False
        if args.skip_existing and (out / "fid_is.csv").exists():
            for row in load_rows(out / "fid_is.csv"):
                if str(row.get("checkpoint_step")) == str(step) and row.get("status") == "ok":
                    fid_done = True
                    break
        if not fid_done:
            existing_samples = 0
            if sample_dir.exists():
                existing_samples = sum(1 for _ in sample_dir.glob('*.png'))
            if existing_samples == args.num_samples:
                print(f"[step {step}] samples exist; metric workers will reuse them", flush=True)
            else:
                sample_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={args.nproc_per_node}",
                    str(repo / "scripts" / "evaluate_fid.py"),
                    "--checkpoint", str(ckpt),
                    "--output_dir", str(out),
                    "--state_key", args.state_key,
                    "--model", "lthc_b4_velocity",
                    "--prediction", "velocity",
                    "--num_samples", str(args.num_samples),
                    "--batch_size", str(args.sample_batch_per_rank),
                    "--steps", str(args.sample_steps),
                    "--cfg", str(args.cfg),
                    "--interval_min", str(args.interval_min),
                    "--interval_max", str(args.interval_max),
                    "--noise_scale", str(args.noise_scale),
                    "--fid_stats", str(args.fid_stats),
                    "--sample_dir", str(sample_dir),
                    "--csv_file", "fid_is.csv",
                    "--keep_samples",
                    "--sample_only",
                ]
                if args.compile:
                    cmd.extend(["--compile", "--compile_mode", args.compile_mode])
                print(f"[step {step}] sampling start", flush=True)
                run(cmd, logs / f"step_{step:08d}_sample.log", env)
        else:
            print(f"[step {step}] FID/IS exists; samples may still be needed for missing feature FD", flush=True)

        missing_feature = [m for m in args.feature_models if not feature_done_for(out, step, m)]
        if (not fid_done) or missing_feature:
            sample_count = sum(1 for _ in sample_dir.glob('*.png')) if sample_dir.exists() else 0
            if sample_count != args.num_samples:
                raise RuntimeError(f"sample count mismatch before metrics for step {step}: expected {args.num_samples}, found {sample_count}")

        jobs = []
        if not fid_done:
            fid_cmd = [
                args.python,
                str(repo / "scripts" / "score_existing_samples.py"),
                "--sample_dir", str(sample_dir),
                "--checkpoint", str(ckpt),
                "--output_dir", str(out),
                "--state_key", args.state_key,
                "--prediction", "velocity",
                "--num_samples", str(args.num_samples),
                "--batch_size", str(args.sample_batch_per_rank),
                "--world_size", str(args.nproc_per_node),
                "--steps", str(args.sample_steps),
                "--cfg", str(args.cfg),
                "--interval_min", str(args.interval_min),
                "--interval_max", str(args.interval_max),
                "--noise_scale", str(args.noise_scale),
                "--fid_stats", str(args.fid_stats),
                "--csv_file", "fid_is.csv",
                "--keep_samples",
            ]
            jobs.append(("fid_is", fid_cmd, logs / f"step_{step:08d}_score_existing.log", env_for_gpu(env, metric_gpu_ids[0])))
        feature_gpu_groups = split_feature_gpus(metric_gpu_ids, fid_done=fid_done)
        for model_alias in missing_feature:
            gpu_group = feature_gpu_groups.get(model_alias, metric_gpu_ids[:1])
            feat_cmd = [
                args.python,
                str(repo / "scripts" / "evaluate_feature_fd_sharded.py"),
                "--real_dir", str(real_dir),
                "--fake_dir", str(sample_dir),
                "--output_dir", str(out),
                "--feature_root", str(args.feature_root),
                "--model_alias", model_alias,
                "--max_images", str(args.num_samples),
                "--batch_size", str(args.feature_batch_size),
                "--num_workers", str(args.feature_num_workers),
                "--checkpoint_step", str(step),
                "--csv_file", f"feature_fd_{model_alias}.csv",
                "--gpu_ids", *[str(g) for g in gpu_group],
            ]
            jobs.append((model_alias, feat_cmd, logs / f"step_{step:08d}_feature_fd_{model_alias}.log", env))
        if jobs:
            print(f"[step {step}] metric workers start: {[name for name, _, _, _ in jobs]}", flush=True)
            run_parallel(jobs)
            sync_feature_csvs(out, args.feature_models)
        else:
            print(f"[step {step}] all metrics exist; skip", flush=True)

        if not args.keep_samples and sample_dir.exists():
            shutil.rmtree(sample_dir)
        write_combined(out)
        plot_curves(out)
        print(f"[step {step}] done", flush=True)

    write_combined(out)
    plot_curves(out)
    print(f"[done] results: {out}", flush=True)


if __name__ == "__main__":
    main()
