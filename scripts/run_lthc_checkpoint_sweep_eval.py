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
    env.setdefault("TORCH_HOME", str(out / "cache" / "torch_home"))
    env.setdefault("XDG_CACHE_HOME", str(out / "cache" / "xdg"))
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

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
            ]
            if args.compile:
                cmd.extend(["--compile", "--compile_mode", args.compile_mode])
            print(f"[step {step}] FID/IS eval start", flush=True)
            run(cmd, logs / f"step_{step:08d}_fid_is.log", env)
        else:
            print(f"[step {step}] FID/IS exists; skip", flush=True)

        feat_done = set()
        if args.skip_existing and (out / "feature_fd.csv").exists():
            for row in load_rows(out / "feature_fd.csv"):
                if str(row.get("checkpoint_step")) == str(step) and row.get("status") == "ok":
                    feat_done.add(row.get("model_alias"))
        missing = [m for m in args.feature_models if m not in feat_done]
        if missing:
            if not sample_dir.exists():
                raise RuntimeError(f"sample_dir missing for feature FD: {sample_dir}")
            cmd = [
                args.python,
                str(repo / "scripts" / "evaluate_feature_fd.py"),
                "--real_dir", str(real_dir),
                "--fake_dir", str(sample_dir),
                "--output_dir", str(out),
                "--feature_root", str(args.feature_root),
                "--models", *missing,
                "--max_images", str(args.num_samples),
                "--batch_size", str(args.feature_batch_size),
                "--num_workers", str(args.feature_num_workers),
                "--checkpoint_step", str(step),
                "--csv_file", "feature_fd.csv",
            ]
            print(f"[step {step}] feature FD start models={missing}", flush=True)
            run(cmd, logs / f"step_{step:08d}_feature_fd.log", env)
        else:
            print(f"[step {step}] feature FD exists; skip", flush=True)

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
