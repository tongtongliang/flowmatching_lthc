#!/usr/bin/env python3
"""Compute feature-space Fréchet distance for generated ImageNet samples.

This script intentionally computes FD only. Precision/recall are useful for
offline analysis but are too expensive for an 8-checkpoint sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--fake_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_root", default="/data/pengrun/tongtong/vision_feature_extract")
    p.add_argument("--models", nargs="+", default=["dinov2_giant_reg", "siglip2_giant_opt"])
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--checkpoint_step", type=int, default=-1)
    p.add_argument("--csv_file", default="feature_fd.csv")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_feature_imports(feature_root: Path) -> None:
    sys.path.insert(0, str(feature_root))
    cache = feature_root / "cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "huggingface" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def select_images(paths: list[Path], max_images: int, seed: int) -> list[Path]:
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(paths), max_images, replace=False))
    return [paths[int(i)] for i in idx]


def stable_tag(paths: list[Path], root: Path, max_images: int, seed: int) -> str:
    h = hashlib.sha1()
    h.update(str(root.resolve()).encode())
    h.update(str(len(paths)).encode())
    h.update(str(max_images).encode())
    h.update(str(seed).encode())
    for p in paths[:128]:
        h.update(str(p.relative_to(root)).encode())
    return h.hexdigest()[:12]


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from feature_models import load_rgb
        return self.transform(load_rgb(self.paths[idx]))


def dtype_from_arg(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


@torch.no_grad()
def extract_features(model_alias: str, image_dir: Path, output_dir: Path, split: str, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    from feature_models import MODEL_SPECS, FeatureExtractor, build_transform

    spec = MODEL_SPECS[model_alias]
    all_paths = list_images(image_dir)
    if not all_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    paths = select_images(all_paths, args.max_images, args.seed)
    tag = stable_tag(paths, image_dir, args.max_images, args.seed)
    feat_dir = output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{split}_{model_alias}_n{len(paths)}_{tag}.npy"
    meta_path = feat_path.with_suffix(".json")
    if feat_path.exists() and meta_path.exists() and not args.overwrite:
        return np.load(feat_path), json.loads(meta_path.read_text())

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_dtype = dtype_from_arg(args.dtype)
    if not device.startswith("cuda"):
        torch_dtype = torch.float32

    extractor = FeatureExtractor(spec, dtype=torch_dtype).to(device).eval()
    transform = build_transform(spec)
    loader = DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    feats = []
    start = time.time()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch_dtype, enabled=device.startswith("cuda") and torch_dtype != torch.float32):
            out = extractor(batch)
        feats.append(out.cpu().numpy().astype(np.float32))
    features = np.concatenate(feats, axis=0)
    meta = {
        "split": split,
        "image_dir": str(image_dir),
        "num_images_total": len(all_paths),
        "num_images_used": len(paths),
        "model_alias": model_alias,
        "model_dir": str(spec.local_dir),
        "resolution": spec.native_resolution,
        "resize_mode": spec.resize_mode,
        "feature_shape": list(features.shape),
        "elapsed_sec": time.time() - start,
        "cache_file": str(feat_path),
    }
    np.save(feat_path, features)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return features, meta


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root)
    setup_feature_imports(feature_root)
    from feature_metrics import frechet_distance

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_alias in args.models:
        print(f"[{model_alias}] extract real", flush=True)
        real, real_meta = extract_features(model_alias, Path(args.real_dir), out, "real", args)
        print(f"[{model_alias}] extract fake", flush=True)
        fake, fake_meta = extract_features(model_alias, Path(args.fake_dir), out, f"fake_step_{args.checkpoint_step:08d}", args)
        start = time.time()
        fd = frechet_distance(real, fake)
        row = {
            "timestamp_unix": time.time(),
            "checkpoint_step": args.checkpoint_step,
            "model_alias": model_alias,
            "fd": fd,
            "real_count": int(real.shape[0]),
            "fake_count": int(fake.shape[0]),
            "resolution": real_meta["resolution"],
            "resize_mode": real_meta["resize_mode"],
            "real_dir": str(args.real_dir),
            "fake_dir": str(args.fake_dir),
            "real_cache": real_meta["cache_file"],
            "fake_cache": fake_meta["cache_file"],
            "metric_sec": time.time() - start,
            "status": "ok",
            "error": "",
        }
        append_csv(out / args.csv_file, row)
        rows.append(row)
        print("FEATURE_FD_JSON=" + json.dumps(row, ensure_ascii=True), flush=True)
    (out / "feature_fd_last.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
