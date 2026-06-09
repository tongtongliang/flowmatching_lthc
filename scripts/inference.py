#!/usr/bin/env python3
"""Minimal image generation entry point for FlowMatching-LTHC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from flowmatching_lthc.checkpoint import load_model_state
from flowmatching_lthc.models import build_model
from flowmatching_lthc.sampling import sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser(description="Generate ImageNet-256 samples with FlowMatching-LTHC.")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("FLOWMATCHING_LTHC_CKPT", ""),
        help="Checkpoint path. Can also be set with FLOWMATCHING_LTHC_CKPT.",
    )
    p.add_argument("--output", default="outputs/flowmatching_lthc_grid.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--class_id", type=int, default=207, help="Single ImageNet class id used if --class_ids is not set.")
    p.add_argument("--class_ids", type=int, nargs="*", default=None, help="Optional explicit ImageNet class ids.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=2.9)
    p.add_argument("--cfg_interval", type=float, nargs=2, default=(0.1, 1.0), metavar=("LOW", "HIGH"))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--naive", action="store_true", help="Use slow PyTorch high-res recurrence; useful for CPU checks.")
    return p.parse_args()


def save_grid(images: torch.Tensor, path: Path, nrow: int = 4) -> None:
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    h, w = images.shape[1:3]
    nrow = min(nrow, len(images))
    ncol = (len(images) + nrow - 1) // nrow
    canvas = Image.new("RGB", (nrow * w, ncol * h))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), ((i % nrow) * w, (i // nrow) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("Provide --checkpoint or set FLOWMATCHING_LTHC_CKPT.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = build_model("lthc_b4_velocity", attn_backend="flash" if device.type == "cuda" else "math")
    load_model_state(model, ckpt, "ema")
    model.to(device).eval()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    if args.naive:
        class NaiveWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x, t, y):
                return self.module(x, t, y, use_lazy=False)

        model_for_sampling = NaiveWrapper(model)
    else:
        model_for_sampling = model

    if args.class_ids:
        labels = torch.tensor(args.class_ids, dtype=torch.long, device=device)
        if labels.numel() < args.batch_size:
            reps = (args.batch_size + labels.numel() - 1) // labels.numel()
            labels = labels.repeat(reps)[: args.batch_size]
        else:
            labels = labels[: args.batch_size]
    else:
        labels = torch.full((args.batch_size,), args.class_id, dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        images = sample_heun(
            model_for_sampling,
            labels,
            image_size=256,
            noise_scale=1.0,
            steps=args.steps,
            cfg_scale=args.cfg,
            interval=tuple(args.cfg_interval),
            num_classes=1000,
            prediction="velocity",
        )
    save_grid(to_uint8(images), Path(args.output))
    print(f"saved {args.output}; checkpoint={args.checkpoint}; labels={labels[:8].tolist()}")


if __name__ == "__main__":
    main()

