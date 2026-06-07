#!/usr/bin/env python3
"""Sample from an ImageNet LTHC checkpoint.

Use ``--naive`` on CPU or for checkpoint validation. The default lazy path uses the fused
Triton final-accumulate kernel and is intended for CUDA inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from imaget_lthc.models import build_model
from imaget_lthc.sampling import sample_heun, to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/sample_grid.png")
    p.add_argument("--model", default="auto")
    p.add_argument("--state_key", default="ema", choices=["ema", "model"])
    p.add_argument("--prediction", default="auto", choices=["auto", "clean", "velocity"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--class_id", type=int, default=207, help="ImageNet class id used for all samples unless --balanced_labels is set.")
    p.add_argument("--balanced_labels", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=2.9)
    p.add_argument("--interval_min", type=float, default=0.1)
    p.add_argument("--interval_max", type=float, default=1.0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--naive", action="store_true", help="Use the PyTorch high-res recurrence instead of the lazy Triton path.")
    return p.parse_args()


def infer_model_name(args, ckpt):
    if args.model != "auto":
        return args.model
    ckpt_args = ckpt.get("args") or {}
    name = ckpt_args.get("model") if isinstance(ckpt_args, dict) else None
    if name == "local_thc_jit_shared_write_fused_final12_shared_adaln_b4":
        return name
    return "lthc_b4_velocity"


def infer_prediction(args, ckpt):
    if args.prediction != "auto":
        return args.prediction
    ckpt_args = ckpt.get("args") or {}
    pred = ckpt_args.get("prediction") if isinstance(ckpt_args, dict) else None
    return pred if pred in {"clean", "velocity"} else "velocity"


def load_state(model, ckpt, key):
    if key == "model":
        model.load_state_dict(ckpt["model"])
        return
    state = model.state_dict()
    ema = ckpt["ema"]
    for name in state:
        if name in ema:
            state[name] = ema[name]
    model.load_state_dict(state)


def save_grid(images, path, nrow=4):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    h, w = images.shape[1], images.shape[2]
    nrow = min(nrow, len(images))
    ncol = (len(images) + nrow - 1) // nrow
    canvas = Image.new("RGB", (nrow * w, ncol * h))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), ((i % nrow) * w, (i // nrow) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = infer_model_name(args, ckpt)
    prediction = infer_prediction(args, ckpt)
    model = build_model(model_name, attn_backend="flash" if device.type == "cuda" else "math")
    load_state(model, ckpt, args.state_key)
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

    if args.balanced_labels:
        labels = torch.arange(args.batch_size, device=device, dtype=torch.long) % 1000
    else:
        labels = torch.full((args.batch_size,), args.class_id, device=device, dtype=torch.long)
    with torch.no_grad():
        images = sample_heun(
            model_for_sampling,
            labels,
            image_size=256,
            noise_scale=args.noise_scale,
            steps=args.steps,
            cfg_scale=args.cfg,
            interval=(args.interval_min, args.interval_max),
            num_classes=1000,
            prediction=prediction,
        )
    out = to_uint8(images)
    save_grid(out, Path(args.output))
    print(f"saved {args.output} model={model_name} state={args.state_key} prediction={prediction} labels={labels[:8].tolist()}")


if __name__ == "__main__":
    main()
