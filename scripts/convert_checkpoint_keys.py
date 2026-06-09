#!/usr/bin/env python3
"""Convert old LocalTHC read/write checkpoint keys without overwriting input."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flowmatching_lthc.checkpoint import migrate_checkpoint_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Existing checkpoint path.")
    p.add_argument("--output", required=True, help="New checkpoint path.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if src.resolve() == dst.resolve():
        raise ValueError("refusing to overwrite the input checkpoint in-place")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")

    ckpt = torch.load(src, map_location="cpu")
    for key in ("model", "ema"):
        if key in ckpt:
            ckpt[key] = migrate_checkpoint_state(ckpt[key])
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"wrote converted checkpoint: {dst}")


if __name__ == "__main__":
    main()

