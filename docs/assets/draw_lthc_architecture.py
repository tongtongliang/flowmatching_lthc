#!/usr/bin/env python3
"""Draw a clean forward-pass LTHC architecture diagram for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("lthc_architecture.png")


def box(ax, xy, wh, text, fc="#f7f2e8", ec="#2f2a25", fs=10, lw=1.35):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.035,rounding_size=0.045",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs, color="#1f1b16")
    return patch


def arrow(ax, p0, p1, text=None, rad=0.0, color="#3a332c", fs=8.8):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.45,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.06, text, ha="center", va="bottom", fontsize=fs, color=color)


def draw_grid(ax, x, y, w, h, nx, ny, fc, ec="#2f2a25", alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.05, alpha=alpha))
    for i in range(1, nx):
        ax.plot([x + w * i / nx, x + w * i / nx], [y, y + h], color=ec, linewidth=0.42, alpha=0.42)
    for j in range(1, ny):
        ax.plot([x, x + w], [y + h * j / ny, y + h * j / ny], color=ec, linewidth=0.42, alpha=0.42)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 170})
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    ax.text(0.35, 6.45, "Local Token Hyper-Connection (LTHC-B/4)", fontsize=18, fontweight="bold", color="#1f1b16")
    ax.text(0.35, 6.12, "Forward pass: keep high-resolution residual state; compute globally in a 16x16 workspace", fontsize=11, color="#5b5148")

    box(ax, (0.45, 4.55), (1.65, 0.78), "image\n256x256", fc="#eadada", fs=10)
    box(ax, (2.45, 4.42), (2.05, 1.05), "patch embed\n4x4 Conv + 1x1 Conv\n3 -> 128 -> 768", fc="#f7e3c4", fs=9.4)
    draw_grid(ax, 5.05, 4.2, 1.95, 1.45, 8, 8, "#cfe8d5")
    ax.text(6.03, 5.86, "X_l", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.03, 3.95, "persistent high-res residual\n64 x 64 x 768", ha="center", fontsize=8.8, color="#5b5148")

    box(ax, (7.55, 4.45), (1.85, 0.95), "shared read R\n4x4 local cell -> 1 token\nalpha[c,r]", fc="#f2d6b3", fs=8.9)
    draw_grid(ax, 10.05, 4.2, 1.55, 1.45, 4, 4, "#c9ddf0")
    ax.text(10.82, 5.86, "Z_l", ha="center", fontsize=12, fontweight="bold")
    ax.text(10.82, 3.95, "workspace\n16 x 16 x 768", ha="center", fontsize=8.8, color="#5b5148")

    arrow(ax, (2.1, 4.94), (2.45, 4.94))
    arrow(ax, (4.5, 4.94), (5.05, 4.94))
    arrow(ax, (7.0, 4.94), (7.55, 4.94), "read")
    arrow(ax, (9.4, 4.94), (10.05, 4.94))

    box(ax, (4.25, 2.15), (2.2, 1.35), "conditioning\nc = time emb + class emb\nshared AdaLN", fc="#f8edc2", fs=9.4)
    box(ax, (7.05, 1.95), (2.65, 1.75), "workspace branch F_l\nRMSNorm\nQK-norm attention + RoPE\nSwiGLU FFN", fc="#e7d7f5", fs=9.2)
    draw_grid(ax, 10.35, 2.15, 1.55, 1.35, 4, 4, "#d6d6ef")
    ax.text(11.12, 3.7, "ΔZ_l", ha="center", fontsize=12, fontweight="bold")
    ax.text(11.12, 1.9, "workspace update", ha="center", fontsize=8.8, color="#5b5148")

    arrow(ax, (10.82, 4.2), (8.35, 3.7), "workspace tokens", rad=0.1)
    arrow(ax, (6.45, 2.83), (7.05, 2.83), "AdaLN gates")
    arrow(ax, (9.7, 2.83), (10.35, 2.83))

    box(ax, (1.0, 1.1), (2.7, 1.05), "layer-specific write P_l\nbeta_l[c,r]\nworkspace update -> local cell", fc="#f2d6b3", fs=9.2)
    box(ax, (4.45, 0.85), (3.25, 1.25), "high-res residual update\nX_{l+1} = X_l + P_l ΔZ_l", fc="#d9ead3", fs=9.4)
    box(ax, (8.6, 0.98), (1.85, 1.0), "final X_L\n64x64x768", fc="#cfe8d5", fs=9.6)
    box(ax, (11.05, 0.98), (1.55, 1.0), "decoder\n4x4 RGB", fc="#eadada", fs=9.6)

    arrow(ax, (11.12, 2.15), (3.7, 1.65), "write", rad=0.22, color="#75507b")
    arrow(ax, (3.7, 1.62), (4.45, 1.48), None, color="#75507b")
    arrow(ax, (7.7, 1.48), (8.6, 1.48), None)
    arrow(ax, (10.45, 1.48), (11.05, 1.48), None)

    ax.text(0.45, 0.35, "Key separation: high-res state stores local detail; workspace branch performs global communication on 256 tokens.", fontsize=10.3, color="#5b5148")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
