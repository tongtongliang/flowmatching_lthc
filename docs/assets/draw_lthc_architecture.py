#!/usr/bin/env python3
"""Draw the LTHC architecture diagram used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("lthc_architecture.png")


def box(ax, xy, wh, text, fc="#f7f2e8", ec="#2f2a25", fs=10, lw=1.4):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs, color="#1f1b16")
    return patch


def arrow(ax, p0, p1, text=None, rad=0.0, color="#3a332c"):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.05, text, ha="center", va="bottom", fontsize=8.5, color=color)


def draw_grid(ax, x, y, w, h, nx, ny, fc, ec="#2f2a25", alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.1, alpha=alpha))
    for i in range(1, nx):
        ax.plot([x + w * i / nx, x + w * i / nx], [y, y + h], color=ec, linewidth=0.45, alpha=0.45)
    for j in range(1, ny):
        ax.plot([x, x + w], [y + h * j / ny, y + h * j / ny], color=ec, linewidth=0.45, alpha=0.45)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 170})
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    ax.text(0.3, 7.15, "Local Token Hyper-Connection (LTHC-B/4)", fontsize=18, fontweight="bold", color="#1f1b16")
    ax.text(0.3, 6.82, "Persistent 64x64 high-resolution residual stream; 16x16 global workspace Transformer", fontsize=11, color="#5b5148")

    draw_grid(ax, 0.55, 4.15, 2.0, 1.55, 8, 8, "#cfe8d5")
    ax.text(1.55, 5.9, "x_l: high-res residual stream", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(1.55, 3.92, "64 x 64 x 768\nlocal cell = 4 x 4", ha="center", fontsize=9, color="#5b5148")

    box(ax, (3.15, 4.45), (1.85, 0.9), "shared read R\nchannel-wise 4x4 pooling\nalpha[c,r]", fc="#f2d6b3", fs=9.2)
    draw_grid(ax, 5.65, 4.15, 1.65, 1.55, 4, 4, "#c9ddf0")
    ax.text(6.47, 5.9, "z_l: workspace", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(6.47, 3.92, "16 x 16 x 768", ha="center", fontsize=9, color="#5b5148")

    box(ax, (7.95, 4.05), (2.35, 1.75), "workspace branch F_l\nRMSNorm + QK-norm Attention + RoPE\nSwiGLU FFN\nshared AdaLN gates", fc="#e7d7f5", fs=9.1)
    draw_grid(ax, 10.9, 4.15, 1.65, 1.55, 4, 4, "#d6d6ef")
    ax.text(11.72, 5.9, "dz_l", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(11.72, 3.92, "workspace update", ha="center", fontsize=9, color="#5b5148")

    arrow(ax, (2.55, 4.93), (3.15, 4.93), "read")
    arrow(ax, (5.0, 4.93), (5.65, 4.93), None)
    arrow(ax, (7.3, 4.93), (7.95, 4.93), None)
    arrow(ax, (10.3, 4.93), (10.9, 4.93), None)

    box(ax, (8.2, 2.25), (2.1, 0.85), "gamma_l[c] = sum_r alpha[c,r] beta_l[c,r]", fc="#f8edc2", fs=8.8)
    arrow(ax, (11.7, 4.15), (10.3, 3.1), "lazy update", rad=-0.16, color="#8a5a00")
    arrow(ax, (8.2, 2.68), (6.55, 4.15), "z_{l+1}=z_l+gamma_l dz_l", rad=-0.1, color="#8a5a00")

    box(ax, (3.1, 1.25), (2.6, 1.0), "layer-specific write P_l\nbeta_l[c,r]", fc="#f2d6b3", fs=9.5)
    box(ax, (6.25, 1.05), (3.25, 1.25), "fused final high-res accumulation\nx_L = x_0 + sum_l P_l dz_l\nTriton: one high-res pass", fc="#d9ead3", fs=9.3)
    box(ax, (10.15, 1.25), (1.95, 1.0), "patch decoder\n4x4 RGB", fc="#eadada", fs=9.5)

    arrow(ax, (11.72, 4.15), (4.4, 2.25), "store dz_l", rad=0.18, color="#75507b")
    arrow(ax, (4.4, 1.25), (6.25, 1.68), None)
    arrow(ax, (9.5, 1.68), (10.15, 1.75), None)

    ax.text(0.55, 0.62, "Why not full patch-4 attention?", fontsize=11, fontweight="bold", color="#1f1b16")
    ax.text(
        0.55,
        0.25,
        "LTHC keeps 4096 high-res residual tokens for local detail, but global attention/FFN runs only on 256 workspace tokens.",
        fontsize=10,
        color="#5b5148",
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
