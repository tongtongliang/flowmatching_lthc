"""Triton kernels for the fixed-shape LocalTHC B/4 read/write path.

These kernels intentionally target the current bottleneck case only:
high_grid=64, workspace_grid=16, cell_size=4, pool_groups=hidden_size.
The goal is to fuse the channel-wise local pooling/broadcast operations that
PyTorch currently lowers to many elementwise/reduce/copy kernels.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import register_autograd, triton_op, wrap_triton


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    return int(value)


@triton.jit
def _write_fwd_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16

    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        w = wx * 4 + ux
        x_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
        w_vals = tl.load(w_ptr + offs_c * 16 + pos, mask=offs_c < channels, other=0.0)
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        acc += x_vals.to(tl.float32) * w_vals[None, :].to(tl.float32)

    z_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(z_ptr + z_offsets, acc, mask=mask)


@triton.jit
def _read_fwd_kernel(
    z_ptr,
    w_ptr,
    x_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // 4096
    r = offs_m - b * 4096
    h = r // 64
    col = r - h * 64
    wy = h // 4
    wx = col // 4
    pos = (h - wy * 4) * 4 + (col - wx * 4)
    n = wy * 16 + wx

    z_offsets = (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
    w_offsets = offs_c[None, :] * 16 + pos[:, None]
    z_vals = tl.load(z_ptr + z_offsets, mask=mask, other=0.0)
    w_vals = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)
    out = z_vals.to(tl.float32) * w_vals.to(tl.float32)

    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(x_ptr + x_offsets, out, mask=mask)


@triton.jit
def _read_add_fwd_kernel(
    z_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // 4096
    r = offs_m - b * 4096
    h = r // 64
    col = r - h * 64
    wy = h // 4
    wx = col // 4
    pos = (h - wy * 4) * 4 + (col - wx * 4)
    n = wy * 16 + wx

    z_offsets = (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
    w_offsets = offs_c[None, :] * 16 + pos[:, None]
    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    z_vals = tl.load(z_ptr + z_offsets, mask=mask, other=0.0)
    w_vals = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    out = x_vals.to(tl.float32) + z_vals.to(tl.float32) * w_vals.to(tl.float32)
    tl.store(out_ptr + x_offsets, out, mask=mask)


@triton.jit
def _read_add_cell_fwd_kernel(
    z_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    offs_p = tl.arange(0, 16)
    mask_c = offs_c < channels

    b = pid_m // 256
    n = pid_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = offs_p // 4
    ux = offs_p - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux

    z = tl.load(z_ptr + pid_m * channels + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(w_ptr + offs_c[None, :] * 16 + offs_p[:, None], mask=mask_c[None, :], other=0.0).to(tl.float32)
    x_offsets = (((b * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    out = x_vals + beta * z[None, :]
    tl.store(out_ptr + x_offsets, out, mask=mask_c[None, :])


@triton.jit
def _cell_weight_grad_kernel(
    high_ptr,
    low_ptr,
    grad_w_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pos = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    w = wx * 4 + ux

    high_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
    low_offsets = (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
    high_vals = tl.load(high_ptr + high_offsets, mask=mask, other=0.0)
    low_vals = tl.load(low_ptr + low_offsets, mask=mask, other=0.0)
    prod = high_vals.to(tl.float32) * low_vals.to(tl.float32)
    grad = tl.sum(prod, axis=0)
    tl.atomic_add(grad_w_ptr + offs_c * 16 + pos, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _final_accumulate_b4_fwd_kernel(
    x0_ptr,
    dz_ptr,
    beta_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    layers: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    offs_p = tl.arange(0, 16)
    mask_c = offs_c < channels

    b = pid_m // 256
    n = pid_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = offs_p // 4
    ux = offs_p - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux

    x_offsets = (((b * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    acc = tl.load(x0_ptr + x_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)

    for layer in tl.static_range(0, layers):
        dz_offsets = ((layer * total_low + pid_m) * channels) + offs_c
        dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
        beta_offsets = (layer * channels + offs_c[None, :]) * 16 + offs_p[:, None]
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
        acc += beta_vals * dz_vals[None, :]

    tl.store(out_ptr + x_offsets, acc, mask=mask_c[None, :])


@triton.jit
def _final_accumulate_b4_grad_dz_kernel(
    grad_out_ptr,
    beta_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    layers: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16

    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        col = wx * 4 + ux
        go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
        beta_offsets = (pid_l * channels + offs_c) * 16 + pos
        go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc += go_vals * beta_vals[None, :]

    gd_offsets = ((pid_l * total_low + offs_m[:, None]) * channels) + offs_c[None, :]
    tl.store(grad_dz_ptr + gd_offsets, acc, mask=mask)


@triton.jit
def _final_accumulate_b4_grad_beta_kernel(
    grad_out_ptr,
    dz_ptr,
    grad_beta_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_lp = tl.program_id(2)
    layer = pid_lp // 16
    pos = pid_lp - layer * 16
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux

    go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = ((layer * total_low + offs_m[:, None]) * channels) + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0)
    dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0)
    grad = tl.sum(go_vals.to(tl.float32) * dz_vals.to(tl.float32), axis=0)
    beta_offsets = (layer * channels + offs_c) * 16 + pos
    tl.atomic_add(grad_beta_ptr + beta_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _final_accumulate_b4_12_fwd_kernel(
    x0_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    beta8_ptr,
    beta9_ptr,
    beta10_ptr,
    beta11_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    offs_p = tl.arange(0, 16)
    mask_c = offs_c < channels

    b = pid_m // 256
    n = pid_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = offs_p // 4
    ux = offs_p - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux
    x_offsets = (((b * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = pid_m * channels + offs_c
    beta_offsets = offs_c[None, :] * 16 + offs_p[:, None]

    acc = tl.load(x0_ptr + x_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    dz0 = tl.load(dz0_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta0 = tl.load(beta0_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta0 * dz0[None, :]
    dz1 = tl.load(dz1_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta1 = tl.load(beta1_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta1 * dz1[None, :]
    dz2 = tl.load(dz2_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta2 = tl.load(beta2_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta2 * dz2[None, :]
    dz3 = tl.load(dz3_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta3 = tl.load(beta3_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta3 * dz3[None, :]
    dz4 = tl.load(dz4_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta4 = tl.load(beta4_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta4 * dz4[None, :]
    dz5 = tl.load(dz5_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta5 = tl.load(beta5_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta5 * dz5[None, :]
    dz6 = tl.load(dz6_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta6 = tl.load(beta6_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta6 * dz6[None, :]
    dz7 = tl.load(dz7_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta7 = tl.load(beta7_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta7 * dz7[None, :]
    dz8 = tl.load(dz8_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta8 = tl.load(beta8_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta8 * dz8[None, :]
    dz9 = tl.load(dz9_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta9 = tl.load(beta9_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta9 * dz9[None, :]
    dz10 = tl.load(dz10_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta10 = tl.load(beta10_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta10 * dz10[None, :]
    dz11 = tl.load(dz11_ptr + dz_offsets, mask=mask_c, other=0.0).to(tl.float32)
    beta11 = tl.load(beta11_ptr + beta_offsets, mask=mask_c[None, :], other=0.0).to(tl.float32)
    acc += beta11 * dz11[None, :]

    tl.store(out_ptr + x_offsets, acc, mask=mask_c[None, :])


@triton.jit
def _final_accumulate_b4_4_grad_dz_kernel(
    grad_out_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16

    acc0 = tl.zeros((block_m, block_c), tl.float32)
    acc1 = tl.zeros((block_m, block_c), tl.float32)
    acc2 = tl.zeros((block_m, block_c), tl.float32)
    acc3 = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        col = wx * 4 + ux
        go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
        go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)

        beta_offsets = offs_c * 16 + pos
        b0 = tl.load(beta0_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        b1 = tl.load(beta1_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        b2 = tl.load(beta2_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        b3 = tl.load(beta3_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc0 += go_vals * b0[None, :]
        acc1 += go_vals * b1[None, :]
        acc2 += go_vals * b2[None, :]
        acc3 += go_vals * b3[None, :]

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    layer_stride = total_low * channels
    tl.store(grad_dz_ptr + out_offsets, acc0, mask=mask)
    tl.store(grad_dz_ptr + layer_stride + out_offsets, acc1, mask=mask)
    tl.store(grad_dz_ptr + 2 * layer_stride + out_offsets, acc2, mask=mask)
    tl.store(grad_dz_ptr + 3 * layer_stride + out_offsets, acc3, mask=mask)


@triton.jit
def _final_accumulate_b4_4_grad_beta_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    grad_beta_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pos = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux

    go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)

    dz0 = tl.load(dz0_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    dz1 = tl.load(dz1_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    dz2 = tl.load(dz2_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    dz3 = tl.load(dz3_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    g0 = tl.sum(go_vals * dz0, axis=0)
    g1 = tl.sum(go_vals * dz1, axis=0)
    g2 = tl.sum(go_vals * dz2, axis=0)
    g3 = tl.sum(go_vals * dz3, axis=0)

    beta_offsets = offs_c * 16 + pos
    layer_stride = channels * 16
    tl.atomic_add(grad_beta_ptr + beta_offsets, g0, sem="relaxed", mask=offs_c < channels)
    tl.atomic_add(grad_beta_ptr + layer_stride + beta_offsets, g1, sem="relaxed", mask=offs_c < channels)
    tl.atomic_add(grad_beta_ptr + 2 * layer_stride + beta_offsets, g2, sem="relaxed", mask=offs_c < channels)
    tl.atomic_add(grad_beta_ptr + 3 * layer_stride + beta_offsets, g3, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _final_accumulate_b4_12_grad_dz_ptr_kernel(
    grad_out_ptr,
    beta0_ptr,
    beta1_ptr,
    beta2_ptr,
    beta3_ptr,
    beta4_ptr,
    beta5_ptr,
    beta6_ptr,
    beta7_ptr,
    beta8_ptr,
    beta9_ptr,
    beta10_ptr,
    beta11_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    beta_ptr = tl.where(
        pid_l == 0,
        beta0_ptr,
        tl.where(
            pid_l == 1,
            beta1_ptr,
            tl.where(
                pid_l == 2,
                beta2_ptr,
                tl.where(
                    pid_l == 3,
                    beta3_ptr,
                    tl.where(
                        pid_l == 4,
                        beta4_ptr,
                        tl.where(
                            pid_l == 5,
                            beta5_ptr,
                            tl.where(
                                pid_l == 6,
                                beta6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    beta7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        beta8_ptr,
                                        tl.where(pid_l == 9, beta9_ptr, tl.where(pid_l == 10, beta10_ptr, beta11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    acc = tl.zeros((block_m, block_c), tl.float32)
    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        col = wx * 4 + ux
        go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
        beta_offsets = offs_c * 16 + pos
        go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
        beta_vals = tl.load(beta_ptr + beta_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc += go_vals * beta_vals[None, :]

    gd_offsets = ((pid_l * total_low + offs_m[:, None]) * channels) + offs_c[None, :]
    tl.store(grad_dz_ptr + gd_offsets, acc, mask=mask)


@triton.jit
def _final_accumulate_b4_12_grad_beta_ptr_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    grad_beta_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_lp = tl.program_id(0)
    pid_l = pid_lp // 16
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    pos = pid_lp - pid_l * 16
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    dz_ptr = tl.where(
        pid_l == 0,
        dz0_ptr,
        tl.where(
            pid_l == 1,
            dz1_ptr,
            tl.where(
                pid_l == 2,
                dz2_ptr,
                tl.where(
                    pid_l == 3,
                    dz3_ptr,
                    tl.where(
                        pid_l == 4,
                        dz4_ptr,
                        tl.where(
                            pid_l == 5,
                            dz5_ptr,
                            tl.where(
                                pid_l == 6,
                                dz6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    dz7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        dz8_ptr,
                                        tl.where(pid_l == 9, dz9_ptr, tl.where(pid_l == 10, dz10_ptr, dz11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    col = wx * 4 + ux
    go_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + col[:, None]) * channels) + offs_c[None, :]
    dz_offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    beta_offsets = (pid_l * channels + offs_c) * 16 + pos
    tl.atomic_add(grad_beta_ptr + beta_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _triangular_accumulate_fwd_kernel(
    base_ptr,
    dz_ptr,
    gamma_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    layers: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    acc = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    for layer in tl.static_range(0, layers):
        dz_offsets = (layer * total_low + offs_m[:, None]) * channels + offs_c[None, :]
        gamma_offsets = layer * channels + offs_c
        dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
        gamma_vals = tl.load(gamma_ptr + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc += dz_vals * gamma_vals[None, :]

    tl.store(out_ptr + offsets, acc, mask=mask)


@triton.jit
def _triangular_accumulate_grad_dz_kernel(
    grad_out_ptr,
    gamma_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    layers: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    go_offsets = offs_m[:, None] * channels + offs_c[None, :]
    gamma_offsets = pid_l * channels + offs_c
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_vals = tl.load(gamma_ptr + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    out = go_vals * gamma_vals[None, :]
    dz_offsets = (pid_l * total_low + offs_m[:, None]) * channels + offs_c[None, :]
    tl.store(grad_dz_ptr + dz_offsets, out, mask=mask)


@triton.jit
def _triangular_accumulate_grad_gamma_kernel(
    grad_out_ptr,
    dz_ptr,
    grad_gamma_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    layers: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    go_offsets = offs_m[:, None] * channels + offs_c[None, :]
    dz_offsets = (pid_l * total_low + offs_m[:, None]) * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + dz_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    gamma_offsets = pid_l * channels + offs_c
    tl.atomic_add(grad_gamma_ptr + gamma_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _all_base_b4_12_fwd_kernel(
    x_ptr,
    alpha_ptr,
    base_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_g = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    layer0 = pid_g * 4
    layer_stride = total_low * channels

    acc0 = tl.zeros((block_m, block_c), tl.float32)
    acc1 = tl.zeros((block_m, block_c), tl.float32)
    acc2 = tl.zeros((block_m, block_c), tl.float32)
    acc3 = tl.zeros((block_m, block_c), tl.float32)

    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        w = wx * 4 + ux
        x_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        alpha_offsets = ((layer0 * channels + offs_c) * 16) + pos
        a0 = tl.load(alpha_ptr + alpha_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        a1 = tl.load(alpha_ptr + alpha_offsets + channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a2 = tl.load(alpha_ptr + alpha_offsets + 2 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a3 = tl.load(alpha_ptr + alpha_offsets + 3 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc0 += x_vals * a0[None, :]
        acc1 += x_vals * a1[None, :]
        acc2 += x_vals * a2[None, :]
        acc3 += x_vals * a3[None, :]

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(base_ptr + layer0 * layer_stride + out_offsets, acc0, mask=mask)
    tl.store(base_ptr + (layer0 + 1) * layer_stride + out_offsets, acc1, mask=mask)
    tl.store(base_ptr + (layer0 + 2) * layer_stride + out_offsets, acc2, mask=mask)
    tl.store(base_ptr + (layer0 + 3) * layer_stride + out_offsets, acc3, mask=mask)


@triton.jit
def _all_base_b4_12_fwd_kernel_group12(
    x_ptr,
    alpha_ptr,
    base_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    """Compute all 12 residual-read bases while each x0 tile is in registers."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    layer_stride = total_low * channels

    acc0 = tl.zeros((block_m, block_c), tl.float32)
    acc1 = tl.zeros((block_m, block_c), tl.float32)
    acc2 = tl.zeros((block_m, block_c), tl.float32)
    acc3 = tl.zeros((block_m, block_c), tl.float32)
    acc4 = tl.zeros((block_m, block_c), tl.float32)
    acc5 = tl.zeros((block_m, block_c), tl.float32)
    acc6 = tl.zeros((block_m, block_c), tl.float32)
    acc7 = tl.zeros((block_m, block_c), tl.float32)
    acc8 = tl.zeros((block_m, block_c), tl.float32)
    acc9 = tl.zeros((block_m, block_c), tl.float32)
    acc10 = tl.zeros((block_m, block_c), tl.float32)
    acc11 = tl.zeros((block_m, block_c), tl.float32)

    for pos in tl.static_range(0, 16):
        uy = pos // 4
        ux = pos - uy * 4
        h = wy * 4 + uy
        w = wx * 4 + ux
        x_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        alpha_offsets = (offs_c * 16) + pos
        a0 = tl.load(alpha_ptr + alpha_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        a1 = tl.load(alpha_ptr + alpha_offsets + channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a2 = tl.load(alpha_ptr + alpha_offsets + 2 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a3 = tl.load(alpha_ptr + alpha_offsets + 3 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a4 = tl.load(alpha_ptr + alpha_offsets + 4 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a5 = tl.load(alpha_ptr + alpha_offsets + 5 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a6 = tl.load(alpha_ptr + alpha_offsets + 6 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a7 = tl.load(alpha_ptr + alpha_offsets + 7 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a8 = tl.load(alpha_ptr + alpha_offsets + 8 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a9 = tl.load(alpha_ptr + alpha_offsets + 9 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a10 = tl.load(alpha_ptr + alpha_offsets + 10 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        a11 = tl.load(alpha_ptr + alpha_offsets + 11 * channels * 16, mask=offs_c < channels, other=0.0).to(tl.float32)
        acc0 += x_vals * a0[None, :]
        acc1 += x_vals * a1[None, :]
        acc2 += x_vals * a2[None, :]
        acc3 += x_vals * a3[None, :]
        acc4 += x_vals * a4[None, :]
        acc5 += x_vals * a5[None, :]
        acc6 += x_vals * a6[None, :]
        acc7 += x_vals * a7[None, :]
        acc8 += x_vals * a8[None, :]
        acc9 += x_vals * a9[None, :]
        acc10 += x_vals * a10[None, :]
        acc11 += x_vals * a11[None, :]

    out_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(base_ptr + out_offsets, acc0, mask=mask)
    tl.store(base_ptr + layer_stride + out_offsets, acc1, mask=mask)
    tl.store(base_ptr + 2 * layer_stride + out_offsets, acc2, mask=mask)
    tl.store(base_ptr + 3 * layer_stride + out_offsets, acc3, mask=mask)
    tl.store(base_ptr + 4 * layer_stride + out_offsets, acc4, mask=mask)
    tl.store(base_ptr + 5 * layer_stride + out_offsets, acc5, mask=mask)
    tl.store(base_ptr + 6 * layer_stride + out_offsets, acc6, mask=mask)
    tl.store(base_ptr + 7 * layer_stride + out_offsets, acc7, mask=mask)
    tl.store(base_ptr + 8 * layer_stride + out_offsets, acc8, mask=mask)
    tl.store(base_ptr + 9 * layer_stride + out_offsets, acc9, mask=mask)
    tl.store(base_ptr + 10 * layer_stride + out_offsets, acc10, mask=mask)
    tl.store(base_ptr + 11 * layer_stride + out_offsets, acc11, mask=mask)


@triton.jit
def _all_base_b4_12_grad_x_kernel(
    grad_base_ptr,
    alpha_ptr,
    grad_x_ptr,
    total_high: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_high) & (offs_c[None, :] < channels)

    b = offs_m // 4096
    r = offs_m - b * 4096
    h = r // 64
    col = r - h * 64
    wy = h // 4
    wx = col // 4
    pos = (h - wy * 4) * 4 + (col - wx * 4)
    n = wy * 16 + wx
    total_low = total_high // 16
    layer_stride = total_low * channels

    acc = tl.zeros((block_m, block_c), tl.float32)
    for layer in tl.static_range(0, 12):
        gb_offsets = layer * layer_stride + (b[:, None] * 256 + n[:, None]) * channels + offs_c[None, :]
        alpha_offsets = (layer * channels + offs_c[None, :]) * 16 + pos[:, None]
        gb_vals = tl.load(grad_base_ptr + gb_offsets, mask=mask, other=0.0).to(tl.float32)
        alpha_vals = tl.load(alpha_ptr + alpha_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += gb_vals * alpha_vals

    x_offsets = offs_m[:, None] * channels + offs_c[None, :]
    tl.store(grad_x_ptr + x_offsets, acc, mask=mask)


@triton.jit
def _all_base_b4_12_grad_alpha_kernel(
    x_ptr,
    grad_base_ptr,
    grad_alpha_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_lp = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_l = pid_lp // 16
    pos = pid_lp - pid_l * 16
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)

    b = offs_m // 256
    n = offs_m - b * 256
    wy = n // 16
    wx = n - wy * 16
    uy = pos // 4
    ux = pos - uy * 4
    h = wy * 4 + uy
    w = wx * 4 + ux
    x_offsets = (((b[:, None] * 64 + h[:, None]) * 64 + w[:, None]) * channels) + offs_c[None, :]
    gb_offsets = (pid_l * total_low + offs_m[:, None]) * channels + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    gb_vals = tl.load(grad_base_ptr + gb_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(x_vals * gb_vals, axis=0)
    alpha_offsets = (pid_l * channels + offs_c) * 16 + pos
    tl.atomic_add(grad_alpha_ptr + alpha_offsets, grad, sem="relaxed", mask=offs_c < channels)


@triton.jit
def _triangular_accumulate_b4_12_ptr_fwd_kernel(
    base_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    gamma_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    gamma_offsets = offs_c
    acc = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    g0 = tl.load(gamma_ptr + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d0 = tl.load(dz0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d0 * g0[None, :]
    g1 = tl.load(gamma_ptr + channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d1 = tl.load(dz1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d1 * g1[None, :]
    g2 = tl.load(gamma_ptr + 2 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d2 = tl.load(dz2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d2 * g2[None, :]
    g3 = tl.load(gamma_ptr + 3 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d3 = tl.load(dz3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d3 * g3[None, :]
    g4 = tl.load(gamma_ptr + 4 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d4 = tl.load(dz4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d4 * g4[None, :]
    g5 = tl.load(gamma_ptr + 5 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d5 = tl.load(dz5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d5 * g5[None, :]
    g6 = tl.load(gamma_ptr + 6 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d6 = tl.load(dz6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d6 * g6[None, :]
    g7 = tl.load(gamma_ptr + 7 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d7 = tl.load(dz7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d7 * g7[None, :]
    g8 = tl.load(gamma_ptr + 8 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d8 = tl.load(dz8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d8 * g8[None, :]
    g9 = tl.load(gamma_ptr + 9 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d9 = tl.load(dz9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d9 * g9[None, :]
    g10 = tl.load(gamma_ptr + 10 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d10 = tl.load(dz10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d10 * g10[None, :]
    g11 = tl.load(gamma_ptr + 11 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
    d11 = tl.load(dz11_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc += d11 * g11[None, :]

    tl.store(out_ptr + offsets, acc, mask=mask)


@triton.jit
def _triangular_accumulate_b4_12_ptr_active_fwd_kernel(
    base_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    gamma_ptr,
    out_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
    active: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    gamma_offsets = offs_c
    acc = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if active >= 1:
        g0 = tl.load(gamma_ptr + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d0 = tl.load(dz0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d0 * g0[None, :]
    if active >= 2:
        g1 = tl.load(gamma_ptr + channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d1 = tl.load(dz1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d1 * g1[None, :]
    if active >= 3:
        g2 = tl.load(gamma_ptr + 2 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d2 = tl.load(dz2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d2 * g2[None, :]
    if active >= 4:
        g3 = tl.load(gamma_ptr + 3 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d3 = tl.load(dz3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d3 * g3[None, :]
    if active >= 5:
        g4 = tl.load(gamma_ptr + 4 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d4 = tl.load(dz4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d4 * g4[None, :]
    if active >= 6:
        g5 = tl.load(gamma_ptr + 5 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d5 = tl.load(dz5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d5 * g5[None, :]
    if active >= 7:
        g6 = tl.load(gamma_ptr + 6 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d6 = tl.load(dz6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d6 * g6[None, :]
    if active >= 8:
        g7 = tl.load(gamma_ptr + 7 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d7 = tl.load(dz7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d7 * g7[None, :]
    if active >= 9:
        g8 = tl.load(gamma_ptr + 8 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d8 = tl.load(dz8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d8 * g8[None, :]
    if active >= 10:
        g9 = tl.load(gamma_ptr + 9 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d9 = tl.load(dz9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d9 * g9[None, :]
    if active >= 11:
        g10 = tl.load(gamma_ptr + 10 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d10 = tl.load(dz10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d10 * g10[None, :]
    if active >= 12:
        g11 = tl.load(gamma_ptr + 11 * channels + gamma_offsets, mask=offs_c < channels, other=0.0).to(tl.float32)
        d11 = tl.load(dz11_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += d11 * g11[None, :]

    tl.store(out_ptr + offsets, acc, mask=mask)


@triton.jit
def _triangular_accumulate_b4_12_ptr_grad_dz_kernel(
    grad_out_ptr,
    gamma_ptr,
    grad_dz_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    go_vals = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_vals = tl.load(gamma_ptr + pid_l * channels + offs_c, mask=offs_c < channels, other=0.0).to(tl.float32)
    out = go_vals * gamma_vals[None, :]
    layer_stride = total_low * channels
    tl.store(grad_dz_ptr + pid_l * layer_stride + offsets, out, mask=mask)


@triton.jit
def _triangular_accumulate_b4_12_ptr_grad_gamma_kernel(
    grad_out_ptr,
    dz0_ptr,
    dz1_ptr,
    dz2_ptr,
    dz3_ptr,
    dz4_ptr,
    dz5_ptr,
    dz6_ptr,
    dz7_ptr,
    dz8_ptr,
    dz9_ptr,
    dz10_ptr,
    dz11_ptr,
    grad_gamma_ptr,
    total_low: tl.constexpr,
    channels: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_c = pid_c * block_c + tl.arange(0, block_c)
    mask = (offs_m[:, None] < total_low) & (offs_c[None, :] < channels)
    offsets = offs_m[:, None] * channels + offs_c[None, :]
    dz_ptr = tl.where(
        pid_l == 0,
        dz0_ptr,
        tl.where(
            pid_l == 1,
            dz1_ptr,
            tl.where(
                pid_l == 2,
                dz2_ptr,
                tl.where(
                    pid_l == 3,
                    dz3_ptr,
                    tl.where(
                        pid_l == 4,
                        dz4_ptr,
                        tl.where(
                            pid_l == 5,
                            dz5_ptr,
                            tl.where(
                                pid_l == 6,
                                dz6_ptr,
                                tl.where(
                                    pid_l == 7,
                                    dz7_ptr,
                                    tl.where(
                                        pid_l == 8,
                                        dz8_ptr,
                                        tl.where(pid_l == 9, dz9_ptr, tl.where(pid_l == 10, dz10_ptr, dz11_ptr)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    go_vals = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dz_vals = tl.load(dz_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.sum(go_vals * dz_vals, axis=0)
    tl.atomic_add(grad_gamma_ptr + pid_l * channels + offs_c, grad, sem="relaxed", mask=offs_c < channels)


class _LocalWriteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b, h, w, c = x.shape
        if h != 64 or w != 64:
            raise ValueError(f"Triton LocalTHC write expects 64x64 high grid, got {h}x{w}")
        if weight.shape != (c, 16):
            raise ValueError(f"Triton LocalTHC write expects weight [{c},16], got {tuple(weight.shape)}")
        z = torch.empty((b, 256, c), device=x.device, dtype=x.dtype)
        block_m = 16
        block_c = 64
        grid = (triton.cdiv(b * 256, block_m), triton.cdiv(c, block_c))
        _write_fwd_kernel[grid](x, weight, z, b * 256, c, block_m, block_c)
        ctx.save_for_backward(x, weight)
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        x, weight = ctx.saved_tensors
        if not grad_z.is_contiguous():
            grad_z = grad_z.contiguous()
        b, _, c = grad_z.shape
        grad_x = torch.empty_like(x)
        block_m = 16
        block_c = 64
        read_grid = (triton.cdiv(b * 4096, block_m), triton.cdiv(c, block_c))
        _read_fwd_kernel[read_grid](grad_z, weight, grad_x, b * 4096, c, block_m, block_c)

        grad_w = torch.zeros((c, 16), device=x.device, dtype=torch.float32)
        grad_grid = (triton.cdiv(b * 256, 64), triton.cdiv(c, block_c), 16)
        _cell_weight_grad_kernel[grad_grid](x, grad_z, grad_w, b * 256, c, 64, block_c)
        return grad_x, grad_w.to(weight.dtype)


class _LocalWriteNoWeightGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b, h, w, c = x.shape
        if h != 64 or w != 64:
            raise ValueError(f"Triton LocalTHC write expects 64x64 high grid, got {h}x{w}")
        z = torch.empty((b, 256, c), device=x.device, dtype=x.dtype)
        block_m = 16
        block_c = 64
        grid = (triton.cdiv(b * 256, block_m), triton.cdiv(c, block_c))
        _write_fwd_kernel[grid](x, weight, z, b * 256, c, block_m, block_c)
        ctx.save_for_backward(weight)
        ctx.batch = b
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        (weight,) = ctx.saved_tensors
        if not grad_z.is_contiguous():
            grad_z = grad_z.contiguous()
        b, _, c = grad_z.shape
        grad_x = torch.empty((b, 64, 64, c), device=grad_z.device, dtype=grad_z.dtype)
        block_m = 16
        block_c = 64
        grid = (triton.cdiv(b * 4096, block_m), triton.cdiv(c, block_c))
        _read_fwd_kernel[grid](grad_z, weight, grad_x, b * 4096, c, block_m, block_c)
        return grad_x, None


class _LocalReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not z.is_contiguous():
            z = z.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b, n, c = z.shape
        if n != 256:
            raise ValueError(f"Triton LocalTHC read expects 256 workspace tokens, got {n}")
        if weight.shape != (c, 16):
            raise ValueError(f"Triton LocalTHC read expects weight [{c},16], got {tuple(weight.shape)}")
        x = torch.empty((b, 64, 64, c), device=z.device, dtype=z.dtype)
        block_m = 16
        block_c = 64
        grid = (triton.cdiv(b * 4096, block_m), triton.cdiv(c, block_c))
        _read_fwd_kernel[grid](z, weight, x, b * 4096, c, block_m, block_c)
        ctx.save_for_backward(z, weight)
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        z, weight = ctx.saved_tensors
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()
        b, _, c = z.shape
        grad_z = torch.empty_like(z)
        block_m = 16
        block_c = 64
        write_grid = (triton.cdiv(b * 256, block_m), triton.cdiv(c, block_c))
        _write_fwd_kernel[write_grid](grad_x, weight, grad_z, b * 256, c, block_m, block_c)

        grad_w = torch.zeros((c, 16), device=z.device, dtype=torch.float32)
        grad_grid = (triton.cdiv(b * 256, 64), triton.cdiv(c, block_c), 16)
        _cell_weight_grad_kernel[grad_grid](grad_x, z, grad_w, b * 256, c, 64, block_c)
        return grad_z, grad_w.to(weight.dtype)


class _LocalReadAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not z.is_contiguous():
            z = z.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if not x.is_contiguous():
            x = x.contiguous()
        b, n, c = z.shape
        if n != 256:
            raise ValueError(f"Triton LocalTHC read_add expects 256 workspace tokens, got {n}")
        out = torch.empty_like(x)
        block_c = 64
        grid = (b * 256, triton.cdiv(c, block_c))
        _read_add_cell_fwd_kernel[grid](z, weight, x, out, b * 256, c, block_c)
        ctx.save_for_backward(z, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z, weight = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        b, _, c = z.shape
        grad_z = torch.empty_like(z)
        block_m = 16
        block_c = 64
        write_grid = (triton.cdiv(b * 256, block_m), triton.cdiv(c, block_c))
        _write_fwd_kernel[write_grid](grad_out, weight, grad_z, b * 256, c, block_m, block_c)

        grad_w = torch.zeros((c, 16), device=z.device, dtype=torch.float32)
        grad_grid = (triton.cdiv(b * 256, 64), triton.cdiv(c, block_c), 16)
        _cell_weight_grad_kernel[grad_grid](grad_out, z, grad_w, b * 256, c, 64, block_c)
        return grad_z, grad_w.to(weight.dtype), grad_out


class _LocalReadAddNoWeightGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not z.is_contiguous():
            z = z.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if not x.is_contiguous():
            x = x.contiguous()
        b, n, c = z.shape
        if n != 256:
            raise ValueError(f"Triton LocalTHC read_add expects 256 workspace tokens, got {n}")
        out = torch.empty_like(x)
        block_c = 64
        grid = (b * 256, triton.cdiv(c, block_c))
        _read_add_cell_fwd_kernel[grid](z, weight, x, out, b * 256, c, block_c)
        ctx.save_for_backward(weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (weight,) = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        b, h, w, c = grad_out.shape
        grad_z = torch.empty((b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
        block_m = 16
        block_c = 64
        write_grid = (triton.cdiv(b * 256, block_m), triton.cdiv(c, block_c))
        _write_fwd_kernel[write_grid](grad_out, weight, grad_z, b * 256, c, block_m, block_c)
        return grad_z, None, grad_out


def local_write(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _LocalWriteFunction.apply(x, weight)


def local_read(z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _LocalReadFunction.apply(z, weight)


def local_write_no_weight_grad(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _LocalWriteNoWeightGradFunction.apply(x, weight)


def local_read_add(z: torch.Tensor, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _LocalReadAddFunction.apply(z, weight, x)


def local_read_add_no_weight_grad(z: torch.Tensor, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _LocalReadAddNoWeightGradFunction.apply(z, weight, x)


@triton_op("jit_thc::final_accumulate_b4_traceable", mutates_args=())
def final_accumulate_b4_traceable(x0: Tensor, dz_stack: Tensor, beta_stack: Tensor) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    layers = dz_stack.shape[0]
    out = torch.empty_like(x0)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_fwd_kernel)[grid](
        x0,
        dz_stack,
        beta_stack,
        out,
        b * 256,
        c,
        layers,
        block_c=64,
    )
    return out


@triton_op("jit_thc::final_accumulate_b4_grad_dz", mutates_args=())
def final_accumulate_b4_grad_dz_traceable(grad_out: Tensor, beta_stack: Tensor) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    layers = beta_stack.shape[0]
    grad_dz = torch.empty((layers, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

    def grid(meta):
        return (layers, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_grad_dz_kernel)[grid](
        grad_out,
        beta_stack,
        grad_dz,
        b * 256,
        c,
        layers,
        block_m=16,
        block_c=64,
    )
    return grad_dz


@triton_op("jit_thc::final_accumulate_b4_grad_beta", mutates_args=())
def final_accumulate_b4_grad_beta_traceable(grad_out: Tensor, dz_stack: Tensor, beta_stack: Tensor) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    layers = beta_stack.shape[0]
    grad_beta = torch.empty((layers, c, 16), device=grad_out.device, dtype=torch.float32)
    grad_beta.zero_()

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]), layers * 16)

    wrap_triton(_final_accumulate_b4_grad_beta_kernel)[grid](
        grad_out,
        dz_stack,
        grad_beta,
        b * 256,
        c,
        block_m=64,
        block_c=64,
    )
    return grad_beta


def _setup_final_accumulate_ctx(ctx, inputs, output):
    _, dz_stack, beta_stack = inputs
    ctx.save_for_backward(dz_stack, beta_stack)


def _backward_final_accumulate(ctx, grad_out):
    dz_stack, beta_stack = ctx.saved_tensors
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_x0 = grad_out
    grad_dz = final_accumulate_b4_grad_dz_traceable(grad_out, beta_stack)
    grad_beta = final_accumulate_b4_grad_beta_traceable(grad_out, dz_stack, beta_stack)
    return grad_x0, grad_dz, grad_beta.to(dtype=beta_stack.dtype)


register_autograd(
    final_accumulate_b4_traceable,
    _backward_final_accumulate,
    setup_context=_setup_final_accumulate_ctx,
)


@triton_op("jit_thc::final_accumulate_b4_12_traceable", mutates_args=())
def final_accumulate_b4_12_traceable(
    x0: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    out = torch.empty_like(x0)
    block_c = _env_int("JIT_THC_FF12_FWD_BLOCK_C", 64)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_12_fwd_kernel)[grid](
        x0,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        out,
        b * 256,
        c,
        block_c=block_c,
    )
    return out


@triton_op("jit_thc::final_accumulate_b4_4_grad_dz", mutates_args=())
def final_accumulate_b4_4_grad_dz_traceable(
    grad_out: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_dz = torch.empty((4, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_4_grad_dz_kernel)[grid](
        grad_out,
        beta0,
        beta1,
        beta2,
        beta3,
        grad_dz,
        b * 256,
        c,
        block_m=16,
        block_c=64,
    )
    return grad_dz


@triton_op("jit_thc::final_accumulate_b4_4_grad_beta", mutates_args=())
def final_accumulate_b4_4_grad_beta_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_beta = torch.empty((4, c, 16), device=grad_out.device, dtype=torch.float32)
    grad_beta.zero_()

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]), 16)

    wrap_triton(_final_accumulate_b4_4_grad_beta_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        grad_beta,
        b * 256,
        c,
        block_m=64,
        block_c=64,
    )
    return grad_beta


@triton_op("jit_thc::final_accumulate_b4_12_grad_dz_ptr", mutates_args=())
def final_accumulate_b4_12_grad_dz_ptr_traceable(
    grad_out: Tensor,
    beta0: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta3: Tensor,
    beta4: Tensor,
    beta5: Tensor,
    beta6: Tensor,
    beta7: Tensor,
    beta8: Tensor,
    beta9: Tensor,
    beta10: Tensor,
    beta11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_dz = torch.empty((12, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)
    block_m = _env_int("JIT_THC_FF12_GDZ_BLOCK_M", 16)
    block_c = _env_int("JIT_THC_FF12_GDZ_BLOCK_C", 64)

    def grid(meta):
        return (12, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_12_grad_dz_ptr_kernel)[grid](
        grad_out,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        beta5,
        beta6,
        beta7,
        beta8,
        beta9,
        beta10,
        beta11,
        grad_dz,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_dz


@triton_op("jit_thc::final_accumulate_b4_12_grad_beta_ptr", mutates_args=())
def final_accumulate_b4_12_grad_beta_ptr_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[3]
    grad_beta = torch.empty((12, c, 16), device=grad_out.device, dtype=torch.float32)
    grad_beta.zero_()
    block_m = _env_int("JIT_THC_FF12_GBETA_BLOCK_M", 64)
    block_c = _env_int("JIT_THC_FF12_GBETA_BLOCK_C", 64)

    def grid(meta):
        return (12 * 16, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_final_accumulate_b4_12_grad_beta_ptr_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        grad_beta,
        b * 256,
        c,
        block_m=block_m,
        block_c=block_c,
    )
    return grad_beta


def _setup_final_accumulate_12_ctx(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _backward_final_accumulate_12(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:12]
    betas = saved[12:]
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    grad_dz_stack = final_accumulate_b4_12_grad_dz_ptr_traceable(grad_out, *betas)
    grad_beta_stack = final_accumulate_b4_12_grad_beta_ptr_traceable(grad_out, *dzs)
    grad_dzs = tuple(grad_dz_stack.unbind(dim=0))
    grad_betas = tuple(g.to(dtype=b.dtype) for g, b in zip(grad_beta_stack.unbind(dim=0), betas, strict=True))
    return (grad_out, *grad_dzs, *grad_betas)


register_autograd(
    final_accumulate_b4_12_traceable,
    _backward_final_accumulate_12,
    setup_context=_setup_final_accumulate_12_ctx,
)


@triton_op("jit_thc::triangular_accumulate_traceable", mutates_args=())
def triangular_accumulate_traceable(base: Tensor, dz_stack: Tensor, gamma: Tensor) -> Tensor:
    b = base.shape[0]
    c = base.shape[2]
    layers = dz_stack.shape[0]
    out = torch.empty_like(base)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_fwd_kernel)[grid](
        base,
        dz_stack,
        gamma,
        out,
        b * 256,
        c,
        layers,
        block_m=16,
        block_c=64,
    )
    return out


@triton_op("jit_thc::triangular_accumulate_grad_dz", mutates_args=())
def triangular_accumulate_grad_dz_traceable(grad_out: Tensor, gamma: Tensor) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    layers = gamma.shape[0]
    grad_dz = torch.empty((layers, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

    def grid(meta):
        return (layers, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_grad_dz_kernel)[grid](
        grad_out,
        gamma,
        grad_dz,
        b * 256,
        c,
        layers,
        block_m=16,
        block_c=64,
    )
    return grad_dz


@triton_op("jit_thc::triangular_accumulate_grad_gamma", mutates_args=())
def triangular_accumulate_grad_gamma_traceable(grad_out: Tensor, dz_stack: Tensor, gamma: Tensor) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    layers = gamma.shape[0]
    grad_gamma = torch.empty((layers, c), device=grad_out.device, dtype=torch.float32)
    grad_gamma.zero_()

    def grid(meta):
        return (layers, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_grad_gamma_kernel)[grid](
        grad_out,
        dz_stack,
        grad_gamma,
        b * 256,
        c,
        layers,
        block_m=64,
        block_c=64,
    )
    return grad_gamma


def _setup_triangular_accumulate_ctx(ctx, inputs, output):
    _, dz_stack, gamma = inputs
    ctx.save_for_backward(dz_stack, gamma)


def _backward_triangular_accumulate(ctx, grad_out):
    dz_stack, gamma = ctx.saved_tensors
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_base = grad_out
    grad_dz = triangular_accumulate_grad_dz_traceable(grad_out, gamma)
    grad_gamma = triangular_accumulate_grad_gamma_traceable(grad_out, dz_stack, gamma)
    return grad_base, grad_dz, grad_gamma.to(dtype=gamma.dtype)


register_autograd(
    triangular_accumulate_traceable,
    _backward_triangular_accumulate,
    setup_context=_setup_triangular_accumulate_ctx,
)


@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_traceable", mutates_args=())
def triangular_accumulate_b4_12_ptr_traceable(
    base: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    gamma: Tensor,
) -> Tensor:
    b = base.shape[0]
    c = base.shape[2]
    if base.shape[1] != 256 or gamma.shape != (12, c):
        raise ValueError(f"triangular_accumulate_b4_12 expects base [B,256,C] and gamma [12,C], got {tuple(base.shape)} and {tuple(gamma.shape)}")
    for idx, dz in enumerate((dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9, dz10, dz11)):
        if dz.shape != base.shape:
            raise ValueError(f"triangular_accumulate_b4_12 dz{idx} shape {tuple(dz.shape)} does not match base {tuple(base.shape)}")
    out = torch.empty_like(base)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_fwd_kernel)[grid](
        base,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        gamma,
        out,
        b * 256,
        c,
        block_m=16,
        block_c=64,
    )
    return out


@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_grad_dz", mutates_args=())
def triangular_accumulate_b4_12_ptr_grad_dz_traceable(grad_out: Tensor, gamma: Tensor) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    grad_dz = torch.empty((12, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

    def grid(meta):
        return (12, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_grad_dz_kernel)[grid](
        grad_out,
        gamma,
        grad_dz,
        b * 256,
        c,
        block_m=16,
        block_c=64,
    )
    return grad_dz


@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_grad_gamma", mutates_args=())
def triangular_accumulate_b4_12_ptr_grad_gamma_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    gamma: Tensor,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    grad_gamma = torch.empty((12, c), device=grad_out.device, dtype=torch.float32)
    grad_gamma.zero_()

    def grid(meta):
        return (12, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_grad_gamma_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        grad_gamma,
        b * 256,
        c,
        block_m=64,
        block_c=64,
    )
    return grad_gamma


def _setup_triangular_accumulate_12_ptr_ctx(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _backward_triangular_accumulate_12_ptr(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:12]
    gamma = saved[12]
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_base = grad_out
    grad_dz_stack = triangular_accumulate_b4_12_ptr_grad_dz_traceable(grad_out, gamma)
    grad_gamma = triangular_accumulate_b4_12_ptr_grad_gamma_traceable(grad_out, *dzs, gamma)
    grad_dzs = tuple(grad_dz_stack.unbind(dim=0))
    return (grad_base, *grad_dzs, grad_gamma.to(dtype=gamma.dtype))


register_autograd(
    triangular_accumulate_b4_12_ptr_traceable,
    _backward_triangular_accumulate_12_ptr,
    setup_context=_setup_triangular_accumulate_12_ptr_ctx,
)


def _make_triangular_accumulate_b4_12_ptr_active_traceable(active: int):
    if not 1 <= active <= 11:
        raise ValueError(f"active must be in [1, 11], got {active}")

    @triton_op(f"jit_thc::triangular_accumulate_b4_12_ptr_active{active}", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_traceable(
        base: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        dz8: Tensor,
        dz9: Tensor,
        dz10: Tensor,
        dz11: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = base.shape[0]
        c = base.shape[2]
        if base.shape[1] != 256 or gamma.shape != (12, c):
            raise ValueError(
                f"triangular_accumulate_b4_12_active expects base [B,256,C] and gamma [12,C], "
                f"got {tuple(base.shape)} and {tuple(gamma.shape)}"
            )
        for idx, dz in enumerate((dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9, dz10, dz11)):
            if dz.shape != base.shape:
                raise ValueError(
                    f"triangular_accumulate_b4_12_active dz{idx} shape {tuple(dz.shape)} "
                    f"does not match base {tuple(base.shape)}"
                )
        out = torch.empty_like(base)

        def grid(meta):
            return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_active_fwd_kernel)[grid](
            base,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz8,
            dz9,
            dz10,
            dz11,
            gamma,
            out,
            b * 256,
            c,
            block_m=16,
            block_c=64,
            active=active,
        )
        return out

    @triton_op(f"jit_thc::triangular_accumulate_b4_12_ptr_active{active}_grad_dz", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_grad_dz_traceable(grad_out: Tensor, gamma: Tensor) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_dz_kernel)[grid](
            grad_out,
            gamma,
            grad_dz,
            b * 256,
            c,
            block_m=16,
            block_c=64,
        )
        return grad_dz

    @triton_op(f"jit_thc::triangular_accumulate_b4_12_ptr_active{active}_grad_gamma", mutates_args=())
    def triangular_accumulate_b4_12_ptr_active_grad_gamma_traceable(
        grad_out: Tensor,
        dz0: Tensor,
        dz1: Tensor,
        dz2: Tensor,
        dz3: Tensor,
        dz4: Tensor,
        dz5: Tensor,
        dz6: Tensor,
        dz7: Tensor,
        dz8: Tensor,
        dz9: Tensor,
        dz10: Tensor,
        dz11: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        b = grad_out.shape[0]
        c = grad_out.shape[2]
        grad_gamma = torch.empty((12, c), device=grad_out.device, dtype=torch.float32)
        grad_gamma.zero_()

        def grid(meta):
            return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_triangular_accumulate_b4_12_ptr_grad_gamma_kernel)[grid](
            grad_out,
            dz0,
            dz1,
            dz2,
            dz3,
            dz4,
            dz5,
            dz6,
            dz7,
            dz8,
            dz9,
            dz10,
            dz11,
            grad_gamma,
            b * 256,
            c,
            block_m=64,
            block_c=64,
        )
        return grad_gamma

    def _setup_active_ctx(ctx, inputs, output):
        ctx.save_for_backward(*inputs[1:])

    def _backward_active(ctx, grad_out):
        saved = ctx.saved_tensors
        dzs = saved[:12]
        gamma = saved[12]
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_base = grad_out
        grad_dz_stack = triangular_accumulate_b4_12_ptr_active_grad_dz_traceable(grad_out, gamma)
        grad_gamma = triangular_accumulate_b4_12_ptr_active_grad_gamma_traceable(grad_out, *dzs, gamma)
        active_grads = tuple(grad_dz_stack.unbind(dim=0))
        grad_dzs = active_grads + (None,) * (12 - active)
        return (grad_base, *grad_dzs, grad_gamma.to(dtype=gamma.dtype))

    register_autograd(
        triangular_accumulate_b4_12_ptr_active_traceable,
        _backward_active,
        setup_context=_setup_active_ctx,
    )

    return triangular_accumulate_b4_12_ptr_active_traceable


# The 11 active-prefix ops are deliberately used by the fast LocalTHC path.
# A separate op per prefix length lets torch.compile see the active prefix as a
# constant and avoids a runtime branch inside the triangular accumulation path.
triangular_accumulate_b4_12_ptr_active_traceable_ops = tuple(
    _make_triangular_accumulate_b4_12_ptr_active_traceable(active) for active in range(1, 12)
)



@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_active", mutates_args=())
def triangular_accumulate_b4_12_ptr_active_traceable(
    base: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    gamma: Tensor,
    active: int,
) -> Tensor:
    if not 1 <= active <= 11:
        raise ValueError(f"active must be in [1, 11], got {active}")
    b = base.shape[0]
    c = base.shape[2]
    if base.shape[1] != 256 or gamma.shape != (12, c):
        raise ValueError(
            f"triangular_accumulate_b4_12_active expects base [B,256,C] and gamma [12,C], "
            f"got {tuple(base.shape)} and {tuple(gamma.shape)}"
        )
    for idx, dz in enumerate((dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9, dz10, dz11)):
        if dz.shape != base.shape:
            raise ValueError(
                f"triangular_accumulate_b4_12_active dz{idx} shape {tuple(dz.shape)} "
                f"does not match base {tuple(base.shape)}"
            )
    out = torch.empty_like(base)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_active_fwd_kernel)[grid](
        base,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        gamma,
        out,
        b * 256,
        c,
        block_m=16,
        block_c=64,
        active=active,
    )
    return out


@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_active_grad_dz", mutates_args=())
def triangular_accumulate_b4_12_ptr_active_grad_dz_traceable(
    grad_out: Tensor,
    gamma: Tensor,
    active: int,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    grad_dz = torch.empty((active, b, 256, c), device=grad_out.device, dtype=grad_out.dtype)

    def grid(meta):
        return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_grad_dz_kernel)[grid](
        grad_out,
        gamma,
        grad_dz,
        b * 256,
        c,
        block_m=16,
        block_c=64,
    )
    return grad_dz


@triton_op("jit_thc::triangular_accumulate_b4_12_ptr_active_grad_gamma", mutates_args=())
def triangular_accumulate_b4_12_ptr_active_grad_gamma_traceable(
    grad_out: Tensor,
    dz0: Tensor,
    dz1: Tensor,
    dz2: Tensor,
    dz3: Tensor,
    dz4: Tensor,
    dz5: Tensor,
    dz6: Tensor,
    dz7: Tensor,
    dz8: Tensor,
    dz9: Tensor,
    dz10: Tensor,
    dz11: Tensor,
    gamma: Tensor,
    active: int,
) -> Tensor:
    b = grad_out.shape[0]
    c = grad_out.shape[2]
    grad_gamma = torch.empty((12, c), device=grad_out.device, dtype=torch.float32)
    grad_gamma.zero_()

    def grid(meta):
        return (active, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_triangular_accumulate_b4_12_ptr_grad_gamma_kernel)[grid](
        grad_out,
        dz0,
        dz1,
        dz2,
        dz3,
        dz4,
        dz5,
        dz6,
        dz7,
        dz8,
        dz9,
        dz10,
        dz11,
        grad_gamma,
        b * 256,
        c,
        block_m=64,
        block_c=64,
    )
    return grad_gamma


def _setup_triangular_accumulate_active_ctx(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:14])
    ctx.active = inputs[14]


def _backward_triangular_accumulate_active(ctx, grad_out):
    saved = ctx.saved_tensors
    dzs = saved[:12]
    gamma = saved[12]
    active = int(ctx.active)
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_base = grad_out
    grad_dz_stack = triangular_accumulate_b4_12_ptr_active_grad_dz_traceable(grad_out, gamma, active)
    grad_gamma = triangular_accumulate_b4_12_ptr_active_grad_gamma_traceable(grad_out, *dzs, gamma, active)
    active_grads = tuple(grad_dz_stack.unbind(dim=0))
    grad_dzs = active_grads + (None,) * (12 - active)
    return (grad_base, *grad_dzs, grad_gamma.to(dtype=gamma.dtype), None)


register_autograd(
    triangular_accumulate_b4_12_ptr_active_traceable,
    _backward_triangular_accumulate_active,
    setup_context=_setup_triangular_accumulate_active_ctx,
)


@triton_op("jit_thc::all_base_b4_12_traceable", mutates_args=())
def all_base_b4_12_traceable(x0: Tensor, alpha_stack: Tensor) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    if x0.shape[1] != 64 or x0.shape[2] != 64 or alpha_stack.shape != (12, c, 16):
        raise ValueError(f"all_base_b4_12 expects x0 [B,64,64,C] and alpha [12,C,16], got {tuple(x0.shape)} and {tuple(alpha_stack.shape)}")
    base = torch.empty((12, b, 256, c), device=x0.device, dtype=x0.dtype)

    if _env_int("JIT_THC_ALL_BASE_GROUP12", 1):
        block_m = _env_int("JIT_THC_ALL_BASE_GROUP12_BLOCK_M", 4)
        block_c = _env_int("JIT_THC_ALL_BASE_GROUP12_BLOCK_C", 32)

        def grid(meta):
            return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

        wrap_triton(_all_base_b4_12_fwd_kernel_group12)[grid](
            x0,
            alpha_stack,
            base,
            b * 256,
            c,
            block_m=block_m,
            block_c=block_c,
        )
    else:
        block_m = _env_int("JIT_THC_ALL_BASE_GROUP4_BLOCK_M", 16)
        block_c = _env_int("JIT_THC_ALL_BASE_GROUP4_BLOCK_C", 64)

        def grid(meta):
            return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]), 3)

        wrap_triton(_all_base_b4_12_fwd_kernel)[grid](
            x0,
            alpha_stack,
            base,
            b * 256,
            c,
            block_m=block_m,
            block_c=block_c,
        )
    return base


@triton_op("jit_thc::all_base_b4_12_grad_x", mutates_args=())
def all_base_b4_12_grad_x_traceable(grad_base: Tensor, alpha_stack: Tensor) -> Tensor:
    b = grad_base.shape[1]
    c = grad_base.shape[3]
    grad_x = torch.empty((b, 64, 64, c), device=grad_base.device, dtype=grad_base.dtype)

    def grid(meta):
        return (triton.cdiv(b * 4096, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_b4_12_grad_x_kernel)[grid](
        grad_base,
        alpha_stack,
        grad_x,
        b * 4096,
        c,
        block_m=16,
        block_c=64,
    )
    return grad_x


@triton_op("jit_thc::all_base_b4_12_grad_alpha", mutates_args=())
def all_base_b4_12_grad_alpha_traceable(x0: Tensor, grad_base: Tensor, alpha_stack: Tensor) -> Tensor:
    b = x0.shape[0]
    c = x0.shape[3]
    grad_alpha = torch.empty((12, c, 16), device=x0.device, dtype=torch.float32)
    grad_alpha.zero_()

    def grid(meta):
        return (12 * 16, triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_all_base_b4_12_grad_alpha_kernel)[grid](
        x0,
        grad_base,
        grad_alpha,
        b * 256,
        c,
        block_m=64,
        block_c=64,
    )
    return grad_alpha


def _setup_all_base_ctx(ctx, inputs, output):
    x0, alpha_stack = inputs
    ctx.save_for_backward(x0, alpha_stack)


def _backward_all_base(ctx, grad_base):
    x0, alpha_stack = ctx.saved_tensors
    if not grad_base.is_contiguous():
        grad_base = grad_base.contiguous()
    grad_x = all_base_b4_12_grad_x_traceable(grad_base, alpha_stack)
    grad_alpha = all_base_b4_12_grad_alpha_traceable(x0, grad_base, alpha_stack)
    return grad_x, grad_alpha.to(dtype=alpha_stack.dtype)


register_autograd(
    all_base_b4_12_traceable,
    _backward_all_base,
    setup_context=_setup_all_base_ctx,
)


@triton_op("jit_thc::local_write_traceable", mutates_args=())
def local_write_traceable(x: Tensor, weight: Tensor) -> Tensor:
    b = x.shape[0]
    c = x.shape[3]
    z = torch.empty((b, 256, c), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_write_fwd_kernel)[grid](x, weight, z, b * 256, c, block_m=16, block_c=64)
    return z


@triton_op("jit_thc::local_read_traceable", mutates_args=())
def local_read_traceable(z: Tensor, weight: Tensor) -> Tensor:
    b = z.shape[0]
    c = z.shape[2]
    x = torch.empty((b, 64, 64, c), device=z.device, dtype=z.dtype)

    def grid(meta):
        return (triton.cdiv(b * 4096, meta["block_m"]), triton.cdiv(c, meta["block_c"]))

    wrap_triton(_read_fwd_kernel)[grid](z, weight, x, b * 4096, c, block_m=16, block_c=64)
    return x


@triton_op("jit_thc::local_weight_grad_traceable", mutates_args=())
def local_weight_grad_traceable(high: Tensor, low: Tensor) -> Tensor:
    b = low.shape[0]
    c = low.shape[2]
    grad_w = torch.empty((c, 16), device=high.device, dtype=torch.float32)
    grad_w.zero_()

    def grid(meta):
        return (triton.cdiv(b * 256, meta["block_m"]), triton.cdiv(c, meta["block_c"]), 16)

    wrap_triton(_cell_weight_grad_kernel)[grid](high, low, grad_w, b * 256, c, block_m=64, block_c=64)
    return grad_w


@triton_op("jit_thc::local_read_add_traceable", mutates_args=())
def local_read_add_traceable(z: Tensor, weight: Tensor, x: Tensor) -> Tensor:
    b = z.shape[0]
    c = z.shape[2]
    out = torch.empty_like(x)

    def grid(meta):
        return (b * 256, triton.cdiv(c, meta["block_c"]))

    wrap_triton(_read_add_cell_fwd_kernel)[grid](z, weight, x, out, b * 256, c, block_c=64)
    return out


def _setup_write_ctx(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _backward_write(ctx, grad_z):
    x, weight = ctx.saved_tensors
    grad_x = local_read_traceable(grad_z, weight)
    grad_w = local_weight_grad_traceable(x, grad_z)
    return grad_x, grad_w.to(dtype=weight.dtype)


def _setup_read_ctx(ctx, inputs, output):
    z, weight = inputs
    ctx.save_for_backward(z, weight)


def _backward_read(ctx, grad_x):
    z, weight = ctx.saved_tensors
    grad_z = local_write_traceable(grad_x, weight)
    grad_w = local_weight_grad_traceable(grad_x, z)
    return grad_z, grad_w.to(dtype=weight.dtype)


def _setup_read_add_ctx(ctx, inputs, output):
    z, weight, x = inputs
    ctx.save_for_backward(z, weight)


def _backward_read_add(ctx, grad_out):
    z, weight = ctx.saved_tensors
    grad_z = local_write_traceable(grad_out, weight)
    grad_w = local_weight_grad_traceable(grad_out, z)
    return grad_z, grad_w.to(dtype=weight.dtype), grad_out


register_autograd(local_write_traceable, _backward_write, setup_context=_setup_write_ctx)
register_autograd(local_read_traceable, _backward_read, setup_context=_setup_read_ctx)
register_autograd(local_read_add_traceable, _backward_read_add, setup_context=_setup_read_add_ctx)
