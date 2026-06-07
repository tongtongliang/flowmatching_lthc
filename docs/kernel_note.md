# Kernel fusion notes

This note explains the system side of the released `SharedWrite-FusedFinal12` model.

## The bandwidth problem

The high-resolution residual stream has shape

```text
[B, 64, 64, 768]
```

or, in local-cell form,

```text
[B, 256, 16, 768]
```

A naive implementation of each layer would do:

```python
z = read_from_residual(x_hi)      # [B, 256, 768]
dz = workspace_branch(z, c)       # [B, 256, 768]
dx = write_to_residual(dz)        # [B, 64, 64, 768]
x_hi = x_hi + dx
```

The branch itself is efficient because it runs on 256 workspace tokens. The bottleneck is repeatedly reading and writing the full high-resolution residual stream and materializing broadcasted `dx` tensors.

## Why shared read helps

With a shared read operator `R`, we can keep a persistent workspace coordinate:

```python
x0 = patch_embed(image)   # high-res stream
z = R(x0)                # initial workspace

for layer in layers:
    dz = F_l(z, c)
    z = z + gamma_l * dz

x_final = x0 + sum_l P_l(dz_l)
```

The key identity is

$R P_l dz_l = \gamma_l \odot dz_l$.

So the loop no longer touches the high-resolution residual stream after the initial read. The high-resolution stream is touched only twice:

1. Initial patch embedding and shared read.
2. Final materialization before the patch decoder.

This is the main system-algorithm co-design point.

## What the Triton kernel fuses

The final materialization is

$X_L[b,m,r,c] = X_0[b,m,r,c] + \sum_{l=0}^{11} \beta_l[c,r] \Delta Z_l[b,m,c]$.

A naive PyTorch implementation loops over 12 layers:

```python
x_hi = x0
for l in range(12):
    dx_l = dz_l[:, :, None, :] * beta_l[None, None, :, :]
    x_hi = x_hi + reshape_to_highres(dx_l)
```

This creates 12 high-resolution broadcast tensors and performs repeated high-resolution reads/writes.

The Triton kernel `final_accumulate_b4_12_traceable` fuses this into one pass:

```text
program id m: one workspace cell m
program id c: one channel block
load x0[b, m, 16, c_block]
for l in 0..11:
    load dz_l[b, m, c_block]
    load beta_l[c_block, 16]
    accumulate beta_l * dz_l into the 16 local high-res positions
store x_final[b, m, 16, c_block]
```

The output is exactly the same algebra as the naive implementation, up to floating-point ordering.

## Why fixed shape

The released kernel specializes to:

```text
input_size      256
patch_size      4
high_grid       64
workspace_grid  16
cell_tokens     16
hidden dim      768
depth           12
```

This makes the kernel simple and avoids dynamic-shape overhead. It also makes the graph easier for `torch.compile` to trace and cache.

## Torch compile interaction

The model uses `torch.library.triton_op` wrappers in `local_thc_triton_kernels.py`. This makes the Triton operation visible to TorchDynamo/Inductor as a graph op instead of hiding it behind arbitrary Python.

The practical setup used in training is:

```bash
--compile --compile_mode auto
```

In `train_imagenet256.py`, `auto` maps to:

- `reduce-overhead` when `grad_accum == 1`,
- `default` when `grad_accum > 1`.

For evaluation/sampling, the reference scripts use:

```bash
--compile --compile_mode default
```

This was the more stable low-risk choice for large 50k-sample evaluation.

## RMSNorm/AdaLN and fusion boundary

The Triton kernel does not fuse RMSNorm, attention, or SwiGLU. Those remain in PyTorch and use optimized SDPA/GEMM kernels. The fusion boundary is intentionally placed around the high-resolution residual interface, because that is the part dominated by memory traffic rather than tensor-core compute.

In other words:

```text
workspace branch:       leave to PyTorch SDPA/GEMM/RMSNorm kernels
high-res write-back:    fuse with Triton
```

This keeps the implementation maintainable while addressing the specific bandwidth bottleneck introduced by LocalTHC.

## Debugging and correctness

The model exposes a naive path:

```python
model(x, t, y, use_lazy=False)
```

This path materializes the high-resolution stream every layer and does not require the fused final accumulation kernel. It is useful for CPU sanity checks and checkpoint-loading checks.

For production CUDA inference/training, use the default lazy path:

```python
model(x, t, y)
```
