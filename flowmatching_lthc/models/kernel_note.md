# Kernel fusion notes

This note explains the system side of the released `SharedRead-FusedFinal12` model at a high level.

## Naming convention

This repository uses the public README convention:

```text
read:   high-resolution residual -> workspace
write:  workspace update -> high-resolution residual
```

The low-level Triton helper names are the one confusing exception. Some helper names still reflect an older internal convention: `read_from_residual()` may call a helper named `triton_local_write()`, while `write_to_residual()` may call `triton_local_read()`. Treat those helper names as historical implementation names, not as the public algorithmic direction.

In the released shared-read model:

```text
read_logits / alpha    high-res -> workspace pooling weights
write_weight / beta    workspace -> high-res write-back weights
```

## High-level idea

The model keeps two coordinate systems:

```text
high-resolution residual stream: [B, 64, 64, C]
low-resolution workspace:        [B, 16*16, C]
```

The Transformer blocks run on the low-resolution workspace. The high-resolution residual stream is there to preserve local image detail, but repeatedly materializing it after every block would be memory-bandwidth heavy.

The shared-read design avoids that repeated high-resolution traffic. It reads the initial residual stream once:

```python
x0 = patch_embed(image)
z = shared_read(x0)
```

Then each block updates the workspace directly:

```python
for block in blocks:
    dz = block.workspace_branch(z, c)
    z = z + gamma_l * dz
```

This is exact, not an approximation. Because the read operator is shared and the local read/write maps are linear, reading after a local write-back is equivalent to multiplying the workspace update by a per-channel scale:

$$
R P_l(\Delta Z_l) = \gamma_l \odot \Delta Z_l .
$$

## What Triton fuses

After the 12 workspace blocks, the model still needs the high-resolution residual state for the patch decoder:

$$
X_L[b,m,r,c] =
X_0[b,m,r,c] +
\sum_{l=0}^{11} \beta_l[c,r]\,\Delta Z_l[b,m,c].
$$

A naive implementation would loop over layers, create 12 high-resolution update tensors, and repeatedly read/write the high-resolution residual. The Triton fast path fuses this final accumulation into one pass:

```text
load X0 for one workspace cell and channel block
for l = 0..11:
    load DeltaZ_l
    load beta_l
    accumulate beta_l * DeltaZ_l into the 16 local positions
store XL once
```

So the fusion boundary is deliberately narrow: it fuses the final high-resolution write-back accumulation. It does not fuse the Transformer block, RMSNorm, AdaLN, attention, or SwiGLU. Those remain in PyTorch and use the usual optimized kernels.

## Fixed-shape scope

The released Triton path is specialized to the public B/4 checkpoint:

```text
image size      256
patch size      4
high-res grid   64 x 64
workspace grid  16 x 16
cell tokens     16
hidden dim      768
depth           12
```

That fixed shape keeps the kernel small, traceable, and easy to compare against the naive path.

## Correctness path

The model still exposes a naive path:

```python
model(x, t, y, use_lazy=False)
```

That path materializes the high-resolution residual stream after every block. It is useful for sanity checks and for comparing the lazy/fused implementation against the literal algorithm.
