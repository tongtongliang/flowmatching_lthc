# Triton kernels in LTHC

The LocalTHC high-resolution residual interface is memory-traffic heavy if implemented as separate PyTorch tensor operations. The released model uses a shared local read-from-residual operator and a fused final accumulation kernel to reduce high-res stream traffic.

## Local maps

Inside each `4 x 4` high-resolution cell, each channel has a local linear read/write map.

Read high-res residual stream into workspace:

```math
z_l[b, m, c] = \sum_r \alpha[c, r] x_l[b, m, r, c]
```

Write workspace update back into high-res residual stream:

```math
x_{l+1}[b, m, r, c] = x_l[b, m, r, c] + \beta_l[c, r] dz_l[b, m, c]
```

The released fast path shares `alpha` across depth. This gives an exact workspace recurrence:

```math
z_{l+1} = z_l + \gamma_l \odot dz_l,
\quad
\gamma_l[c] = \sum_r \alpha[c, r] \beta_l[c, r].
```

The model no longer needs to materialize the high-resolution stream between all layers during the block loop. It computes branch updates in workspace coordinates, then materializes the final high-res stream once:

```math
x_L = x_0 + \sum_{l=0}^{L-1} P_l dz_l.
```

## Fused final accumulation

The important fixed-shape kernel is `final_accumulate_b4_12_traceable` in `imaget_lthc/models/local_thc_triton_kernels.py`.

It computes, for depth 12 and B/4 geometry:

```math
x_L[b,m,r,c] = x_0[b,m,r,c] + \sum_{l=0}^{11} \beta_l[c,r] dz_l[b,m,c].
```

Doing this in one Triton kernel avoids creating 12 broadcasted high-resolution update tensors and avoids repeatedly reading/writing the full `64 x 64 x 768` high-res residual stream.

## Kernel scope

The current kernels intentionally target the released fixed shape:

```text
input_size      256
patch_size      4
high_grid       64
workspace_grid  16
cell tokens     16
hidden dim      768
depth           12
```

This makes the kernel simpler and helps `torch.compile` see a stable graph. Generalizing to other shapes is possible but should be done as a separate engineering pass.

## Debug path

For debugging and CPU smoke tests, the model still has a naive PyTorch path:

```python
model(x, t, y, use_lazy=False)
```

The naive path is exact but slower; it avoids Triton so it can be used to verify checkpoint loading on CPU.
