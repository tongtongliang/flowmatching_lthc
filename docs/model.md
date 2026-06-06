# Local Token Hyper-Connection model

This note fixes the terminology used in the released code.

- **Read** means reading from the persistent high-resolution residual stream into the low-resolution workspace.
- **Write** means writing a workspace branch update back into the persistent high-resolution residual stream.

## State and workspace

The model keeps a high-resolution residual stream:

```math
x_l \in \mathbb{R}^{B \times M \times R \times C},
```

where `M = 16 * 16` workspace cells, `R = 4 * 4` high-resolution tokens per cell, and `C = 768` channels.

The workspace is:

```math
z_l \in \mathbb{R}^{B \times M \times C}.
```

## Block update

A generic LocalTHC block is:

```math
z_l = R_l x_l,
```

```math
dz_l = F_l(z_l, c),
```

```math
x_{l+1} = x_l + P_l dz_l.
```

Here `F_l` is a normal JiT-style workspace Transformer block with RMSNorm, QK RMSNorm attention, RoPE, SwiGLU FFN, and AdaLN gates.

The current released model uses shared AdaLN conditioning:

```math
c = e_t(t) + e_y(y).
```

## Shared-read fast path

The released fast path shares the read operator across depth:

```math
R_l = R.
```

Then:

```math
z_{l+1} = R x_{l+1} = R(x_l + P_l dz_l) = z_l + R P_l dz_l.
```

For the local channel-wise maps used here, `R P_l` is a per-channel scale:

```math
(RP_l dz_l)[b,m,c] = \gamma_l[c] dz_l[b,m,c],
```

```math
\gamma_l[c] = \sum_{r=1}^{16} \alpha[c,r] \beta_l[c,r].
```

So the layer loop can run entirely in workspace coordinates:

```math
z_{l+1} = z_l + \gamma_l \odot dz_l.
```

The final high-resolution residual stream is materialized once:

```math
x_L = x_0 + \sum_l P_l dz_l.
```

This final accumulation is fused in Triton for the fixed B/4, depth-12 shape.

## Training target

Training uses velocity prediction:

```math
t = \sigma(0.8 \epsilon - 0.8), \quad \epsilon \sim \mathcal{N}(0,1),
```

```math
z = t x + (1 - t) n, \quad n \sim \mathcal{N}(0, I),
```

```math
v = \frac{x - z}{\max(1 - t, 0.05)}.
```

The model output is trained directly as `v`.
