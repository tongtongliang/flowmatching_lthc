# Local Token Hyper-Connection architecture

This document gives the full model definition used by the released `lthc_b4_velocity` alias.

## Naming convention

The code keeps some historical parameter names for checkpoint compatibility, but the paper/blog notation should use this convention:

- **Read**: high-resolution residual stream to low-resolution workspace.
- **Write**: workspace update back to high-resolution residual stream.

The released model uses a shared read operator and layer-specific write operators.

## Geometry

Input image:

$X_{img} \in \mathbb{R}^{B \times 3 \times 256 \times 256}$.

Patch embedding uses `patch_size=4`:

$X_0 \in \mathbb{R}^{B \times 64 \times 64 \times C}$,

with `C=768`. We also use cell notation:

$X_l \in \mathbb{R}^{B \times M \times R \times C}$,

where

$M = 16 \times 16 = 256$

is the workspace grid flattened into cells, and

$R = 4 \times 4 = 16$

is the number of high-resolution patch tokens inside one workspace cell.

The workspace state is

$Z_l \in \mathbb{R}^{B \times M \times C}$.

## Generic LTHC block

A generic LTHC block is

$Z_l = \mathcal{R}_l(X_l)$

$\Delta Z_l = F_l(Z_l, c)$

$X_{l+1} = X_l + \mathcal{P}_l(\Delta Z_l)$.

Here `F_l` is a standard workspace Transformer branch, while `R_l` and `P_l` are local cell-wise linear maps.

The local read is channel-wise:

$Z_l[b,m,c] = \sum_{r=1}^{R} \alpha_l[c,r] X_l[b,m,r,c]$.

The local write is also channel-wise:

$\mathcal{P}_l(\Delta Z_l)[b,m,r,c] = \beta_l[c,r] \Delta Z_l[b,m,c]$.

In the released model, the read weights are shared across layers:

$\alpha_l = \alpha$.

The write weights remain layer-specific:

$\beta_l \neq \beta_j \quad \text{for } l \neq j$.

## Exact lazy workspace recurrence

Because the read operator is shared,

$Z_{l+1} = \mathcal{R}(X_{l+1})$.

Substitute the residual update:

$Z_{l+1} = \mathcal{R}(X_l + \mathcal{P}_l(\Delta Z_l))$.

Linearity gives

$Z_{l+1} = Z_l + \mathcal{R}\mathcal{P}_l(\Delta Z_l)$.

For the channel-wise local maps,

$(\mathcal{R}\mathcal{P}_l(\Delta Z_l))[b,m,c] = \gamma_l[c] \Delta Z_l[b,m,c]$,

where

$\gamma_l[c] = \sum_{r=1}^{16} \alpha[c,r] \beta_l[c,r]$.

Therefore the layer loop can be executed exactly in workspace coordinates:

$Z_{l+1} = Z_l + \gamma_l \odot \Delta Z_l$.

This is not an approximation. It is algebraically equivalent to repeatedly materializing the high-resolution residual stream, as long as the read operator is shared and the local maps stay linear.

## Final materialization

The final high-resolution stream is

$X_L = X_0 + \sum_{l=0}^{L-1} \mathcal{P}_l(\Delta Z_l)$.

In index form:

$X_L[b,m,r,c] = X_0[b,m,r,c] + \sum_{l=0}^{L-1} \beta_l[c,r] \Delta Z_l[b,m,c]$.

The released model uses `L=12`, and this final accumulation is implemented by a fixed-shape Triton kernel.

## Workspace branch

The conditioning vector is

$c = e_t(t) + e_y(y)$.

The shared AdaLN head produces

$(s_a, q_a, g_a, s_m, q_m, g_m) = W_{6C}(\mathrm{SiLU}(c))$.

Here `s` is shift, `q` is scale, and `g` is gate. For a tensor `U`, AdaLN modulation is

$\mathrm{AdaLN}(U; s, q) = U \odot (1 + q) + s$.

The attention update is

$A_l = g_a \odot \mathrm{Attn}(\mathrm{AdaLN}(\mathrm{RMSNorm}(Z_l); s_a, q_a))$.

The MLP update is

$M_l = g_m \odot \mathrm{SwiGLU}(\mathrm{AdaLN}(\mathrm{RMSNorm}(Z_l + A_l); s_m, q_m))$.

The branch output is

$\Delta Z_l = A_l + M_l$.

Attention uses:

- multi-head self-attention on the `16 x 16` workspace only,
- RMSNorm on Q and K,
- 2D RoPE on workspace tokens,
- PyTorch SDPA with flash backend preference.

## Training target

The model is trained as a velocity predictor. The training time is sampled from a logit-normal distribution:

$u \sim \mathcal{N}(0,1)$

$t = \sigma(0.8u - 0.8)$.

Noisy input:

$\epsilon \sim \mathcal{N}(0, I)$

$Z_t = tX + (1-t)\epsilon$.

Velocity target:

$v = \frac{X - Z_t}{\max(1-t, 0.05)}$.

The model output is directly trained to match `v`.
