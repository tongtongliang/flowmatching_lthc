# ImageNet LTHC

Local Token Hyper-Connection (LTHC) is a diffusion backbone for class-conditional ImageNet-256 generation. The main idea is to keep a persistent high-resolution residual state while running expensive global computation on a much smaller workspace.

![LTHC architecture](docs/assets/lthc_architecture.png)

This repository is a cleaned-up release snapshot of the model used in the `SharedRead-FusedFinal12 LocalTHC-B/4 velocity` experiments. It contains the model, training loop, Heun sampler, FID/IS evaluation entry point, and the Triton kernels used by the fast shared-read implementation.

## Why LTHC?

A direct patch-4 Transformer on 256x256 images has `64 * 64 = 4096` tokens. Full attention on that sequence is expensive enough that long ImageNet diffusion training becomes impractical in this setup. A patch-16 model has only `16 * 16 = 256` tokens, but it discards most high-resolution residual detail at the representation level.

LTHC decouples these two resolutions:

```text
persistent residual state: 64 x 64 high-resolution tokens
global workspace compute: 16 x 16 low-resolution tokens
```

The workspace branch is still a normal JiT-style Transformer block with RMSNorm, QK RMSNorm attention, RoPE, SwiGLU, and AdaLN gates. The difference is that each layer reads a local 4x4 high-resolution cell into one workspace token, computes a workspace update, and writes that update back into the high-resolution residual stream.

## Released model

Public alias:

```text
lthc_b4_velocity
```

Checkpoint-compatible internal name:

```text
local_thc_jit_shared_read_fused_final12_shared_adaln_b4
```

Legacy run/checkpoint alias, still accepted by `build_model()`:

```text
local_thc_jit_shared_write_fused_final12_shared_adaln_b4
```

The legacy alias says `shared_write` because early research code used the
opposite naming convention. In this repository, `read` means
high-resolution residual to workspace and `write` means workspace update back
to the residual stream.

Architecture summary:

```text
input image:        256 x 256 x 3
high-res patch:       4 x 4
high-res grid:       64 x 64 tokens
workspace grid:      16 x 16 tokens
hidden dim:         768
depth:               12
heads:               12
patch embed:       Conv patchify 3 -> 128, then 1x1 Conv 128 -> 768
conditioning:       timestep embedding + class embedding -> shared AdaLN
objective:          velocity prediction
sampler:            Heun, 50 steps, CFG
```

## Mathematical form

Let the high-resolution residual state at layer `l` be

$X_l \in \mathbb{R}^{B \times M \times R \times C}$,

where `B` is batch size, `M = 16 * 16` is the number of workspace cells, `R = 4 * 4` is the number of high-resolution tokens inside each local cell, and `C = 768` is channel width.

The low-resolution workspace is

$Z_l \in \mathbb{R}^{B \times M \times C}$.

A generic LocalTHC block has three steps:

$Z_l = \mathcal{R}_l(X_l)$

$\Delta Z_l = F_l(Z_l, c)$

$X_{l+1} = X_l + \mathcal{P}_l(\Delta Z_l)$

Here `read` means high-res residual to workspace, and `write` means workspace update back to high-res residual. `F_l` is the workspace Transformer branch.

For this released model, the read operator is shared across depth:

$\mathcal{R}_l = \mathcal{R}$.

Inside a local cell, the channel-wise maps are

$Z_l[b,m,c] = \sum_{r=1}^{16} \alpha[c,r] X_l[b,m,r,c]$

and

$\mathcal{P}_l(\Delta Z_l)[b,m,r,c] = \beta_l[c,r] \Delta Z_l[b,m,c]$.

Because `alpha` is shared, the workspace state can be advanced exactly without materializing the high-resolution stream after every layer:

$Z_{l+1} = Z_l + \gamma_l \odot \Delta Z_l$

with

$\gamma_l[c] = \sum_{r=1}^{16} \alpha[c,r] \beta_l[c,r]$.

The final high-resolution state is materialized once:

$X_L = X_0 + \sum_{l=0}^{L-1} \mathcal{P}_l(\Delta Z_l)$.

This is why the shared-read design is more than a code optimization. It gives a stable workspace coordinate across depth and makes the lazy recurrence exact.

## Conditioning and normalization

The released model uses shared AdaLN conditioning:

$c = e_t(t) + e_y(y)$.

One shared modulation head produces the six vectors used by all workspace blocks:

$(\mathrm{shift}_{attn}, \mathrm{scale}_{attn}, \mathrm{gate}_{attn}, \mathrm{shift}_{mlp}, \mathrm{scale}_{mlp}, \mathrm{gate}_{mlp}) = W_{adaLN}(\mathrm{SiLU}(c))$.

A workspace branch update is

$U_{attn} = \mathrm{gate}_{attn} \odot \mathrm{Attention}(\mathrm{AdaLN}(\mathrm{RMSNorm}(Z_l)))$

$U_{mlp} = \mathrm{gate}_{mlp} \odot \mathrm{SwiGLU}(\mathrm{AdaLN}(\mathrm{RMSNorm}(Z_l + U_{attn})))$

$\Delta Z_l = U_{attn} + U_{mlp}$.

Attention uses QK RMSNorm and RoPE on the `16 x 16` workspace grid. There are no time/class/register prefix tokens in the released LTHC model; class conditioning enters through AdaLN via the class embedding.

## Triton fusion in one paragraph

The lazy workspace recurrence avoids per-layer high-resolution traffic, but the final state still requires

$X_L = X_0 + \sum_l \mathcal{P}_l(\Delta Z_l)$.

A naive PyTorch implementation creates one broadcasted high-resolution update per layer and repeatedly reads/writes the full `64 x 64 x 768` residual stream. The release uses a fixed-shape Triton kernel for B/4, depth 12 that fuses all 12 write-back operations plus the residual add into one high-resolution pass. This reduces memory traffic and gives `torch.compile` a stable graph. See `imaget_lthc/models/kernel_note.md` for details.

## Results from the reference run

Reference run:

```text
im256_local_thc_shared_write_fused_final12_b4_velocity_gpus4567_bs128_accum2_20260531_055810
```

The run directory keeps its original legacy name; the architecture is the
shared-read model described above.

50k-sample ImageNet validation FID/IS, Heun 50, CFG 2.9. EMA is the main evaluation path after 250k; raw/model numbers are included where they were run.

| step | raw/model FID50k | EMA FID50k | EMA IS |
|---:|---:|---:|---:|
| 200k | 19.87 | not run | not run |
| 250k | 17.32 | 14.30 | 104.57 |
| 300k | 17.10 | 12.52 | 114.42 |
| 350k | not run | 11.31 | 122.80 |
| 400k | not run | 10.63 | 127.59 |

These numbers document the training setup that this code is meant to reproduce; they are not hard-coded in the code.

The compact CSV/plot snapshot is stored in:

```text
results/lthc_patch4_ema50k/
```

## Installation

Use an environment with PyTorch 2.x and Triton. The CUDA fast path expects GPUs supported by PyTorch SDPA/FlashAttention.

```bash
cd imaget_lthc
pip install -e .
```

Core dependencies are listed in `pyproject.toml`.

## Checkpoint inference

Generate a sample grid from a checkpoint:

```bash
python scripts/sample_checkpoint.py \
  --checkpoint /path/to/step_00400000.pt \
  --state_key ema \
  --output outputs/lthc_sample.png \
  --device cuda \
  --batch_size 16 \
  --steps 50 \
  --cfg 2.9
```

CPU checkpoint-loading sanity check using the non-Triton naive path:

```bash
python scripts/sample_checkpoint.py \
  --checkpoint /path/to/step_00400000.pt \
  --state_key ema \
  --output outputs/sanity.png \
  --device cpu \
  --batch_size 1 \
  --steps 1 \
  --naive
```

The CUDA default path uses the lazy recurrence and Triton fused final accumulation. The CPU `--naive` path is exact but slow.

## Training

Example 8-GPU launch:

```bash
DATA_PATH=/path/to/imagenet256 \
RUN_DIR=runs/lthc_b4_velocity \
WANDB_PROJECT=jit-imagenet256 \
WANDB_ENTITY=your-wandb-entity \
bash scripts/launch_lthc_b4_velocity_8gpu.sh
```

Important training parameters:

```text
prediction        velocity
optimizer         AdamW, betas=(0.9, 0.95), fused=True
lr                2e-4
warmup_steps      6250
weight_decay      0.0
EMA               0.9999
batch/GPU         128
grad_accum        1
compile           torch.compile, mode auto
attention         PyTorch SDPA with flash backend preference
```

Noise schedule used during training:

```python
u = torch.randn(batch)
t = sigmoid(0.8 * u - 0.8)
eps = randn_like(x) * 1.0
z = t * x + (1 - t) * eps
v = (x - z) / clamp(1 - t, min=0.05)
```

Equivalent config is stored in `configs/lthc_b4_velocity_imagenet256.json`.

## Evaluation

The evaluation script samples images and can compute FID/IS if torch-fidelity and an ImageNet-256 statistics file are available:

```bash
torchrun --nproc_per_node=8 scripts/evaluate_fid.py \
  --checkpoint /path/to/step_00400000.pt \
  --output_dir runs/eval_lthc \
  --state_key ema \
  --model lthc_b4_velocity \
  --prediction velocity \
  --num_samples 50000 \
  --batch_size 256 \
  --steps 50 \
  --cfg 2.9 \
  --interval_min 0.1 \
  --interval_max 1.0 \
  --noise_scale 1.0 \
  --compile \
  --compile_mode default
```

## Repository layout

```text
imaget_lthc/
  imaget_lthc/
    models/
      local_thc.py
      local_thc_triton_kernels.py
      jit_shared_adaln.py
    imagenet.py
    sampling.py
    optim/
  scripts/
    train_imagenet256.py
    evaluate_fid.py
    sample_checkpoint.py
    launch_lthc_b4_velocity_8gpu.sh
  configs/
    lthc_b4_velocity_imagenet256.json
  docs/
    model.md
    kernel_note.md
    assets/lthc_architecture.png
```

## Notes

- No checkpoint is committed by default. Put checkpoints under `checkpoints/` or pass an absolute path.
- No dataset is committed. `DATA_PATH` should point to ImageNet-256 in ImageFolder form or a supported zip layout.
- No license has been selected yet. Add a license before publishing publicly.
