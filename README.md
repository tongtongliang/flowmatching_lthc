# ImageNet LTHC

Local Token Hyper-Connection (LTHC) diffusion backbone for class-conditional ImageNet-256 generation.

This repository is a cleaned-up release snapshot of the model used in the `SharedWrite-FusedFinal12 LocalTHC-B/4 velocity` experiments. It contains the model, training loop, Heun sampler, FID/IS evaluation entry point, and the Triton kernels used by the fast shared-write implementation.

## Current released model

Public alias:

```text
lthc_b4_velocity
```

Checkpoint-compatible internal name:

```text
local_thc_jit_shared_write_fused_final12_shared_adaln_b4
```

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

The high-resolution residual stream is persistent. Each block reads a local `4 x 4` cell from the high-res stream into a low-resolution workspace token, runs a JiT-style attention/FFN branch on the workspace, and writes the update back to the high-res residual stream.

The released fast model uses a depth-shared local read-from-residual operator and layer-specific write-back operators, enabling an exact lazy workspace recurrence and a fused final high-resolution accumulation kernel.

## Results from the reference run

Reference run:

```text
im256_local_thc_shared_write_fused_final12_b4_velocity_gpus4567_bs128_accum2_20260531_055810
```

EMA, 50k samples, Heun 50, CFG 2.9:

| step | FID50k | IS |
|---:|---:|---:|
| 300k | 12.52 | 114.42 |
| 350k | 11.31 | 122.80 |
| 400k | 10.63 | 127.59 |

These numbers are included to document the training setup that this code is meant to reproduce; they are not hard-coded in the code.

## Installation

Use an environment with PyTorch 2.x and Triton. The code was developed with CUDA GPUs that support PyTorch SDPA/FlashAttention.

```bash
cd imaget_lthc
pip install -e .
```

Core dependencies are listed in `pyproject.toml`.

## Checkpoint inference

Generate a small sample grid from a checkpoint:

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

CPU/debug smoke test using the non-Triton naive path:

```bash
python scripts/sample_checkpoint.py \
  --checkpoint /path/to/step_00400000.pt \
  --state_key ema \
  --output outputs/smoke.png \
  --device cpu \
  --batch_size 1 \
  --steps 1 \
  --naive
```

The CUDA default path uses the lazy recurrence and Triton fused final accumulation. The CPU `--naive` path is for correctness/debug only and is slow.

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
      local_thc.py              # LTHC model family; released alias builds fused-final12 B/4
      local_thc_triton_kernels.py
      jit_shared_adaln.py       # RMSNorm, RoPE, shared AdaLN/JiT utilities
    imagenet.py                 # ImageNet-256 ImageFolder/zip loader
    sampling.py                 # Heun sampler + CFG
    optim/                      # AdamW/Muon support from the research code
  scripts/
    train_imagenet256.py
    evaluate_fid.py
    sample_checkpoint.py
    launch_lthc_b4_velocity_8gpu.sh
  configs/
    lthc_b4_velocity_imagenet256.json
  docs/
    triton_kernels.md
```

## Notes

- No checkpoint is committed by default. Put checkpoints under `checkpoints/` or pass an absolute path.
- No dataset is committed. `DATA_PATH` should point to ImageNet-256 in ImageFolder form or a supported zip layout.
- No license has been selected yet. Add a license before publishing publicly.
