#!/usr/bin/env bash
set -euo pipefail

# Example 8-GPU launch. Set DATA_PATH and RUN_DIR before running if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DATA_PATH="${DATA_PATH:-/path/to/imagenet256}"
export RUN_DIR="${RUN_DIR:-runs/lthc_b4_velocity_$(date -u +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-flowmatching-lthc}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_ID="${WANDB_ID:-lthc_b4_velocity_$(date -u +%Y%m%d_%H%M%S)}"
export TMPDIR="${TMPDIR:-${PWD}/.cache/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PWD}/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/.cache/triton}"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${RUN_DIR}"

WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_id "${WANDB_ID}" --wandb_group lthc-b4-velocity --run_name "$(basename "${RUN_DIR}")")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

torchrun --nproc_per_node=8 scripts/train_imagenet256.py \
  --data_path "${DATA_PATH}" \
  --run_dir "${RUN_DIR}" \
  --model lthc_b4_velocity \
  --prediction velocity \
  --batch_size 128 \
  --grad_accum 1 \
  --num_workers 16 \
  --max_steps 200000 \
  --save_every 10000 \
  --log_every 100 \
  --lr 2e-4 \
  --warmup_steps 6250 \
  --weight_decay 0.0 \
  --ema_decay 0.9999 \
  --P_mean -0.8 \
  --P_std 0.8 \
  --noise_scale 1.0 \
  --t_eps 0.05 \
  --label_drop_prob 0.1 \
  --attn_backend flash \
  --compile \
  --compile_mode auto \
  "${WANDB_ARGS[@]}"
