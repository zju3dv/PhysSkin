#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_ROOT:?Set DATASET_ROOT to the processed dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

NUM_SELF_ATTN=1
HANDLES_NUM=16
NUM_HANDLE_TOKENS=32
BATCH_TRANSFORM=1024
RUN_NAME="point_shape_handle_cross_only_rignet_class_dp${NUM_SELF_ATTN}_handle${HANDLES_NUM}_token${NUM_HANDLE_TOKENS}_dir_batch${BATCH_TRANSFORM}"

"${PYTHON_BIN}" train.py \
    --dataset_root "${DATASET_ROOT}" \
    --split_dir "${SPLIT_DIR:-${REPO_ROOT}/data/splits}" \
    --log_path "${LOG_DIR:-${REPO_ROOT}/logs/${RUN_NAME}}" \
    --ckpt_path "${CKPT_DIR:-${REPO_ROOT}/checkpoints/${RUN_NAME}}" \
    --num_epochs 300 \
    --batch_obj 1 \
    --batch_transform "${BATCH_TRANSFORM}" \
    --num_self_attn "${NUM_SELF_ATTN}" \
    --handles_num "${HANDLES_NUM}" \
    --num_handle_tokens "${NUM_HANDLE_TOKENS}" \
    --gpu_ids "${GPU_ID_ARRAY[@]}"
