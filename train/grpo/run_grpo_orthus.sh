#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/grpo_orthus.py"

# Optional debug switch: pass "--debug" as the first argument or set DEBUG=1
DEBUG_MODE=${DEBUG:-0}
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_MODE=1
  shift
fi

# 环境变量设置
export WANDB_PROJECT="${WANDB_PROJECT:-orthus-grpo-project}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# --- 可配置参数 ---
MODEL_PATH="${MODEL_PATH:-/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-base-sample80b100ep500l1e-5-weight-F}"
DATASET_PATH="${DATASET_PATH:-/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json}"
RUN_NAME="${RUN_NAME:-orthus-7b-grpo-geneval-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/oujingfeng/project/twgi/checkpoints/grpo/${RUN_NAME}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:-2000}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1400}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
BETA="${BETA:-0}"
USE_CPU_FLAG="${USE_CPU_FLAG:-False}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw_torch}"
INTERLEAVE_GENERATION="${INTERLEAVE_GENERATION:-False}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"
LOG_FILE="${LOG_FILE:-${DEFAULT_LOG_FILE}}"

if [[ "${DEBUG_MODE}" == "1" ]]; then
  echo "[DEBUG] Enabling debug configuration."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_DEBUG:-}"
  RUN_NAME="${RUN_NAME}-debug"
  OUTPUT_DIR="${OUTPUT_DIR}-debug"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS_DEBUG:-1}"
  MAX_STEPS="${MAX_STEPS_DEBUG:-2}"
  NUM_PROCESSES="${NUM_PROCESSES_DEBUG:-1}"
  NUM_GENERATIONS="${NUM_GENERATIONS_DEBUG:-2}"
  GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE_DEBUG:-2}"
  PER_DEVICE_BATCH="${PER_DEVICE_BATCH_DEBUG:-1}"
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH_DEBUG:-1400}"
  MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH_DEBUG:-256}"
  BETA="${BETA_DEBUG:-0.0}"
  USE_CPU_FLAG="${USE_CPU_FLAG_DEBUG:-True}"
fi

if [[ -z "${DEEPSPEED_CONFIG_PATH:-}" ]]; then
  if (( NUM_PROCESSES > 1 )); then
    # Shared config already pins DeepSpeed to ZeRO stage 2.
    DEEPSPEED_CONFIG_PATH="/data1/oujingfeng/project/twgi/Orthus/accelerate_config_stage3_offload.yaml"
  else
    DEEPSPEED_CONFIG_PATH="/data1/oujingfeng/project/twgi/Orthus/accelerate_config_single.yaml"
  fi
fi

BATCH_PER_STEP=$(( PER_DEVICE_BATCH * NUM_PROCESSES ))
DEFAULT_GENERATION_BATCH_SIZE=$(( BATCH_PER_STEP * GRAD_ACCUM_STEPS ))
if (( DEFAULT_GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  if (( BATCH_PER_STEP <= 0 )); then
    echo "[ERROR] Invalid batch configuration: per_device_batch (${PER_DEVICE_BATCH}) * num_processes (${NUM_PROCESSES}) must be positive." >&2
    exit 1
  fi
  # Increase by whole training batches until we hit a multiple of num_generations.
  ADJUSTED_BATCH_SIZE=${DEFAULT_GENERATION_BATCH_SIZE}
  while (( ADJUSTED_BATCH_SIZE % NUM_GENERATIONS != 0 )); do
    ADJUSTED_BATCH_SIZE=$(( ADJUSTED_BATCH_SIZE + BATCH_PER_STEP ))
  done
  DEFAULT_GENERATION_BATCH_SIZE=${ADJUSTED_BATCH_SIZE}
fi
if [[ -z "${GENERATION_BATCH_SIZE:-}" ]]; then
  GENERATION_BATCH_SIZE="${DEFAULT_GENERATION_BATCH_SIZE}"
fi

mkdir -p "${OUTPUT_DIR}"

if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "[ERROR] generation_batch_size (${GENERATION_BATCH_SIZE}) must be divisible by num_generations (${NUM_GENERATIONS})." >&2
  exit 1
fi

if (( BATCH_PER_STEP <= 0 )); then
  echo "[ERROR] Invalid batch configuration: per_device_batch (${PER_DEVICE_BATCH}) * num_processes (${NUM_PROCESSES}) must be positive." >&2
  exit 1
fi

if (( GENERATION_BATCH_SIZE % BATCH_PER_STEP != 0 )); then
  echo "[ERROR] generation_batch_size (${GENERATION_BATCH_SIZE}) must be divisible by the global batch size (${BATCH_PER_STEP})." >&2
  exit 1
fi

if (( NUM_PROCESSES > 1 )); then
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

BF16_ARGS=(--bf16 True --torch_dtype "bfloat16" --attn_implementation "flash_attention_2")
USE_CPU_ARGS=()
if [[ "${USE_CPU_FLAG}" == "True" ]]; then
  BF16_ARGS=()
  USE_CPU_ARGS=(--use_cpu True)
fi

ACC_CMD=(accelerate launch --config_file "${DEEPSPEED_CONFIG_PATH}" --num_processes "${NUM_PROCESSES}" "${PYTHON_SCRIPT}"
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --dataset_name "${DATASET_PATH}" \
  "${BF16_ARGS[@]}" \
  --gradient_checkpointing True \
  "${USE_CPU_ARGS[@]}" \
  \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate 1e-6 \
  --max_steps "${MAX_STEPS}" \
  --logging_steps 1 \
  --save_steps 500 \
  \
  --temperature 1.0 \
  --num_generations "${NUM_GENERATIONS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --beta "${BETA}" \
  --optim "${OPTIMIZER_NAME}" \
  --interleave_generation "${INTERLEAVE_GENERATION}" \
  \
  --reward_funcs "answer_correctness" "format" \
  --reward_smooth True \
  --kl_reweight True \
  --update_ref False \
  --entropy_reward True)

if [[ $# -gt 0 ]]; then
  ACC_CMD+=("$@")
fi

echo "[INFO] Launch command: ${ACC_CMD[*]}" | tee "${LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[INFO] Dry run enabled; skipping actual execution." | tee -a "${LOG_FILE}"
  exit 0
fi

"${ACC_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
