#!/usr/bin/env bash
# 统一管理 Orthus × Verl 训练入口：test/ train 共用 8 卡配置，test 仅缩短步数/轮次
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python}

if [[ $# -lt 1 ]]; then
    echo "用法: $0 {test|train} [额外的 Hydra 覆盖参数]" >&2
    exit 1
fi

MODE=$1
shift

declare -a CMD
CMD=("$PYTHON_BIN" "$SCRIPT_DIR/run_orthus_grpo.py")
declare -a OVERRIDES=()

LOG_ROOT=${ORTHUS_VERL_LOG_DIR:-"$SCRIPT_DIR/../../logs"}
mkdir -p "$LOG_ROOT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_ROOT/${MODE}_${TIMESTAMP}.log"

# 默认使用 8 卡；若用户已有设置则尊重
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ORTHUS_NUM_GPUS="${ORTHUS_NUM_GPUS:-8}"

case "$MODE" in
    test)
        export ORTHUS_MAX_SAMPLES="${ORTHUS_MAX_SAMPLES:--1}"
        export ORTHUS_NUM_GENERATIONS="${ORTHUS_NUM_GENERATIONS:-2}"
        TEST_TOTAL_EPOCHS=${ORTHUS_TEST_TOTAL_EPOCHS:-1}
        TEST_TOTAL_STEPS=${ORTHUS_TEST_TOTAL_STEPS:-10}
        OVERRIDES+=("trainer.total_epochs=${TEST_TOTAL_EPOCHS}")
        OVERRIDES+=("trainer.total_training_steps=${TEST_TOTAL_STEPS}")
        ;;
    train)
        ;;
    *)
        echo "未知模式: $MODE" >&2
        echo "用法: $0 {test|train} [额外的 Hydra 覆盖参数]" >&2
        exit 1
        ;;
esac

# 默认 PPO micro batch 尺寸（每卡），缺省取 1；可设置 ORTHUS_PPO_MICRO_BATCH_SIZE_PER_GPU 或命令行覆写
PPO_MICRO_BATCH_SIZE_PER_GPU=${ORTHUS_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
OVERRIDES+=("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}")

# 默认总 mini batch 尺寸，若未指定则与每卡尺寸×GPU 数保持一致
DEFAULT_MINI_BATCH=$((PPO_MICRO_BATCH_SIZE_PER_GPU * ORTHUS_NUM_GPUS))
PPO_MINI_BATCH_SIZE=${ORTHUS_PPO_MINI_BATCH_SIZE:-$DEFAULT_MINI_BATCH}
OVERRIDES+=("actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}")

# Rollout log-prob 计算的 micro batch，默认与 actor 设置一致，避免校验失败
ROLLOUT_LOGPROB_BATCH=${ORTHUS_ROLLOUT_LOGPROB_BATCH:-$PPO_MICRO_BATCH_SIZE_PER_GPU}
OVERRIDES+=("actor_rollout_ref.rollout.log_prob_micro_batch_size=${ROLLOUT_LOGPROB_BATCH}")

if [[ $# -gt 0 ]]; then
    OVERRIDES+=("$@")
fi

FULL_CMD=("${CMD[@]}" "${OVERRIDES[@]}")

echo "[Orthus] 运行模式: $MODE"
echo "[Orthus] 日志文件: $LOG_FILE"
echo "[Orthus] 命令: ${FULL_CMD[*]}"

"${FULL_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
