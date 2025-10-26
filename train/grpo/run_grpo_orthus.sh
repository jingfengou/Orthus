#!/bin/bash

# 环境变量设置
export WANDB_PROJECT="orthus-grpo-project"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# --- 可配置参数 ---
# 基础模型路径，应替换为您微调好的 Orthus SFT 模型路径
MODEL_PATH="/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-base-sample80b100ep500l1e-5-weight-F"
# 数据集路径
DATASET_PATH="/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json"
# DeepSpeed 配置文件路径
DEEPSPEED_CONFIG_PATH="/data1/oujingfeng/project/twgi/Orthus/accelerate_config.yaml"
# 奖励模型配置文件路径
# REWARD_CONFIG_PATH="./reward_paths.json"
# 输出目录
RUN_NAME="orthus-7b-grpo-geneval-v1"
OUTPUT_DIR="/data1/oujingfeng/project/twgi/checkpoints/grpo/${RUN_NAME}"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# --- torchrun/accelerate launch 命令 ---
# 使用 accelerate launch 更为推荐，因为它与 transformers Trainer 结合更紧密
accelerate launch --config_file ${DEEPSPEED_CONFIG_PATH} --num_processes=8 grpo_orthus.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    --dataset_name ${DATASET_PATH} \
    --bf16 \
    --torch_dtype "bfloat16" \
    --gradient_checkpointing \
    --attn_implementation "flash_attention_2" \
    \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --max_steps 2000 \
    --logging_steps 1 \
    --save_steps 100 \
    \
    --temperature 1.0 \
    --num_generations 4 \
    --beta 0.05 \
    \
    --reward_funcs "answer_correctness" "format" \
    --reward_smooth True \
    --kl_reweight True \
    --update_ref False \
    --entropy_reward True