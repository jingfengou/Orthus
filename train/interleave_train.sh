# # NPROC_PER_NODE 是你希望使用的 GPU 数量
# export NPROC_PER_NODE=8

# accelerate launch --num_processes ${NPROC_PER_NODE} train_interleave_orthus.py \
#     --ckpt_path "SJTU-Deng-Lab/Orthus-7B-instruct" \
#     --train_file "/data1/oujingfeng/project/twgi/datasets/SpatialViz/datasets/train.jsonl" \
#     --eval_file "/data1/oujingfeng/project/twgi/datasets/SpatialViz/datasets/test.jsonl" \
#     --image_folder "/data1/oujingfeng/project/twgi/datasets/SpatialViz" \
#     --output_dir "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-think-v1" \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --learning_rate 5e-6 \
#     --warmup_ratio 0.03 \
#     --logging_steps 5 \
#     --eval_steps 50 \
#     --save_steps 200 \
#     --save_total_limit 3 \
#     --bf16 \
#     --gradient_checkpointing \
#     --report_to "wandb" \
#     # --generation_log_file "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-v4/generation_log.jsonl"

#!/bin/bash

# 默认设置（用于多GPU正式训练）
LAUNCH_CMD="accelerate launch --num_processes 8"
BATCH_SIZE=1
# REPORT_TO="none"     # <--- 关闭 wandb
REPORT_TO="wandb"
EPOCHS=100
GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"
DEBUG_MODE_FLAG=""

# 检查第一个参数是否为 "debug"
if [ "$1" == "debug" ]; then
  echo ">>> Running in DEBUG mode <<<"
  # --- 调试模式设置 ---
  LAUNCH_CMD="python"  # <--- 使用 python 直接启动，单进程单GPU
  BATCH_SIZE=1         # <--- 使用极小的批量大小
  REPORT_TO="none"     # <--- 关闭 wandb
  EPOCHS=1             # <--- 只训练一个 epoch
  GRADIENT_CHECKPOINTING_FLAG="" # <--- 在调试时通常不开启
  DEBUG_MODE_FLAG="--debug_mode" # <--- 传递给 python 脚本的标志
else
  echo ">>> Running in Multi-GPU TRAINING mode <<<"
fi

# 使用变量执行命令，保持代码整洁
$LAUNCH_CMD train_interleave_orthus.py \
    --ckpt_path "SJTU-Deng-Lab/Orthus-7B-instruct" \
    --train_file "/data1/oujingfeng/project/twgi/datasets/mydatasets/metadata.json" \
    --eval_file "/data1/oujingfeng/project/twgi/datasets/mydatasets/metadata.json" \
    --image_folder "/data1/oujingfeng/project/twgi/datasets/mydatasets" \
    --output_dir "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v1" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 \
    --report_to $REPORT_TO \
    $GRADIENT_CHECKPOINTING_FLAG \
    $DEBUG_MODE_FLAG \
    # --generation_log_file "/data1/oujingfeng/project/twgi/checkpoints/orthus-7b-sft-v4/generation_log.jsonl"