#!/bin/bash

# RL training script for Orthus model with visual-textual understanding

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your available GPUs
export TOKENIZERS_PARALLELISM=false

# Training parameters
CKPT_PATH="SJTU-Deng-Lab/Orthus-7B-base"
TRAIN_FILE="/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json"
EVAL_FILE="/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json"
IMAGE_FOLDER="/data1/oujingfeng/project/twgi/datasets/mydatasets"
OUTPUT_DIR="/data1/oujingfeng/project/twgi/checkpoints/orthus-rl-test"
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-6

# PPO-specific parameters
PPO_EPOCHS=4
CLIP_EPSILON=0.2
GAMMA=0.99
LAM=0.95
ENTROPY_COEF=0.01

# Reward model parameters
ALIGNMENT_WEIGHT=0.4
REASONING_WEIGHT=0.3
ACCURACY_WEIGHT=0.3

# Training configuration
NUM_EPOCHS=10
SAVE_STEPS=100
LOG_STEPS=10

# Debug mode - if first argument is "debug", use smaller dataset
if [ "$1" == "debug" ]; then
    echo ">>> Running in DEBUG mode <<<"
    TRAIN_FILE="/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json"  # Use a small subset
    EVAL_FILE="/data1/oujingfeng/project/twgi/datasets/mydatasets/modified_data.json"   # Use a small subset
    NUM_EPOCHS=2
    BATCH_SIZE=1
    SAVE_STEPS=10
    LOG_STEPS=5
else
    echo ">>> Running in FULL TRAINING mode <<<"
fi

# Run the training script
accelerate launch --config_file /data1/oujingfeng/project/twgi/Orthus/accelerate_config.yaml \
    train_rl.py \
    --ckpt_path "$CKPT_PATH" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --ppo_epochs $PPO_EPOCHS \
    --num_train_epochs $NUM_EPOCHS \
    --clip_epsilon $CLIP_EPSILON \
    --gamma $GAMMA \
    --lam $LAM \
    --entropy_coef $ENTROPY_COEF \
    --alignment_weight $ALIGNMENT_WEIGHT \
    --reasoning_weight $REASONING_WEIGHT \
    --accuracy_weight $ACCURACY_WEIGHT \
    --save_steps $SAVE_STEPS \
    --log_steps $LOG_STEPS \
    --use_wandb true

echo "Training completed!"