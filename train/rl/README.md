# Reinforcement Learning Training for Orthus Model

This directory contains the implementation for training the Orthus model using Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO).

## Overview

The RL training approach transforms the original supervised fine-tuning (SFT) task into a reinforcement learning problem where:
- The **policy** is the Orthus language model that generates text and interacts with images
- The **environment** consists of the visual-textual understanding tasks from the dataset
- The **reward function** evaluates the quality of generated responses based on:
  - Visual-text alignment (using CLIP)
  - Reasoning consistency 
  - Answer accuracy
  - (Optional) Aesthetic quality of generated images

## File Structure

```
rl/
├── reward_model.py     # Defines reward functions for visual-textual understanding
├── rl_dataset.py       # RL-specific dataset wrapper
├── rl_trainer.py       # PPO trainer implementation
├── train_rl.py         # Main training script
├── train_rl.sh         # Training shell script
└── README.md           # This file
```

## Reward Model Components

The reward model combines multiple components:

1. **Visual-Text Alignment** (`alignment_weight`): Measures how well the generated text describes the input image using CLIP similarity
2. **Reasoning Consistency** (`reasoning_weight`): Evaluates if the generated reasoning aligns with the question
3. **Answer Accuracy** (`accuracy_weight`): Checks if the final answer matches the expected response

## Training Process

The PPO training loop follows these steps:

1. **Rollout Phase**: 
   - Sample a batch of prompts from the dataset
   - Generate responses using the current policy
   - Compute rewards using the reward model
   - Calculate old policy log probabilities

2. **Optimization Phase**:
   - Update the policy using PPO objective with clipped surrogate function
   - Apply entropy regularization to encourage exploration

## Usage

### Quick Start

```bash
cd /data1/oujingfeng/project/twgi/Orthus/train/rl
bash train_rl.sh
```

### Debug Mode

```bash
bash train_rl.sh debug
```

### Custom Training

```bash
python train_rl.py \
    --ckpt_path "SJTU-Deng-Lab/Orthus-7B-base" \
    --train_file "/path/to/train.jsonl" \
    --eval_file "/path/to/eval.jsonl" \
    --image_folder "/path/to/images" \
    --output_dir "/path/to/output" \
    --learning_rate 1e-6 \
    --batch_size 4 \
    --ppo_epochs 4 \
    --num_train_epochs 10
```

## Configuration Options

- `learning_rate`: PPO learning rate (default: 1e-6)
- `batch_size`: Batch size for training (default: 4)
- `ppo_epochs`: Number of PPO epochs per batch (default: 4)
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `gamma`: Discount factor for rewards (default: 0.99)
- `lam`: Lambda for GAE (default: 0.95)
- `entropy_coef`: Entropy regularization coefficient (default: 0.01)

Reward weights:
- `alignment_weight`: Weight for visual-text alignment (default: 0.4)
- `reasoning_weight`: Weight for reasoning consistency (default: 0.3)
- `accuracy_weight`: Weight for answer accuracy (default: 0.3)

## Requirements

- PyTorch >= 1.12
- Transformers
- Datasets
- wandb (for logging)
- accelerate

## Notes

- The current implementation focuses on text generation and evaluation of visual-text understanding
- Image generation capabilities can be added by extending the reward model to include aesthetic metrics
- For best results, tune reward weights based on your specific task requirements
- Monitor reward values during training to detect reward hacking or optimization issues