"""
Main training script for RL-based Orthus model training.
"""
import argparse
import torch
import os
import sys
from datasets import load_dataset

# Add the root directory to sys.path to import custom modules
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)

from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
from reward_model import VisualTextualRewardModel, RewardConfig
from rl_dataset import InterleaveRLDataset
from rl_trainer import PPORLTrainer, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="RL training script for Orthus model with visual-textual understanding.")
    
    # Model and data paths
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data (train.jsonl).")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation data (test.jsonl).")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the base folder containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for PPO.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Mini-batch size for PPO updates.")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per batch.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument("--lam", type=float, default=0.95, help="Lambda for GAE.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient.")
    
    # Reward model configuration
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", 
                        help="Name of the CLIP model to use for reward computation.")
    parser.add_argument("--alignment_weight", type=float, default=0.4, help="Weight for alignment reward.")
    parser.add_argument("--reasoning_weight", type=float, default=0.3, help="Weight for reasoning reward.")
    parser.add_argument("--accuracy_weight", type=float, default=0.3, help="Weight for answer accuracy reward.")
    parser.add_argument("--use_aesthetic_score", type=str, choices=['true', 'false'], default='false',
                        help="Whether to use aesthetic score for generated images.")
    
    # Training configuration
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--use_wandb", type=str, choices=['true', 'false'], default='true',
                        help="Whether to use wandb for logging.")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    
    args = parser.parse_args()
    
    # Convert string args to boolean
    args.use_aesthetic_score = args.use_aesthetic_score.lower() == 'true'
    args.use_wandb = args.use_wandb.lower() == 'true'
    
    return args


def main():
    args = parse_args()
    
    print("Loading model and processor...")
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",  # Comment out to control device placement manually
    )
    
    # Move model to device
    model.to(args.device)
    
    print("Loading reward model...")
    reward_config = RewardConfig(
        clip_model_name=args.clip_model_name,
        use_aesthetic_score=args.use_aesthetic_score,
        alignment_weight=args.alignment_weight,
        reasoning_weight=args.reasoning_weight,
        accuracy_weight=args.accuracy_weight
    )
    reward_model = VisualTextualRewardModel(**reward_config.__dict__)
    reward_model.to(args.device)
    
    print("Loading datasets...")
    train_dataset_raw = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset_raw = load_dataset("json", data_files=args.eval_file, split="train")
    
    # Limit dataset size for debugging
    # train_dataset_raw = train_dataset_raw.select(range(100))  # Use full dataset in actual training
    # eval_dataset_raw = eval_dataset_raw.select(range(20))   # Use full dataset in actual training
    
    train_dataset = InterleaveRLDataset(
        dataset=train_dataset_raw,
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.vqmodel,  # Access the vqmodel from the model
        max_length=args.max_length
    )
    
    eval_dataset = InterleaveRLDataset(
        dataset=eval_dataset_raw,
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.vqmodel,  # Access the vqmodel from the model
        max_length=args.max_length
    )
    
    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        gamma=args.gamma,
        lam=args.lam,
        entropy_coef=args.entropy_coef,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )
    
    # Create trainer
    trainer = PPORLTrainer(
        model=model,
        reward_model=reward_model,
        config=ppo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor
    )
    
    print("Starting RL training...")
    trainer.train(num_epochs=args.num_train_epochs)
    
    print("Training completed!")
    
    # Final evaluation
    print("Running final evaluation...")
    final_reward = trainer.evaluate()
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    processor.save_pretrained(final_output_dir)
    print(f"Final model saved to {final_output_dir}")
    
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()