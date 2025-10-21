"""
RL trainer for visual-textual understanding in Orthus model.
Implements PPO-based training for RLHF.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import PreTrainedModel
from typing import Dict, List, Tuple, Optional
import wandb
import os
from tqdm import tqdm

# Import from our modules
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
from .reward_model import VisualTextualRewardModel, RewardConfig
from .rl_dataset import InterleaveRLDataset, custom_rl_data_collator


class PPOConfig:
    """
    Configuration for PPO training
    """
    def __init__(
        self,
        learning_rate: float = 1e-6,
        batch_size: int = 4,
        mini_batch_size: int = 2,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.25,
        max_grad_norm: float = 1.0,
        use_wandb: bool = True,
        output_dir: str = "./rl_checkpoints",
        save_steps: int = 100,
        log_steps: int = 10,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.log_steps = log_steps


class PPORLTrainer:
    """
    PPO trainer for Orthus model with visual-textual understanding
    """
    def __init__(
        self,
        model: OrthusForConditionalGeneration,
        reward_model: VisualTextualRewardModel,
        config: PPOConfig,
        train_dataset: InterleaveRLDataset,
        eval_dataset: Optional[InterleaveRLDataset] = None,
        processor=None,
    ):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processor = processor

        # Move models to device
        self.device = next(model.parameters()).device
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup logging
        if config.use_wandb:
            wandb.init(project="orthus-rl", config=config.__dict__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Store generated sequences for reward computation
        self.stored_sequences = []
        self.stored_rewards = []
    
    def generate_with_grad(self, input_ids, attention_mask, image_latents, max_new_tokens=256):
        """
        Generate tokens with gradient tracking enabled.
        This method performs autoregressive generation with gradients enabled.
        """
        self.model.eval()  # Set to eval to disable dropout etc.
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Expand latents if needed
        if image_latents.size(0) != batch_size:
            image_latents = image_latents.expand(batch_size, -1, -1, -1)
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        cur_len = input_ids.size(1)
        max_len = cur_len + max_new_tokens

        # Create attention mask for the full sequence
        attention_mask = attention_mask.clone()
        
        with torch.enable_grad():
            for i in range(max_new_tokens):
                # Get model outputs
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    image_latents=image_latents,
                    use_cache=False  # Disable cache for consistent gradients
                )
                
                # Get next token logits (last position)
                next_token_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                next_token_logits = next_token_logits / 0.7  # temperature
                
                # Sample next token
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)  # (batch, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Update attention mask
                new_attn_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
                
                attention_mask = new_attn_mask
                
                # Check if all sequences have generated EOS token
                if torch.all(next_token == self.processor.tokenizer.eos_token_id):
                    break
        
        return generated

    def rollout(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Generate sequences and compute rewards
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        image_latents = batch['image_latents'].to(self.device)
        
        # Generate sequences
        generated_sequences = self.generate_with_grad(
            input_ids, 
            attention_mask, 
            image_latents,
            max_new_tokens=256  # Adjust based on your task
        )
        
        # Decode generated sequences for reward model
        generated_texts = []
        prompt_lengths = [f for f in batch['prompt_length']]  # Get actual prompt lengths
        
        for i, seq in enumerate(generated_sequences):
            # Extract only the generated part (after prompt)
            prompt_len = prompt_lengths[i]
            generated_part = seq[prompt_len:]
            decoded_text = self.processor.tokenizer.decode(generated_part, 
                                                          skip_special_tokens=True)
            generated_texts.append(decoded_text)
        
        # Compute rewards using reward model
        reference_images = batch['question_image']  # PIL images
        questions = batch['question_text']
        correct_answers = batch['ground_truth_answer']
        
        rewards_dict = self.reward_model(
            generated_texts=generated_texts,
            reference_images=reference_images,
            questions=questions,
            correct_answers=correct_answers
        )
        
        rewards = rewards_dict['total_reward'].to(self.device)
        
        # Calculate old log probabilities for the generated sequences
        # We need the log probabilities that the current policy assigned to the actions it took
        with torch.no_grad():
            # Create attention mask for the generated sequences
            full_attention_mask = torch.ones_like(generated_sequences).to(self.device)
            full_attention_mask[generated_sequences == self.processor.tokenizer.pad_token_id] = 0
            
            # Run the current policy on the generated sequences to get log probabilities
            old_outputs = self.model(
                input_ids=generated_sequences,
                attention_mask=full_attention_mask,
                image_latents=image_latents.expand(generated_sequences.size(0), -1, -1, -1),  # Expand to match batch size
            )
            old_logits = old_outputs.logits
            old_log_probs_all = F.log_softmax(old_logits, dim=-1)
            
            # Get the log probabilities for the tokens that were actually generated
            # The logits are for predicting the next token, so we align them with the generated tokens
            generated_tokens = generated_sequences[:, 1:]  # Tokens we generated (excluding first)
            old_log_probs_selected = torch.gather(
                old_log_probs_all[:, :-1, :],  # Logits for next tokens (excluding last)
                dim=-1,
                index=generated_tokens.unsqueeze(-1)  # Shape: [batch, seq_len-1, 1]
            ).squeeze(-1)  # Shape: [batch, seq_len-1]
        
        return generated_sequences, old_log_probs_selected, rewards, generated_texts

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages. In our simple implementation, just return normalized rewards.
        For a complete implementation, you would need a value function.
        """
        # Normalize rewards to have zero mean and unit variance
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    def ppo_step(self, generated_sequences: torch.Tensor, old_log_probs: torch.Tensor, rewards: torch.Tensor):
        """
        Perform a single PPO optimization step
        """
        # Get the original input IDs and attention mask from the current batch
        input_ids = self.orig_input_ids
        image_latents = self.current_image_latents
        batch_size = input_ids.size(0)
        
        # Get the attention mask for the full generated sequence
        attention_mask = torch.ones_like(generated_sequences).to(self.device)
        attention_mask[generated_sequences == self.processor.tokenizer.pad_token_id] = 0
        
        # Run model on generated sequences to get new logits
        # Expand image_latents to match the batch size of generated sequences
        expanded_image_latents = image_latents.expand(generated_sequences.size(0), -1, -1, -1)
        
        outputs = self.model(
            input_ids=generated_sequences,
            attention_mask=attention_mask,
            image_latents=expanded_image_latents,
        )
        new_logits = outputs.logits
        new_log_probs_all = F.log_softmax(new_logits, dim=-1)
        
        # Get new log probs for the tokens that were actually generated (excluding the first one to align)
        generated_tokens = generated_sequences[:, 1:]  # Skip first token to align with logits
        new_log_probs_selected = torch.gather(
            new_log_probs_all[:, :-1, :],  # Remove last logit to align with tokens
            dim=-1, 
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute ratio between new and old policies
        # Ensure both tensors have the same shape
        if new_log_probs_selected.shape != old_log_probs.shape:
            min_len = min(new_log_probs_selected.shape[1], old_log_probs.shape[1])
            new_log_probs_selected = new_log_probs_selected[:, :min_len]
            old_log_probs = old_log_probs[:, :min_len]
        
        ratio = torch.exp(new_log_probs_selected - old_log_probs.detach())  # Detach old_log_probs to avoid backprop
        
        # Compute advantages based on rewards
        advantages = self.compute_advantages(rewards)
        
        # Expand advantages to match token dimensions
        # Each sequence gets its own advantage value replicated across its tokens
        seq_advantages = advantages.unsqueeze(1).expand(-1, new_log_probs_selected.size(1))
        
        # PPO objective
        surr1 = ratio * seq_advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * seq_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus to encourage exploration
        entropy = -(new_log_probs_all * torch.exp(new_log_probs_all)).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        # Backpropagate
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item()
        }

    def train(self, num_epochs: int = 10):
        """
        Main training loop
        """
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=custom_rl_data_collator
        )

        global_step = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Rollout: generate sequences and compute rewards
                generated_sequences, old_log_probs, rewards, generated_texts = self.rollout(batch)
                
                # Store current batch data for PPO step
                self.current_image_latents = batch['image_latents'].to(self.device)
                self.orig_input_ids = batch['input_ids'].to(self.device)
                
                # Perform multiple PPO epochs
                for ppo_epoch in range(self.config.ppo_epochs):
                    loss_dict = self.ppo_step(generated_sequences, old_log_probs, rewards)
                
                # Logging
                if global_step % self.config.log_steps == 0:
                    avg_reward = rewards.mean().item()
                    print(f"Step {global_step}, Reward: {avg_reward:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "epoch": epoch,
                            "step": global_step,
                            "reward": avg_reward,
                            "policy_loss": loss_dict['policy_loss'],
                            "entropy_loss": loss_dict['entropy_loss'],
                            "total_loss": loss_dict['total_loss'],
                            "entropy": loss_dict['entropy']
                        })
                
                # Save checkpoints
                if global_step % self.config.save_steps == 0 and global_step > 0:
                    checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    self.model.save_pretrained(checkpoint_path)
                    self.processor.save_pretrained(checkpoint_path)
                
                global_step += 1

    def evaluate(self):
        """
        Evaluation function
        """
        if self.eval_dataset is None:
            print("No evaluation dataset provided")
            return
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=custom_rl_data_collator
        )
        
        self.model.eval()
        total_reward = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                _, _, rewards, _ = self.rollout(batch)
                total_reward += rewards.sum().item()
                num_samples += len(rewards)
        
        avg_reward = total_reward / num_samples
        print(f"Evaluation Average Reward: {avg_reward:.4f}")
        
        if self.config.use_wandb:
            wandb.log({"eval_avg_reward": avg_reward})
        
        return avg_reward