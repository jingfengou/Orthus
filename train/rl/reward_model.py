"""
Reward model for visual-textual understanding in Orthus model.
This module implements reward functions for RL training.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

class VisualTextualRewardModel(nn.Module):
    """
    A reward model that evaluates the quality of generated text and images
    in the context of visual-textual understanding tasks.
    """
    
    def __init__(self, 
                 clip_model_name="openai/clip-vit-base-patch32",
                 use_aesthetic_score=True,
                 weights={'alignment': 0.4, 'reasoning': 0.3, 'accuracy': 0.3}):
        super().__init__()
        
        # Load CLIP model for visual-text alignment
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        self.use_aesthetic_score = use_aesthetic_score
        self.weights = weights
        
        # Initialize aesthetic predictor (simplified version)
        if self.use_aesthetic_score:
            self.aesthetic_predictor = nn.Linear(512, 1)  # CLIP vision feature dim is 512 for base model
        
    def compute_visual_text_alignment(self, generated_texts, reference_images):
        """
        Compute alignment between generated text and reference images using CLIP.
        
        Args:
            generated_texts: List of generated text strings
            reference_images: List of PIL Images or tensors
            
        Returns:
            alignment_scores: Tensor of shape (batch_size,)
        """
        # Process images
        inputs = self.clip_processor(
            text=generated_texts,
            images=reference_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.clip_model.device)
        
        # Get similarity score
        outputs = self.clip_model(**inputs)
        logits_per_text = outputs.logits_per_text  # Shape: (batch_size, num_images)
        
        # For our case, we have one image per text, so take diagonal
        alignment_scores = torch.diag(logits_per_text)
        
        # Normalize to [0, 1] range using sigmoid
        alignment_scores = torch.sigmoid(alignment_scores)
        
        return alignment_scores
    
    def compute_reasoning_consistency(self, generated_reasoning, question):
        """
        Compute consistency of generated reasoning with the question.
        This is a simplified implementation - in practice, you might use 
        a more sophisticated reasoning evaluator.
        
        Args:
            generated_reasoning: Generated reasoning text
            question: Original question text
            
        Returns:
            consistency_scores: Tensor of consistency scores
        """
        # Simple keyword overlap as a placeholder
        # In practice, use a trained reasoning evaluator or entailment model
        batch_size = len(generated_reasoning)
        consistency_scores = torch.zeros(batch_size)
        
        for i, (reason, q) in enumerate(zip(generated_reasoning, question)):
            # Simple overlap ratio (placeholder)
            reason_words = set(reason.lower().split())
            question_words = set(q.lower().split())
            overlap = len(reason_words.intersection(question_words))
            total = len(question_words)
            if total > 0:
                consistency_scores[i] = min(overlap / total, 1.0)
        
        return consistency_scores
    
    def compute_answer_accuracy(self, generated_answers, correct_answers):
        """
        Compute accuracy of generated answers compared to correct answers.
        
        Args:
            generated_answers: List of generated answer strings
            correct_answers: List of correct answer strings
            
        Returns:
            accuracy_scores: Tensor of accuracy scores (0-1)
        """
        batch_size = len(generated_answers)
        accuracy_scores = torch.zeros(batch_size)
        
        for i, (gen_ans, corr_ans) in enumerate(zip(generated_answers, correct_answers)):
            # Simple exact match (can be extended to semantic similarity)
            if gen_ans.strip().lower() == corr_ans.strip().lower():
                accuracy_scores[i] = 1.0
            else:
                # Partial credit for similar answers
                # This is a simple heuristic - can be replaced with better metrics
                gen_words = set(gen_ans.lower().split())
                corr_words = set(corr_ans.lower().split())
                if len(corr_words) > 0:
                    overlap = len(gen_words.intersection(corr_words))
                    accuracy_scores[i] = overlap / len(corr_words)
        
        return accuracy_scores
    
    def compute_aesthetic_score(self, generated_images):
        """
        Compute aesthetic quality of generated images.
        Args:
            generated_images: Tensor of generated images (batch, channels, height, width)
        Returns:
            aesthetic_scores: Tensor of aesthetic scores (0-1)
        """
        if not self.use_aesthetic_score or generated_images is None:
            return torch.zeros(generated_images.shape[0] if generated_images is not None else 1)
        
        # Get image features from CLIP visual encoder
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                pixel_values=generated_images
            )
        
        # Predict aesthetic score
        aesthetic_scores = torch.sigmoid(self.aesthetic_predictor(image_features))
        return aesthetic_scores.squeeze(-1)
    
    def forward(self, 
                generated_texts=None, 
                generated_images=None,
                reference_images=None,
                questions=None,
                correct_answers=None):
        """
        Compute total reward for generated outputs.
        
        Args:
            generated_texts: Generated text outputs (list of strings or tensors)
            generated_images: Generated image outputs (tensors, optional)
            reference_images: Reference/input images
            questions: Original questions
            correct_answers: Correct answers for evaluation
            
        Returns:
            total_rewards: Tensor of total rewards for each sample in batch
        """
        batch_size = len(generated_texts) if generated_texts is not None else 1
        
        # Initialize reward components
        alignment_reward = torch.zeros(batch_size)
        reasoning_reward = torch.zeros(batch_size)
        accuracy_reward = torch.zeros(batch_size)
        aesthetic_reward = torch.zeros(batch_size)
        
        # Compute alignment reward
        if generated_texts is not None and reference_images is not None:
            alignment_reward = self.compute_visual_text_alignment(generated_texts, reference_images)
        
        # Compute reasoning reward
        if generated_texts is not None and questions is not None:
            reasoning_reward = self.compute_reasoning_consistency(generated_texts, questions)
        
        # Compute answer accuracy reward
        if generated_texts is not None and correct_answers is not None:
            accuracy_reward = self.compute_answer_accuracy(generated_texts, correct_answers)
        
        # Compute aesthetic reward for generated images
        if generated_images is not None:
            aesthetic_reward = self.compute_aesthetic_score(generated_images)
        
        # Combine rewards with weights
        total_rewards = (
            self.weights['alignment'] * alignment_reward +
            self.weights['reasoning'] * reasoning_reward +
            self.weights['accuracy'] * accuracy_reward +
            aesthetic_reward  # Only add if we have generated images
        )
        
        return {
            'total_reward': total_rewards,
            'alignment': alignment_reward,
            'reasoning': reasoning_reward,
            'accuracy': accuracy_reward,
            'aesthetic': aesthetic_reward
        }


class RewardConfig:
    """
    Configuration class for reward model parameters.
    """
    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch32",
                 use_aesthetic_score=True,
                 weights=None,
                 alignment_weight=0.4,
                 reasoning_weight=0.3,
                 accuracy_weight=0.3):
        
        self.clip_model_name = clip_model_name
        self.use_aesthetic_score = use_aesthetic_score
        self.weights = weights or {
            'alignment': alignment_weight,
            'reasoning': reasoning_weight,
            'accuracy': accuracy_weight
        }