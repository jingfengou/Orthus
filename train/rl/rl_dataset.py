"""
RL dataset for visual-textual understanding in Orthus model.
This dataset provides interface for RL training.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from datasets import load_dataset

class InterleaveRLDataset(Dataset):
    """
    Dataset for RL training of Orthus model in visual-textual understanding tasks.
    This dataset provides the necessary information for computing rewards.
    """
    
    def __init__(self, 
                 dataset,  # Loaded dataset (from datasets.load_dataset)
                 image_base_dir,
                 processor,
                 vqmodel,
                 max_length=4096):
        """
        Args:
            dataset: The loaded dataset object from datasets.load_dataset
            image_base_dir: Base directory containing images
            processor: OrthusProcessor instance
            vqmodel: VQVAE model from OrthusForConditionalGeneration
            max_length: Maximum sequence length
        """
        self.data = dataset
        self.image_base_dir = image_base_dir
        self.processor = processor
        self.vqmodel = vqmodel
        self.max_length = max_length
        
        print(f"Initialized RL dataset with {len(self.data)} examples.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 1. Load Images ---
        task = item.get('Task', '')
        image_id = item.get('Image_id', '')
        
        # Question image
        question_image_path = os.path.join(self.image_base_dir, task, image_id, item.get('Combined_image', ''))
        try:
            question_image = Image.open(question_image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image at {question_image_path} not found. Using a blank image.")
            question_image = Image.new('RGB', (224, 224), (255, 255, 255))
        
        # Step images for multi-step reasoning
        step_images = []
        for step in item.get('Rotation_steps', []):
            step_image_path = os.path.join(self.image_base_dir, task, image_id, step.get('image', ''))
            try:
                img = Image.open(step_image_path).convert("RGB")
            except FileNotFoundError:
                print(f"Warning: Step image at {step_image_path} not found. Using a blank image.")
                img = Image.new('RGB', (224, 224), (255, 255, 255))
            step_images.append(img)
        
        # --- 2. Build Text Components ---
        instruction = (
            "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
            "The reasoning process and the answer are enclosed within <reasoning></reasoning> and <answer></answer> tags, "
            "respectively, i.e., <reasoning>reasoning process</reasoning>, <answer>answer</answer>."
        )
        question = item.get('Question', '')
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(item.get('Choices', []))])
        
        prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
        
        # Store ground truth for reward calculation
        ground_truth_answer = item.get('Answer', '')
        explanation_text = item.get('Explanation', '')
        
        # --- 3. Prepare Model Inputs ---
        # Combine question and step images
        all_images = [question_image] + step_images
        
        # Process with processor but don't add labels yet (for RL sampling)
        model_inputs = self.processor(
            text=prompt_text,
            images=all_images,
            vqmodel=self.vqmodel,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Extract image latents
        image_latents = model_inputs["image_latents"]
        input_image_latents = image_latents[0:1]  # Question image
        target_image_latents = image_latents[1:]  # Step images (if any)
        
        # Prepare the final data dict with all information needed for RL
        sample_data = {
            # Input components
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "image_latents": input_image_latents.squeeze(0),
            
            # Additional information for reward computation
            "prompt_text": prompt_text,
            "question_text": question,
            "choices_text": choices_text,
            "ground_truth_answer": ground_truth_answer,
            "explanation_text": explanation_text,
            
            # Reference images for alignment reward
            "question_image": question_image,
            "step_images": step_images,
            
            # Index for separating prompt from generation
            "prompt_length": len(self.processor.tokenizer.encode(prompt_text)),
            
            # If target images are needed for image generation reward
            "target_image_latents": target_image_latents if target_image_latents.size(0) > 0 else None
        }
        
        return sample_data


def custom_rl_data_collator(features):
    """
    Custom data collator for RL dataset.
    """
    # Separate the components that need special handling
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    image_latents = torch.stack([f["image_latents"] for f in features])
    
    # Other metadata that doesn't need stacking
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_latents": image_latents,
        
        # Keep other fields as lists for reward computation
        "prompt_text": [f["prompt_text"] for f in features],
        "question_text": [f["question_text"] for f in features],
        "choices_text": [f["choices_text"] for f in features],
        "ground_truth_answer": [f["ground_truth_answer"] for f in features],
        "explanation_text": [f["explanation_text"] for f in features],
        "question_image": [f["question_image"] for f in features],
        "step_images": [f["step_images"] for f in features],
        "prompt_length": [f["prompt_length"] for f in features],
        "target_image_latents": [f["target_image_latents"] for f in features]
    }
    
    return batch