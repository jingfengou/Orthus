import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from custom_rewards import answer_correctness_reward, format_reward
from grpo_trainer_orthus import (
    GRPOConfig,
    Geneval_score,
    OrthusGRPOTrainer,
    reward_funcs_registry as trainer_reward_registry,
)

# ------------------------------------------------------------------
#  这个 GRPOConfig 继承自 TRL 的配置，并增加了 Orthus 特有的参数
#  它对标您项目中的 grpo.py 里的 GRPOConfig
# ------------------------------------------------------------------
@dataclass
class OrthusGRPOConfig(GRPOConfig):
    """
    为 Orthus GRPO 训练脚本定制的配置类。
    """
    # 奖励模型和数据集相关
    # reward_ckpt_path_file: str = field(default=None, metadata={"help": "包含奖励模型路径的 JSON 文件。"})
    
    # Orthus 生成特定参数
    cfg_weight: float = field(default=5.0, metadata={"help": "图像生成时使用的 CFG (Classifier-Free Guidance) 权重。"})
    
    # GRPO 算法变体控制
    reward_smooth: bool = field(default=False, metadata={"help": "是否使用基于图像相似度的奖励平滑。"})
    kl_reweight: bool = field(default=False, metadata={"help": "是否使用基于图像相似度的 KL 散度重加权。"})
    update_ref: bool = field(default=False, metadata={"help": "是否在训练中途更新参考模型。"})
    entropy_reward: bool = field(default=False, metadata={"help": "是否加入熵奖励项。"})
    image_latent_cache_size: int = field(
        default=256,
        metadata={"help": "图像 latent 的 LRU 缓存大小（0 表示禁用）。"},
    )
    interleave_generation: bool = field(
        default=False,
        metadata={"help": "是否使用文本-图像交替生成（仅对文本部分计算奖励和损失）。"},
    )

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    脚本参数，定义要使用的奖励函数。
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["geneval"],
        metadata={"help": "要使用的奖励函数列表，例如: 'geneval', 'hps' 等。"},
    )
    image_base_dir: str = field(
        default="/data1/oujingfeng/project/twgi/datasets/mydatasets",
        metadata={"help": "图像根目录，用于加载题目图片。"},
    )

def set_seed(seed: int):
    """为保证可复现性设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


reward_funcs_registry = {
    "geneval": Geneval_score,
    "answer_correctness": answer_correctness_reward,
    "format": format_reward,
}
reward_funcs_registry.update(trainer_reward_registry)

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # 1. 加载数据集
    print("--- [Step 1/3] Loading dataset... ---")
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    train_len = len(dataset.get(script_args.dataset_train_split, []))
    print(f"Dataset '{script_args.dataset_name}' loaded successfully with {train_len} samples.")

    # 2. 数据集预处理/格式化
    #    这个函数将原始数据格式化为模型需要的对话格式
    def format_dataset(example):
        instruction = (
            "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
            "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
            "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        )
        question = example.get('Question', '')
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(example.get('Choices', []))])

        # 核心 Prompt 结构，与 SFT 和推理时保持一致
        prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
        image_rel = Path(example.get('Task', '')) / example.get('Image_id', '') / example.get('Combined_image', '')
        image_base = Path(script_args.image_base_dir).expanduser()
        image_path = (image_base / image_rel).resolve()

        return {
            "prompt": prompt_text,  # Trainer 将使用这个字段
            "image_path": str(image_path),
            "metadata": dict(example),  # Geneval 等奖励函数需要原始元数据
        }

    dataset = dataset.map(
        format_dataset,
        num_proc=getattr(script_args, 'dataset_num_proc', 1),
    )
    print("Dataset formatted.")

    # 3. 检查并恢复 Checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # ### 新增 ###
    # 在初始化 Trainer 之前，解析并设置奖励函数
    # 这样 Trainer 就能知道要调用哪些函数了
    print(f"Enabled reward functions: {script_args.reward_funcs}")
    enabled_reward_funcs = []
    for func_name in script_args.reward_funcs:
        if func_name in reward_funcs_registry:
            # 如果是模型类（如 Geneval），则实例化；如果是函数，则直接引用
            if isinstance(reward_funcs_registry[func_name], type):
                 enabled_reward_funcs.append(reward_funcs_registry[func_name](training_args))
            else:
                 enabled_reward_funcs.append(reward_funcs_registry[func_name])
        else:
            raise ValueError(f"Unknown reward function: {func_name}")

    # 4. 初始化自定义 Trainer
    print("--- [Step 2/3] Initializing OrthusGRPOTrainer... ---")
    trainer = OrthusGRPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=None, # Trainer 内部会自动创建
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset.get(script_args.dataset_test_split),
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        # ### 新增 ###
        # 将解析好的奖励函数实例/引用列表传递给 Trainer
        reward_funcs=enabled_reward_funcs,
        # 传递额外的脚本参数给 Trainer
        script_args=script_args,
    )
    print("Trainer initialized.")

    # 5. 开始训练
    print("--- [Step 3/3] Starting GRPO training... ---")
    checkpoint_to_resume = None
    if training_args.resume_from_checkpoint is True and last_checkpoint is not None:
        checkpoint_to_resume = last_checkpoint
        
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 6. 保存最终模型
    print("--- Training finished. Saving final model... ---")
    trainer.save_model(training_args.output_dir)
    trainer.processor.save_pretrained(training_args.output_dir)
    print(f"Final model and processor saved to {training_args.output_dir}")


if __name__ == "__main__":
    # 使用 TRL 的解析器来同时解析三种配置
    parser = TrlParser((GRPOScriptArguments, OrthusGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
