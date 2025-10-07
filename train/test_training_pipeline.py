import torch
import os
import sys
import argparse
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
# --- 引用我们之前编写的 SFT 脚本中的核心类 ---
# 确保此脚本与 sft_orthus.py 在同一个文件夹下
try:
    from sft_orthus import InterleaveSFTDataset, InterleaveSFTTrainer
    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
except ImportError:
    print("Error: Could not import from sft_orthus.py or models.")
    print("Make sure this script is in the 'train/' directory and run from the project root.")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from sft_orthus import InterleaveSFTDataset, InterleaveSFTTrainer
    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration


def run_pipeline_test(args):
    """
    对 SFT 训练流程进行一个简短的端到端测试。
    """
    print("--- [Step 1/5] Loading processor and model... ---")
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        attn_implementation="flash_attention_2",
    )
    print("Model and processor loaded.")

    print("\n--- [Step 2/5] Initializing small datasets for testing... ---")
    # 为了快速测试，我们只加载少量样本
    # .select(range(N)) 是 datasets 库的一个功能，用于选取前N个样本
    # 【核心修改】: 先加载和切分，再初始化
    # a. 使用 datasets 库加载完整数据
    full_train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    full_eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")

    # b. 使用 .select() 方法切分出小子集
    train_subset = full_train_dataset.select(range(args.num_test_samples))
    eval_subset = full_eval_dataset.select(range(args.num_test_samples))
    
    print(f"Using {args.num_test_samples} samples for train and eval test.")
    # print(f"Train subset example: {train_subset[0]}")
    # c. 将切分好的小子集传入我们自定义的 SFTDataset 类
    train_dataset = InterleaveSFTDataset(
        dataset=train_subset, # <-- 传入的是dataset对象
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.vqmodel
    )
    # print(f"Target image latents shape: {train_dataset[0]['target_image_latents'].shape}")
    eval_dataset = InterleaveSFTDataset(
        dataset=eval_subset, # <-- 传入的是dataset对象
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.vqmodel
    )
    # --- 3. 定义一个“迷你”的训练配置 ---
    print("\n--- [Step 3/5] Configuring a 'mini' training job... ---")
    training_args = TrainingArguments(
        output_dir="./temp_test_trainer", # 临时输出目录
        max_steps=3,                      # 【关键】只运行3个训练步骤
        logging_steps=1,                  # 每一步都打印loss
        per_device_train_batch_size=1,    # 使用最小批次以测试
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",      # 按步骤进行评估
        eval_steps=2,                     # 在第2步后进行一次评估
        save_strategy="no",               # 测试时不保存模型
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=True,
        # attn_implementation="flash_attention_2",
    )

    def custom_data_collator(features):
        for feature in features:
            if "vqmodel" in feature:
                feature.pop("vqmodel")
        return torch.utils.data.dataloader.default_collate(features)

    # --- 4. 初始化我们自定义的 Trainer ---
    print("\n--- [Step 4/5] Initializing the custom InterleaveSFTTrainer... ---")
    trainer = InterleaveSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
        processor=processor, # 传入 processor 以便评估时解码
    )

    # --- 5. 运行短暂的训练 ---
    print("\n--- [Step 5/5] Starting the test run (3 training steps)... ---")
    try:
        trainer.train()
        print("\n\n✅ [Success] The SFT training pipeline test completed successfully!")
        print("This confirms that:")
        print("  - Data loading and preprocessing are correct.")
        print("  - The model's forward pass can compute a loss with your data.")
        print("  - The training and evaluation loops can run without crashing.")
        print("You are now ready for a full training run!")
        
    except Exception as e:
        print(f"\n\n❌ [Failure] The test run failed with an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the full SFT pipeline for Orthus.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the train.jsonl file.")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the test.jsonl file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--num_test_samples", type=int, default=16, help="Number of samples to use for the test.")
    
    args = parser.parse_args()
    
    run_pipeline_test(args)