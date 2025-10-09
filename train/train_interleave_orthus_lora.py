import torch
import os
import sys
import argparse
import logging
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from typing import Dict, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig # 用于QLoRA
# --- 引用项目中的核心类 ---
# 确保此脚本与 sft_orthus.py 在同一个文件夹下，或项目根目录已添加到 PYTHONPATH
try:
    from interleave_sft_orthus import InterleaveSFTDataset, InterleaveSFTTrainer
    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
except ImportError:
    print("Error: Could not import custom modules.")
    print("Attempting to add project root to sys.path...")
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    print(f"Added '{root_path}' to path.")
    try:
        from sft_orthus import InterleaveSFTDataset, InterleaveSFTTrainer
        from models.processing_orthus import OrthusProcessor
        from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
    except ImportError as e:
        print(f"Failed to import after path adjustment: {e}")
        sys.exit(1)

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    主训练函数，负责解析参数、加载数据和模型、启动训练流程。
    """
    parser = argparse.ArgumentParser(description="Formal SFT training script for Orthus model.")
    
    # --- 数据和模型路径参数 ---
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrained base model checkpoint.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data (train.jsonl).")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation data (test.jsonl).")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the base folder containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    # --- 【新增代码】: 添加 debug_mode 标志 ---
    parser.add_argument("--debug_mode", action='store_true', help="Enable debug mode: uses a small subset of data for quick checks.")

    # --- 核心训练超参数 ---
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU for evaluation.")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of total steps for learning rate warmup.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type (e.g., 'linear', 'cosine').")

    # --- 训练控制参数 ---
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Evaluation strategy.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps (if evaluation_strategy is 'steps').")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Checkpoint saving strategy.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Limit the total number of checkpoints to save.")
    parser.add_argument("--bf16", action='store_true', default=True, help="Enable bfloat16 training.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="The integration to report results to (e.g., 'wandb', 'tensorboard').")
    parser.add_argument("--generation_log_file", type=str, default=None, help="Path to save the generated outputs for debugging.")
    # parser.add_argument("--enable_generation_log", action='store_true', help="Enable logging of generation outputs during training for debugging.")
    args = parser.parse_args()

    # # --- 1. 加载模型和处理器 ---
    # logger.info("--- [Step 1/5] Loading processor and model... ---")
    # processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    # model = OrthusForConditionalGeneration.from_pretrained(
    #     args.ckpt_path,
    #     torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    #     attn_implementation="flash_attention_2",
    # )
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    # --- 1. 加载模型和处理器 ---
    # logger.info("--- [Step 1/5] Loading processor and model for 16-bit LoRA... ---")
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)

    # 1. 直接以 bfloat16 精度加载模型
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16, # 直接指定16位精度
        # device_map="auto",         # 推荐与PEFT一起使用，它会自动处理设备放置
        attn_implementation="flash_attention_2",
    )
    logger.info("Base model loaded in bfloat16.")
    
    # 2. 定义LoRA配置 (这部分与QLoRA版本完全相同)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. 将LoRA应用到模型上
    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA to the model.")

    # # 4. (可选但推荐) 打印可训练参数的数量和比例
    # model.print_trainable_parameters()
        
    # logger.info("Model and processor loaded successfully for 16-bit LoRA training.")
    
    # 5. 如果您开启了梯度检查点，请保留此设置
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # 对于LoRA+梯度检查点的组合，通常需要下面这行来确保正常工作
        model.enable_input_require_grads() 

    logger.info("Model and processor loaded successfully.")

    # --- 2. 加载并初始化完整数据集 ---
    logger.info("\n--- [Step 2/5] Initializing full datasets... ---")
    train_dataset_raw = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset_raw = load_dataset("json", data_files=args.eval_file, split="train")
    # --- 【新增代码】: 如果是调试模式，则截取一小部分数据 ---
    if args.debug_mode:
        # --- 主要修改这里 ---
        logger.warning("--- [SINGLE DATA TEST] --- Using only one data sample for training and evaluation.")
        # .select() 方法可以高效地创建一个只包含指定索引的子集
        train_dataset_raw = train_dataset_raw.select(range(1))
        eval_dataset_raw = eval_dataset_raw.select(range(1))
    logger.info(f"Full train dataset size: {len(train_dataset_raw)}")
    logger.info(f"Full eval dataset size: {len(eval_dataset_raw)}")
    
    train_dataset = InterleaveSFTDataset(
        dataset=train_dataset_raw,
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.model.vqmodel    # lora
        # vqmodel=model.model.vqmodel
    )
    eval_dataset = InterleaveSFTDataset(
        dataset=eval_dataset_raw,
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.model.vqmodel    # lora
        # vqmodel=model.model.vqmodel
    )
    logger.info("Custom SFT datasets created.")

    # --- 3. 定义训练参数 ---
    logger.info("\n--- [Step 3/5] Configuring Training Arguments... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["labels"], # 确保 Trainer 知道哪个是标签
    )

    # --- 自定义数据整理器 (Data Collator) ---
    def custom_data_collator(features: list) -> Dict[str, Any]:
        """
        在将样本组合成批次之前，移除不需要的 'vqmodel' 键。
        """
        for feature in features:
            if "vqmodel" in feature:
                feature.pop("vqmodel")
        # 使用 PyTorch 的默认 collate 函数处理剩余的张量
        return torch.utils.data.dataloader.default_collate(features)

    # --- 4. 初始化自定义 Trainer ---
    logger.info("\n--- [Step 4/5] Initializing the custom InterleaveSFTTrainer... ---")
    trainer = InterleaveSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
        processor=processor,
        enable_generation_log=(args.generation_log_file is not None),
        generation_log_file=args.generation_log_file,
    )

    # --- 5. 启动训练 ---
    logger.info("\n--- [Step 5/5] Starting the formal training run... ---")
    try:
        # 如果 output_dir 中有断点，可以从断点恢复训练
        trainer.train(resume_from_checkpoint=True if os.path.isdir(args.output_dir) and any(fn.startswith("checkpoint-") for fn in os.listdir(args.output_dir)) else False)
        
        logger.info("\n\n✅ [Success] Training completed successfully!")
        
        # 保存最终的模型和处理器状态
        logger.info("Saving final model...")
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        logger.info(f"Final model and processor saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"\n\n❌ [Failure] Training failed with an error: {e}", exc_info=True)


if __name__ == "__main__":
    main()