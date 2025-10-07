import torch
import os
import sys
import argparse
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import json

# --- 引用我们刚刚创建的数据集类 ---
# 假设 sft_orthus.py 和 test_dataset.py 在同一个文件夹
from sft_orthus import InterleaveSFTDataset 

# # --- 确保能从您的项目路径导入Orthus模块 ---
# try:
#     from models.processing_orthus import OrthusProcessor
#     from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
# except ImportError:
#     print("Error: Make sure to run this script from the root of the Orthus project.")
#     sys.exit(1)


def test_dataset(args):
    """
    加载数据集并检查单个样本的构造是否正确。
    """
    print("--- [Step 1/4] Loading processor and model (for vqmodel)... ---")
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    # 只需要加载模型来获取vqmodel，可以放在CPU上以节省显存
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True # 尝试减少CPU内存使用
    )
    processor.vqmodel = model.model.vqmodel
    print("Processor and model components loaded.")

    print("\n--- [Step 2/4] Initializing dataset... ---")
    # 【核心修改】: 在创建 Dataset 实例时，传入 model.model.vqmodel
    dataset = InterleaveSFTDataset(
        data_file=args.data_file,
        image_base_dir=args.image_folder,
        processor=processor,
        vqmodel=model.model.vqmodel, # <-- 增加这一行
        max_length=args.max_length
    )
    if len(dataset) == 0:
        print("Dataset is empty. Cannot perform test.")
        return

    print(f"\n--- [Step 3/4] Fetching first sample (dataset[0])... ---")
    sample = dataset[0]

    print("\n--- [Step 4/4] Analyzing the sample's structure and content... ---")
    
    # 检查1：输出的键是否完整
    print("\n[Check 1: Keys in the sample]")
    print(list(sample.keys()))

    # 检查2：各个张量的形状
    print("\n[Check 2: Tensor Shapes]")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key}: {value.shape}")

    # 检查3：检查 labels 的屏蔽（masking）是否正确
    print("\n[Check 3: Label Masking Verification]")
    
    # 将 labels 中的 -100 替换为 pad_token_id 以便解码
    labels_for_decoding = sample['labels'].clone()
    labels_for_decoding[labels_for_decoding == -100] = processor.tokenizer.pad_token_id
    
    # 解码原始输入和处理后的标签
    # 【核心修改】: 在解码labels时，设置 skip_special_tokens=False
    decoded_input_ids = processor.decode(sample['input_ids'], skip_special_tokens=True)
    decoded_labels = processor.decode(labels_for_decoding, skip_special_tokens=True) # <--- 修改在这里


    print("\n---------- DECODED INPUT_IDS (What model sees) ----------")
    print(decoded_input_ids)
    print("----------------------------------------------------------")

    print("\n---------- DECODED LABELS (What model learns from) ----------")
    print(decoded_labels)
    print("---------------------------------------------------------------")
    
    print("\nVerification finished.")
    print("Please check if the 'DECODED LABELS' output correctly masks the prompt part.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test run for the SFT Dataset class.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the train.jsonl or test.jsonl file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the SpatialViz dataset root directory.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    
    args = parser.parse_args()
    # 1. 获取当前脚本的绝对路径，并找到上一级目录（即项目根目录）
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 2. 将项目根目录临时添加到Python解释器的“模块搜索路径”列表中
    sys.path.append(root_path)

    # 3. 现在可以从根目录开始，成功地导入任何模块了
    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
    
    test_dataset(args)