import torch
import os
import sys
import argparse
from PIL import Image
import torchvision

# 确保能导入项目中的模块
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="Decode saved image latents into PNG images.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint folder (the one containing config.json)."
    )
    parser.add_argument(
        "--latents_file", 
        type=str, 
        required=True, 
        help="Path to the .pt file containing the saved latents."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="decoded_images",
        help="Directory to save the decoded images."
    )
    args = parser.parse_args()

    print("--- [Step 1/4] Setting up device ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"--- [Step 2/4] Loading model from {args.model_path} ---")
    # 以 bfloat16 格式加载模型以匹配训练时的精度
    model = OrthusForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    ).to(device).eval() # 设置为评估模式
    print("Model loaded successfully.")

    print(f"--- [Step 3/4] Loading latents from {args.latents_file} ---")
    latents_dict = torch.load(args.latents_file, map_location=device)
    pred_latents_all = latents_dict['predicted'].to(dtype=torch.bfloat16)
    true_latents_all = latents_dict['target'].to(dtype=torch.bfloat16)
    print(f"Latents loaded successfully. Shape of predicted latents: {pred_latents_all.shape}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.latents_file))[0]


    # ==================== 核心修改：处理批次维度 ====================
    # 如果张量是3维的 (batch, seq_len, dim)，则将其 reshape 为 (batch * seq_len, dim)
    if pred_latents_all.dim() == 3:
        print("Reshaping 3D tensor to 2D by flattening batch and sequence dimensions.")
        pred_latents_all = pred_latents_all.reshape(-1, pred_latents_all.shape[-1])
        true_latents_all = true_latents_all.reshape(-1, true_latents_all.shape[-1])
        print(f"New shape: {pred_latents_all.shape}")

    # ==================== 核心修改：循环解码 ====================
    # 计算文件中包含多少张图片 (每张图片1024个token)
    num_images = pred_latents_all.shape[0] // 1024
    print(f"Found {num_images} image(s) in the latents file.")

    print("--- [Step 4/4] Decoding and saving images ---")
    with torch.no_grad():
        for i in range(num_images):
            print(f"  - Decoding image {i+1}/{num_images}...")
            
            # 1. 切片，获取当前图片的特征
            start_idx = i * 1024
            end_idx = (i + 1) * 1024
            current_pred_latents = pred_latents_all[start_idx:end_idx, :]
            current_true_latents = true_latents_all[start_idx:end_idx, :]

            # 2. 解码预测的图片
            predicted_pixels = model.decode_image_latents(current_pred_latents)
            pred_save_path = os.path.join(args.output_dir, f"{base_filename}_predicted_{i}.png")
            torchvision.utils.save_image(predicted_pixels, pred_save_path, normalize=True)
            print(f"    - Saved predicted image to: {pred_save_path}")

            # 3. 解码真实的图片
            true_pixels = model.decode_image_latents(current_true_latents)
            true_save_path = os.path.join(args.output_dir, f"{base_filename}_target_{i}.png")
            torchvision.utils.save_image(true_pixels, true_save_path, normalize=True)
            print(f"    - Saved target image to: {true_save_path}")
    # =========================================================
        
    print("\n✅ Decoding complete!")

if __name__ == "__main__":
    main()