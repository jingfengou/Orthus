####### multimodality understanding ########
import torch
import sys
import os
import random

import numpy as np
import json
from tqdm import tqdm
import torch.nn.functional as F
import requests
from PIL import Image
from safetensors.torch import load_file
import argparse
# ckpt_path = "SJTU-Deng-Lab/Orthus-7B-instruct"
# processor = OrthusProcessor.from_pretrained(ckpt_path)
# model = OrthusForConditionalGeneration.from_pretrained(
#     ckpt_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation='flash_attention_2',
#     device_map="auto",
# )

# ### Example usage ###
# prompt = "<image>Can you please tell me what kind of farm equipment would be essential for this kind of farm?"
# image = Image.open(os.path.join(root_path, "inference/mmu_demo/Grain-production-wheat.jpg"))
# images=[image]

# inputs = processor(prompt, images=images, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
# if len(images) >=2:
#     inputs['image_latents']=inputs['image_latents'].unsqueeze(dim=0)

# generated_ids = model.generate(input_ids=inputs['input_ids'], \
#                                attention_mask=inputs['attention_mask'], cfg_scale=None, \
#                                image_latents=inputs['image_latents'], max_new_tokens=512, do_sample=False, use_cache=False) 
# out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(f'Response: {out}')



# --- Utility Functions ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Orthus Interleave Generation with Progress Bar")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path or name of the model checkpoint.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    args = parser.parse_args()

    set_seed(50)

    # --- 1. Model and Processor Loading ---
    print(f"Loading model from {args.ckpt_path}...")
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Prepare directories for output images ---
    output_dir = os.path.dirname(args.output_file)
    image_output_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # --- 3. Read input data ---
    with open(args.input_file, 'r') as f:
        input_data = [json.loads(line) for line in f]

    # --- 4. Process data with a progress bar ---
    # tqdm will write its progress bar to stderr by default
    with open(args.output_file, 'w') as f_out:
        for item in tqdm(input_data, desc=f"Processing {os.path.basename(args.input_file)}"):
            prompt_con = item['prompt']
            sample_id = item.get('id', 'unknown_id')
            # 更健壮的修改
            images_list = item.get("images", [])
            if images_list: # 确保列表不为空
                image_path = images_list[0]
                images = Image.open(image_path).convert("RGB")
            else:
                # 在这里处理没有图片的情况，例如跳过或者记录日志
                print(f"警告：样本 {item.get('id')} 的 'images' 列表为空。")
                continue # 或者其他处理逻辑
            inputs = processor(prompt_con, images=[images], return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)

            


            with torch.no_grad():

                text_tokens = model.generate(input_ids=inputs['input_ids'], \
                                            attention_mask=inputs['attention_mask'], cfg_scale=None, \
                                            image_latents=inputs['image_latents'], max_new_tokens=512, do_sample=True, use_cache=True) 
                full_text_response = processor.batch_decode(text_tokens, skip_special_tokens=True)[0]


                # --- 6. Decode Text and Write Output ---
                # full_text_response = ""
                # if text_tokens:
                #     text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
                #     full_text_response = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]
                
                result_obj = {
                    "id": sample_id,
                    "prompt": prompt_con,
                    "response": full_text_response
                    # "generated_images": saved_image_paths
                }
                f_out.write(json.dumps(result_obj) + '\n')

if __name__ == "__main__":
    # Add root path for custom module imports
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_path)

    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus import OrthusForConditionalGeneration
    # from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
    
    main()