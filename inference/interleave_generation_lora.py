# filename: inference/interleave_generation.py
import torch
import os
import sys
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path or name of the lora model checkpoint.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path or name of the base model.")
    args = parser.parse_args()

    set_seed(50)

    # --- 1. Model and Processor Loading ---
    print(f"Loading model from {args.base_model_path}...")
    processor = OrthusProcessor.from_pretrained(args.base_model_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
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
            # interleave_inputs_con = processor(prompt_con, images=[images],padding=True, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
            interleave_inputs_con = processor([prompt_con], images=[images],padding=True, return_tensors="pt", vqmodel=model.model.model.vqmodel).to(model.device, torch.bfloat16)
            interleave_input_ids_con = interleave_inputs_con['input_ids']
            
            # Unconditional prompt for CFG
            prompt_uncon = "generate an origenal images"
            # interleave_inputs_uncon = processor(prompt_uncon, images=[images], return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
            interleave_inputs_uncon = processor([prompt_uncon], images=[images],padding=True, return_tensors="pt", vqmodel=model.model.model.vqmodel).to(model.device, torch.bfloat16)
            interleave_input_ids_uncon = interleave_inputs_uncon['input_ids']

            kwargs_con = {
                "input_ids": interleave_input_ids_con,
                "image_latents": interleave_inputs_con['image_latents'],
                "cfg_scale": 3.0,
                "interleave_output_format": True,
                "max_new_tokens": 2048,
                "do_sample": True,
                "attention_mask": interleave_inputs_con['attention_mask'],
                "use_cache": True,
            }
            kwargs_uncon = {
                "input_ids": interleave_input_ids_uncon,
                "cfg_scale": 1.0,
                "attention_mask": interleave_inputs_uncon['attention_mask'],
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = model.generate(
                        multimodal_generation_mode_list=["interleaved-text-image","image-only"],
                        kwargs_list=[kwargs_con, kwargs_uncon],
                )
                # print(outputs)
                text_tokens = []
                all_image_embeds_wo_quant = []
                
                for output in outputs:
                    if len(output.shape) == 1:
                        if torch.sum(output == 8196) == 0 and torch.sum(output == 8197) == 0:
                            text_tokens.append(output)
                    else:
                        all_image_embeds_wo_quant.append(output)

                # --- 5. Decode and Save Images ---
                num_images = len(all_image_embeds_wo_quant) // 1024
                saved_image_paths = []
                for img_idx in range(num_images):
                    image_embeds_wo_quant = torch.cat(all_image_embeds_wo_quant[img_idx * 1024:(img_idx + 1) * 1024], dim=0).to(model.device)

                    emb_dim = model.model.vqmodel.quantize.embedding.weight.shape[-1]
                    image_embeds_wo_quant = image_embeds_wo_quant.view((1, *model.model.vqmodel.quantize.quant_state_dims, emb_dim))
                    image_embeds_wo_quant = image_embeds_wo_quant.permute(0, 3, 1, 2).contiguous()

                    hidden_states = model.model.vqmodel.post_quant_conv(image_embeds_wo_quant.to(model.model.vqmodel.post_quant_conv.weight.dtype))
                    pixel_values_wo_quant = model.model.vqmodel.decoder(hidden_states)
                    images_wo_quant = processor.postprocess_pixel_values(pixel_values_wo_quant)

                    from torchvision.transforms.functional import to_pil_image
                    pil_image = to_pil_image(images_wo_quant[0].detach().cpu())
                    
                    # Save image to a structured path
                    image_save_path = os.path.join(image_output_dir, f"{sample_id}_image_{img_idx}.png")
                    pil_image.save(image_save_path)
                    saved_image_paths.append(image_save_path)

                # --- 6. Decode Text and Write Output ---
                full_text_response = ""
                if text_tokens:
                    text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
                    full_text_response = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]
                # full_text_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                result_obj = {
                    "id": sample_id,
                    "prompt": prompt_con,
                    "response": full_text_response,
                    "generated_images": saved_image_paths
                }
                f_out.write(json.dumps(result_obj) + '\n')

if __name__ == "__main__":
    # Add root path for custom module imports
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_path)

    from models.processing_orthus import OrthusProcessor
    from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
    # from models.modeling_orthus import OrthusForConditionalGeneration
    main()