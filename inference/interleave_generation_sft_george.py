import torch
import os
import sys
import random
import numpy as np
import shutil
# set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
import torch.nn.functional as F
import json
from PIL import Image
from tqdm import tqdm
# Orthus的特殊Token
DEFAULT_IMAGE_TOKEN = "<image>"
from datasets import load_dataset

def load_inference_data(data_file_path, limit=None, random_seed=42):
    """
    加载数据集中用于推理的最后10%部分。
    如果 'limit' 被设置，将从这10%的数据中随机采样指定数量的样本。

    Args:
        data_file_path (str): .jsonl 文件的路径。
        limit (int, optional): 要随机采样的样本数量。
        random_seed (int): 随机种子，用于确保采样可复现。

    Returns:
        datasets.Dataset: 包含最后10%数据中随机样本的Hugging Face Dataset对象。
    """
    print(f"Loading full dataset from {data_file_path} to select the inference split...")
    dataset_raw = load_dataset("json", data_files=data_file_path, split="train")

    num_samples = len(dataset_raw)
    # eval_split_index = int(num_samples * 0.9)
    train_split_index = int(num_samples * 0.8)
    # 1. 首先，和之前一样，选出完整的最后10%作为推理集
    # inference_dataset = dataset_raw.select(range(eval_split_index, num_samples))
    inference_dataset = dataset_raw.select(range(train_split_index))
    inference_set_size = len(inference_dataset)
    print(f"Total samples in file: {num_samples}")
    print(f"Full inference split (last 10%) contains {inference_set_size} samples.")

    # --- 【核心修改】 ---
    # 2. 如果提供了 'limit' 并且 'limit' 小于推理集的大小
    if limit is not None and limit < inference_set_size:
        print(f"Randomly sampling {limit} items from the inference set (seed={random_seed})...")
        
        # 先打乱 (shuffle) 推理集，然后选择前 'limit' 个
        inference_dataset = inference_dataset.shuffle(seed=random_seed).select(range(limit))
        
    elif limit is not None:
        print(f"Warning: Requested limit ({limit}) is >= inference set size ({inference_set_size}). Using all {inference_set_size} inference samples.")
    
    print(f"Loaded {len(inference_dataset)} samples for inference.")
    return inference_dataset


ckpt_path = "SJTU-Deng-Lab/Orthus-7B-base"
data_path = "/data1/oujingfeng/project/twgi/datasets/StoryStream_dataset/George/processed_train.jsonl"
limit = 10
image_base_dir = "/data1/oujingfeng/project/twgi/datasets/StoryStream_dataset/George"
processor = OrthusProcessor.from_pretrained(ckpt_path)

model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
)

exp_dir = os.path.join(root_path, "results/george/orthus-7b-base-george-train")
os.makedirs(exp_dir, exist_ok=True)

set_seed(42)


# --- 2. 加载推理数据 (现在会调用新函数) ---
items = load_inference_data(data_path, limit)

# --- 3. 循环处理每个样本并进行推理 ---
for item in tqdm(items, desc="Generating stories"):
    # (后续代码无需改动，因为'items'对象仍然支持索引访问)
    # --- a. 从样本中提取信息 ---
    story_id = item.get("story_id")
    order = item.get("order")
    prompt_image_name = item.get("prompt_image")
    prompt_text = item.get("prompt_text")

    # ... (您脚本中剩余的推理和保存逻辑保持不变) ...
    # --- b. 构建Prompt ---
    image_path = os.path.join(image_base_dir, prompt_image_name)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}. Skipping sample.")
        continue
        
    image = Image.open(image_path).convert("RGB")
    
    instruction = "Please continue this story:"
    prompt = DEFAULT_IMAGE_TOKEN + prompt_text + instruction


    interleave_inputs_con = processor([prompt], images=[image],padding=True, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
    interleave_input_ids_con = interleave_inputs_con['input_ids']

    prompt_uncon="generate images"
    interleave_inputs_uncon = processor([prompt_uncon], return_tensors="pt")
    interleave_input_ids_uncon = interleave_inputs_uncon['input_ids'].to(model.device)

    kwargs_con = {
        "input_ids": interleave_input_ids_con,
        "cfg_scale": 1.0,    # 忽略uncon的影响
        "interleave_output_format": True,
        "max_new_tokens": 4096,
        "do_sample": True,
        "attention_mask": interleave_inputs_con['attention_mask'].to(model.device),
        "use_cache": True,
    }
    kwargs_uncon = {
        "input_ids": interleave_input_ids_uncon,
        "cfg_scale": 1.0,  # 冗余参数 没有被使用
        "attention_mask": interleave_inputs_uncon['attention_mask'].to(model.device),
        "use_cache": True,
    }


    outputs = model.generate(
        multimodal_generation_mode_list=["interleaved-text-image","image-only"],
        kwargs_list=[kwargs_con, kwargs_uncon],
    )

    text_tokens = []
    all_image_embeds_wo_quant = []

    with torch.no_grad():
        for output in outputs:
            if len(output.shape) == 1:
                if torch.sum(output == 8196) == 0 and torch.sum(output == 8197) == 0:
                    text_tokens.append(output)
            else:
                all_image_embeds_wo_quant.append(output)

        # decode image one by one
        num_images = len(all_image_embeds_wo_quant) // 1024


        # 1. First, define the path for the sample's output directory
        sample_dir = os.path.join(exp_dir, f'{story_id}_{order}')

        # 2. Create this directory if it doesn't exist. This is the key fix. 📂
        os.makedirs(sample_dir, exist_ok=True)
        for id in range(num_images):
            image_embeds_wo_quant = torch.cat(all_image_embeds_wo_quant[id * 1024:(id + 1) * 1024], dim=0).to(model.device)

            emb_dim = model.model.vqmodel.quantize.embedding.weight.shape[-1]
            image_embeds_wo_quant = image_embeds_wo_quant.view((1, *model.model.vqmodel.quantize.quant_state_dims, emb_dim))
            image_embeds_wo_quant = image_embeds_wo_quant.permute(0, 3, 1, 2).contiguous()

            hidden_states = model.model.vqmodel.post_quant_conv(image_embeds_wo_quant.to(model.model.vqmodel.post_quant_conv.weight.dtype))
            pixel_values_wo_quant = model.model.vqmodel.decoder(hidden_states)
            images_wo_quant = processor.postprocess_pixel_values(pixel_values_wo_quant)

            from torchvision.transforms.functional import to_pil_image
            images_wo_quant = [to_pil_image(img.detach().cpu()) for img in images_wo_quant]

            images_wo_quant[0].save(os.path.join(sample_dir, f'{id+1}_cfg3.jpg'))

        # # decode generated text
        # text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
        # text = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]
        # --- 【核心修改開始】 ---
        # 不再直接合併所有 text_tokens，而是逐個解碼




        # 3. 保存原始輸入 (Prompt)
        image.save(os.path.join(sample_dir, 'prompt_image.jpg'))
        with open(os.path.join(sample_dir, "prompt_text.txt"), "w", encoding="utf-8") as file:
            file.write(prompt_text)

        # --- 【核心修改：使用 shutil.copy() 複製標準答案】 ---

        # 4. 從 item 中獲取標準答案的文件名和文本
        gt_image_name_1 = item.get("label_image_1")
        gt_image_name_2 = item.get("label_image_2")
        gt_text_1 = item.get("label_text_1", "")
        gt_text_2 = item.get("label_text_2", "")

        # 5. 複製第一張標準答案圖片
        if gt_image_name_1:
            source_path = os.path.join(image_base_dir, gt_image_name_1)
            destination_path = os.path.join(sample_dir, 'ground_truth_image_1.jpg')
            try:
                shutil.copy(source_path, destination_path)
            except FileNotFoundError:
                print(f"Warning: Ground truth image not found at {source_path}. Cannot copy.")

        # 6. 複製第二張標準答案圖片
        if gt_image_name_2:
            source_path = os.path.join(image_base_dir, gt_image_name_2)
            destination_path = os.path.join(sample_dir, 'ground_truth_image_2.jpg')
            try:
                shutil.copy(source_path, destination_path)
            except FileNotFoundError:
                print(f"Warning: Ground truth image not found at {source_path}. Cannot copy.")
        
        # 7. 將標準答案的文本保存
        combined_gt_text = f"{gt_text_1}\n{gt_text_2}"
        with open(os.path.join(sample_dir, "ground_truth_text.txt"), "w", encoding="utf-8") as file:
            file.write(combined_gt_text)

        # decode generated text
        text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
        text = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]

        # --- 【核心修改結束】 ---
        with open(os.path.join(sample_dir,"text.txt"), "w", encoding="utf-8") as file:
            file.write(str(story_id)+"_"+ str(order) + '\n' + text)