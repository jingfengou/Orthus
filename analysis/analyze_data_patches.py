# analyze_data_patches.py
import torch
from PIL import Image
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
import json
import os
from tqdm import tqdm

# --- 參數設定 ---
CKPT_PATH = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v010"
DATA_FILE = "/data1/oujingfeng/project/twgi/datasets/mydatasets/metadata.json"
IMAGE_BASE_DIR = "/data1/oujingfeng/project/twgi/datasets/mydatasets"
INDICES_TO_PICK = [i * 2 for i in range(10)] # -> [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
OUTPUT_FILE = "multi_intervention_data.pt" # 儲存分析結果的檔案

# --- 載入模型和處理器 ---
print("Loading model and processor...")
processor = OrthusProcessor.from_pretrained(CKPT_PATH)
model = OrthusForConditionalGeneration.from_pretrained(CKPT_PATH, device_map="auto", torch_dtype=torch.bfloat16)
vqmodel = model.model.vqmodel

# --- 生成 "標準空白" 特徵作為參考 ---
print("Generating 'standard blank' latents reference...")
white_image = Image.new('RGB', (224, 224), (255, 255, 255))
with torch.no_grad():
    instruction = (
    "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
    "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
    "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )

    question = "First, rotate this cube stack along the X axis by 270 degrees. Then, rotate it along the Z axis by 270 degrees. Which option shows the correct final result?"
    choices_text = "\nA: This option uses incorrect rotation angles.\nB: Correct: this is the result after performing all rotation steps correctly.\nC: This option rotates along wrong axes.\nD: This option performs the rotations in wrong order."
    prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
    white_inputs = processor(text=[prompt_text],images=[white_image], return_tensors="pt", vqmodel=vqmodel)
standard_blank_latents_mean = white_inputs['image_latents'].view(-1,256).squeeze(0).mean(dim=0).to("cpu", torch.float32)


# --- 遍歷數據集並分析 ---
print("Analyzing training data patch distribution...")
total_patches = 0
blank_patches = 0

with open(DATA_FILE, 'r') as f:

    item = json.load(f)
    
    # 找到所有的目標圖片 (這裡假設是 'Rotation_steps' 中的圖片)
    target_image_paths = []
    image_id = item.get('Image_id', '')
    for step in item.get('Rotation_steps', []):
        step_image_filename = step.get('image', '')
        if step_image_filename:
            target_image_paths.append(os.path.join(IMAGE_BASE_DIR, image_id, step_image_filename))
    cnt = 0
    for img_path in target_image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            continue
        # print(f"image.shape: {image}")
        # 編碼圖片
        with torch.no_grad():
            inputs = processor(text=[prompt_text],images=[image], return_tensors="pt", vqmodel=vqmodel)
        
        # [1, 1024, 256] -> [1024, 256]
        latents = inputs['image_latents'].view(-1,256).to("cpu", torch.float32)

        total_patches += latents.shape[0]
        print("standard_blank_latents_mean.shape",standard_blank_latents_mean.shape)
        # 計算與標準空白的相似度
        print("latents shape",latents.shape)
        similarities = torch.nn.functional.cosine_similarity(latents, standard_blank_latents_mean.unsqueeze(0), dim=1)

        # 計算被視為空白的 patch 數量
        blank_patches += (similarities > 0.99).sum().item()


        print("similarities shape",similarities.shape)
        # 【修改】: 找到所有非空白 patch 的索引

        all_non_blank_indices = (similarities < 0.99).nonzero(as_tuple=True)[0]     
        print(f"Image '{img_path}' - Found {len(all_non_blank_indices)} non-blank patches.")
        if len(all_non_blank_indices) == 0:
            print("Warning: No non-blank patches found.")
            intervention_indices = []
        else:
            # 【修改】: 從所有非空白 patch 中，挑選出我們想要的間隔索引
            intervention_indices = all_non_blank_indices

            print(f"Found {len(all_non_blank_indices)} non-blank patches.")
            print(f"Selected intervention indices: {intervention_indices}")

        # 準備要儲存的數據
        data_to_save = {
            'intervention_indices': intervention_indices, # 儲存索引列表
            'target_image_latents': latents.to(torch.bfloat16)
        }

        torch.save(data_to_save, "step_" + str(cnt) + OUTPUT_FILE)
        print(f"Multi-intervention data saved successfully to 'step_{cnt}{OUTPUT_FILE}'")
        cnt += 1


# --- 報告結果 ---
print("\n--- Analysis Complete ---")
if total_patches > 0:
    blank_percentage = (blank_patches / total_patches) * 100
    print(f"Total patches analyzed: {total_patches}")
    print(f"Patches identified as 'blank': {blank_patches}")
    print(f"Percentage of blank patches in target images: {blank_percentage:.2f}%")

    if blank_percentage > 50:
        print("\n[Conclusion]: The training data contains a high percentage of blank patches. This strongly supports the 'blank inertia' hypothesis.")
    else:
        print("\n[Conclusion]: The percentage of blank patches is not dominant. The issue might be less related to data bias.")
else:
    print("No target images found to analyze.")




# # analyze_and_save_multi_patch_info.py
# import torch
# from PIL import Image
# import os
# import sys
# import json
# from tqdm import tqdm

# # --- 專案路徑設定 ---
# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(root_path)
# from models.processing_orthus import OrthusProcessor
# from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

# # --- 參數設定 ---
# CKPT_PATH = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v010"
# DATA_FILE = "/data1/oujingfeng/project/twgi/datasets/mydatasets/metadata.json"
# IMAGE_BASE_DIR = "/data1/oujingfeng/project/twgi/datasets/mydatasets"
# TARGET_IMAGE_ID = "sample_0001"
# OUTPUT_FILE = "multi_intervention_data.pt" # 儲存分析結果的檔案

# # 【修改】: 定義要挑選的非空白 patch 的索引 (第1個, 第3個, 第5個, ...)
# # 這裡我們挑選前10個間隔的 patch 進行干預
# INDICES_TO_PICK = [i * 2 for i in range(10)] # -> [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# instruction = (
# "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
# "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
# "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
# )

# question = "First, rotate this cube stack along the X axis by 270 degrees. Then, rotate it along the Z axis by 270 degrees. Which option shows the correct final result?"
# choices_text = "\nA: This option uses incorrect rotation angles.\nB: Correct: this is the result after performing all rotation steps correctly.\nC: This option rotates along wrong axes.\nD: This option performs the rotations in wrong order."
# prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
# # --- 載入模型和處理器 ---
# print("Loading model and processor...")
# processor = OrthusProcessor.from_pretrained(CKPT_PATH)
# model = OrthusForConditionalGeneration.from_pretrained(CKPT_PATH, device_map="auto", torch_dtype=torch.bfloat16)
# vqmodel = model.model.vqmodel

# # --- 生成 "標準空白" 特徵作為參考 ---
# print("Generating 'standard blank' latents reference...")
# white_image = Image.new('RGB', (224, 224), (255, 255, 255))
# with torch.no_grad():
#     white_inputs = processor(text=[prompt_text],images=[white_image], return_tensors="pt", vqmodel=vqmodel)
# standard_blank_latents_mean = white_inputs['image_latents'].squeeze(0).mean(dim=0).to("cpu", torch.float32)

# # --- 尋找並分析指定的樣本 ---
# print(f"Searching for target sample '{TARGET_IMAGE_ID}' in {DATA_FILE}...")
# found = False
# with open(DATA_FILE, 'r') as f:
#     # for line in f:

    
#         item = json.load(f)
#         if item.get('Image_id') == TARGET_IMAGE_ID:
#             print(f"Found target sample '{TARGET_IMAGE_ID}'. Analyzing its first target image...")
            
#             first_step = item.get('Rotation_steps', [{}])[0]
#             step_image_filename = first_step.get('image')
            
#             if not step_image_filename:
#                 print("Error: No target image found in the first rotation step.")
#                 exit()

#             img_path = os.path.join(IMAGE_BASE_DIR, TARGET_IMAGE_ID, step_image_filename)
#             image = Image.open(img_path).convert("RGB")

#             with torch.no_grad():
#                 inputs = processor(text=[prompt_text],images=[image], return_tensors="pt", vqmodel=vqmodel)
            
#             latents = inputs['image_latents'].squeeze(0).to("cpu", torch.float32)
#             similarities = torch.nn.functional.cosine_similarity(latents, standard_blank_latents_mean.unsqueeze(0), dim=1)
            
#             # 【修改】: 找到所有非空白 patch 的索引
#             all_non_blank_indices = (similarities < 0.99).nonzero(as_tuple=True)[0]
            
#             if len(all_non_blank_indices) == 0:
#                 print("Warning: No non-blank patches found.")
#                 intervention_indices = []
#             else:
#                 # 【修改】: 從所有非空白 patch 中，挑選出我們想要的間隔索引
#                 intervention_indices = [
#                     all_non_blank_indices[i].item() 
#                     for i in INDICES_TO_PICK 
#                     if i < len(all_non_blank_indices) # 確保不超出範圍
#                 ]
#                 print(f"Found {len(all_non_blank_indices)} non-blank patches.")
#                 print(f"Selected intervention indices: {intervention_indices}")

#             # 準備要儲存的數據
#             data_to_save = {
#                 'intervention_indices': intervention_indices, # 儲存索引列表
#                 'target_image_latents': latents.to(torch.bfloat16)
#             }
            
#             torch.save(data_to_save, OUTPUT_FILE)
#             print(f"Multi-intervention data saved successfully to '{OUTPUT_FILE}'")
            
#             found = True
#             # break

# if not found:
#     print(f"Error: Could not find sample with Image_id '{TARGET_IMAGE_ID}'.")