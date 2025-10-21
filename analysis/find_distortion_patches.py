# find_distortion_patches.py
import torch
from PIL import Image
import os
import sys
import json
import numpy as np

# --- 專案路徑設定 ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

# --- 參數設定 ---
CKPT_PATH = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v010"
DATA_FILE = "/data1/oujingfeng/project/twgi/datasets/mydatasets/metadata.json"
IMAGE_BASE_DIR = "/data1/oujingfeng/project/twgi/datasets/mydatasets"
TARGET_IMAGE_ID = "sample_0001"
OUTPUT_DIR = "/data1/oujingfeng/project/twgi/Orthus/analysis/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 載入模型和處理器 ---
print("Loading model and processor...")
processor = OrthusProcessor.from_pretrained(CKPT_PATH)
model = OrthusForConditionalGeneration.from_pretrained(CKPT_PATH, device_map="auto", torch_dtype=torch.bfloat16)
vqmodel = model.model.vqmodel
instruction = (
"You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
"The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
"respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
)

question = "First, rotate this cube stack along the X axis by 270 degrees. Then, rotate it along the Z axis by 270 degrees. Which option shows the correct final result?"
choices_text = "\nA: This option uses incorrect rotation angles.\nB: Correct: this is the result after performing all rotation steps correctly.\nC: This option rotates along wrong axes.\nD: This option performs the rotations in wrong order."
prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
# --- 生成 "標準空白" 特徵作為參考 ---
print("Generating 'standard blank' latents reference...")
white_image = Image.new('RGB', (224, 224), (255, 255, 255))
with torch.no_grad():
    white_inputs = processor(text=[prompt_text], images=[white_image], return_tensors="pt", vqmodel=vqmodel)
standard_blank_latents_mean = white_inputs['image_latents'].squeeze(0).view(1024,256).mean(dim=0).to("cpu", torch.float32)
print(f"Standard blank latents mean shape: {standard_blank_latents_mean.shape}")  # 應為 (latent_dim,)

# --- 尋找並分析指定的樣本 ---
# (此處省略了遍歷 JSON 的部分，直接使用範例圖片路徑)
print(f"Analyzing target sample '{TARGET_IMAGE_ID}'...")
image_id = TARGET_IMAGE_ID
step_image_filename = "step_1.png" # 假設我們分析第一步的結果圖
img_path = os.path.join(IMAGE_BASE_DIR, image_id,"steps", step_image_filename)
image = Image.open(img_path).convert("RGB")

with torch.no_grad():
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", vqmodel=vqmodel)

latents = inputs['image_latents'].squeeze(0).view(1024,256).to("cpu", torch.float32)
print(f"Latents shape: {latents.shape}")  # 應為 (1024, latent_dim)
similarities = torch.nn.functional.cosine_similarity(latents, standard_blank_latents_mean, dim=1)
print(f"Similarities shape: {similarities.shape}")  # 應為 (1024,)
# 1. 將 1024 個 patch 分類為「空白」(True) 或「內容」(False)
is_blank = (similarities > 0.99)
is_blank_grid = is_blank.view(32, 32) # 將其重塑為 32x32 的網格

# 2. 找到邊界/畸變區域
# 初始化一個全為 False 的遮罩
distortion_mask = torch.zeros_like(is_blank_grid, dtype=torch.bool)

# 檢查水平方向的鄰居是否有變化
# 如果 is_blank_grid[:, :-1] 和 is_blank_grid[:, 1:] 不一樣，說明中間存在一條垂直邊界
horizontal_changes = is_blank_grid[:, :-1] != is_blank_grid[:, 1:]
distortion_mask[:, :-1] |= horizontal_changes # 將邊界左側的 patch 標記為 True
distortion_mask[:, 1:] |= horizontal_changes  # 將邊界右側的 patch 標記為 True

# 檢查垂直方向的鄰居是否有變化
# 如果 is_blank_grid[:-1, :] 和 is_blank_grid[1:, :] 不一樣，說明中間存在一條水平邊界
vertical_changes = is_blank_grid[:-1, :] != is_blank_grid[1:, :]
distortion_mask[:-1, :] |= vertical_changes # 將邊界上側的 patch 標記為 True
distortion_mask[1:, :] |= vertical_changes  # 將邊界下側的 patch 標記為 True

# 3. 獲取畸變 patch 的索引
distortion_indices = distortion_mask.flatten().nonzero(as_tuple=True)[0].tolist()

# --- 報告與可視化結果 ---
print("\n--- Analysis Complete ---")
num_distortion_patches = len(distortion_indices)
distortion_percentage = (num_distortion_patches / 1024) * 100
print(f"Total patches: 1024")
print(f"Number of distortion (boundary) patches identified: {num_distortion_patches}")
print(f"Percentage of distortion patches: {distortion_percentage:.2f}%")
# print(f"Indices of distortion patches: {distortion_indices}")

# 簡單可視化
print("\n--- Visualization of Patch Grid ---")
print("'.': Blank, '#': Content, 'X': Distortion/Boundary")
grid_vis = np.full((32, 32), ' ')
grid_vis[is_blank_grid.numpy()] = '.'
grid_vis[~is_blank_grid.numpy()] = '#'
grid_vis[distortion_mask.numpy()] = 'X'
for row in grid_vis:
    print(' '.join(row))

# 儲存結果以供訓練使用
output_file = os.path.join(OUTPUT_DIR, f"{image_id}_{step_image_filename}_distortion_info.pt")
torch.save({'distortion_indices': distortion_indices}, output_file)
print(f"\nDistortion patch indices saved to '{output_file}'")