import torch
import os
import sys
import random
import numpy as np

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

ckpt_path = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-think-v010"
processor = OrthusProcessor.from_pretrained(ckpt_path)

model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

exp_dir = os.path.join(root_path, "results/mydatasets/sftv010ana3")
os.makedirs(exp_dir, exist_ok=True)

set_seed(50)

instruction = (
"You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
"The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
"respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
)

question = "First, rotate this cube stack along the X axis by 270 degrees. Then, rotate it along the Z axis by 270 degrees. Which option shows the correct final result?"
choices_text = "\nA: This option uses incorrect rotation angles.\nB: Correct: this is the result after performing all rotation steps correctly.\nC: This option rotates along wrong axes.\nD: This option performs the rotations in wrong order."
prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
image_path = "/data1/oujingfeng/project/twgi/datasets/mydatasets/sample_0001/combined.png"
images = Image.open(image_path).convert("RGB")

interleave_inputs_con = processor([prompt_text], images=[images],padding=True, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
interleave_input_ids_con = interleave_inputs_con['input_ids']

prompt_uncon="generate images"
interleave_inputs_uncon = processor([prompt_uncon], return_tensors="pt")
interleave_input_ids_uncon = interleave_inputs_uncon['input_ids'].to(model.device)

# ==========================================================
#               【策略配置區】
# ==========================================================
# 1. 選擇策略: 'even_indices', 'spaced_interval', 'burst_and_gap'
STRATEGY = 'burst_and_gap'

# 2. 策略對應的參數
SPACED_INTERVAL_STEP = 5  # 對於 'spaced_interval'：每隔 10 個內容 patch 干預一次
BURST_LENGTH = 5         # 對於 'burst_and_gap'：連續干預 5 個 patch
GAP_LENGTH = 15          # 對於 'burst_and_gap'：然後跳過 15 個 patch
# ==========================================================


# 【修改】 1. 載入多個步驟的數據，並根據策略生成干預列表
# ==========================================================
steps_to_load = ["step_0", "step_1"] # 您希望在推理中生成幾張圖，就載入幾個
ANALYSIS_DIR = "/data1/oujingfeng/project/twgi/Orthus/analysis/"
# INTERVENTION_DATA_FILE = "multi_intervention_data.pt"
intervention_indices = []
target_latents_for_intervention = []
for step in steps_to_load:
    INTERVENTION_DATA_FILE = "/data1/oujingfeng/project/twgi/Orthus/analysis/" + step + "multi_intervention_data.pt"
    print(f"Loading multi-intervention data from '{INTERVENTION_DATA_FILE}'...")
    try:
        intervention_data = torch.load(INTERVENTION_DATA_FILE)
        all_non_blank_indices = intervention_data['intervention_indices']
        # intervention_indices.append(intervention_data['intervention_indices']) # 載入索引列表
        target_latents_for_intervention.append(intervention_data['target_image_latents'].to(model.device, torch.bfloat16))
        
        final_intervention_indices = []
        if STRATEGY == 'even_indices':
            # 策略一：只取偶數索引的內容 patch (第 1, 3, 5, ... 個)
            final_intervention_indices = [all_non_blank_indices[i] for i in range(len(all_non_blank_indices)) if i % 2 == 0]

        elif STRATEGY == 'spaced_interval':
            # 策略二：每隔 N 個內容 patch 干預一次
            final_intervention_indices = [all_non_blank_indices[i] for i in range(SPACED_INTERVAL_STEP - 1, len(all_non_blank_indices), SPACED_INTERVAL_STEP)]

        elif STRATEGY == 'burst_and_gap':
            # 策略三：連續干預 B 個，然後跳過 G 個
            i = 0
            while i < len(all_non_blank_indices):
                # 連續干預 B 個
                burst = all_non_blank_indices[i : i + BURST_LENGTH]
                final_intervention_indices.extend(burst)
                # 跳過 B + G 個，移動到下一個 burst 的起點
                i += (BURST_LENGTH + GAP_LENGTH)
        else:
            raise ValueError(f"Unknown strategy: {STRATEGY}")

        intervention_indices.append(final_intervention_indices)
        print(f"Strategy '{STRATEGY}' selected. {len(final_intervention_indices)} intervention points for {step}.")
        
    except FileNotFoundError:
        print(f"Error: Analysis data file not found: {INTERVENTION_DATA_FILE}")
        exit()
# ==========================================================

kwargs_con = {
    "input_ids": interleave_input_ids_con,
    "cfg_scale": 1.0,    # 忽略uncon的影响
    "interleave_output_format": True,
    "max_new_tokens": 4096,
    "do_sample": True,
    "attention_mask": interleave_inputs_con['attention_mask'].to(model.device),
    "use_cache": True,
    "intervention_indices":intervention_indices,
    "target_latents_for_intervention":target_latents_for_intervention,
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
        images_wo_quant[0].save(os.path.join(exp_dir, f'./sample0001_img{id+1}_cfg3.jpg'))

    # decode generated text
    text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
    text = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]

    with open(os.path.join(exp_dir, f'./sample0001_text.txt'), "w", encoding="utf-8") as file:
        file.write("sample0001" + '\n' + text)