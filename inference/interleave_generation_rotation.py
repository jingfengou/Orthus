import torch
import os
import sys
import random
import numpy as np
from datasets import load_dataset

# set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
import json
from PIL import Image
from torchvision.transforms.functional import to_pil_image



ckpt_path = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-base-sample80b100ep500l1e-5-weight-F"
processor = OrthusProcessor.from_pretrained(ckpt_path)

model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

exp_dir = os.path.join(root_path, "results/mydatasets/sft-myb-base-sample80b100ep500l1e-5-weight-F-train-modified-test")
os.makedirs(exp_dir, exist_ok=True)

set_seed(42)

# Load dataset - you need to specify the correct dataset path here
# This example assumes the dataset is in JSON format
# Replace with your actual dataset path
dataset_path = "/data1/oujingfeng/project/twgi/datasets/mydatasets"  # Update this path as needed
dataset = load_dataset("json", data_files=f"{dataset_path}/modified_data.json", split="train")
dataset = dataset.select(range(1))  # Reduce sample count for quick smoke test
# Define the instruction template
instruction = (
"You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
"The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
"respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
)

# Process each item in the dataset
for idx, item in enumerate(dataset):
    # Extract fields from the dataset - adjust these field names based on your dataset structure
    question = item.get('Question', '')  # Adjust field name as needed
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(item.get('Choices', []))])
    # If choices is a list, format it appropriately
    if isinstance(choices_text, list):
        choices_text = "\n" + "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices_text)])
    elif not choices_text.startswith('\n'):
        choices_text = "\n" + choices_text  # Add newline if not present
    
    # Construct the prompt text with only the input (no answer/ground truth for inference)
    prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
    
    # Load the image
    image_path = item.get('image_path', '') or item.get('image', '')  # Adjust field name as needed
    question_image_path = os.path.join(dataset_path, item.get('Task', ''), item.get('Image_id', ''), item.get('Combined_image', ''))
    try:
        question_image = Image.open(question_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Warning: Image at {question_image_path} not found. Using a blank image.")
        question_image = Image.new('RGB', (224, 224), (255, 255, 255))
    
    interleave_inputs_con = processor([prompt_text], images=[question_image],padding=True, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
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
    SPACED_INTERVAL_STEP = 10  # 對於 'spaced_interval'：每隔 10 個內容 patch 干預一次
    BURST_LENGTH = 5         # 對於 'burst_and_gap'：連續干預 5 個 patch
    GAP_LENGTH = 15          # 對於 'burst_and_gap'：然後跳過 15 個 patch
    # ==========================================================
    
    
    # # 【修改】 1. 載入多個步驟的數據，並根據策略生成干預列表
    # # ==========================================================
    # steps_to_load = ["step_0", "step_1"] # 您希望在推理中生成幾張圖，就載入幾個
    # ANALYSIS_DIR = "/data1/oujingfeng/project/twgi/Orthus/analysis/"
    # # INTERVENTION_DATA_FILE = "multi_intervention_data.pt"
    # intervention_indices = []
    # target_latents_for_intervention = []
    # for step in steps_to_load:
    #     INTERVENTION_DATA_FILE = "/data1/oujingfeng/project/twgi/Orthus/analysis/" + step + "multi_intervention_data.pt"
    #     print(f"Loading multi-intervention data from '{INTERVENTION_DATA_FILE}'...")
    #     try:
    #         intervention_data = torch.load(INTERVENTION_DATA_FILE)
    #         all_non_blank_indices = intervention_data['intervention_indices']
    #         # intervention_indices.append(intervention_data['intervention_indices']) # 載入索引列表
    #         target_latents_for_intervention.append(intervention_data['target_image_latents'].to(model.device, torch.bfloat16))
        
    #         final_intervention_indices = []
    #         if STRATEGY == 'even_indices':
    #             # 策略一：只取偶數索引的內容 patch (第 1, 3, 5, ... 個)
    #             final_intervention_indices = [all_non_blank_indices[i] for i in range(len(all_non_blank_indices)) if i % 2 == 0]
    
    #         elif STRATEGY == 'spaced_interval':
    #             # 策略二：每隔 N 個內容 patch 干預一次
    #             final_intervention_indices = [all_non_blank_indices[i] for i in range(SPACED_INTERVAL_STEP - 1, len(all_non_blank_indices), SPACED_INTERVAL_STEP)]
    
    #         elif STRATEGY == 'burst_and_gap':
    #             # 策略三：連續干預 B 個，然後跳過 G 個
    #             i = 0
    #             while i < len(all_non_blank_indices):
    #                 # 連續干預 B 個
    #                 burst = all_non_blank_indices[i : i + BURST_LENGTH]
    #                 final_intervention_indices.extend(burst)
    #                 # 跳過 B + G 個，移動到下一個 burst 的起點
    #                 i += (BURST_LENGTH + GAP_LENGTH)
    #         else:
    #             raise ValueError(f"Unknown strategy: {STRATEGY}")
    
    #         intervention_indices.append(final_intervention_indices)
    #         print(f"Strategy '{STRATEGY}' selected. {len(final_intervention_indices)} intervention points for {step}.")
        
    #     except FileNotFoundError:
    #         print(f"Error: Analysis data file not found: {INTERVENTION_DATA_FILE}")
    #         exit()
    # # ==========================================================
    
    kwargs_con = {
        "input_ids": interleave_input_ids_con,
        # "cfg_scale": 3.0,    # 忽略uncon的影响
        "cfg_scale": None,    # 忽略uncon的影响
        "interleave_output_format": True,
        "max_new_tokens": 4096,
        "do_sample": True,
        "attention_mask": interleave_inputs_con['attention_mask'].to(model.device),
        "use_cache": True,
        # "intervention_indices":intervention_indices,
        # "target_latents_for_intervention":target_latents_for_intervention,
    }
    kwargs_uncon = {
        "input_ids": interleave_input_ids_uncon,
        "cfg_scale": 1.0,  # 冗余参数 没有被使用
        "attention_mask": interleave_inputs_uncon['attention_mask'].to(model.device),
        "use_cache": True,
    }
    
    
    # outputs = model.generate(
    #     multimodal_generation_mode_list=["interleaved-text-image","image-only"],
    #     kwargs_list=[kwargs_con, kwargs_uncon],
    # )
    outputs = model.generate(
        multimodal_generation_mode_list=["interleaved-text-image"],
        kwargs_list=[kwargs_con],
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
            image_latents = torch.cat(
                all_image_embeds_wo_quant[id * 1024 : (id + 1) * 1024], dim=0
            ).to(model.device)
            pixel_values_uint8 = model.decode_image_latents_processed(
                image_latents, processor.image_processor
            )

            images_wo_quant = [to_pil_image(img.cpu()) for img in pixel_values_uint8]
            images_wo_quant[0].save(os.path.join(exp_dir, f"./sample{idx:04d}_img{id+1}_cfg3.jpg"))
    
        # decode generated text
        text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
        text = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]
    
        with open(os.path.join(exp_dir, f'./sample{idx:04d}_text.txt'), "w", encoding="utf-8") as file:
            file.write(f"sample{idx:04d}" + '\n' + text)
            
    print(f"Processed sample {idx+1}")
