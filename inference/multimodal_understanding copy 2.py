####### multimodality understanding ########
import torch
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus import OrthusForConditionalGeneration
import torch.nn.functional as F
import requests
from PIL import Image
from safetensors.torch import load_file

ckpt_path = "SJTU-Deng-Lab/Orthus-7B-instruct"
processor = OrthusProcessor.from_pretrained(ckpt_path)
model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    device_map="auto",
)

### Example usage ###
prompt = "<image>Can you please tell me what kind of farm equipment would be essential for this kind of farm?"
image = Image.open(os.path.join(root_path, "inference/mmu_demo/Grain-production-wheat.jpg"))
images=[image]

inputs = processor(prompt, images=images, return_tensors="pt", vqmodel=model.model.vqmodel).to(model.device, torch.bfloat16)
if len(images) >=2:
    inputs['image_latents']=inputs['image_latents'].unsqueeze(dim=0)

generated_ids = model.generate(input_ids=inputs['input_ids'], \
                               attention_mask=inputs['attention_mask'], cfg_scale=None, \
                               image_latents=inputs['image_latents'], max_new_tokens=512, do_sample=False, use_cache=False) 
out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'Response: {out}')
