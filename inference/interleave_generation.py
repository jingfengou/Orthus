import torch
import os
import sys
import random
import numpy as np
from torchvision.transforms.functional import to_pil_image

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
from models.modeling_orthus import OrthusForConditionalGeneration
import torch.nn.functional as F
import json
from PIL import Image

ckpt_path = "SJTU-Deng-Lab/Orthus-7B-base"
processor = OrthusProcessor.from_pretrained(ckpt_path)
model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

exp_dir = os.path.join(root_path, "results/interleave_generation")
os.makedirs(exp_dir, exist_ok=True)

set_seed(1)

prompt = "Generate an overview of famous historical landmarks. Include pictures and descriptions of each."

# Get inputs
interleave_inputs = processor([prompt], return_tensors="pt")
interleave_input_ids = interleave_inputs['input_ids'].to(model.device)

# Generate a list of image latents & text tokens, cfg_guidance is not supported in this version.
outputs = model.generate(
    input_ids=interleave_input_ids,
    cfg_scale=1.0,
    multimodal_generation_mode="interleaved-text-image",
    interleave_output_format=True,
    do_sample=True,
    use_cache=True,
    max_new_tokens=4096,
)

# Decode outputs
with torch.no_grad():
    # split text tokens and image latents
    text_tokens = []
    all_image_embeds_wo_quant = []
    for output in outputs:
        if len(output.shape) == 1:
            if torch.sum(output == 8196) == 0 and torch.sum(output == 8197) == 0:
                text_tokens.append(output)
        else:
            all_image_embeds_wo_quant.append(output)

    # decode image one by one
    num_images = len(all_image_embeds_wo_quant) // 1024
    for id in range(num_images):
        # decode image_latents to pixel_values
        image_latents = torch.cat(all_image_embeds_wo_quant[id * 1024:(id + 1) * 1024], dim=0).to(model.device)
        pixel_values = model.decode_image_latents(image_latents)

        # convert raw image to pil image
        images = processor.postprocess_pixel_values(pixel_values)
        images = [to_pil_image(img.detach().cpu()) for img in images]

        # save image
        images[0].save(os.path.join(exp_dir, f'./{prompt}_img{id+1}.jpg'))

    # decode generated text
    text_tokens = torch.cat(text_tokens, dim=0).to(model.device)
    text = processor.batch_decode(text_tokens.unsqueeze(0), skip_special_tokens=True)[0]

    # save text
    with open(os.path.join(exp_dir, f'./{prompt}_text.txt'), "w", encoding="utf-8") as file:
        file.write(prompt + '\n' + text)
