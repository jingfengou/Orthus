######## text-to-image generation ########
import torch
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus import OrthusForConditionalGeneration
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from torchvision.transforms.functional import to_pil_image

# set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ckpt_path = "SJTU-Deng-Lab/Orthus-7B-instruct"
processor = OrthusProcessor.from_pretrained(ckpt_path)
model = OrthusForConditionalGeneration.from_pretrained(
    ckpt_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

exp_dir = os.path.join(root_path, "results/t2i_generation")
os.makedirs(exp_dir, exist_ok=True)

# Prepare a prompt list
prompt_list = [ "Portrait of a digital shaman.", \
                "The image features an ancient Chinese landscape with a mountain, waterfalls, willow trees, and arch bridges set against a blue background.", \
                "Portrait of a monkey wearing an astronaut helmet.", \
                "A neon-colored frog in a cyberpunk setting.", \
                "A painting of a Persian cat dressed as a Renaissance king, standing around a skyscraper overlooking a city.", \
                "A girl looks out from the edge of a mountain onto a large city at night.", \
                "An anime-style depiction of a boy that showcases impressive artistic skill.", \
                "A cobblestone street with a tree over the sea at sunset, illuminated by sun rays, in a colorful illustration by Peter Chan on Artstation.", \
                ]

set_seed(12)
exp_dir = os.path.join(root_path, f"results/t2i_generation")
os.makedirs(exp_dir, exist_ok=True)

for prompt in tqdm(prompt_list):
    # We set 'Generate an image.' as the uncondtion_text_prompt
    prompt = [prompt, 'Generate an image.']
    # Preprocess the prompt
    inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)
    # Generate continuous image latents
    output = model.generate(
        **inputs,
        cfg_scale=5.0,
        multimodal_generation_mode="image-only",
        use_cache=True,
    )

    # Deconde image latents with vision decoder
    with torch.no_grad():
        # decode image_latents to pixel_values
        pixel_values = model.decode_image_latents(output)

        # convert raw image to pil image
        images = processor.postprocess_pixel_values(pixel_values)
        images = [to_pil_image(img.detach().cpu()) for img in images]

        # Save the image
        save_path = os.path.join(exp_dir, f'{prompt[0]}.jpg')
        images[0].save(save_path)
        print(f'Images saved successfully in {save_path}')
