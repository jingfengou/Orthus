"""Multimodal understanding demo with per-token attention heatmaps."""
import json
import math
import numpy as np
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration


def sanitize_token_for_filename(token: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in token)
    return clean or "token"


def compute_attention_heatmap(attentions, target_idx: int, image_positions: torch.Tensor) -> torch.Tensor:
    last_layer = attentions[-1][0]
    mean_attn = last_layer.mean(dim=0)
    return mean_attn[target_idx, image_positions]




def postprocess_pixels(processor, pixel_values):
    image_processor = processor.image_processor
    if hasattr(image_processor, "postprocess"):
        return image_processor.postprocess(pixel_values, output_type="pil")

    arr = pixel_values.clone().to(torch.float32)
    if arr.ndim == 3:
        arr = arr.unsqueeze(0)

    do_normalize = getattr(image_processor, "do_normalize", True)
    do_rescale = getattr(image_processor, "do_rescale", True)
    image_mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]), dtype=arr.dtype).view(1, -1, 1, 1)
    image_std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]), dtype=arr.dtype).view(1, -1, 1, 1)
    if do_normalize:
        arr = arr * image_std + image_mean

    rescale_factor = getattr(image_processor, "rescale_factor", 1.0 / 255.0)
    if do_rescale and rescale_factor not in (0, None):
        arr = arr / rescale_factor

    arr = arr.clamp(0, 255).to(torch.uint8)
    pil_images = []
    for sample in arr:
        np_img = sample.permute(1, 2, 0).cpu().numpy()
        pil_images.append(Image.fromarray(np_img))
    return pil_images

def draw_patch_overlay(base_image: Image.Image, patch_indices, scores, grid_size: int, output_path: Path) -> None:
    if not patch_indices:
        base_image.save(output_path)
        return

    patch_side = base_image.width // grid_size
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    s_min, s_max = min(scores), max(scores)
    denom = (s_max - s_min) or 1.0

    for idx, score in zip(patch_indices, scores):
        row = idx // grid_size
        col = idx % grid_size
        x0 = col * patch_side
        y0 = row * patch_side
        x1 = x0 + patch_side
        y1 = y0 + patch_side
        norm = (score - s_min) / denom
        alpha = int(80 + 120 * norm)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 102, 0, alpha), outline=(255, 255, 255, 200))

    Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def visualize_token_attentions(processor, model, inputs, generated_ids, output_dir: Path, topk: int = 50):
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_len = inputs["input_ids"].shape[1]
    full_input_ids = generated_ids
    full_attention_mask = torch.zeros_like(full_input_ids)
    full_attention_mask[:, :prompt_len] = inputs["attention_mask"]
    full_attention_mask[:, prompt_len:] = 1

    with torch.no_grad():
        attn_outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            image_latents=inputs.get("image_latents"),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
            # train_mode="interleave",
            mode="discrete",
        )

    attentions = attn_outputs.attentions
    seq = full_input_ids[0]
    boi_id = getattr(processor.tokenizer, "boi_token_id", 8197)
    image_token_id = getattr(processor.tokenizer, "image_token_id", 8711)
    eoi_id = getattr(processor.tokenizer, "eoi_token_id", 8196)
    pad_id = processor.tokenizer.pad_token_id

    boi_positions = (seq == boi_id).nonzero(as_tuple=True)[0]
    if len(boi_positions) == 0:
        raise RuntimeError("Prompt does not contain BOI token.")

    image_seq_length = model.model.image_seq_length
    question_boi = int(boi_positions[0].item())
    question_image_positions = torch.arange(
        question_boi + 1,
        question_boi + 1 + image_seq_length,
        device=full_input_ids.device,
    )
    grid_size = int(math.sqrt(image_seq_length))

    pixel_values = inputs["pixel_values"].detach().to(torch.float32).cpu()
    processed_visual = postprocess_pixels(processor, pixel_values)[0]

    records = []
    sequence = seq.tolist()

    for idx, token_id in enumerate(sequence):
        if token_id in {boi_id, eoi_id, image_token_id, pad_id}:
            continue
        decoded = processor.tokenizer.decode([token_id], skip_special_tokens=False)
        decoded = decoded if decoded.strip() else f"id_{token_id}"

        scores = compute_attention_heatmap(attentions, idx, question_image_positions)
        k = min(topk, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=k)
        patch_indices = top_indices.cpu().tolist()
        patch_scores = top_scores.to(torch.float32).cpu().tolist()

        safe_name = sanitize_token_for_filename(decoded)
        source = "prompt" if idx < prompt_len else "generation"
        overlay_path = output_dir / f"{source}_pos{idx:04d}_{safe_name}_top{k}.png"
        draw_patch_overlay(processed_visual, patch_indices, patch_scores, grid_size, overlay_path)

        records.append(
            {
                "token": decoded,
                "position": idx,
                "source": source,
                "top_patches": [
                    {
                        "rank": rank + 1,
                        "patch_index": int(p_idx),
                        "row": int(p_idx // grid_size),
                        "col": int(p_idx % grid_size),
                        "attention": float(score),
                    }
                    for rank, (p_idx, score) in enumerate(zip(patch_indices, patch_scores))
                ],
            }
        )

    summary_path = output_dir / "attention_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def main():
    ckpt_path = "SJTU-Deng-Lab/Orthus-7B-instruct"
    output_dir = Path(root_path) / "analysis_outputs/mmu_full"

    processor = OrthusProcessor.from_pretrained(ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )

    prompt = (
        "<image>Can you please tell me what kind of farm equipment would be essential for this kind of farm?"
    )
    image = Image.open(os.path.join(root_path, "inference/mmu_demo/Grain-production-wheat.jpg")).convert("RGB")
    images = [image]

    inputs = processor(prompt, images=images, return_tensors="pt", vqmodel=model.model.vqmodel)
    processed_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                processed_inputs[key] = value.to(model.device, torch.bfloat16)
            else:
                processed_inputs[key] = value.to(model.device)
        else:
            processed_inputs[key] = value
    inputs = processed_inputs
    if len(images) >= 2 and inputs.get("image_latents") is not None:
        inputs["image_latents"] = inputs["image_latents"].unsqueeze(dim=0)

    kwargs_con = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "image_latents": inputs.get("image_latents"),
        "pixel_values": inputs.get("pixel_values"),
        "max_new_tokens": 512,
        "use_cache": True,
        "cfg_scale": None,
        "interleave_output_format": False,
    }

    raw_outputs = model.generate(
        multimodal_generation_mode_list=["text-only"],
        kwargs_list=[kwargs_con],
    )

    text_tokens = [entry for entry in raw_outputs if entry.ndim == 1]
    if not text_tokens:
        raise RuntimeError("No text tokens generated.")

    generated_suffix = torch.cat(text_tokens, dim=0).unsqueeze(0).to(model.device)
    full_ids = torch.cat([inputs["input_ids"], generated_suffix], dim=1)

    out = processor.batch_decode(generated_suffix, skip_special_tokens=True)[0]
    print(f"Response: {out}")

    visualize_token_attentions(processor, model, inputs, full_ids, output_dir, topk=50)

if __name__ == "__main__":
    main()
