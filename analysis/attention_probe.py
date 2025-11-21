#!/usr/bin/env python3
"""
Probe Orthus attentions for specific tokens within an interleaved-generation sample.

Example:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python attention_probe.py \
  --ckpt_path /data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-base-sample4000b100e15l1e-5weight-F \
  --dataset_path /data1/oujingfeng/project/twgi/datasets/mydatasets/dataset \
  --dataset_file data_modified_with_subject.json \
  --sample_idx 4501 \
  --output_dir analysis_outputs/sample004501 \
  --target_tokens "First,rotate,the,original,cube,stack,clockwise,along,the,Y,axis,by,270,degrees,Let's,generate,an,image,to,visualize,the,state,of,the,object,after,rotation" \
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import os
import sys
import torch
from PIL import Image, ImageDraw, ImageOps
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(instruction: str, item: dict) -> str:
    choices = item.get("Choices", [])
    choices_text = "\n".join(f"{chr(65 + idx)}) {opt}" for idx, opt in enumerate(choices))
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text
    question = item.get("Question", "")
    return f"{instruction}<image>\n\nQuestion: {question}{choices_text}\n\nAnswer: "


def to_device(batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def sanitize_token_for_filename(token: str) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in token)
    return sanitized or "token"


def tokenize_targets(tokenizer, targets: Sequence[str]) -> Dict[str, List[List[int]]]:
    token_map: Dict[str, List[List[int]]] = {}
    for token in targets:
        variants: List[List[int]] = []
        for prefix in (" ", ""):
            encoded = tokenizer.encode(f"{prefix}{token}", add_special_tokens=False)
            if encoded and encoded not in variants:
                variants.append(encoded)
        if not variants:
            raise ValueError(f"Token '{token}' could not be encoded in any form.")
        token_map[token] = variants
    return token_map


def locate_token_positions(sequence: Sequence[int], pattern: Sequence[int], offset: int) -> List[int]:
    """Return absolute indices (within the full sequence) for pattern matches."""
    if len(pattern) == 1:
        target_id = pattern[0]
        return [offset + idx for idx, token in enumerate(sequence) if token == target_id]

    positions: List[int] = []
    pattern_len = len(pattern)
    for idx in range(len(sequence) - pattern_len + 1):
        if list(sequence[idx : idx + pattern_len]) == list(pattern):
            positions.append(offset + idx + pattern_len - 1)
    return positions


def compute_attention_heatmap(
    attentions: torch.Tensor,
    target_idx: int,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    """
    attentions: [num_layers, batch, num_heads, seq_len, seq_len]
    return mean head attention from target_idx to image Positions using last layer.
    """
    last_layer = attentions[-1][0]  # [num_heads, seq_len, seq_len]
    mean_over_heads = last_layer.mean(dim=0)  # [seq_len, seq_len]
    scores = mean_over_heads[target_idx, image_positions]
    return scores


def draw_patch_overlay(
    image: Image.Image,
    patch_indices: Sequence[int],
    scores: Sequence[float],
    grid_size: int,
    output_path: Path,
) -> None:
    if not patch_indices:
        image.save(output_path)
        return

    patch_side = image.width // grid_size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    score_min = min(scores)
    score_max = max(scores)
    denom = (score_max - score_min) or 1.0

    for idx, score in zip(patch_indices, scores):
        row = idx // grid_size
        col = idx % grid_size
        x0 = col * patch_side
        y0 = row * patch_side
        x1 = x0 + patch_side
        y1 = y0 + patch_side
        norm = (score - score_min) / denom
        alpha = int(80 + 120 * norm)
        color = (255, 102, 0, alpha)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 200))

    blended = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    blended.save(output_path)


def save_report(
    output_path: Path,
    sample_idx: int,
    generated_text: str,
    per_token_records: Dict[str, Dict[str, object]],
) -> None:
    payload = {
        "sample_idx": sample_idx,
        "generated_text": generated_text,
        "tokens": per_token_records,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Orthus attentions on interleaved generation.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Checkpoint directory.")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--dataset_file", type=str, default="data.json", help="Dataset JSON file inside dataset_path.")
    parser.add_argument("--sample_idx", type=int, required=True, help="Zero-based sample index (matches sampleXXXX).")
    parser.add_argument(
        "--instruction",
        type=str,
        default=(
            "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
            "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
            "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        ),
    )
    parser.add_argument(
        "--target_tokens",
        type=str,
        default="this,cube",
        help="Comma-separated tokens to probe (e.g., 'this,cube').",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate for context.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG scale (None disables CFG).")
    parser.add_argument("--topk", type=int, default=50, help="Number of image patches to keep per token.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store reports/visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = args.dataset_path / args.dataset_file
    with dataset_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not (0 <= args.sample_idx < len(data)):
        raise ValueError(f"sample_idx {args.sample_idx} outside dataset range 0..{len(data)-1}")
    item = data[args.sample_idx]

    question_image_path = (
        args.dataset_path
        / "data"
        / item.get("Task", "")
        / item.get("Level", "")
        / item.get("Image_id", "")
        / item.get("Combined_image", "")
    )
    if not question_image_path.exists():
        raise FileNotFoundError(f"Question image not found at {question_image_path}")
    processor = OrthusProcessor.from_pretrained(str(args.ckpt_path))
    question_image = Image.open(question_image_path).convert("RGB")
    if question_image.width != question_image.height:
        side = max(question_image.size)
        mean = getattr(processor.image_processor, "image_mean", [1.0, 1.0, 1.0])
        pad_color = tuple(int(max(0, min(255, round(m * 255)))) for m in mean)
        padded_question_image = Image.new("RGB", (side, side), color=pad_color)
        offset = ((side - question_image.width) // 2, (side - question_image.height) // 2)
        padded_question_image.paste(question_image, offset)
    else:
        padded_question_image = question_image

    LOGGER.info("Loading processor/model from %s", args.ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        str(args.ckpt_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    model.config.output_attentions = True

    prompt = build_prompt(args.instruction, item)
    cond_inputs = processor(
        [prompt],
        images=[padded_question_image],
        padding=True,
        return_tensors="pt",
        vqmodel=model.model.vqmodel,
    )
    try:
        processed_visual = processor.image_processor.postprocess(
            cond_inputs["pixel_values"].clone().detach(),
            output_type="pil",
        )[0]
    except Exception:
        processed_visual = question_image.resize((512, 512))
    cond_inputs = to_device(cond_inputs, model.device, torch.bfloat16)

    kwargs_con = {
        "input_ids": cond_inputs["input_ids"],
        "attention_mask": cond_inputs.get("attention_mask"),
        "image_latents": cond_inputs.get("image_latents"),
        "pixel_values": cond_inputs.get("pixel_values"),
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "cfg_scale": args.cfg_scale,
        "interleave_output_format": True,
    }

    LOGGER.info("Running generation to obtain context tokens...")
    with torch.no_grad():
        outputs = model.generate(
            multimodal_generation_mode_list=["interleaved-text-image"],
            kwargs_list=[kwargs_con],
        )

    boi_id = getattr(processor.tokenizer, "boi_token_id", 8197)
    eoi_id = getattr(processor.tokenizer, "eoi_token_id", 8196)

    text_tokens: List[torch.Tensor] = []
    image_latents_out: List[torch.Tensor] = []
    for entry in outputs:
        if entry.ndim == 1:
            if torch.sum(entry == boi_id) == 0 and torch.sum(entry == eoi_id) == 0:
                text_tokens.append(entry)
        else:
            image_latents_out.append(entry)

    if not text_tokens:
        raise RuntimeError("No text tokens generated; cannot probe attention.")

    generated_ids = torch.cat(text_tokens, dim=0).unsqueeze(0).to(model.device)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    LOGGER.info("Generated text (truncated): %s", generated_text[:200] + ("..." if len(generated_text) > 200 else ""))

    prompt_len = cond_inputs["input_ids"].shape[1]
    full_input_ids = torch.cat([cond_inputs["input_ids"], generated_ids], dim=1)
    full_attention_mask = torch.cat(
        [
            cond_inputs["attention_mask"],
            torch.ones_like(generated_ids, device=model.device),
        ],
        dim=1,
    )

    with torch.no_grad():
        attn_outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            image_latents=cond_inputs.get("image_latents"),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
            train_mode="interleave",
            mode="discrete",
        )
    attentions = attn_outputs.attentions  # tuple[num_layers] each [batch, num_heads, seq, seq]

    token_map = tokenize_targets(processor.tokenizer, [tok.strip() for tok in args.target_tokens.split(",") if tok.strip()])
    generated_sequence = generated_ids[0].tolist()
    records: Dict[str, Dict[str, object]] = {}

    seq = full_input_ids[0]
    boi_positions = (seq == boi_id).nonzero(as_tuple=True)[0]
    if len(boi_positions) == 0:
        raise RuntimeError("Prompt does not contain BOI token.")
    image_seq_length = model.model.image_seq_length
    question_boi = int(boi_positions[0].item())
    question_image_positions = torch.arange(
        question_boi + 1,
        question_boi + 1 + image_seq_length,
        device=model.device,
    )
    grid_size = int(math.sqrt(image_seq_length))

    for token_text, patterns in token_map.items():
        all_positions: List[int] = []
        for pattern in patterns:
            all_positions.extend(locate_token_positions(generated_sequence, pattern, prompt_len))
        if not all_positions:
            LOGGER.warning("Token '%s' not found in generated text.", token_text)
            continue
        all_positions.sort()
        target_idx = all_positions[0]
        scores = compute_attention_heatmap(attentions, target_idx, question_image_positions)
        topk = min(args.topk, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=topk)
        patch_indices = top_indices.cpu().tolist()
        patch_scores = top_scores.to(dtype=torch.float32).cpu().tolist()

        token_key = f"{token_text}_pos{target_idx}"
        records[token_key] = {
            "absolute_position": int(target_idx),
            "prompt_length": int(prompt_len),
            "top_patches": [
                {
                    "rank": rank + 1,
                    "patch_index": int(idx),
                    "row": int(idx // grid_size),
                    "col": int(idx % grid_size),
                    "attention": float(score),
                }
                for rank, (idx, score) in enumerate(zip(patch_indices, patch_scores))
            ],
        }

        safe_name = sanitize_token_for_filename(token_text)
        overlay_path = args.output_dir / f"sample{args.sample_idx:04d}_{safe_name}_pos{target_idx}_top{topk}.png"
        draw_patch_overlay(
            processed_visual,
            patch_indices,
            patch_scores,
            grid_size=grid_size,
            output_path=overlay_path,
        )
        LOGGER.info("Saved overlay for token '%s' to %s", token_text, overlay_path)

    report_path = args.output_dir / f"sample{args.sample_idx:04d}_attention.json"
    save_report(report_path, args.sample_idx, generated_text, records)
    LOGGER.info("Attention report saved to %s", report_path)


if __name__ == "__main__":
    main()
