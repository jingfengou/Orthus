import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in os.sys.path:
    os.sys.path.insert(0, str(root_path))

from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interleave inference for rotation reasoning.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the fine-tuned checkpoint.")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Directory containing dataset json/images.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data.json",
        help="Dataset json file name inside dataset_path.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick runs.")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting index when slicing the dataset.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store generated outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation max_new_tokens.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="CFG scale for interleaved generation. Use None to disable CFG.",
    )
    parser.add_argument(
        "--latents_cache_size",
        type=int,
        default=128,
        help="Enable processor latent cache with given size; set 0 to disable.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose outputs already exist on disk.",
    )
    parser.add_argument(
        "--debug_every",
        type=int,
        default=0,
        help="Log debug information every N samples (0 to disable).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=(
            "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
            "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
            "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        ),
        help="Instruction prefix for prompts.",
    )
    parser.add_argument(
        "--uncond_prompt",
        type=str,
        default="generate images",
        help="Prompt for unconditional branch when CFG is used.",
    )
    return parser.parse_args()


def build_prompt(instruction: str, item: dict) -> str:
    question = item.get("Question", "")
    choices = item.get("Choices", [])
    choices_text = "\n".join(f"{chr(65 + idx)}) {opt}" for idx, opt in enumerate(choices))
    if choices_text and not choices_text.startswith("\n"):
        choices_text = "\n" + choices_text
    return f"{instruction}<image>\n\nQuestion: {question}{choices_text}\n\nAnswer: "


def load_image_or_blank(path: Path) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    LOGGER.warning("Image missing at %s, using blank placeholder.", path)
    return Image.new("RGB", (224, 224), (255, 255, 255))


def to_device(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def decode_and_save_images(
    image_latents_list: Sequence[torch.Tensor],
    model: OrthusForConditionalGeneration,
    processor: OrthusProcessor,
    output_dir: Path,
    sample_idx: int,
) -> None:
    if not image_latents_list:
        return

    latents_per_image = 1024
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    stacked_latents = torch.cat(image_latents_list, dim=0)
    num_images = stacked_latents.shape[0] // latents_per_image
    image_latents_list.clear()

    for image_id in range(num_images):
        slice_latents = stacked_latents[
            image_id * latents_per_image : (image_id + 1) * latents_per_image
        ].to(device)
        pixel_values_uint8 = model.decode_image_latents_processed(slice_latents, processor.image_processor)
        pil_images = [to_pil_image(img.cpu()) for img in pixel_values_uint8]
        if not pil_images:
            continue
        image_path = output_dir / f"sample{sample_idx:04d}_img{image_id + 1}.jpg"
        pil_images[0].save(image_path)


def save_text(output_dir: Path, sample_idx: int, text: str) -> None:
    text_path = output_dir / f"sample{sample_idx:04d}_text.txt"
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading processor and model from %s", args.ckpt_path)
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    if args.latents_cache_size > 0:
        processor.enable_latents_cache(args.latents_cache_size)

    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    dataset_file = args.dataset_path / args.dataset_file
    LOGGER.info("Loading dataset from %s (split=%s)", dataset_file, args.split)
    ds = load_dataset("json", data_files=str(dataset_file), split=args.split)
    if args.sample_offset or args.max_samples:
        end = args.sample_offset + args.max_samples if args.max_samples else None
        ds = ds.select(range(args.sample_offset, end))
    LOGGER.info("Total samples to process: %d", len(ds))

    uncond_inputs = processor([args.uncond_prompt], return_tensors="pt")
    uncond_inputs = to_device(uncond_inputs, model.device, torch.bfloat16)
    uncond_kwargs = {
        "input_ids": uncond_inputs["input_ids"],
        "attention_mask": uncond_inputs.get("attention_mask"),
        "use_cache": True,
    }

    images_root = args.dataset_path / "data"

    with torch.no_grad():
        for local_idx, item in enumerate(ds):
            sample_idx = args.sample_offset + local_idx
            text_output_path = args.output_dir / f"sample{sample_idx:04d}_text.txt"
            image_output_path = args.output_dir / f"sample{sample_idx:04d}_img1.jpg"
            if args.skip_existing and text_output_path.exists() and image_output_path.exists():
                continue

            prompt_text = build_prompt(args.instruction, item)
            question_image_path = images_root / item.get("Task", "") / item.get("Level", "") / item.get("Image_id", "") / item.get("Combined_image", "")
            question_image = load_image_or_blank(question_image_path)

            cond_inputs = processor(
                [prompt_text],
                images=[question_image],
                padding=True,
                return_tensors="pt",
                vqmodel=model.model.vqmodel,
            )
            cond_inputs = to_device(cond_inputs, model.device, torch.bfloat16)

            kwargs_con = {
                "input_ids": cond_inputs["input_ids"],
                "max_new_tokens": args.max_new_tokens,
                "attention_mask": cond_inputs.get("attention_mask"),
                "use_cache": True,
                "cfg_scale": args.cfg_scale,
                "interleave_output_format": True,
            }

            outputs = model.generate(
                multimodal_generation_mode_list=["interleaved-text-image"],
                kwargs_list=[kwargs_con],
            )

            text_tokens: List[torch.Tensor] = []
            image_latents: List[torch.Tensor] = []
            for output in outputs:
                if output.ndim == 1:
                    if torch.sum(output == 8196) == 0 and torch.sum(output == 8197) == 0:
                        text_tokens.append(output)
                else:
                    image_latents.append(output)

            decode_and_save_images(image_latents, model, processor, args.output_dir, sample_idx)

            if text_tokens:
                concatenated = torch.cat(text_tokens, dim=0).to(model.device)
                decoded_text = processor.batch_decode(concatenated.unsqueeze(0), skip_special_tokens=True)[0]
                save_text(args.output_dir, sample_idx, decoded_text)

            if args.debug_every and (local_idx + 1) % args.debug_every == 0:
                LOGGER.debug("Processed %d samples", local_idx + 1)

    LOGGER.info("Inference completed. Outputs stored in %s", args.output_dir)


if __name__ == "__main__":
    main()
