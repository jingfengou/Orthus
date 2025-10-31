import argparse
import math
import os
import random
import sys
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
from models.processing_orthus import OrthusProcessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_distributed() -> int:
    if "LOCAL_RANK" not in os.environ:
        return -1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank


def parse_args() -> argparse.Namespace:
    default_ckpt = "/data1/oujingfeng/project/twgi/checkpoints/mydatasets/orthus-7b-sft-base-sample4000b100e100l1e-5weight-F"
    default_dataset_path = "/data1/oujingfeng/project/twgi/datasets/mydatasets/dataset"
    default_output_dir = os.path.join(
        root_path,
        "results/test_mydatasets/sft-myb-base-sample4000b100ep20-tail10",
    )

    parser = argparse.ArgumentParser(
        description="Interleave generation for the last 10% of the dataset using up to 8 GPUs."
    )
    parser.add_argument("--ckpt-path", default=default_ckpt, help="Checkpoint path.")
    parser.add_argument(
        "--dataset-path", default=default_dataset_path, help="Dataset directory."
    )
    parser.add_argument("--data-file", default="data_modified.json", help="Dataset json filename.")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory.")
    parser.add_argument(
        "--save-image-interval",
        type=int,
        default=50,
        help="Save generated images every N samples (global index). Set <=0 to skip saving images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens for generation.",
    )
    return parser.parse_args()


def get_rank_world() -> (int, int):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def broadcast_directory_setup(output_dir: str) -> None:
    rank, _ = get_rank_world()
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def chunk_indices(indices: List[int], world_size: int, rank: int) -> List[int]:
    return indices[rank::world_size]


def load_image(dataset_path: str, item: dict) -> Image.Image:
    image_path = (
        item.get("image_path", "") or item.get("image", "")
    )
    if image_path and os.path.isabs(image_path) and os.path.isfile(image_path):
        return Image.open(image_path).convert("RGB")

    task = item.get("Task", "")
    level = item.get("Level", "")
    image_id = item.get("Image_id", "")
    combined_image = item.get("Combined_image", "")
    composed_path = os.path.join(dataset_path, "data", task, level, image_id, combined_image)
    try:
        return Image.open(composed_path).convert("RGB")
    except FileNotFoundError:
        print(f"Warning: Image at {composed_path} not found. Using a blank image.")
    return Image.new("RGB", (224, 224), (255, 255, 255))


def main() -> None:
    args = parse_args()
    local_rank = init_distributed()

    rank, world_size = get_rank_world()
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cuda" if torch.cuda.is_available() else "cpu")

    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    dataset = load_dataset(
        "json",
        data_files=os.path.join(args.dataset_path, args.data_file),
        split="train",
    )
    total_samples = len(dataset)
    tail_count = max(1, math.ceil(total_samples * 0.1))
    start_idx = max(total_samples - tail_count, 0)
    target_indices = list(range(start_idx, total_samples))

    assigned_indices = chunk_indices(target_indices, world_size, rank)
    if not assigned_indices:
        if rank == 0:
            print("No samples assigned for processing. Check dataset size.")
        return

    broadcast_directory_setup(args.output_dir)

    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    for global_idx in assigned_indices:
        item = dataset[int(global_idx)]
        question = item.get("Question", "")
        answer = item.get("Answer", "")
        image_id = item.get("Image_id", "")
        choices = item.get("Choices", [])

        choices_text = "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)]) if choices else ""
        if choices_text and not choices_text.startswith("\n"):
            choices_text = "\n" + choices_text

        prompt_text = (
            instruction
            + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
        )

        question_image = load_image(args.dataset_path, item)

        interleave_inputs_con = processor(
            [prompt_text],
            images=[question_image],
            padding=True,
            return_tensors="pt",
            vqmodel=model.model.vqmodel,
        ).to(device, torch.bfloat16)
        interleave_input_ids_con = interleave_inputs_con["input_ids"]

        kwargs_con = {
            "input_ids": interleave_input_ids_con,
            "cfg_scale": None,
            "interleave_output_format": True,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "attention_mask": interleave_inputs_con["attention_mask"].to(device),
            "use_cache": True,
        }

        outputs = model.generate(
            multimodal_generation_mode_list=["interleaved-text-image"],
            kwargs_list=[kwargs_con],
        )

        text_tokens = []
        all_image_embeds_wo_quant = []
        for output in outputs:
            if len(output.shape) == 1:
                if torch.sum(output == 8196) == 0 and torch.sum(output == 8197) == 0:
                    text_tokens.append(output)
            else:
                all_image_embeds_wo_quant.append(output)

        save_images = (
            args.save_image_interval > 0
            and all_image_embeds_wo_quant
            and (global_idx % args.save_image_interval == 0)
        )

        if save_images:
            num_images = len(all_image_embeds_wo_quant) // 1024
            for local_image_idx in range(num_images):
                image_latents = torch.cat(
                    all_image_embeds_wo_quant[
                        local_image_idx * 1024 : (local_image_idx + 1) * 1024
                    ],
                    dim=0,
                ).to(device)
                pixel_values_uint8 = model.decode_image_latents_processed(
                    image_latents, processor.image_processor
                )
                images_wo_quant = [to_pil_image(img.cpu()) for img in pixel_values_uint8]
                image_name = f"sample{global_idx:05d}_img{local_image_idx + 1}.jpg"
                images_wo_quant[0].save(os.path.join(args.output_dir, image_name))

        generated_text = ""
        if text_tokens:
            text_tensor = torch.cat(text_tokens, dim=0).to(device)
            decoded = processor.batch_decode(
                text_tensor.unsqueeze(0), skip_special_tokens=True
            )
            generated_text = decoded[0]

        text_filename = f"sample{global_idx:05d}_text.txt"
        with open(
            os.path.join(args.output_dir, text_filename),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"sample{global_idx:05d}\n")
            file.write(f"Image_id: {image_id}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Ground truth: {answer}\n")
            file.write("Generated:\n")
            file.write(generated_text)

        print(f"[Rank {rank}] Processed sample {global_idx}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
