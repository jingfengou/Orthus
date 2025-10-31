import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import List

import sys
import json

from PIL import Image
from datasets import load_dataset

import torch
import torch.distributed as dist

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration


logger = logging.getLogger(__name__)


def get_distributed_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return world_size, rank, local_rank


def maybe_init_distributed():
    if dist.is_available() and not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def build_image_cache_key(question_image_path: str, step_image_paths: List[str]) -> str:
    """
    Reuses the same cache key definition as `InterleaveSFTDataset`.
    """
    key_parts = [os.path.abspath(question_image_path)]
    for path in step_image_paths:
        key_parts.append(os.path.abspath(path))
    return "|".join(key_parts)


def compute_digest(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()


def load_image_or_blank(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except FileNotFoundError:
        logger.warning("Image missing at %s, substituting blank image.", path)
        return Image.new("RGB", (224, 224), (255, 255, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Orthus image latents for SFT datasets.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path or identifier of the pretrained base model checkpoint.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the JSON/JSONL data file that will be used during SFT.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Root folder containing all images referenced by the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store the precomputed latents. "
        "Defaults to <image_folder>/latents_cache if not provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent files instead of skipping them.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for datasets map (for large datasets).",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    world_size, rank, local_rank = get_distributed_env()
    maybe_init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.image_folder) / "latents_cache"
    ensure_output_dir(output_dir)
    if rank == 0:
        logger.info("Latents will be stored under %s", output_dir)

    if rank == 0:
        logger.info("Loading processor and base model from %s", args.ckpt_path)
    processor = OrthusProcessor.from_pretrained(args.ckpt_path)
    model = OrthusForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    vqmodel = model.model.vqmodel.to(device)
    vqmodel.eval()
    del model
    torch.cuda.empty_cache()

    if rank == 0:
        logger.info("Loading dataset from %s", args.data_file)
    ext = Path(args.data_file).suffix.lower()
    if ext == ".jsonl":
        dataset = load_dataset("json", data_files=args.data_file, split="train")
    else:
        with open(args.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = [data[i] for i in range(len(data))]

    total = len(dataset)
    if rank == 0:
        logger.info("Total number of samples: %d", total)

    for idx, item in enumerate(dataset):
        if idx % world_size != rank:
            continue

        category = item.get("Category", "")
        task = item.get("Task", "")
        level = item.get("Level", "")
        image_id = item.get("Image_id", "")

        question_image_path = os.path.join(
            args.image_folder, task, level, image_id, item.get("Combined_image", "")
        )
        step_image_paths = []
        for step in item.get("Rotation_steps", []):
            step_image_paths.append(
                os.path.join(args.image_folder, task, level, image_id, step.get("image", ""))
            )

        cache_key = build_image_cache_key(question_image_path, step_image_paths)
        digest = compute_digest(cache_key)
        latent_path = output_dir / f"{digest}.pt"

        if latent_path.exists() and not args.overwrite:
            if idx % 100 == 0 and rank == 0:
                logger.info("[%d/%d] Skipping existing latent %s", idx, total, latent_path.name)
            continue

        images = [load_image_or_blank(question_image_path)]
        for step_path in step_image_paths:
            images.append(load_image_or_blank(step_path))

        prompt_placeholder = "<image>"
        with torch.no_grad():
            outputs = processor(
                text=prompt_placeholder,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=4096,
                vqmodel=vqmodel,
                image_cache_key=cache_key,
            )
            latents = outputs["image_latents"]

        torch.save(
            {
                "image_latents": latents.cpu(),
                "num_images": latents.shape[0],
            },
            latent_path,
        )

        if idx % 100 == 0:
            logger.info("[rank %d] [%d/%d] Saved latents to %s", rank, idx, total, latent_path.name)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info("Completed latent precomputation.")


if __name__ == "__main__":
    main()
