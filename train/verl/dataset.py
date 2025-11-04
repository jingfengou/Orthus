import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from safetensors.torch import safe_open
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.configuration_chameleon import ChameleonVQVAEConfig
from models.modeling_orthus import ChameleonVQVAE
from models.processing_orthus import OrthusProcessor


_PRINTED_PROCESSOR_MSG = False
_VQMODEL_CACHE: Dict[str, torch.nn.Module] = {}


def _load_vq_model(model_path: str) -> torch.nn.Module:
    """Load Orthus VQ model once and reuse across workers."""

    global _VQMODEL_CACHE

    cached = _VQMODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    model_dir = Path(model_path)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"[OrthusRLDataset] Warning: missing config.json in {model_path},无法加载 VQ 模型。")
        return None

    print(f"[OrthusRLDataset] Loading VQ model from {model_path}")
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if "vq_config" not in config_data:
            raise ValueError("config.json 中缺少 `vq_config` 字段。")

        vq_config = ChameleonVQVAEConfig(**config_data["vq_config"])
        vqmodel = ChameleonVQVAE(vq_config)

        index_file = model_dir / "model.safetensors.index.json"
        weight_map: Dict[str, str] = {}
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
        else:
            single_file = model_dir / "model.safetensors"
            if single_file.exists():
                weight_map = {f"model.vqmodel.{name}": single_file.name for name in vqmodel.state_dict().keys()}
            else:
                raise ValueError("未找到 safetensors 权重索引。")

        shard_to_keys: Dict[str, List[str]] = defaultdict(list)
        for param_name, shard_name in weight_map.items():
            if param_name.startswith("model.vqmodel."):
                shard_to_keys[shard_name].append(param_name)

        if not shard_to_keys:
            raise ValueError("权重中未找到 `model.vqmodel` 前缀对应的参数。")

        state_dict: Dict[str, torch.Tensor] = {}
        for shard_name, keys in shard_to_keys.items():
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"缺少权重分片: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
                for full_key in keys:
                    tensor = shard_file.get_tensor(full_key)
                    sub_key = full_key.replace("model.vqmodel.", "", 1)
                    state_dict[sub_key] = tensor

        load_result = vqmodel.load_state_dict(state_dict, strict=True)
        if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
            raise ValueError(
                f"加载 VQ 模型时存在缺失或多余参数: missing={getattr(load_result, 'missing_keys', [])}, "
                f"unexpected={getattr(load_result, 'unexpected_keys', [])}"
            )

        vqmodel.to("cpu")
        vqmodel.eval()
        for param in vqmodel.parameters():
            param.requires_grad_(False)

        _VQMODEL_CACHE[model_path] = vqmodel
        return vqmodel
    except Exception as exc:  # noqa: BLE001
        print(f"[OrthusRLDataset] Warning: 读取 VQ 模型失败 ({exc})，将回退到零 latent。")
        return None


def _build_prompt(sample: Dict[str, Any]) -> str:
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    question = sample.get("Question", "")
    choices = sample.get("Choices", [])
    lines = [f"{chr(65 + idx)}) {opt}" for idx, opt in enumerate(choices)]
    choices_text = "\n" + "\n".join(lines) if lines else ""
    return f"{instruction}<image>\n\nQuestion: {question}{choices_text}\n\nAnswer: "


def _load_image_or_blank(path: Path) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


class OrthusRLDataset(Dataset):
    """Dataset adapter that prepares Orthus multimodal prompts for Verl."""

    def __init__(
        self,
        data_files: List[str] | str,
        tokenizer,
        config,
        processor: Optional[OrthusProcessor] = None,
        max_samples: int = -1,
        vqmodel: Optional[torch.nn.Module] = None,
    ):
        self.processor = processor
        self.vqmodel = vqmodel
        model_path = config.get("model_path") or os.environ.get("ORTHUS_MODEL_PATH")

        if self.processor is None:
            if not model_path:
                raise ValueError("OrthusRLDataset needs `processor` or `data.model_path`/`ORTHUS_MODEL_PATH`.")
            global _PRINTED_PROCESSOR_MSG
            rank = int(os.getenv("RANK", "0"))
            if not _PRINTED_PROCESSOR_MSG and rank == 0:
                print(f"[OrthusRLDataset] Loading processor from {model_path}")
                _PRINTED_PROCESSOR_MSG = True
            self.processor = OrthusProcessor.from_pretrained(model_path)

        if self.vqmodel is None:
            vq_path = os.environ.get("ORTHUS_VQMODEL_PATH", model_path)
            if vq_path:
                self.vqmodel = _load_vq_model(vq_path)

        if self.vqmodel is not None and hasattr(self.vqmodel, "quantize"):
            self.latent_dtype = self.vqmodel.quantize.embedding.weight.dtype
        else:
            self.latent_dtype = torch.float32

        if hasattr(self.processor, "enable_latents_cache"):
            self.processor.enable_latents_cache(0)

        self.use_precomputed_latents = False
        self.precomputed_latents_dir = None

        if isinstance(data_files, str):
            data_files = [data_files]

        dataset = load_dataset("json", data_files=data_files, split="train")
        if max_samples and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        image_root = Path(config.get("image_root", ""))
        if not image_root:
            raise ValueError("`data.image_root` must be configured for OrthusRLDataset.")

        self.records: List[Dict[str, Any]] = []
        for idx, sample in enumerate(dataset):
            prompt = _build_prompt(sample)
            image_rel = Path(sample.get("Task", "")) / sample.get("Level", "") / sample.get("Image_id", "") / sample.get(
                "Combined_image", ""
            )
            image_path = (image_root / image_rel).resolve()
            cache_key = hashlib.sha1(str(image_path).encode("utf-8"), usedforsecurity=False).hexdigest()
            record = {
                "uid": idx,
                "prompt": prompt,
                "image_path": image_path,
                "metadata": dict(sample),
                "cache_key": cache_key,
            }
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        question_image = _load_image_or_blank(record["image_path"])

        precomputed_latents = None

        image_latents = None
        if self.vqmodel is not None:
            processor_inputs = self.processor(
                text=[record["prompt"]],
                images=[question_image],
                return_tensors="pt",
                padding="longest",
                truncation=False,
                vqmodel=self.vqmodel,
                image_cache_key=record["cache_key"],
            )
            image_latents = processor_inputs.pop("image_latents")[0].to(dtype=self.latent_dtype)
        else:
            processor_inputs = self.processor(
                text=[record["prompt"]],
                images=[question_image],
                return_tensors="pt",
                padding="longest",
                truncation=False,
            )
            image_latents = torch.zeros((1, 32, 32, 256), dtype=self.latent_dtype)

        input_ids = processor_inputs.pop("input_ids")[0]
        attention_mask = processor_inputs.pop("attention_mask")[0]

        position_ids = torch.arange(input_ids.size(0), dtype=torch.long)

        multi_modal_inputs: Dict[str, Any] = {}
        if "pixel_values" in processor_inputs:
            multi_modal_inputs["pixel_values"] = processor_inputs["pixel_values"][0]
        multi_modal_inputs["image_latents"] = image_latents

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
            "prompt": record["prompt"],
            "image_path": str(record["image_path"]),
            "metadata": record["metadata"],
            "uid": torch.tensor(record["uid"], dtype=torch.long),
        }
        return sample


__all__ = ["OrthusRLDataset"]
