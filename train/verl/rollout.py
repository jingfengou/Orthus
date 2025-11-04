import copy
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout, _ROLLOUT_REGISTRY

from models.modeling_orthus import OrthusForConditionalGeneration

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - PEFT 是延迟依赖
    LoraConfig = None
    TaskType = None
    get_peft_model = None


class OrthusRollout(BaseRollout):
    """Rollout engine tailored for Orthus multimodal generation."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

        self.tokenizer = model_config.tokenizer
        self.processor = model_config.get_processor() if hasattr(model_config, "get_processor") else None

        self.module: Optional[nn.Module] = None
        self._device = get_torch_device()
        self._state_device = torch.device("cpu")
        self.sleep_level = 0  # align with vLLM rollout contract
        self._printed_resume = False
        self._update_calls = 0
        self._generate_calls = 0

    # ---------------------------------------------------------------------
    # 生命周期控制
    # ---------------------------------------------------------------------
    async def resume(self, tags: list[str]):
        """恢复权重或 KV cache。当前实现只在 GPU 上保持权重常驻。"""
        if self.module is None:
            return
        if "weights" in tags:
            self.module.to(self._device)
        if "kv_cache" in tags:
            torch.cuda.empty_cache()

    async def release(self):
        """释放 GPU 显存，权重回迁至 CPU。"""
        if self.module is None:
            return
        self.module.to(self._state_device)
        torch.cuda.empty_cache()

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """将 FSDP 广播来的参数同步至 rollout 模型。"""
        module = self._ensure_module()

        state_updates: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, tensor in weights:
            if isinstance(tensor, torch.Tensor):
                state_updates[name] = tensor.to(self._device, non_blocking=True)

        self._update_calls += 1
        if self._update_calls <= 3:
            print(f"[OrthusRollout] update_weights call #{self._update_calls} | tensors={len(state_updates)}")

        missing = module.load_state_dict(state_updates, strict=False)
        if missing.missing_keys:
            print(f"[OrthusRollout] Warning: missing keys {missing.missing_keys[:4]} ...")
        if missing.unexpected_keys:
            print(f"[OrthusRollout] Warning: unexpected keys {missing.unexpected_keys[:4]} ...")

        del state_updates
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 核心生成逻辑
    # ---------------------------------------------------------------------
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self._generate_calls += 1
        module = self._ensure_module()

        batch_size = prompts.batch.batch_size[0]
        micro_batch = max(getattr(self.config, "micro_batch_size", batch_size), 1)
        num_chunks = max(batch_size // micro_batch, 1)

        if self._generate_calls <= 3:
            print(
                f"[OrthusRollout] generate_sequences call #{self._generate_calls} "
                f"| batch={batch_size} micro_batch={micro_batch} chunks={num_chunks}"
            )

        outputs = [self._generate_minibatch(module, chunk) for chunk in prompts.chunk(num_chunks)]
        return DataProto.concat(outputs)

    # ---------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------
    def _ensure_module(self) -> nn.Module:
        if self.module is not None:
            return self.module

        hf_config = copy.deepcopy(self.model_config.hf_config)
        module = OrthusForConditionalGeneration(hf_config)

        if get_peft_model is not None and self.model_config.lora_rank and self.model_config.lora_rank > 0:
            target_modules = self.model_config.target_modules
            lora_cfg = {
                "task_type": TaskType.CAUSAL_LM if TaskType is not None else "CAUSAL_LM",
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": target_modules,
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_cfg))

        module.to(self._device)
        module.eval()
        self.module = module
        return module

    def _prepare_multi_modal_inputs(self, prompts: DataProto, device: torch.device) -> Dict[str, Any]:
        if "multi_modal_inputs" not in prompts.non_tensor_batch:
            return {}

        raw_inputs = prompts.non_tensor_batch["multi_modal_inputs"]
        if isinstance(raw_inputs, torch.Tensor):
            raw_inputs = raw_inputs.tolist()
        elif hasattr(raw_inputs, "tolist"):
            raw_inputs = raw_inputs.tolist()

        indices = prompts.non_tensor_batch.get("multi_modal_inputs_idx")
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        stacked = extract_multi_modal_inputs(raw_inputs, indices)
        result: Dict[str, Any] = {}
        for key, value in stacked.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device, non_blocking=True)
            else:
                result[key] = [tensor.to(device, non_blocking=True) for tensor in value]
        return result

    def _generate_minibatch(self, module: nn.Module, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", getattr(self.config, "top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", getattr(self.config, "top_k", 0)))

        generation_config = GenerationConfig(
            do_sample=do_sample,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
        )

        module.eval()
        multi_modal_inputs = self._prepare_multi_modal_inputs(prompts, idx.device)

        branch_kwargs = {
            "input_ids": idx,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "max_new_tokens": response_length,
            "use_cache": True,
        }
        branch_kwargs.update(multi_modal_inputs)

        autocast_device = get_device_name()
        if self._generate_calls <= 3:
            print(
                f"[OrthusRollout] minibatch shapes input={tuple(idx.shape)} "
                f"response_len={response_length} has_image_latents={'image_latents' in branch_kwargs}"
            )
        with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            output = module.generate(
                inputs=idx,
                generation_config=generation_config,
                multimodal_generation_mode_list=["text-only"],
                kwargs_list=[branch_kwargs],
            )

        if hasattr(output, "sequences"):
            seq = output.sequences
        elif isinstance(output, list):
            seq = output[0]
        else:
            seq = output

        prompt_length = idx.size(1)
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            padding = torch.full(
                (seq.size(0), delta_length),
                fill_value=pad_token_id,
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat((seq, padding), dim=1)

        generated_batch_size = seq.size(0)
        response = seq[:, prompt_length:]
        response_length = response.size(1)

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": seq[:, :prompt_length],
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        module.train()
        return DataProto(batch=batch)


_ROLLOUT_REGISTRY[("orthus", "sync")] = __name__ + ".OrthusRollout"
