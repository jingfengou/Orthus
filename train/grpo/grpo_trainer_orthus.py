import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, Trainer
from trl import GRPOConfig
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp

from custom_rewards import answer_correctness_reward, format_reward

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
from models.processing_orthus import OrthusProcessor


class Geneval_score:
    """Placeholder Geneval implementation used during GRPO bring-up."""

    def __init__(self, args):
        print("Initializing Dummy Geneval Reward Model...")
        self.device = "cpu"

    def __call__(self, images: List[Image.Image], prompts: List[str], metadatas: List[Dict]) -> List[float]:
        print(f"  - [Dummy Reward] Received {len(images)} images for reward calculation.")
        return [random.uniform(0.5, 0.95) for _ in range(len(images))]

    def load_to_device(self, device):
        self.device = device
        print(f"  - [Dummy Reward] Moved to device: {device}")


reward_funcs_registry = {
    "geneval": Geneval_score,
    "answer_correctness": answer_correctness_reward,
    "format": format_reward,
}


class OrthusGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Optional[List[Callable]] = None,
        ref_model: Optional[PreTrainedModel] = None,
        args: Optional[GRPOConfig] = None,
        script_args: Any = None,
        peft_config: Optional[Dict] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        print("--- OrthusGRPOTrainer Initializing ---")
        self.script_args = script_args
        self.reward_funcs = reward_funcs or []

        self.processor = OrthusProcessor.from_pretrained(model if isinstance(model, str) else model.config._name_or_path)

        cache_size = getattr(args, "image_latent_cache_size", 0)
        if cache_size and hasattr(self.processor, "enable_latents_cache"):
            self.processor.enable_latents_cache(cache_size)

        if isinstance(model, str):
            torch_dtype = None
            if getattr(args, "bf16", False):
                torch_dtype = torch.bfloat16
            elif getattr(args, "fp16", False):
                torch_dtype = torch.float16
            elif getattr(args, "torch_dtype", None) is not None:
                torch_dtype = args.torch_dtype
            model = OrthusForConditionalGeneration.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )
        self.model = model

        if ref_model is None and args.beta != 0:
            self.ref_model = create_reference_model(self.model)
        else:
            self.ref_model = ref_model

        self._freeze_model_parts()

        super().__init__(model=self.model, args=args, data_collator=self.collate_fn, **kwargs)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

            # Ensure reference model stays frozen after accelerator wrapping.
            ref_module = getattr(self.ref_model, "module", self.ref_model)
            for param in ref_module.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        print(f"Loaded {len(self.reward_funcs)} reward functions: {[f.__name__ for f in self.reward_funcs]}")

        self.beta = args.beta
        self.num_generations = args.num_generations
        self.cfg_weight = args.cfg_weight
        self.reward_smooth = args.reward_smooth
        self.kl_reweight = args.kl_reweight
        self.entropy_reward = args.entropy_reward
        self.image_base_dir = getattr(script_args, "image_base_dir", None)
        self.debug_mode = os.getenv("ORTHUS_GRPO_DEBUG", "0") == "1"

        if self.processor is not None:
            self.processor.padding_side = "left"

        self.completion_log_path: Optional[Path] = None
        if self.is_world_process_zero():
            log_dir_override = (
                os.getenv("ORTHUS_COMPLETION_LOG_DIR")
                or os.getenv("LOG_DIR")
                or getattr(args, "completion_log_dir", None)
            )
            default_log_dir = Path(__file__).resolve().parent / "logs"
            log_dir = Path(log_dir_override) if log_dir_override else default_log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            run_name = getattr(args, "run_name", "orthus-grpo")
            self.completion_log_path = log_dir / f"{run_name}_completions.txt"
            with self.completion_log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"\n===== New session started at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def _freeze_model_parts(self):
        print("Freezing non-trainable parts of the model...")
        for name, param in self.model.named_parameters():
            if "vqmodel" in name or "mlp_head" in name:
                param.requires_grad = False
                print(f"  - Froze policy model parameter: {name}")

        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        print("Model parts frozen.")

    def collate_fn(self, features):
        return {key: [d[key] for d in features] for key in features[0]}

    def _log_debug(self, message: str):
        if getattr(self, "debug_mode", False) and self.is_world_process_zero():
            print(f"[GRPO DEBUG] {message}", flush=True)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        device = self.accelerator.device
        prompts = inputs["prompt"]
        image_paths = inputs["image_path"]
        metadatas = inputs.get("metadata", [{} for _ in prompts])
        batch_size = len(prompts)

        self._log_debug(f"compute_loss start | batch_size={batch_size} num_generations={self.num_generations}")

        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        eos_token_id = self.processor.tokenizer.eos_token_id

        generated_texts: List[str] = []
        metadata_for_rewards: List[Dict] = []
        full_sequences: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        completion_lengths: List[int] = []
        logged_prompt_metadata: List[Dict[str, Any]] = []

        unwrap_model = self.accelerator.unwrap_model(model)
        original_mode = model.training
        gen_start = time.time()

        with torch.no_grad():
            model.eval()
            for prompt_text, image_path, metadata in zip(prompts, image_paths, metadatas):
                image = self._load_image(image_path)
                proc_inputs = self.processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt",
                    vqmodel=self.model.model.vqmodel,
                    image_cache_key=str(image_path),
                )
                proc_inputs = {k: v.to(device) for k, v in proc_inputs.items()}

                prompt_ids = proc_inputs["input_ids"][0].clone()
                prompt_len = prompt_ids.size(0)
                prompt_meta = self._prepare_prompt_metadata(str(image_path), prompt_ids)

                generation_kwargs = {
                    "input_ids": proc_inputs["input_ids"],
                    "attention_mask": proc_inputs.get("attention_mask"),
                    "image_latents": proc_inputs.get("image_latents"),
                    "interleave_output_format": False,
                    "max_new_tokens": self.args.max_completion_length,
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "use_cache": True,
                }

                for gen_idx in range(self.num_generations):
                    self._log_debug(f"generate call {gen_idx + 1}/{self.num_generations}")
                    outputs = unwrap_model.generate(
                        multimodal_generation_mode_list=["text-only"],
                        kwargs_list=[generation_kwargs],
                    )
                    sequence = self._extract_generated_sequence(outputs).to(device)

                    completion_ids = sequence[prompt_len:].clone()
                    completion_ids = self._trim_after_eos(completion_ids, eos_token_id, pad_token_id)

                    full_sequence = torch.cat([prompt_ids, completion_ids], dim=0)
                    full_sequences.append(full_sequence)
                    prompt_lengths.append(prompt_len)
                    completion_lengths.append(completion_ids.size(0))

                    logged_prompt_metadata.append(dict(prompt_meta))
                    generated_texts.append(
                        self.processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    )
                    metadata_for_rewards.append(metadata)

        self._log_debug(
            f"generation finished | total_samples={len(generated_texts)} elapsed={time.time() - gen_start:.2f}s"
        )
        if getattr(self, "debug_mode", False) and self.is_world_process_zero():
            for idx, (prompt_text, decoded_text) in enumerate(zip(prompts, generated_texts)):
                if idx >= 2:
                    break
                preview = decoded_text[:200].replace("\n", "\\n")
                self._log_debug(f"decoded completion[{idx}] => {preview}")

        self._log_samples(logged_prompt_metadata, generated_texts)

        if original_mode:
            model.train()

        if not full_sequences:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero, {}) if return_outputs else zero

        padded_full = pad_sequence(full_sequences, batch_first=True, padding_value=pad_token_id).to(device)
        attention_mask = (padded_full != pad_token_id).long()

        was_training = model.training
        if was_training:
            model.eval()

        self._log_debug(
            f"logprobs input shapes | padded={tuple(padded_full.shape)} attention={tuple(attention_mask.shape)}"
        )
        log_probs_policy_full = self.get_text_per_token_logps(model, padded_full, attention_mask)
        if was_training:
            model.train()
        if self.ref_model is not None:
            with torch.no_grad():
                log_probs_ref_full = self.get_text_per_token_logps(self.ref_model, padded_full, attention_mask)
        else:
            log_probs_ref_full = torch.zeros_like(log_probs_policy_full)

        policy_segments: List[torch.Tensor] = []
        ref_segments: List[torch.Tensor] = []
        completion_masks: List[torch.Tensor] = []
        for idx, (prompt_len, completion_len) in enumerate(zip(prompt_lengths, completion_lengths)):
            start = max(prompt_len - 1, 0)
            end = start + completion_len

            if completion_len == 0:
                policy_segments.append(torch.zeros(1, device=device))
                ref_segments.append(torch.zeros(1, device=device))
                completion_masks.append(torch.zeros(1, device=device))
                continue

            policy_segments.append(log_probs_policy_full[idx, start:end])
            ref_segments.append(log_probs_ref_full[idx, start:end])
            completion_masks.append(torch.ones(completion_len, device=device))

        policy_tensor = pad_sequence(policy_segments, batch_first=True, padding_value=0.0)
        ref_tensor = pad_sequence(ref_segments, batch_first=True, padding_value=0.0)
        mask_tensor = pad_sequence(completion_masks, batch_first=True, padding_value=0.0)

        rewards_tensor = torch.zeros(len(generated_texts), device=device)
        for reward_func in self.reward_funcs:
            rewards = reward_func(generated_texts=generated_texts, metadatas=metadata_for_rewards)
            rewards_tensor += torch.tensor(rewards, device=device, dtype=rewards_tensor.dtype)

        if rewards_tensor.numel() > 0:
            self._log_debug(
                f"reward stats | mean={rewards_tensor.mean().item():.4f} std={rewards_tensor.std().item():.4f}"
            )

        if batch_size == 0 or len(generated_texts) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero, {}) if return_outputs else zero

        if len(generated_texts) % batch_size != 0:
            raise ValueError(
                f"Generated samples ({len(generated_texts)}) not divisible by batch size ({batch_size}). "
                "Check generation loop."
            )

        generations_per_sample = len(generated_texts) // batch_size
        if generations_per_sample != self.num_generations:
            warnings.warn(
                f"Expected {self.num_generations} generations per prompt but produced {generations_per_sample}.",
                RuntimeWarning,
            )

        rewards_grouped = rewards_tensor.view(batch_size, generations_per_sample)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - mean_rewards) / (std_rewards + 1e-8)).reshape(-1).detach()

        advantages = advantages.to(device)
        pg_loss_per_token = -advantages.unsqueeze(1) * policy_tensor
        kl_div_per_token = policy_tensor - ref_tensor
        loss_per_token = pg_loss_per_token + self.beta * kl_div_per_token
        denom = mask_tensor.sum().clamp_min(1.0)
        loss = (loss_per_token * mask_tensor).sum() / denom

        self._log_debug(
            f"loss ready | denom={denom.item():.2f} mask_nonzero={int(mask_tensor.sum().item())} "
            f"policy_shape={tuple(policy_tensor.shape)}"
        )

        self._record_grpo_metrics(
            rewards_tensor.detach(),
            advantages.detach(),
            pg_loss_per_token.detach(),
            kl_div_per_token.detach(),
            mask_tensor.detach(),
        )

        if return_outputs:
            return loss, {"rewards": rewards_tensor}
        return loss

    def _prepare_prompt_metadata(self, image_path: str, prompt_ids: torch.Tensor) -> Dict[str, Any]:
        max_prompt_len = getattr(self.args, "max_prompt_length", None)
        prompt_len = prompt_ids.size(0)
        truncated = prompt_ids
        truncated_flag = False
        if max_prompt_len is not None and prompt_len > max_prompt_len:
            truncated = prompt_ids[:max_prompt_len]
            truncated_flag = True

        decoded_prompt = self.processor.tokenizer.decode(truncated, skip_special_tokens=False)
        return {
            "image": image_path or "N/A",
            "token_count": int(prompt_len),
            "truncated": truncated_flag,
            "prompt": decoded_prompt.strip(),
        }

    def _log_samples(self, prompt_metadata: List[Dict[str, Any]], generated_texts: List[str]):
        if self.completion_log_path is None or not self.is_world_process_zero():
            return

        step = getattr(self.state, "global_step", 0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.completion_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"\n--- Step {step} @ {timestamp} ---\n")
            for idx, (prompt_info, text) in enumerate(zip(prompt_metadata, generated_texts)):
                sanitized = text.replace("\r", "").strip()
                fp.write(
                    f"[{idx}] prompt(image={prompt_info['image']}, tokens={prompt_info['token_count']}"
                    f"{' truncated' if prompt_info['truncated'] else ''}):\n{prompt_info['prompt']}\n"
                )
                fp.write(f"    completion:\n{sanitized}\n")

    def get_text_per_token_logps(self, model, input_ids, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            train_mode="mmu",
            mode="discrete",
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return per_token_logps

    def _record_grpo_metrics(
        self,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        pg_loss: torch.Tensor,
        kl_div: torch.Tensor,
        mask: torch.Tensor,
    ):
        if not self.is_world_process_zero():
            return

        mask_sum = mask.sum().clamp_min(1.0).item()
        metrics = {
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std(unbiased=False).item(),
            "policy/advantage_mean": advantages.mean().item(),
            "policy/advantage_std": advantages.std(unbiased=False).item(),
            "loss/pg_loss": (pg_loss * mask).sum().item() / mask_sum,
            "loss/kl_div": (kl_div * mask).sum().item() / mask_sum,
        }
        self.log(metrics)

    def _load_image(self, image_path: str) -> Image.Image:
        resolved_path = image_path
        if not os.path.isabs(resolved_path) and self.image_base_dir:
            resolved_path = os.path.join(self.image_base_dir, resolved_path)
        with Image.open(resolved_path) as image:
            return image.convert("RGB")

    @staticmethod
    def _extract_generated_sequence(outputs) -> torch.LongTensor:
        if isinstance(outputs, list):
            sequence = outputs[0]
        elif hasattr(outputs, "sequences"):
            sequence = outputs.sequences[0]
        else:
            sequence = outputs[0] if getattr(outputs, "ndim", 0) > 1 else outputs
        return sequence.squeeze(0)

    @staticmethod
    def _trim_after_eos(
        completion_ids: torch.Tensor, eos_token_id: Optional[int], pad_token_id: int
    ) -> torch.Tensor:
        completion_ids = completion_ids.flatten()
        if eos_token_id is not None:
            eos_positions = torch.nonzero(completion_ids == eos_token_id, as_tuple=False)
            if eos_positions.numel() > 0:
                completion_ids = completion_ids[: eos_positions[0].item()]
        completion_ids = completion_ids[completion_ids != pad_token_id]
        return completion_ids

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        return inputs
