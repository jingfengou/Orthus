import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

# Ensure Verl与本地模块可被加载
ORTHUS_ROOT = Path(__file__).resolve().parents[2]
if str(ORTHUS_ROOT) not in sys.path:
    sys.path.append(str(ORTHUS_ROOT))

VERL_ROOT = Path(__file__).resolve().parents[3] / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.append(str(VERL_ROOT))

# Propagate Python path给 Ray 子进程
python_path_entries = [str(ORTHUS_ROOT), str(VERL_ROOT)]
existing_python_path = os.environ.get("PYTHONPATH")
if existing_python_path:
    python_path_entries.append(existing_python_path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_path_entries)

from verl.trainer.main_ppo import run_ppo
from verl.workers.rollout import base as rollout_base


def _default(path_env: str, fallback: str) -> str:
    value = os.getenv(path_env)
    if value:
        return value
    return fallback


@hydra.main(config_path=str(VERL_ROOT / "verl" / "trainer" / "config"), config_name="ppo_trainer", version_base=None)
def main(cfg):
    rollout_base._ROLLOUT_REGISTRY[("orthus", "sync")] = "train.verl.rollout.OrthusRollout"

    OmegaConf.set_struct(cfg, False)

    # Project root corresponds to repository顶层（twgi），需要向上三级
    project_root = Path(__file__).resolve().parents[3]

    cfg.data.custom_cls.path = str(Path(__file__).resolve().parent / "dataset.py")
    cfg.data.custom_cls.name = "OrthusRLDataset"
    cfg.data.image_root = _default(
        "ORTHUS_IMAGE_ROOT", str(project_root / "datasets" / "mydatasets" / "dataset" / "data")
    )
    cfg.data.precomputed_latents_dir = _default(
        "ORTHUS_LATENTS_DIR", str(project_root / "datasets" / "mydatasets" / "dataset" / "latents_cache")
    )
    cfg.data.train_files = _default(
        "ORTHUS_RL_DATA", str(project_root / "datasets" / "mydatasets" / "dataset" / "data.json")
    )
    cfg.data.val_files = _default(
        "ORTHUS_RL_VAL_DATA", str(project_root / "datasets" / "mydatasets" / "dataset" / "data.json")
    )
    cfg.data.train_max_samples = int(os.getenv("ORTHUS_RL_TRAIN_MAX_SAMPLES", "0"))
    cfg.data.val_max_samples = int(os.getenv("ORTHUS_RL_VAL_MAX_SAMPLES", "0"))
    cfg.data.max_prompt_length = 1400
    cfg.data.max_response_length = 256
    cfg.data.train_batch_size = 16
    cfg.data.max_samples = int(os.getenv("ORTHUS_MAX_SAMPLES", "-1"))
    cfg.data.shuffle = True
    cfg.data.return_multi_modal_inputs = True
    cfg.actor_rollout_ref.model.path = _default(
        "ORTHUS_MODEL_PATH",
        str(project_root / "checkpoints" / "mydatasets" / "orthus-7b-sft-base-sample4000b100e100l1e-5weight-F"),
    )
    cfg.actor_rollout_ref.model.tokenizer_path = cfg.actor_rollout_ref.model.path
    cfg.actor_rollout_ref.model.enable_gradient_checkpointing = True
    cfg.actor_rollout_ref.model.override_config = {
        "attn_implementation": "flash_attention_2",
    }
    cfg.actor_rollout_ref.model.external_lib = "train.verl.rollout"

    # LoRA configuration
    cfg.actor_rollout_ref.model.lora_rank = int(os.getenv("ORTHUS_LORA_RANK", "32"))
    cfg.actor_rollout_ref.model.lora_alpha = int(os.getenv("ORTHUS_LORA_ALPHA", "32"))

    default_lora_targets = "all-linear"
    env_lora_targets = os.getenv("ORTHUS_LORA_TARGET")
    if env_lora_targets:
        cfg.actor_rollout_ref.model.target_modules = env_lora_targets.strip()
    else:
        cfg.actor_rollout_ref.model.target_modules = default_lora_targets

    cfg.actor_rollout_ref.model.exclude_modules = os.getenv("ORTHUS_LORA_EXCLUDE", "lm_head")

    cfg.actor_rollout_ref.actor.optim.lr = float(os.getenv("ORTHUS_LORA_LR", "3e-5"))

    cfg.actor_rollout_ref.actor.fsdp_config.wrap_policy = {
        "transformer_layer_cls_to_wrap": ["ChameleonDecoderLayer"]
    }
    cfg.actor_rollout_ref.actor.fsdp_config.param_offload = True
    cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    cfg.actor_rollout_ref.actor.fsdp_config.use_torch_compile = False
    cfg.actor_rollout_ref.actor.fsdp_config.model_dtype = "bf16"
    if hasattr(cfg.actor_rollout_ref, "ref") and hasattr(cfg.actor_rollout_ref.ref, "fsdp_config"):
        cfg.actor_rollout_ref.ref.fsdp_config.model_dtype = "bf16"
    if hasattr(cfg.actor_rollout_ref, "ref") and hasattr(cfg.actor_rollout_ref.ref, "fsdp_config"):
        cfg.actor_rollout_ref.ref.fsdp_config.model_dtype = "bf16"

    cfg.data.model_path = os.path.abspath(cfg.actor_rollout_ref.model.path)
    print("[Orthus] model path:", cfg.actor_rollout_ref.model.path)

    default_generations = int(os.getenv("ORTHUS_NUM_GENERATIONS", "2"))
    rollout_name = os.getenv("ORTHUS_ROLLOUT_NAME", "orthus")
    rollout_mode = os.getenv("ORTHUS_ROLLOUT_MODE", "sync")
    cfg.actor_rollout_ref.rollout.name = rollout_name
    cfg.actor_rollout_ref.rollout.mode = rollout_mode
    cfg.actor_rollout_ref.rollout.do_sample = True
    rollout_n = getattr(cfg.actor_rollout_ref.rollout, "n", None)
    if rollout_n is None or rollout_n == 1:
        cfg.actor_rollout_ref.rollout.n = default_generations
    cfg.actor_rollout_ref.rollout.n = int(cfg.actor_rollout_ref.rollout.n)
    cfg_scale_env = os.getenv("ORTHUS_CFG_SCALE")
    if cfg_scale_env is not None:
        try:
            cfg.actor_rollout_ref.rollout.cfg_scale = float(cfg_scale_env)
        except ValueError:
            cfg.actor_rollout_ref.rollout.cfg_scale = cfg_scale_env

    cfg.algorithm.adv_estimator = "grpo"

    cfg.trainer.n_gpus_per_node = int(os.getenv("ORTHUS_NUM_GPUS", "8"))

    cfg.data.prompt_key = "prompt"
    cfg.data.max_prompt_length = cfg.data.max_prompt_length
    cfg.data.max_response_length = cfg.data.max_response_length

    # Ensure Ray workers继承必要环境
    if "ray_kwargs" not in cfg:
        cfg.ray_kwargs = {}
    if "ray_init" not in cfg.ray_kwargs or cfg.ray_kwargs.ray_init is None:
        cfg.ray_kwargs.ray_init = {}
    if "runtime_env" not in cfg.ray_kwargs.ray_init or cfg.ray_kwargs.ray_init.runtime_env is None:
        cfg.ray_kwargs.ray_init.runtime_env = {}
    if "env_vars" not in cfg.ray_kwargs.ray_init.runtime_env or cfg.ray_kwargs.ray_init.runtime_env.env_vars is None:
        cfg.ray_kwargs.ray_init.runtime_env.env_vars = {}
    cfg.ray_kwargs.ray_init.runtime_env.env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]
    cfg.ray_kwargs.ray_init.runtime_env.env_vars["VERL_SUPPRESS_CONFIG_LOG"] = os.environ.get(
        "VERL_SUPPRESS_CONFIG_LOG", "1"
    )

    # 允许通过环境变量开启 Ray local_mode 以便调试（single-process 执行）
    if os.environ.get("ORTHUS_RAY_LOCAL_MODE", "0").lower() not in {"0", "false"}:
        cfg.ray_kwargs.ray_init.local_mode = True
        print("[Orthus] Ray local_mode 已开启（ORTHUS_RAY_LOCAL_MODE）。")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        rollout_cfg = cfg.actor_rollout_ref.rollout
        rollout_cfg.tensor_model_parallel_size = 1
        rollout_cfg.data_parallel_size = 1
        rollout_cfg.pipeline_model_parallel_size = 1
        rollout_cfg.expert_parallel_size = 1
        actor_fsdp = cfg.actor_rollout_ref.actor.fsdp_config
        actor_fsdp.fsdp_size = 1
        actor_fsdp.wrap_policy = {"disable": True}
        actor_fsdp.use_orig_params = True
        actor_fsdp.param_offload = True
        actor_fsdp.optimizer_offload = True
        actor_fsdp.mixed_precision = {
            "param_dtype": "bfloat16",
            "reduce_dtype": "float32",
            "buffer_dtype": "float32",
        }
        actor_fsdp.use_torch_compile = False
        actor_fsdp.forward_prefetch = False
        actor_fsdp.reshard_after_forward = True
        actor_fsdp.offload_policy = True
        cfg.actor_rollout_ref.actor.use_torch_compile = False
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
        cfg.actor_rollout_ref.model.lora_rank = 0
        cfg.actor_rollout_ref.model.override_config["attn_implementation"] = "eager"
        cfg.actor_rollout_ref.model.enable_gradient_checkpointing = False

    os.environ.setdefault("VERL_SUPPRESS_CONFIG_LOG", "1")

    run_ppo(cfg)


if __name__ == "__main__":
    main()
