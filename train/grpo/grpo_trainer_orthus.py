import os
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
from transformers import Trainer, PreTrainedModel
from accelerate.utils import unwrap_model, wait_for_everyone
from trl.models import create_reference_model
import json
# 导入我们刚刚创建的函数
from custom_rewards import answer_correctness_reward, format_reward
# 导入 Orthus 相关的类
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 2. 将项目根目录临时添加到Python解释器的“模块搜索路径”列表中
sys.path.append(root_path)
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration
from models.processing_orthus import OrthusProcessor
# 导入您项目中的奖励函数注册表和实现 (假设它们在 utils 目录下)
# from utils.reward_geneval import Geneval_score
# ... 其他奖励函数 ...

# --- 为了演示，我们先定义一个假的 Geneval 类 ---
# --- 在您的实际项目中，请替换为真实的奖励函数实现 ---
class Geneval_score:
    def __init__(self, args):
        print("Initializing Dummy Geneval Reward Model...")
        # 在真实场景中，这里会加载模型等
        self.device = "cpu"
    def __call__(self, images: List[Image.Image], prompts: List[str], metadatas: List[Dict]) -> List[float]:
        print(f"  - [Dummy Reward] Received {len(images)} images for reward calculation.")
        # 返回一些随机奖励值用于演示
        return [random.uniform(0.5, 0.95) for _ in range(len(images))]
    def load_to_device(self, device):
        self.device = device
        print(f"  - [Dummy Reward] Moved to device: {device}")

# 奖励函数注册表
reward_funcs_registry = {
    "geneval": Geneval_score,
    "answer_correctness": answer_correctness_reward,
    "format": format_reward,
}

# ----------------------------------------------------

# 导入在 grpo_orthus.py 中定义的配置类
from grpo_orthus import OrthusGRPOConfig as GRPOConfig

class OrthusGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: List[Callable],
        ref_model: Optional[PreTrainedModel] = None,
        args: GRPOConfig = None,
        script_args = None,
        peft_config: Optional[Dict] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        # 1. 初始化模型、参考模型和处理器
        print("--- OrthusGRPOTrainer Initializing ---")
        self.script_args = script_args
        

        # ### 修改 ###
        # 将解析好的奖励函数列表直接保存
        self.reward_funcs = reward_funcs
        # 增加图像内在奖励的权重
        self.image_reward_weight = 1.0 # 这是一个可调整的超参数

        # 加载处理器
        self.processor = OrthusProcessor.from_pretrained(model if isinstance(model, str) else model.config._name_or_path)
        
        # 加载主模型 (策略模型)
        if isinstance(model, str):
            model = OrthusForConditionalGeneration.from_pretrained(
                model,
                torch_dtype=args.torch_dtype,
                attn_implementation=attn_implementation
            )
        self.model = model
        
        # 创建参考模型
        if ref_model is None and args.beta != 0:
            self.ref_model = create_reference_model(self.model)
        else:
            self.ref_model = ref_model
            
        # 冻结不需要训练的部分
        self._freeze_model_parts()

        # 初始化父类 Trainer
        super().__init__(model=self.model, args=args, data_collator=self.collate_fn, **kwargs)

        # ------------------- 主要修改点 -------------------
        # 2. 直接保存从主脚本传入的奖励函数列表
        self.reward_funcs = reward_funcs if reward_funcs is not None else []
        print(f"Loaded {len(self.reward_funcs)} reward functions: {[f.__name__ for f in self.reward_funcs]}")
        # --------------------------------------------------
        
        # 3. 保存 GRPO 特定参数
        self.beta = args.beta
        self.num_generations = args.num_generations
        self.cfg_weight = args.cfg_weight
        self.reward_smooth = args.reward_smooth
        self.kl_reweight = args.kl_reweight
        self.entropy_reward = args.entropy_reward
        self.image_base_dir = self.train_dataset.image_base_dir # 从数据集中获取图像根目录
        
        # 确保 Trainer 内部的 processor 属性被设置
        if self.processor is not None:
            self.processor.padding_side = "left"

    def _freeze_model_parts(self):
        """冻结 VQVAE 和奖励模型等不需要训练的部分。"""
        print("Freezing non-trainable parts of the model...")
        # 遍历策略模型（主模型）的参数
        for name, param in self.model.named_parameters():
            # ### 修改 ###
            # 冻结 vqmodel 和 mlp_head
            if 'vqmodel' in name or 'mlp_head' in name:
                param.requires_grad = False
                print(f"  - Froze policy model parameter: {name}")
        
        # 确保参考模型处于评估模式且不计算梯度
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        print("Model parts frozen.")
    def get_text_per_token_logps(self, model, input_ids, attention_mask):
        """
        辅助函数，用于计算给定文本 token 序列的每个 token 的对数概率。
        对标 Janus-Pro 项目中的 _get_per_token_logps_part。
        """
        # 模型前向传播获取 logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits

        # 平移 logits 和 labels 以进行下一个 token 的预测
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # 计算 log_softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 使用 gather 获取对应 token 的 log_prob
        per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return per_token_logps
    # def _load_reward_models(self):
    #     """根据配置加载和准备奖励模型。"""
    #     print("Loading reward models...")
    #     with open(self.args.reward_ckpt_path_file, "r") as f:
    #         reward_paths = json.load(f)

    #     # 将路径设置到 args 中，以便奖励模型初始化时可以访问
    #     for key, value in reward_paths.items():
    #         setattr(self.args, key, value)
            
    #     self.reward_funcs = []
    #     for func_name in self.script_args.reward_funcs:
    #         if func_name in reward_funcs_registry:
    #             reward_func_class = reward_funcs_registry[func_name]
    #             # 将整个 args 对象传递给奖励函数，以便它能找到所需路径
    #             self.reward_funcs.append(reward_func_class(self.args))
    #         else:
    #             raise ValueError(f"Unknown reward function: {func_name}")
        
    #     print(f"Loaded {len(self.reward_funcs)} reward models.")

    def collate_fn(self, features):
        """自定义数据整理器，将数据转换为一个字典。"""
        # 这个整理器很简单，因为大部分工作在 Dataset 的 __getitem__ 中完成了
        return {key: [d[key] for d in features] for key in features[0]}

#     def compute_loss(self, model, inputs, return_outputs=False):
#         # ==========================================================
#         #  阶段 1: 生成文本和图像 (无梯度)
#         # ==========================================================
#         torch.set_grad_enabled(False)
#         model.eval()

#         prompts = inputs["prompt"]
#         image_paths = inputs["image_path"]
#         metadatas = inputs["metadata"]
#         batch_size = len(prompts)
        
#         generated_latents_list = []
#         generated_images_list = []
#         generated_texts_list = []
        
#         for i in range(batch_size):
#             image = Image.open(os.path.join(self.image_base_dir, image_paths[i])).convert("RGB")
            
#             # 为每个 prompt 生成 G 次
#             for _ in range(self.num_generations):
#                 proc_inputs = self.processor(
#                     text=prompts[i], 
#                     images=image,
#                     return_tensors="pt"
#                 ).to(self.accelerator.device, dtype=self.args.torch_dtype)

#                 # ### 核心修改 ###
#                 # 使用 'interleaved-text-image' 模式生成，让模型自由决定生成文本还是图像
#                 gen_kwargs = {
#                     "max_new_tokens": 1500, # 足够生成一张图 + 文本
#                     "do_sample": True,
#                     "use_cache": True,
#                     "interleave_output_format": True
#                 }
#                 outputs = unwrap_model(model).generate(
#                     **proc_inputs,
#                     multimodal_generation_mode_list=["interleaved-text-image"],
#                     kwargs_list=[gen_kwargs]
#                 )
                
#                 # --- 解析生成结果 ---
#                 text_tokens = [out for out in outputs if isinstance(out, torch.Tensor) and out.dim() == 1]
#                 image_latents = [out for out in outputs if isinstance(out, torch.Tensor) and out.dim() == 2]

#                 # 解码文本
#                 if text_tokens:
#                     full_text_tokens = torch.cat(text_tokens, dim=0)
#                     decoded_text = self.processor.tokenizer.decode(full_text_tokens.tolist(), skip_special_tokens=True)
#                 else:
#                     decoded_text = ""
#                 generated_texts_list.append(decoded_text)
                
#                 # 解码图像 (只处理第一张生成的图)
#                 if len(image_latents) >= 1024:
#                     single_image_latents = torch.cat(image_latents[:1024], dim=0)
#                     generated_latents_list.append(single_image_latents)
#                     decoded_image = unwrap_model(model).decode_image_latents(single_image_latents)
#                     generated_images_list.append(self.processor.postprocess_pixel_values(decoded_image)[0])
#                 else:
#                     # 如果没有生成足够的图像块，我们添加一个空潜变量和空白图像作为占位符
#                     generated_latents_list.append(torch.zeros(1024, 256, device=self.accelerator.device))
#                     generated_images_list.append(Image.new('RGB', (512, 512), (255, 255, 255)))

#         # ==========================================================
#         #  阶段 2: 计算奖励 (无梯度)
#         # ==========================================================
#         if not generated_images_list:
#             return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

#         # 准备奖励函数输入
#         num_total_generations = len(generated_images_list)
#         reward_prompts = [p for p in prompts for _ in range(self.num_generations)]
#         reward_metadatas = [m for m in metadatas for _ in range(self.num_generations)]
        
#         # 计算所有外部奖励
#         total_extrinsic_rewards = torch.zeros(num_total_generations, device=self.accelerator.device)
#         for reward_func in self.reward_funcs:
#             # 根据函数名判断是文本奖励还是图像奖励
#             if "answer" in reward_func.__name__ or "format" in reward_func.__name__:
#                  rewards = reward_func(generated_texts=generated_texts_list, metadatas=reward_metadatas)
#             else: # 假设其他都是图像奖励，如 Geneval
#                  rewards = reward_func(images=generated_images_list, prompts=reward_prompts, metadatas=reward_metadatas)
#             total_extrinsic_rewards += torch.tensor(rewards, device=self.accelerator.device)

#         # ==========================================================
#         #  阶段 3: 计算 GRPO 损失 (有梯度)
#         # ==========================================================
#         torch.set_grad_enabled(True)
#         model.train() # 切换回训练模式

#         # 准备模型 forward pass 的输入
#         # 我们需要重新处理输入，这次是为了计算损失，而不是生成
#         input_prompts = [p for p in prompts for _ in range(self.num_generations)]
#         input_images = []
#         for i in range(batch_size):
#             image = Image.open(os.path.join(self.image_base_dir, image_paths[i])).convert("RGB")
#             input_images.extend([image] * self.num_generations)
            
#         full_text_list = [p + "<image>" for p in input_prompts] # 构造一个包含目标图像占位符的文本

#         loss_inputs = self.processor(
#             text=full_text_list,
#             images=input_images,
#             return_tensors="pt"
#         ).to(self.accelerator.device)

#         target_latents = torch.stack(generated_latents_list).to(self.accelerator.device) # Shape: [B*G, 1024, 256]

# # 计算策略模型的 diff_loss
#         _, diff_loss_policy = model(**loss_inputs, target_image_latents=target_latents.unsqueeze(1), train_mode='interleave')
        
#         # ### 新增：计算内在图像奖励 ###
#         # 使用指数衰减将 loss 转换为 reward，loss 越低，reward 越高
#         # alpha 是一个超参数，控制奖励的敏感度，可以设为 1.0
#         alpha = 1.0 
#         intrinsic_image_reward = torch.exp(-alpha * diff_loss_policy.detach())

#         # ### 核心修改：合并所有奖励 ###
#         total_rewards = total_extrinsic_rewards + self.image_reward_weight * intrinsic_image_reward
        
#         # 计算优势 (Advantage)
#         rewards_grouped = total_rewards.view(batch_size, self.num_generations)
#         mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
#         std_rewards = rewards_grouped.std(dim=1, keepdim=True)
#         advantages = ((rewards_grouped - mean_rewards) / (std_rewards + 1e-8)).flatten().detach()
#         # 计算参考模型的 diff_loss
#         with torch.no_grad():
#             if self.ref_model is not None:
#                 _, diff_loss_ref = self.ref_model(
#                     **loss_inputs,
#                     target_image_latents=target_latents.unsqueeze(1),
#                     train_mode='interleave'
#                 )
#             else:
#                 diff_loss_ref = torch.zeros_like(diff_loss_policy)
        
#         # 核心 GRPO 损失计算
#         # log π_θ(y|x) ≈ -diff_loss_policy
#         # log π_ref(y|x) ≈ -diff_loss_ref
#         log_probs_policy = -diff_loss_policy
#         log_probs_ref = -diff_loss_ref.detach()
        
#         # 策略梯度项: -Advantage * log π_θ
#         pg_loss = -advantages * log_probs_policy
        
#         # KL 惩罚项: KL(π_ref || π_θ) ≈ log π_ref - log π_θ
#         kl_div = log_probs_ref - log_probs_policy
        
#         # 总损失
#         loss = (pg_loss + self.beta * kl_div).mean()
        
#         # 日志记录
#         self.log_metrics(total_rewards, advantages, pg_loss, kl_div)

#         return loss
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        核心 GRPO 损失计算函数 - 纯文本模式。
        """
        # ==========================================================
        #  阶段 1: 生成文本 (无梯度)
        # ==========================================================
        torch.set_grad_enabled(False)
        model.eval()

        prompts_text = inputs["prompt"]
        image_paths = inputs["image_path"]
        metadatas = inputs["metadata"]
        batch_size = len(prompts_text)
        
        generated_texts_list = []
        prompt_ids_list = []
        completion_ids_list = []

        for i in range(batch_size):
            image = Image.open(os.path.join(self.image_base_dir, image_paths[i])).convert("RGB")
            
            proc_inputs = self.processor(
                text=prompts_text[i], 
                images=image,
                return_tensors="pt"
            ).to(self.accelerator.device)
            
            prompt_ids = proc_inputs['input_ids']
            prompt_length = prompt_ids.shape[1]

            # ### 核心修改：切换为 text-only 生成模式 ###
            gen_kwargs = {
                "max_new_tokens": self.args.max_completion_length,
                "do_sample": True,
                "temperature": self.args.temperature,
                "use_cache": True,
            }

            # 为每个 prompt 生成 G 次
            generated_ids = unwrap_model(model).generate(
                **proc_inputs,
                num_return_sequences=self.num_generations,
                multimodal_generation_mode_list=["text-only"],
                kwargs_list=[gen_kwargs]
            )
            
            # 分离 prompt 和 completion
            completions = generated_ids[:, prompt_length:]
            
            # 解码以用于奖励计算
            decoded_completions = self.processor.batch_decode(completions, skip_special_tokens=True)
            
            generated_texts_list.extend(decoded_completions)
            prompt_ids_list.extend([prompt_ids[0]] * self.num_generations) # 存储 G 份 prompt_ids
            completion_ids_list.append(completions)

        # ==========================================================
        #  阶段 2: 计算奖励 (无梯度)
        # ==========================================================
        if not generated_texts_list:
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        reward_metadatas = [m for m in metadatas for _ in range(self.num_generations)]
        
        total_rewards = torch.zeros(len(generated_texts_list), device=self.accelerator.device)
        for reward_func in self.reward_funcs:
            # 只调用文本奖励函数
            rewards = reward_func(generated_texts=generated_texts_list, metadatas=reward_metadatas)
            total_rewards += torch.tensor(rewards, device=self.accelerator.device)
            
        # 计算优势 (Advantage)
        rewards_grouped = total_rewards.view(batch_size, self.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - mean_rewards) / (std_rewards + 1e-8)).flatten().detach()

        # ==========================================================
        #  阶段 3: 计算 GRPO 损失 (有梯度)
        # ==========================================================
        torch.set_grad_enabled(True)
        model.train()

        # 准备模型 forward pass 的输入
        # 拼接 prompt 和 completion
        prompt_ids_all = torch.stack(prompt_ids_list)
        completion_ids_all = torch.cat(completion_ids_list, dim=0)
        full_ids = torch.cat([prompt_ids_all, completion_ids_all], dim=1)
        full_attention_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
        prompt_len_for_mask = prompt_ids_all.shape[1]

        # 计算策略模型的 log_probs
        log_probs_policy = self.get_text_per_token_logps(model, full_ids, full_attention_mask)
        # 只保留 completion 部分的 log_probs
        log_probs_policy = log_probs_policy[:, prompt_len_for_mask-1:]

        # 计算参考模型的 log_probs
        with torch.no_grad():
            if self.ref_model is not None:
                log_probs_ref = self.get_text_per_token_logps(self.ref_model, full_ids, full_attention_mask)
                log_probs_ref = log_probs_ref[:, prompt_len_for_mask-1:]
            else:
                log_probs_ref = torch.zeros_like(log_probs_policy)
        
        # 创建 completion 部分的掩码
        completion_mask = (completion_ids_all != self.processor.tokenizer.pad_token_id).long()

        # 核心 GRPO 损失计算
        pg_loss_per_token = -advantages.unsqueeze(1) * log_probs_policy
        kl_div_per_token = log_probs_policy - log_probs_ref
        
        loss_per_token = pg_loss_per_token + self.beta * kl_div_per_token
        
        # 应用掩码并计算平均损失
        loss = (loss_per_token * completion_mask).sum() / completion_mask.sum()
        
        # 日志记录 (简化版)
        self.log_metrics(total_rewards, advantages, pg_loss_per_token, kl_div_per_token, completion_mask)
        
        return loss
    # def log_metrics(self, rewards, advantages, pg_loss, kl_div):
    #     """辅助函数，用于聚合和记录指标。"""
    #     if not self.is_world_process_zero():
    #         return
            
    #     metrics = {
    #         "reward/mean": rewards.mean().item(),
    #         "reward/std": rewards.std().item(),
    #         "policy/advantage_mean": advantages.mean().item(),
    #         "loss/pg_loss": pg_loss.mean().item(),
    #         "loss/kl_div": kl_div.mean().item(),
    #     }
    #     self.log(metrics)
    def log_metrics(self, rewards, advantages, pg_loss, kl_div, mask):
        """辅助函数，用于聚合和记录指标。"""
        if not self.is_world_process_zero():
            return
            
        metrics = {
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std().item(),
            "policy/advantage_mean": advantages.mean().item(),
            "loss/pg_loss": (pg_loss * mask).sum().item() / mask.sum().item(),
            "loss/kl_div": (kl_div * mask).sum().item() / mask.sum().item(),
        }
        self.log(metrics)
    # 覆盖 _prepare_inputs 以避免 Trainer 默认的数据到设备操作，因为我们在 compute_loss 中手动处理
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        return inputs