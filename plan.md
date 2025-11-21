# Orthus 优化路线图

## 代码结构速览
- `models/`：核心模型定义与生成逻辑，包含 `modeling_orthus.py`、`modeling_orthus_for_inteleave_cfg.py`、`orthus_generation_mixin.py`、`processing_orthus.py` 等文件。
- `train/`：各类训练脚本与数据处理工具，`train_orthus.py`、`sft_orthus.py`、`grpo/grpo_trainer_orthus.py` 是训练主干。
- `inference/`：推理与演示脚本，主流程集中在 `interleave_generation_rotation.py` 及其衍生版本。
- `tests/`：当前仅有 `tests/test_orthus_generation_mixin.py` 覆盖生成混入逻辑，缺乏对数据管线与训练脚本的测试。
- `transformers/`：项目依赖的定制 Transformers 版本。
- 其余目录用于评估、分析、模型权重与实验资产。

-## 训练/推理链路现状
-**SFT 训练（train/interleave_sft_orthus.py）**：基于 HuggingFace `Trainer` 的自定义实现，`InterleaveSFTDataset` 在每个样本上重新编码问题图与步骤图（`__getitem__` 中逐图使用 `processor(..., vqmodel)`），并构建复杂的畸变权重、目标 latent、标签遮罩。梯度检查点需在模型内部另行开启，多处调试日志写入频繁。
-**SFT 数据集细节（train/interleave_sft_orthus.py:31-330）**：支持 `distortion_weight` 与 `return_analysis`，会针对每张目标图计算畸变掩码并生成 `distortion_indices`、`distortion_weights`、`distortion_lens` 等张量，但缺乏缓存机制。标签 mask 依据 prompt 长度、图片 token 位置手动裁剪，未使用 `image_cache_key`，同时频繁创建零填充张量。
-**GRPO 训练（train/grpo/grpo_trainer_orthus.py:50-352）**：自定义 `Trainer.compute_loss` 顺序处理 batch，每个样本单独执行 `processor` 编码与 `generate`；参考模型默认同构常驻 GPU，`processor.enable_latents_cache()` 需额外配置；loss 聚合基于 token 级 logits。
-**推理脚本（inference/interleave_generation_rotation.py:43-202）**：针对旋转任务逐样本处理，仅加载问题图，支持干预策略但当前注释；`model.generate` 仅调用 CFG 主分支且 `max_new_tokens` 固定 4096；输出解析与 `decode_image_latents_processed` 逐张进行，无批量/缓存/错误恢复。
- **显存基线**：训练脚本默认 ZeRO stage-2、`per_device_train_batch_size=1`、`gradient_accumulation_steps=2`、`generation_batch_size=16`；参考模型（`beta>0`）同构常驻 GPU；模型未启用层级梯度检查点；冻结的 VQ-VAE 仍常驻显存。

## 优化阶段规划
### 阶段 0：基线度量与工具
- **步骤 0.1 基线跑通与显存/性能记录**
  - 目标：建立训练与推理基准日志，记录显存峰值、迭代耗时，为后续优化提供对照。
  - 依赖：`train/grpo/run_grpo_orthus.sh`、`train/interleave_sft_orthus.py`、`inference/interleave_generation_rotation.py`；需要可用的 FlashAttention 环境。
  - 测试：运行一次 SFT 训练单 step、一次 GRPO 小批次，以及一次推理流程，记录指标。
  - 注意事项：固定随机种子，保证与后续优化使用相同模型权重与配置。

### 阶段 1：训练数据与前处理提效
- **步骤 1.1 VQ-VAE latent 缓存与离线预计算**
  - 目标：避免 `train/sft_orthus.py:88-103` 每次重算 image latent，支持 Processor LRU 或磁盘缓存。
  - 依赖：`models/processing_orthus.py` 的缓存能力；训练数据读写权限。
  - 测试：构建 DataLoader 运行两个 epoch，确认第二次 epoch 几乎无新增 VQ 编码调用，并比较训练 step 时间。
  - 注意事项：缓存 dtype/device 需与模型一致；处理缓存失效与图像缺失的回退逻辑。
- **步骤 1.2 DataLoader 并行与日志精简**
  - 目标：启用 `num_workers`、`pin_memory` 等选项，移除 `train/sft_orthus.py:154-165` 的频繁打印，降低 I/O 阻塞。
  - 依赖：`train/interleave_sft_orthus.py` 中的 `TrainingArguments` 与自定义 Trainer 配置。
  - 测试：SFT 训练跑单 step，对比优化前后数据准备耗时；确保可通过调试开关恢复详细日志。
  - 注意事项：多进程加载下需处理 PIL 线程安全及缓存共享。
- **步骤 1.3 文本截断与标签策略优化**
  - 目标：集中封装 prompt 模板（可基于 tokenizer 的 `apply_chat_template`），改进 `InterleaveSFTDataset` 中的标签遮罩逻辑，利用张量索引统一屏蔽 `<image>` token 和 prompt 部分，替换手写切片。
  - 依赖：`InterleaveSFTDataset`、tokenizer 行为。
  - 测试：构建包含多种回答长度的单元测试；运行 `tests/test_orthus_generation_mixin.py` 确认生成逻辑不受影响。
  - 注意事项：保持 `<think>` / `<answer>` 标签约定；兼容未来多图片样本。
- **步骤 1.4 畸变权重预处理与矢量化**
  - 目标：将 `InterleaveSFTDataset.__getitem__` 中逐样本计算畸变掩码与权重的流程迁移到数据预处理阶段，或通过 NumPy/矢量化手段批量生成 `distortion_indices`、`distortion_weights`、`distortion_lens` 张量，减少频繁 pad/stack。
  - 依赖：预处理脚本（可新增）或数据加载初始化；需要可靠的缓存命名方案。
  - 测试：对比优化前后 DataLoader 吞吐与 CPU 占用；验证训练过程读取的畸变权重与旧实现一致。
  - 注意事项：缓存文件需包含版本号与任务标识；设计缺失或损坏文件的回退策略。

### 阶段 2：训练循环与显存优化
- **步骤 2.1 模型层梯度检查点**
  - 目标：在 `models/modeling_orthus.py` 与 `modeling_orthus_for_inteleave_cfg.py` 的 Transformer block 上增加梯度检查点开关。
  - 依赖：PyTorch checkpoint API；前向逻辑需区分训练/推理。
  - 测试：运行 `bash train/grpo/run_grpo_orthus.sh` 完成至少一个训练 step，记录显存下降幅度。
  - 注意事项：仅在 `use_cache=False` 时启用；验证与 FlashAttention 版本兼容。
- **步骤 2.2 冻结模块低精与 CPU 驻留**
  - 目标：将已冻结的 VQ-VAE 模块转换为 float16/bfloat16，并提供 CPU 驻留选项。
  - 依赖：`model.model.vqmodel` 在 SFT、GRPO、推理脚本中的调用点。
  - 测试：训练与推理各执行一次，确认图像编码/解码结果保持一致。
  - 注意事项：处理 CPU/GPU 之间的非阻塞拷贝；确保缓存命中时 dtype 不丢失。
- **步骤 2.3 参考模型轻量化路径**
  - 目标：当 `beta>0` 时提供 LoRA-off 或 8-bit 参考模型加载选项，降低显存占用。
  - 依赖：`trl.create_reference_model`、bitsandbytes；`run_grpo_orthus.sh` 配置。
  - 测试：设置 `BETA=0.05` 运行 GRPO，监控 loss 与 reward 变化。
  - 注意事项：保证 KL 计算仍在 GPU 上完成，梯度正确传播。
- **步骤 2.4 训练循环内显存回收**
  - 目标：在 `OrthusGRPOTrainer.compute_loss` 中及时释放或转移大张量，并记录显存指标。
  - 依赖：PyTorch CUDA API；现有 `_log_debug`、`_record_grpo_metrics`。
  - 测试：重复相同 batch，比较显存峰值；在 `CUDA_LAUNCH_BLOCKING=1` 下确认无引用残留。
  - 注意事项：避免破坏梯度图；兼容 Accelerator 包装。
- **步骤 2.5 优化器与参数管理**
  - 目标：训练脚本增加 `paged_adamw_8bit`、CPU offload 开关并做好依赖检测与回退。
  - 依赖：bitsandbytes、Accelerate/DeepSpeed 配置。
  - 测试：分别在 8bit 与 CPU offload 模式运行若干 step，验证稳定性。
  - 注意事项：网络受限环境需提前准备 wheel；提供失败时的回退路径。
- **步骤 2.6 Trainer 调试与监控开关**
  - 目标：为 `InterleaveSFTTrainer` 引入配置化调试/监控机制（如自定义 `TrainerCallback` 输出样例、`torch.cuda.max_memory_allocated` 显存观测），将 `compute_loss` 中的打印与文件写入移动至可控钩子。
  - 依赖：`transformers.TrainerCallback`、现有日志目录。
  - 测试：在开启/关闭调试模式下分别运行一个 epoch，确认日志正常写入且训练稳定；检查显存指标是否符合预期。
  - 注意事项：默认关闭重型调试逻辑；在分布式环境中避免重复输出。

### 阶段 3：推理链路优化
- **步骤 3.1 批量推理与流水线化**
  - 目标：改造 `inference/interleave_generation_rotation.py`，支持批量处理与流水线调度，减少 Python 循环与重复开销。
  - 依赖：模型显存、`model.generate` 接口；可能需要 `torch.cuda.Stream`。
  - 测试：相同输入集下比较吞吐与延迟，确保输出顺序一致。
  - 注意事项：控制 batch 大小以适配显存；保留 CFG 生成路径。
- **步骤 3.2 推理 latent 缓存与预取**
  - 目标：在推理阶段复用 `models/processing_orthus.py` 缓存能力或预编码数据集，避免重复 VQ-VAE 前向。
  - 依赖：缓存存储介质；推理脚本的输入格式。
  - 测试：连续运行两次推理，确认第二次命中缓存；对比生成图像一致性。
  - 注意事项：缓存键需包含图像变换参数；考虑多进程共享。
- **步骤 3.3 生成配置与停止条件细化**
  - 目标：将 `max_new_tokens`、停止 token、`return_dict_in_generate` 参数化，减少手动解析与无效生成。
  - 依赖：`OrthusGenerationMixin`、`GenerationConfig`。
  - 测试：扩充 `tests/test_orthus_generation_mixin.py` 覆盖文本/图像混合边界；实际推理验证输出。
  - 注意事项：兼容现有脚本；输出需包含调试元数据。
- **步骤 3.4 量化与部署形态评估**
  - 目标：评估推理时的 8bit/4bit 权重加载与 KV cache 压缩方案，形成部署建议。
  - 依赖：bitsandbytes 或 相关量化库；部署硬件限制。
  - 测试：在量化权重下运行 QA 用例，比对生成质量与吞吐。
  - 注意事项：提供回退方案；关注多 GPU/CPU 推理适配。

### 阶段 4：质量保障与监控
- **步骤 4.1 测试集扩充与自动化**
  - 目标：为数据集构建、推理脚本、训练管线补充单元/集成测试，引入 CI。
  - 依赖：`pytest`、示例数据；现有测试目录结构。
  - 测试：新增测试纳入 CI；现有 `tests/test_orthus_generation_mixin.py` 保持通过。
  - 注意事项：测试数据需脱敏；控制运行时长。
- **步骤 4.2 性能/显存仪表板**
  - 目标：统一记录训练与推理的显存、时延、奖励指标，输出到日志或 JSON。
  - 依赖：`wandb`、`logging` 或自定义监控模块；日志目录权限。
  - 测试：执行一次训练+推理，确认产出指标文件并可视化。
  - 注意事项：避免多进程写冲突；与缓存策略协同。

## TODO 清单
- [ ] 0.1 基线跑通与显存/性能记录
- [x] 1.1 VQ-VAE latent 缓存与离线预计算
- [x] 1.2 DataLoader 并行与日志精简
- [x] 1.3 文本截断与标签策略优化
- [x] 1.4 畸变权重预处理与矢量化
- [ ] 2.1 模型层梯度检查点
- [ ] 2.2 冻结模块低精与 CPU 驻留
- [ ] 2.3 参考模型轻量化路径
- [ ] 2.4 训练循环内显存回收
- [ ] 2.5 优化器与参数管理
- [ ] 2.6 Trainer 调试与监控开关
- [ ] 3.1 批量推理与流水线化
- [ ] 3.2 推理 latent 缓存与预取
- [ ] 3.3 生成配置与停止条件细化
- [ ] 3.4 量化与部署形态评估
- [ ] 4.1 测试集扩充与自动化
- [ ] 4.2 性能/显存仪表板

---

## Orthus × Verl 集成计划（需逐步确认执行）

> 目标：使 Orthus 模型能够在 Verl 框架下执行 GRPO / GSPO 训练。每一步在实施前需获得确认。
> 代码实现须集中在 `train/verl/` 目录下，所有新增脚本、配置与适配层均放置于此。

### 当前阻塞问题汇总
- **FSDP 显存溢出**：按现有配置加载 Orthus 7B 全参时，FSDP 展平参数一次性申请约 26GiB 显存，单卡无法容纳，运行 `run_orthus_grpo.py` 会在初始化阶段 OOM。待后续改用 LoRA 或手工 wrap 策略降内存。
- **CPU 路径需独立策略**：将 `n_gpus_per_node` 设为 0 时，Verl 校验逻辑仍按 GPU 公式计算 `ppo_micro_batch_size_per_gpu`，导致 0 除错误；若要用 CPU 验证，需要单独添加 CPU actor 配置或跳过相关校验。
- **Ray 在无 GPU 模式初始化失败**：禁用 GPU 后，Ray 尝试初始化本地节点时因权限限制无法创建 socket，未能落地 multi-process CPU 模式，后续需配置容器权限或改用单进程脚本验证；2025-02-15 在 orthus 环境下尝试 `run_orthus_grpo.py`（LoRA 配置、1 卡调试）同样触发 `PermissionError: [Errno 1] Operation not permitted`，说明当前容器仍无法开放 UDP socket。
- **路径修复**：`run_orthus_grpo.py` 现将仓库根目录定位为 `twgi`（`parents[3]`），解决 tokenizer 加载阶段把 `/data1/oujingfeng/project/twgi/Orthus/...` 作为 HuggingFace repo id 导致的 `HFValidationError`。

### 最新排查记录
- **2025-11-03 VQ 模型缺失**：`OrthusRLDataset` 在多进程加载时未能通过 `AutoModelForCausalLM` 找到 `ChameleonConfig`，导致 processor 报错 "`vqmodel` must be provided"。已改为直接解析 checkpoint 的 `vq_config` 并仅加载 `model.vqmodel.*` 对应的 safetensors 分片（`train/verl/dataset.py`），确保 `vqmodel` 常驻 CPU、参数冻结，避免再次打印词典或回退零 latent。
- **2025-11-03 Rollout 卡死排查**：为确认权重同步与采样调用，`OrthusRollout` 新增了前 3 次 `update_weights`/`generate_sequences` 的诊断打印；同时收紧 Verl FSDP 初始化日志，避免整张词表写入导致日志刷屏（`verl/workers/fsdp_workers.py`），以便后续分析真正阻塞点。

### 阶段 0：现状评估与接口梳理
1. **步骤 0.1：接口差异清点**  
   - 目标：列出 Orthus 模型与 HuggingFace 标准接口的差异（如 `generate` 返回混合张量、载入 VQ-VAE 等）。  
   - 依赖：`models/modeling_orthus_for_inteleave_cfg.py`、`inference/interleave_generation_rotation.py`、Verl `HFRollout` 代码。  
   - 注意事项：输出混合列表、`train_mode` 额外参数、`processor` 对 VQ-VAE 的依赖都需记录。

2. **步骤 0.2：Verl 侧扩展点确认**  
   - 目标：明确 Verl 中需要扩展的模块（数据集、rollout、actor、Hydra 配置）。  
   - 依赖：`verl/utils/dataset/rl_dataset.py`、`verl/workers/rollout/hf_rollout.py`、`verl/workers/actor/dp_actor.py`、`verl/trainer/ppo/core_algos.py`。  
   - 注意事项：确认是否需要注册新的 model loader / processor，避免破坏现有模型注册逻辑。

### 阶段 1：数据与处理器接入
3. **步骤 1.1：自定义数据适配器**  
   - 目标：实现 Orthus 专用 `RLHFDataset` 派生类或前置转换脚本，确保样本包含 `input_ids`、`attention_mask`、`position_ids`、`responses` 及 `multi_modal_inputs`。  
   - 依赖：Orthus `InterleaveSFTDataset` 处理逻辑、Verl `DataProto` 结构。  
   - 注意事项：加载图片时需将 `vqmodel` 引入 Processor；考虑 latent 缓存（磁盘/内存）与缓存一致性。

4. **步骤 1.2：VQ-VAE 缓存与预处理**  
   - 目标：提供可在 Verl 数据加载阶段生成/缓存 image latents 的机制，减少在线编码。  
   - 依赖：`models/processing_orthus.py` 缓存 API。  
   - 注意事项：缓存精度（bf16/fp16）需与训练 dtype 一致；需要处理缺失图像的兜底逻辑。

### 阶段 2：Rollout 与生成流程
5. **步骤 2.1：Orthus Rollout Worker**  
   - 目标：基于 `verl/workers/rollout/hf_rollout.py` 实现新 worker，支持 `multimodal_generation_mode_list` 并拆解文本/latent。  
   - 依赖：`inference/interleave_generation_rotation.py` 中的输出解析流程。  
   - 注意事项：需维护 CFG 参数、保证批处理能力，生成后清理 GPU 缓存。

6. **步骤 2.2：生成结果适配**  
   - 目标：定义生成结果到 Verl `DataProto` 的映射（包括文本 completion 与必要的调试信息），确保后续 log-prob 计算只依赖文本 token。  
   - 依赖：步骤 2.1 中的 rollout worker、`verl/workers/actor/dp_actor.py` 的输入格式。  
   - 注意事项：保留图像 latent 仅用于分析，不参与优势计算；确认 `response_mask` 构造与 Orthus 标签约定一致。

### 阶段 3：训练循环接入
7. **步骤 3.1：Actor 端前向适配**  
   - 目标：在计算 log-prob 时调用 Orthus 模型的 `train_mode="mmu"` 分支，确保只涉及文本 logits。  
   - 依赖：`train/grpo/grpo_trainer_orthus.py` 现有实现、`verl/workers/actor/dp_actor.py`。  
   - 注意事项：处理 `multi_modal_inputs` 的透传，确保梯度检查点配置兼容；数据集中会自动加载 `vqmodel`（优先使用 `ORTHUS_VQMODEL_PATH`，否则退回 `data.model_path`），因此需保证本地 checkpoint 包含 VQ-VAE 模块。
   - 辅助工具：`train/verl/launch_grpo.sh` 提供 `test` / `train` 双模式入口；两者默认均使用 8 张 GPU 与相同配置，`test` 仅通过 `trainer.total_epochs` / `trainer.total_training_steps` 缩短运行时长（默认 1 epoch / 10 steps，可通过环境变量覆盖）。脚本默认注入 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`、`actor_rollout_ref.actor.ppo_mini_batch_size=ORTHUS_NUM_GPUS`，并同步设置 `actor_rollout_ref.rollout.log_prob_micro_batch_size=ppo_micro_batch_size_per_gpu`（可由 `ORTHUS_PPO_MICRO_BATCH_SIZE_PER_GPU` / `ORTHUS_PPO_MINI_BATCH_SIZE` / `ORTHUS_ROLLOUT_LOGPROB_BATCH` 覆写）。`run_orthus_grpo.py` 默认将训练/验证数据文件指向 `datasets/mydatasets/dataset/data.json`（可用 `ORTHUS_RL_DATA` / `ORTHUS_RL_VAL_DATA` 覆写），采样数可由 `ORTHUS_RL_TRAIN_MAX_SAMPLES` / `ORTHUS_RL_VAL_MAX_SAMPLES` 控制，并透传 `PYTHONPATH`/`VERL_SUPPRESS_CONFIG_LOG` 到 Ray runtime；演员与参考模型 FSDP dtype 均设置为 bfloat16。

8. **步骤 3.2：Hydra 配置集成**  
   - 目标：新增 Verl 配置（或 recipe）以载入 Orthus checkpoint、processor、数据源与自定义 worker。  
   - 依赖：Verl `examples/grpo_trainer`、`recipe/gspo` 的示例配置。  
   - 注意事项：配置需支持切换 GRPO/GSPO；标注必要的环境变量与资源需求。

9. **步骤 3.3：GRPO 小规模验证**  
   - 目标：以小样本运行若干 step，验证奖励计算、优势归一化、梯度更新是否正常。  
   - 依赖：步骤 1-3 所有成果。  
   - 注意事项：对齐 `reward_funcs`（正确率、格式），观察损失、KL、奖励曲线是否合理。

10. **步骤 3.4：GSPO 扩展验证**  
    - 目标：在 GRPO 成功基础上，将 policy loss 切换为 `gspo`，调优 `clip_ratio`、`loss_agg_mode`。  
    - 依赖：Verl GSPO 实现（`core_algos.py`），步骤 3.3 的配置。  
    - 注意事项：密切监控训练稳定性，必要时调整梯度裁剪和学习率。

### 阶段 4：完善与交付
11. **步骤 4.1：性能与显存基准记录**  
    - 目标：记录 Orthus 在 Verl 下的训练耗时、显存、吞吐，与原生 GRPO 脚本对比。  
    - 依赖：阶段 3 的可运行管线。  
    - 注意事项：使用统一硬件环境、固定随机种子。

12. **步骤 4.2：文档与测试补充**  
    - 目标：更新 README/内部文档，增加数据适配、rollout、配置示例；补充关键单元/集成测试。  
    - 依赖：所有前续步骤完成输出。  
    - 注意事项：文档中需明确依赖项（FlashAttention、DeepSpeed 等）和使用限制。

### 阶段 5：图像结构语义对齐
13. **步骤 5.1：结构特征离线准备**  
    - 目标：针对 `/data1/oujingfeng/project/twgi/datasets/mydatasets/dataset/data_modified.json` 中的 `Rotation_steps[*].image` 生成与 latent 网格 (32×32) 对齐的结构标签。先用脚本遍历 `task/level/image_id/steps/*.png`，利用 OpenCV/PyTorch 算子提取：  
        1) 边缘/线条（Sobel/Canny→Hough，或直接对梯度幅值做多阈值编码）；  
        2) 角点/特征点（Harris/LoG）。  
      输出统一保存为 `.pt`（包含 edge_map、line_map、corner_map，float32），并记录 metadata 供 Dataset 映射。  
    - 依赖：`train/precompute_latents.py` 的遍历逻辑，可复用其 cache key 与路径解析。  
    - 注意事项：保证算子输出可被下游重用（尺寸与 latent patch 对齐，或在保存前做双线性缩放），并在脚本中支持 `--overwrite`、断点续算。

14. **步骤 5.2：数据/模型管线扩展**  
    - 目标：在 `InterleaveSFTDataset` 中加载上述结构标签，并与 `target_image_latents` 对齐（同样数量的 step）。返回字段新增 `structure_edge_targets` / `structure_corner_targets`。  
    - 模型端：在 `OrthusForConditionalGeneration` 内新增可选的结构投影辅助头，方法：  
        - 将 diffusion head 的输出 latent 还原为 `(B, num_imgs, 32, 32, C)`；  
        - 通过固定卷积核（Sobel、二阶差分）实现可微的 edge/point 映射；  
        - 或在 forward 中返回 `pred_latents`，由 Trainer 使用相同算子在 PyTorch 中生成结构预测。  
    - 注意事项：保持算子可微；需要对 `pred_latents` 与 ground-truth latent 做 dtype/device 对齐，并允许关闭该特性（CLI `--structure_loss_weight 0`）。

15. **步骤 5.3：Loss 嵌入与日志**  
    - 目标：在 `InterleaveSFTTrainer.compute_loss` 中，当 `pred_latents`/`true_latents` 可用时，计算：  
      `L_edge = ||Edge(pred) - Edge(true)||₁`，`L_corner = BCE(Sigmoid(Corner(pred)), Corner(true))`，再以新超参 γ_edge、γ_corner 融入总损失：`loss = α·text + β·diff + γ_edge·L_edge + γ_corner·L_corner`。  
    - 要求：  
        - CLI (`train_interleave_orthus.py`) 暴露权重与开关（如 `--enable_structure_loss`）；  
        - 训练日志/评估输出单独记录每个结构项，便于调参。  
    - 注意事项：在 `bf16` 下计算结构 loss 需谨慎，可在算子后转 fp32；确保多卡情况下 `structure_targets` 不引入额外通信。

16. **步骤 5.4：验证与推理对齐**  
    - 目标：新增评估脚本，对训练后权重在验证集上重新生成 step 图像，对比结构指标（edge IoU、corner Precision）。  
    - 推理侧（`inference/interleave_generation_rotation.py`）添加可选开关：若传入 `--dump_structure_metrics`，对生成的每张图片运行同样算子并输出指标，辅助观测语义一致性。  
    - 注意事项：保证离线/在线算子一致；指标计算尽量放在 CPU 以免占用训练 GPU 资源。

### Orthus × Verl TODO 列表
- [x] 0.1 梳理 Orthus 与 HF 接口差异
- [x] 0.2 明确 Verl 扩展点
- [x] 1.1 完成 Orthus 数据适配器
- [x] 1.2 实现 VQ-VAE latent 缓存策略
- [x] 2.1 开发 Orthus 专用 rollout worker
- [x] 2.2 适配生成结果到 DataProto
- [x] 3.1 调整 actor 端 log-prob 前向
- [x] 3.2 集成 Hydra/recipe 配置
- [ ] 3.3 验证 Orthus GRPO 训练
- [ ] 3.4 验证 Orthus GSPO 训练
- [ ] 4.1 输出性能/显存基准
- [ ] 4.2 补充文档与测试

### 最新修复记录（2025-11-03）
- [x] 修复 GRPO LoRA `target_modules` 命令行仅保留最后一项导致 `o_proj` 未匹配的问题：统一在 `run_grpo_orthus.sh` 传递逗号分隔字符串，并在 `grpo_orthus.py` 端解析为列表。

### 当前待解决问题
- GRPO 训练阶段推理与策略前向仍在同一进程执行，生成输出（`completion_ids`、`full_sequences` 等）长期保留在 GPU，显存峰值无法下降；需参考 Verl 架构，将生成阶段输出及时迁移到 CPU 或拆到独立 worker。
