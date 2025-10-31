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
