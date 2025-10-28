# Orthus VRAM 优化路线图

## 基线现状
- 训练脚本 `train/grpo/run_grpo_orthus.sh` 当前默认以 ZeRO stage-2、`per_device_train_batch_size=1`、`gradient_accumulation_steps=2` 和 `generation_batch_size=16` 运行。
- 参考模型在 `beta>0` 时与主模型同构，默认会常驻 GPU；虽然近期改为 `beta=0`，但代码路径仍会在 `beta>0` 时加载完整副本。
- `OrthusForConditionalGeneration` 和 `OrthusGRPOTrainer` 未显式启用分层梯度检查点，长序列生成时需要保留完整激活图。
- VQ-VAE 图像分支已冻结，但仍随主体模型一起驻留在 GPU，处理图像时会分配额外激活。

## 优化阶段规划

1. **模型层梯度检查点**
   - 在 `OrthusModel` / `ChameleonDecoderLayer` 上增加配置开关，允许在训练时通过 `torch.utils.checkpoint.checkpoint` 对 Transformer block 启用梯度检查点。
   - 确保只在 `self.training` 且 `use_cache=False` 时启用，兼容现有生成流程。
   - 验证：运行 `bash train/grpo/run_grpo_orthus.sh` 确认为默认配置仍能完成一个训练 step。

2. **冻结模块低精与 CPU 驻留**
   - 将已冻结的 VQ-VAE 模块在加载后转换为 `torch.float16/bfloat16`，并提供选项将其参数常驻 CPU，在前向时按需搬运输入输出。
   - 处理 `OrthusProcessor` 与模型 forward，使其在 VQ-VAE 驻留 CPU 时保持功能正确。
   - 验证：同上脚本 + 额外调用推理路径，确保图像编码/解码无误。

3. **参考模型轻量化路径**
   - 当 `beta>0` 时，提供自动 LoRA-off 参考或 8-bit 权重加载选项；默认维持原逻辑但允许用户切换至低显存模式。
   - 需确保 KL 计算仍在 GPU 上进行且梯度正确传播。
   - 验证：设置 `BETA=0.05` 跑训练脚本，确认指标和 loss 更新正常。

4. **训练循环内的显存回收**
   - 在 `OrthusGRPOTrainer.compute_loss` 中，显式 `del` / `detach` 大体积张量，并在可能的地方将中间结果转移到 CPU，避免长期占用 GPU 缓冲。
   - 为 reward 评估与日志记录加入 `torch.cuda.reset_peak_memory_stats()` 辅助接口，便于监控。
   - 验证：默认脚本运行，通过日志和 `nvidia-smi` 观察峰值显存是否下降。

5. **优化器与参数管理**
   - 在训练脚本中增加对 `paged_adamw_8bit` / CPU offload 的安全开关，确保依赖存在时自动启用，以进一步压缩优化器状态。
   - 同时补充依赖检测与回退路径，避免因缺少 bitsandbytes 或缺乏权限导致运行失败。
   - 验证：分别在 8bit 与 CPU offload 模式跑训练脚本，确认能够稳定完成若干 step。

## 验证流程
- 每完成一项改动，运行 `bash train/grpo/run_grpo_orthus.sh`（必要时调整环境变量，例如恢复 `BETA=0.05`）确认训练能完成至少一个更新 step。
- 如某项修改影响推理，还需执行一次纯推理脚本（若无单独脚本，可用调试模式 `--debug` 跑一轮生成）确保输出正确。
- 记录显存基准，便于量化每项优化收益。

