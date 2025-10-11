import torch
import os
import sys
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import json
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import traceback
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. 将项目根目录临时添加到Python解释器的“模块搜索路径”列表中
sys.path.append(root_path)

# 3. 现在可以从根目录开始，成功地导入任何模块了
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

# --- 数据集初始化代码 ---
class InterleaveSFTDataset(Dataset):
    """
    为 Orthus 的图文交错模式进行SFT微调的数据集。
    这个版本只为文本logit loss准备标签。
    """
    def __init__(self, dataset, image_base_dir, processor, vqmodel, max_length=2048):


        """
        :param data_file: train.jsonl 或 test.jsonl 文件的路径。
        :param image_base_dir: SpatialViz 数据集的根目录。
        :param processor: OrthusProcessor 实例。
        :param max_length: 序列最大长度。
        """
        self.data = dataset # <-- 直接使用传入的 dataset 对象
        self.image_base_dir = image_base_dir
        self.processor = processor
        self.vqmodel = vqmodel
        self.max_length = max_length
        print(f"Initialized dataset with {len(self.data)} examples.")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 1. 根据您的逻辑构建图片路径 ---
        category = item.get('Category', '')
        task = item.get('Task', '')
        level = item.get('Level', '')
        image_id = item.get('Image_id', '')
        image_path = os.path.join(self.image_base_dir, category, task, level, f"{image_id}.png")
        # print(f"Loading image from: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image at {image_path} not found. Using a blank image.")
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        # --- 2. 构建模型的输入和期望的输出 ---
        instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        )
        question = item.get('Question', '')
        # --- 【核心修改】直接获取字符串类型的 Explanation ---
        # 因为我们已经确认所有样本的 Explanation 都是字符串，所以不再需要isinstance判断
        explanation_text = item.get('Explanation', '')
        answer = item.get('Answer', '')
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(item.get('Choices', []))])

        prompt_text = instruction + f"<image>\n\nQuestion: {question}\n{choices_text}\n\nAnswer: "
        
        # 根据您的要求，label是纯文本，我们教模型先输出解释，再输出答案
        part_before_answer = f"<think>{explanation_text}</think> The final answer is <answer>"
        answer_text = answer
        part_after_answer = f"</answer></s>"
        
        full_text = prompt_text + part_before_answer + answer_text + part_after_answer


        # --- 3. 预处理 & Tokenize ---
        # 【修改点 2】: 在调用 processor 时传入 self.vqmodel
        model_inputs = self.processor(
            text=full_text, 
            images=[image],
            vqmodel=self.vqmodel,  # <-- 将保存的 vqmodel 传入
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        # print(f"Input IDs: {model_inputs['input_ids']}")

                # 【关键补充】: 创建 target_image_latents 以满足 'interleave' 模式的需要
        # 我们使用输入图像的latents作为“伪”目标
        if 'image_latents' in model_inputs:
            model_inputs["target_image_latents"] = model_inputs['image_latents'].clone()
        else:
            # 处理特殊情况，如果processor没有返回latents，则创建一个空的
            model_inputs["target_image_latents"] = torch.zeros((1, 1024, 256))
        # # print(f"Target image latents shape: {model_inputs['target_image_latents'].shape}")
        # # --- 4. 【核心修改】创建健壮的文本 Labels ---
        # labels = model_inputs['input_ids'].clone()
        # # print(f"Original labels: {len(labels[0])}")
        # # a. 获取 answer 的真实长度 (不含特殊token)
        # target_len_no_special = len(self.processor.tokenizer(target_text2, add_special_tokens=False).input_ids)

        # # d. 执行屏蔽
        # #    labels[0, :mask_len] 正是屏蔽了从0开始到prompt结束的所有部分
        # #    对于左填充来说，这部分恰好就是 [PAD, PAD, ..., PROMPT]
        # labels[0, :-(target_len_no_special+1)] = -100

        # # print(f"Masked labels: {labels}")
        # model_inputs["labels"] = labels
        # ==================== 实现【只为 {answer} 计算 Loss】的正确逻辑 ====================
        # ==================== 【修正方案】使用從後往前的子序列搜索 ====================
        
        labels = torch.full_like(model_inputs['input_ids'], -100)

        answer_text = item.get('Answer', '')
        answer_tokens = self.processor.tokenizer(answer_text, add_special_tokens=False).input_ids

        full_input_ids_list = model_inputs['input_ids'][0].tolist()
        
        found = False
        # 【核心修改】使用 reversed() 從後往前搜索，確保找到的是最後一個匹配項
        for i in reversed(range(len(full_input_ids_list) - len(answer_tokens) + 1)):
            if full_input_ids_list[i : i + len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                answer_end_pos = i + len(answer_tokens)
                
                labels[0, answer_start_pos:answer_end_pos] = model_inputs['input_ids'][0, answer_start_pos:answer_end_pos]
                found = True
                break

        if not found and idx < 5:
             print(f"--- [Sample {idx} WARNING] ---")
             print(f"Could not find answer subsequence '{answer_text}' (tokens: {answer_tokens}) in the input.")
             print("--------------------------")

        model_inputs["labels"] = labels
        # ========================================================================

        # ========================================================================

        # ==================== 【新增】验证逻辑：解码label并打印 ====================
        # 只对前5个样本进行打印，避免刷屏
        if idx < 5:
            # 创建一个副本用于解码
            labels_for_verification = labels.clone()
            # 将 -100 替换为 pad_token_id，以便 tokenizer 解码
            labels_for_verification[labels_for_verification == -100] = self.processor.tokenizer.pad_token_id
            # 解码并打印
            decoded_labels = self.processor.tokenizer.decode(labels_for_verification[0], skip_special_tokens=True)
            print(f"--- [Sample {idx} Verification] ---")
            print(f"Ground Truth Answer: '{answer_text}'")
            print(f"Decoded Label for Loss: '{decoded_labels.strip()}'") # 使用 strip() 去除可能的多余空格
            print("--------------------------")
        # ========================================================================
  



        # 清理工作
        if "vqmodel" in model_inputs:
            model_inputs.pop("vqmodel")

        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.squeeze(0)

        return model_inputs
    

# --- 2. 自定义 Trainer ---
# --- 2. 自定义 Trainer (修正版) ---
# class InterleaveSFTTrainer(Trainer):
#     # 【核心修改】: 在参数列表中添加 **kwargs，以接收所有未预期的额外参数
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         """
#         重写 compute_loss 方法。
#         我们将在这里调用 train_mode='interleave'，并手动只计算文本部分的损失。
#         """
#         # 从输入中分离出 labels 和 target_image_latents
#         labels = inputs.pop("labels")
#         target_image_latents = inputs.pop("target_image_latents")
        
#         if model.training:
#             # 训练时，使用 'interleave' 模式
#             outputs_tuple = model(**inputs, target_image_latents=target_image_latents, train_mode='interleave')
#             logits, diff_loss = outputs_tuple
#             # (这里可以根据需要组合 loss)
#         else:
#             # 评估时，使用 'discrete' 模式来获取 logits
#             outputs = model(**inputs, mode='discrete')
#             logits = outputs.logits.float()
#         # 我们在这里【忽略】diff_loss，只计算文本的损失
        
#         # 手动计算交叉熵损失
#         loss_fct = CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        
#         # 为了与 Trainer 的其他功能（如日志记录）兼容，我们构建一个简单的输出对象
#         # class TempOutput:
#         #     def __init__(self, loss, logits):
#         #         self.loss = loss
#         #         self.logits = logits
        
#         # outputs = TempOutput(loss, logits)
        
#         # Trainer 传入的 num_items_in_batch 会被 **kwargs “吸收”掉，但我们用不到它
#         return (loss, outputs) if return_outputs else loss
# --- 2. 自定义 Trainer (修改版，增加调试功能) ---
class InterleaveSFTTrainer(Trainer):
    """
    自定义Trainer，增加了在每一步训练中打印标签和生成输出的调试功能。
    """
    # 【修改1】: 修改 __init__ 方法以接收并保存 processor
    def __init__(self, *args, processor=None, generation_log_file=None, enable_generation_log=False, **kwargs):
        super().__init__(*args, **kwargs)
        # 将 processor 保存为类的属性，以便在 compute_loss 中使用
        self.processor = processor
        self.generation_log_file = generation_log_file
        self.enable_generation_log = enable_generation_log

        # 如果开启日志，在训练开始前清空旧文件
        if self.enable_generation_log and self.is_world_process_zero():
            with open(self.generation_log_file, "w") as f:
                f.write("") # 清空文件
    # 【修改2】: 大幅重写 compute_loss 方法以添加详细的调试打印
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法。
        增加了在训练时打印标签、 logits预测 和 生成输出 的功能。
        """
        # 从输入中分离出 labels 和 target_image_latents
        labels = inputs.pop("labels")
        target_image_latents = inputs.pop("target_image_latents")

        # ==================== 调试代码块 (开始) ====================
        # 只在训练和评估时打印，不在 generate 时打印
        # model.training 是训练状态， not model.training 是评估状态
        should_debug = (self.is_world_process_zero() and
                        (self.state.global_step + 1) % 20 == 0 and
                        (model.training or not self.is_in_train))
        
        if should_debug:
            print(f"\n{'='*40} [DEBUGGING AT GLOBAL STEP: {self.state.global_step}] ({'TRAIN' if model.training else 'EVAL'}) {'='*40}")

            # --- 1. 检查 Labels 是否正确 ---
            print("\n--- [1. Ground Truth Labels] ---")
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
            
            # ✅ 【修正】: 循环处理批次中的每个样本
            for i, single_label in enumerate(labels_for_decode):
                decoded_label = self.processor.tokenizer.decode(single_label.tolist(), skip_special_tokens=True)
                print(f"  - Sample {i} Label: \033[92m{decoded_label}\033[0m")
        # ==========================================================


        # --- 模型前向传播 ---
        if model.training:
            # outputs_tuple = model(**inputs, target_image_latents=target_image_latents, train_mode='interleave')
            # logits, diff_loss = outputs_tuple
            outputs_tuple = model(**inputs, train_mode='mmu')
            logits = outputs_tuple
            model_outputs_for_return = outputs_tuple
        else:
            outputs = model(**inputs, mode='discrete')
            logits = outputs.logits.float()
            # model_outputs_for_return = (logits,)

        # --- 计算损失 ---
        shift_logits = logits[:, :-1, :].contiguous()  # 去掉最后一个位置
        shift_labels = labels[:, 1:].contiguous()      # 去掉第一个位置

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        model_outputs_for_return = loss
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        # print(loss)
        # ==================== 调试代码块 (继续) ====================
        if should_debug:
            # --- 2. 检查模型从 Logits 中的预测 ---
            print("\n--- [2. Model Prediction from Logits] ---")
            predicted_ids = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            predicted_ids_masked = torch.where(mask, predicted_ids, self.processor.tokenizer.pad_token_id)

            # ✅ 【修正】: 循环处理批次中的每个样本
            for i, single_pred in enumerate(predicted_ids_masked):
                decoded_pred = self.processor.tokenizer.decode(single_pred.tolist(), skip_special_tokens=True)
                print(f"  - Sample {i} Pred: \033[93m{decoded_pred}\033[0m")
            
            print(f"{'='*107}\n")
        # ==================== 调试代码块 (结束) ====================

            # --- 3. 在每一步解码一个完整的生成输出 ---

            
        # ==================== 实时生成与记录模块 ====================
        # 只在训练时，且日志开关已开启时执行
        # if model.training and self.enable_generation_log:
        #     try:
        #         print("\n--- [3. Live Generation Output] ---")
        #         # --- 准备生成任务的输入 ---
        #         prompt_end_idx = (labels[0] != -100).nonzero(as_tuple=True)[0][0]
        #         prompt_input_ids = inputs['input_ids'][:, :prompt_end_idx]
        #         prompt_attention_mask = inputs['attention_mask'][:, :prompt_end_idx]
                
        #         kwargs_con = {
        #             "input_ids": prompt_input_ids,
        #             "image_latents": inputs['image_latents'],
        #             "target_image_latents": target_image_latents,
        #             "cfg_scale": None, "interleave_output_format": False, "max_new_tokens": 512,
        #             "do_sample": False, "attention_mask": prompt_attention_mask, "use_cache": True,
        #         }

        #         # --- 执行生成 ---
        #         with torch.no_grad():
        #             generated_ids = self.model.generate(
        #                 multimodal_generation_mode_list=["text-only"],
        #                 kwargs_list=[kwargs_con],
        #             )
                
        #         # --- 解码并准备记录 ---
        #         prompt_len = prompt_input_ids.shape[1]
        #         # 解码生成结果
        #         decoded_generation = self.processor.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
                
        #         # 解码对应的真实标签以供对比
        #         labels_for_decode = labels[0].clone() # 只处理当前批次的第一个样本
        #         labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
        #         decoded_label = self.processor.tokenizer.decode(labels_for_decode.tolist(), skip_special_tokens=True)

        #         # ✅ 【核心修改】只在主进程 (rank 0) 写入文件
        #         if self.state.is_world_process_zero:
        #             log_data = {
        #                 "global_step": self.state.global_step,
        #                 "ground_truth": decoded_label,
        #                 "generated_output": decoded_generation
        #             }
        #             # 以追加模式写入JSONL文件
        #             with open(self.generation_log_file, "a", encoding="utf-8") as f:
        #                 f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        #     except Exception as e:
        #         # 打印错误，但不在每个进程中都打印
        #         if self.state.is_world_process_zero:
        #             print(f"\033[91mCould not generate and log output at step {self.state.global_step}: {e}\033[0m")
        #             traceback.print_exc()

        # ==================== 调试代码块 (结束) ====================

        # 确保返回值格式正确，以兼容Trainer的日志记录等功能
        return (loss, model_outputs_for_return) if return_outputs else loss

# # --- 3. 主训练逻辑 (与之前的sft_orthus.py基本相同) ---
# def main():
#     # ... (这部分代码与上一版的 sft_orthus.py 完全相同) ...
#     # ... (包括 argparse, 模型加载, 数据集加载等) ...
#     parser = argparse.ArgumentParser(description="Finetune Orthus model for VQA in Interleave Mode (Text Loss Only)")
#     parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrained model.")
#     parser.add_argument("--train_file", type=str, required=True, help="Path to the train.jsonl file.")
#     parser.add_argument("--test_file", type=str, required=True, help="Path to the test.jsonl file.")
#     parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the finetuned model.")
#     parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
#     parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device.")
#     parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps.")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    
#     args = parser.parse_args()

#     print(f"Loading model and processor from {args.ckpt_path}...")
#     processor = OrthusProcessor.from_pretrained(args.ckpt_path)
#     model = OrthusForConditionalGeneration.from_pretrained(
#         args.ckpt_path,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2"
#         # device_map="auto",
#     )
#     print("Model and processor loaded.")

#     print("Preparing datasets...")
#     train_dataset = InterleaveSFTDataset(
#         data_file=args.train_file,
#         image_base_dir=args.image_folder,
#         processor=processor,
#         vqmodel=model.model.vqmodel
#     )
#     eval_dataset = InterleaveSFTDataset(
#         data_file=args.test_file,
#         image_base_dir=args.image_folder,
#         processor=processor,
#         vqmodel=model.model.vqmodel
#     )
#     print("Datasets prepared.")

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         num_train_epochs=args.epochs,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum_steps,
#         learning_rate=args.learning_rate,
#         bf16=True, 
#         logging_steps=10,
#         save_total_limit=2,
#         save_strategy="epoch",
#         evaluation_strategy="epoch",
#         remove_unused_columns=False,
#         label_names=["labels"],
#         # attn_implementation="flash_attention_2",
#     )
    
#     def custom_data_collator(features):
#         # 移除 vqmodel, 因为它不是模型 forward 的直接输入
#         for feature in features:
#             if "vqmodel" in feature:
#                 feature.pop("vqmodel")
#         return torch.utils.data.dataloader.default_collate(features)

#     # 使用我们新的、为 interleave 模式定制的 Trainer
#     trainer = InterleaveSFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=custom_data_collator,
#     )

#     print("Starting training...")
#     trainer.train()
#     print("Training finished.")

#     print(f"Saving final model to {args.output_dir}...")
#     trainer.save_model(args.output_dir)
#     processor.save_pretrained(args.output_dir)
#     print("Model and processor saved successfully.")


# if __name__ == "__main__":
#     main()