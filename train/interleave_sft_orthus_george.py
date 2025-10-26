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
import numpy as np
import random
# ==================== 新增的 imports ====================
import torchvision
# ========================================================

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IMAGE_TOKEN = "<image>"
# 2. 将项目根目录临时添加到Python解释器的“模块搜索路径”列表中
sys.path.append(root_path)

# 3. 现在可以从根目录开始，成功地导入任何模块了
from models.processing_orthus import OrthusProcessor
from models.modeling_orthus_for_inteleave_cfg import OrthusForConditionalGeneration

# --- 数据集初始化代码 ---
class InterleaveSFTGeorgeDataset(Dataset):
    """
    为 Orthus 的图文交错模式进行SFT微调的数据集。
    这个版本只为文本logit loss准备标签。
    """
    def __init__(self, dataset, image_base_dir, processor, vqmodel, distortion_weight=False, return_analysis=False, max_length=4096):


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
        # 【新增】為動態權重增加一個基數和一個 epsilon 防止除零
        self.base_distortion_weight = 10.0 # 可以設為可配置參數
        self.epsilon = 1e-6 # 一個極小值，防止除以零    
        self.distortion_weight = distortion_weight
        self.return_analysis = return_analysis




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 1. 根据预处理好的信息构建图片路径并加载 ---
        story_id = item.get('story_id')
        
        # 定义需要加载的图片文件名
        image_filenames = [
            item.get('prompt_image'),
            item.get('label_image_1'),
            item.get('label_image_2')
        ]
        
        images = []
        for filename in image_filenames:
            # 假设图片路径结构为: {image_base_dir}/{story_id}/{image_name}
            image_path = os.path.join(self.image_base_dir, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            except FileNotFoundError:
                print(f"Warning: Image at {image_path} not found. Returning a random sample instead.")
                # 如果有图片缺失，随机返回另一个样本以避免训练中断
                return self.__getitem__(random.randint(0, len(self) - 1))

        # --- 2. 构建模型的输入 (prompt) 和期望的输出 (label) ---
        prompt_text = item['prompt_text'] + DEFAULT_IMAGE_TOKEN + "Please continue this story:"
        
        label_text = (
            item['label_text_1'] + DEFAULT_IMAGE_TOKEN + 
            item['label_text_2'] + DEFAULT_IMAGE_TOKEN
        )


        prompt_image_num = prompt_text.count('<image>')
        # 根据您的要求，label是纯文本，我们教模型先输出解释，再输出答案

        full_text = prompt_text + label_text

        # --- 3. 预处理 & Tokenize ---
        # 将问题图片和所有步骤图片拼在一起传入
        all_images = images

        model_inputs = self.processor(
            text=full_text,
            images=all_images,
            vqmodel=self.vqmodel,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        # --- 4. 拆分 image_latents ---
        # image_latents: [num_images, 32, 32, 256]
        image_latents = model_inputs["image_latents"]
        # print(f"image_latents.shape: {image_latents.shape}")
        input_image_latents = image_latents[0:1]  # 问题图片
        target_image_latents = image_latents[1:]  # 步骤图片

            # 【新增】: 即時計算畸變 patch 索引
            # ==========================================================
        if self.distortion_weight == True: 
            white_image = Image.new('RGB', (224, 224), (255, 255, 255))
            with torch.no_grad():
                white_inputs = self.processor(text=[prompt_text], images=[white_image], return_tensors="pt", vqmodel=self.vqmodel)
            standard_blank_latents_mean = white_inputs['image_latents'].squeeze(0).view(1024,256).mean(dim=0)
            # 【核心修改】: 計算每張圖的動態權重並傳遞
            # ==========================================================
            max_num_images = target_image_latents.shape[0]
            distortion_info_list = [] # 儲存每張圖的 {'indices': ..., 'weight': ...}
            for i in range(target_image_latents.shape[0]): # 遍歷每一張目標圖
                latents = target_image_latents[i].view(1024,256).to("cpu", torch.float32)
                
                # --- 畸變分析 (邏輯不變) ---
                similarities = torch.nn.functional.cosine_similarity(latents, standard_blank_latents_mean, dim=1)
                is_blank_grid = (similarities > 0.99).view(32, 32)
                distortion_mask = torch.zeros_like(is_blank_grid, dtype=torch.bool)
                h_changes = is_blank_grid[:, :-1] != is_blank_grid[:, 1:]
                distortion_mask[:, :-1] |= h_changes
                distortion_mask[:, 1:] |= h_changes
                v_changes = is_blank_grid[:-1, :] != is_blank_grid[1:, :]
                distortion_mask[:-1, :] |= v_changes
                distortion_mask[1:, :] |= v_changes
                
                distortion_indices = distortion_mask.flatten().nonzero(as_tuple=True)[0]

                # --- 動態權重計算 ---
                distortion_ratio = len(distortion_indices) / 1024.0
                # 反比加權：佔比越小，權重越高。加上 epsilon 防止除以零。
                dynamic_weight = self.base_distortion_weight / (distortion_ratio + self.epsilon)
                # ==================== 【新增修复代码】 ====================
                #  对 distortion_indices 进行填充，使其长度统一
                #  设定一个足够大的最大长度，例如 1024 (所有patch都可能是边界的极端情况)
                max_indices_len = 1024 
                padded_indices = torch.nn.functional.pad(
                    distortion_indices, 
                    (0, max_indices_len - len(distortion_indices)), # (pad_left, pad_right)
                    mode='constant', 
                    value=-100  # 使用一个特殊的负值填充，方便后续在loss计算中忽略
                )
                # ==========================================================
                distortion_info_list.append({
                    # "indices": distortion_indices, # <-- 使用填充后的张量
                    "indices": padded_indices,
                    "weight": dynamic_weight,
                    # 【新增】同时传递原始的、未填充的长度，这在后续处理中非常有用
                    "original_len": len(distortion_indices)
                })
                # # 3. 獲取畸變 patch 的索引
                # distortion_indices = distortion_mask.flatten().nonzero(as_tuple=True)[0].tolist()

                # # --- 報告與可視化結果 ---
                # print("\n--- Analysis Complete ---")
                # num_distortion_patches = len(distortion_indices)
                # distortion_percentage = (num_distortion_patches / 1024) * 100
                # print(f"Total patches: 1024")
                # print(f"Number of distortion (boundary) patches identified: {num_distortion_patches}")
                # print(f"Percentage of distortion patches: {distortion_percentage:.2f}%")
                # # print(f"Indices of distortion patches: {distortion_indices}")

                # # 簡單可視化
                # print("\n--- Visualization of Patch Grid ---")
                # print("'.': Blank, '#': Content, 'X': Distortion/Boundary")
                # grid_vis = np.full((32, 32), ' ')
                # grid_vis[is_blank_grid.numpy()] = '.'
                # grid_vis[~is_blank_grid.numpy()] = '#'
                # grid_vis[distortion_mask.numpy()] = 'X'
                # for row in grid_vis:
                #     print(' '.join(row))


            

            # ==================== 【↓↓↓ 全新的替换逻辑 ↓↓↓】 ====================
            # 将这个資訊列表加入 model_inputs
            # model_inputs["distortion_info"] = distortion_info_list # <--- 删掉或注释掉这一行

            # 1. 初始化用于存储所有图片信息的列表
            all_indices = []
            all_weights = []
            all_lens = []
            
            num_target_images = len(distortion_info_list)

            # 2. 从临时列表中提取信息
            for info in distortion_info_list:
                all_indices.append(info["indices"])
                all_weights.append(info["weight"])
                all_lens.append(info["original_len"])

            # 3. 将列表转换为张量
            # 如果 distortion_info_list 不为空，则堆叠；否则创建空张量
            if num_target_images > 0:
                indices_tensor = torch.stack(all_indices)
                weights_tensor = torch.tensor(all_weights)
                lens_tensor = torch.tensor(all_lens, dtype=torch.long)
            else: # 处理没有目标图片的情况
                indices_tensor = torch.empty(0, 1024, dtype=torch.long)
                weights_tensor = torch.empty(0, dtype=torch.float)
                lens_tensor = torch.empty(0, dtype=torch.long)

            # 4. 对 "图片数量" 这个维度进行填充
            num_images_padding = max_num_images - num_target_images
            
            # Pad a. 索引张量 (在第0维填充)
            indices_tensor = torch.nn.functional.pad(indices_tensor, (0, 0, 0, num_images_padding), mode='constant', value=-100)
            # Pad b. 权重张量 (在第0维填充)
            weights_tensor = torch.nn.functional.pad(weights_tensor, (0, num_images_padding), mode='constant', value=0)
            # Pad c. 长度张量 (在第0维填充)
            lens_tensor = torch.nn.functional.pad(lens_tensor, (0, num_images_padding), mode='constant', value=0)

            # 5. 将这些可以直接被 default_collate 处理的张量加入 model_inputs
            model_inputs["distortion_indices"] = indices_tensor
            model_inputs["distortion_weights"] = weights_tensor
            model_inputs["distortion_lens"] = lens_tensor
            model_inputs["num_target_images"] = torch.tensor(num_target_images, dtype=torch.long) # 也作为一个张量传入



        # 兼容后续代码，拼回到 batch 维度
        # model_inputs["image_latents"] = input_image_latents
        model_inputs["target_image_latents"] = target_image_latents

        # 1. 获取 attention_mask 张量
        #    因为 batch_size 为 1，我们直接取第一个（也是唯一一个）样本
        attention_mask = model_inputs['attention_mask'][0]
        
        # 2. 计算真实内容的长度（即 attention_mask 中 1 的数量）
        actual_content_length = attention_mask.sum().item()
        
        # 3. 获取序列的总长度 (等于 self.max_length)
        total_length = len(attention_mask)
        
        # 4. 计算填充的 token 数量
        padding_length = total_length - actual_content_length
        # --- 5. 构建 labels ---
        # 步骤 1: 定义图片 token 的相关常量
        try:
            boi_token_id = self.processor.tokenizer.boi_token_id
        except AttributeError:
            boi_token_id = 8197 
            # print(f"Warning: processor.tokenizer.boi_token_id not found. Using default ID: {boi_token_id}")
        image_token_len = 1024
        labels = model_inputs['input_ids'].clone()
        prompt_len = len(self.processor.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        prompt_len += 1024 * prompt_image_num + padding_length
        labels[0, :(prompt_len+1)] = -100  # +1 to include the last token of the prompt
        # 这里的 +1 是为了确保 prompt 的最后一个 token 也被忽略掉
        # print("label prompt position:", labels[0, prompt_len])
        # 步骤 4: 屏蔽（忽略）所有图片 token
        image_start_indices = (model_inputs['input_ids'][0] == boi_token_id).nonzero(as_tuple=True)[0]
        for start_idx in image_start_indices:
            end_idx = start_idx + image_token_len
            # 这部分代码的意图是：【告诉模型，不要对任何图片 token 计算 loss】
            # 这会覆盖掉 target 中穿插的图片部分
            # print("label start:", labels[0, start_idx])
            # print("label end:", labels[0, end_idx])
            labels[0, (start_idx+1):(end_idx+2)] = -100  # start+1 为了预测BOI, end+2 为了不预测EOI
            # print("label start:", labels[0, start_idx])
            # print("label end:", labels[0, end_idx])

        model_inputs["labels"] = labels


        # torch.set_printoptions(profile="full")

        # print("input_ids:", model_inputs['input_ids'])
        # print("attention_mask:", model_inputs['attention_mask'])
        # print("labels:", model_inputs['labels'])
        # print("prompt len:", model_inputs['input_ids'][0][prompt_len])
        # # 獲取所有需要儲存的變數
        # input_ids_val = model_inputs['input_ids']
        # attention_mask_val = model_inputs['attention_mask']
        # labels_val = model_inputs['labels']
        # prompt_len_token_val = model_inputs['input_ids'][0][prompt_len]

        # # 使用 with open 將所有內容寫入一個檔案
        # with open("model_inputs_log.txt", "w") as f:
        #     f.write("--- input_ids ---\n")
        #     f.write(str(input_ids_val))
        #     f.write("\n\n") # 空兩行方便閱讀

        #     f.write("--- attention_mask ---\n")
        #     f.write(str(attention_mask_val))
        #     f.write("\n\n")

        #     f.write("--- labels ---\n")
        #     f.write(str(labels_val))
        #     f.write("\n\n")

        #     f.write("--- Token at prompt_len ---\n")
        #     f.write(str(prompt_len_token_val.item())) # .item() 獲取純數字

        # print("資料已成功寫入 model_inputs_log.txt")
        # 找到第一个不为 -100 的索引，即为 target 的起始位置
        # target_start_idx = np.where(labels[0].cpu() != -100)[0][0] # 加上 .cpu() 以防 labels 在GPU上
        model_inputs["target_start_idx"] = torch.tensor(prompt_len, dtype=torch.long)

        # print("target_start_idx", model_inputs["target_start_idx"])
        # 清理
        if "vqmodel" in model_inputs:
            model_inputs.pop("vqmodel")
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.squeeze(0)
                # print(f"{key}.shape: {model_inputs[key].shape}")

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
    def __init__(self, *args, processor=None, generation_log_file=None, enable_generation_log=False, alpha=1.0, beta=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 将 processor 保存为类的属性，以便在 compute_loss 中使用
        self.processor = processor
        self.generation_log_file = generation_log_file
        self.enable_generation_log = enable_generation_log

        # ==================== 新增代码：创建调试图片目录 ====================
        self.debug_image_dir = "debug_images"
        if self.is_world_process_zero():
            os.makedirs(self.debug_image_dir, exist_ok=True)
        # =================================================================
        self.debug_output_dir = "debug_outputs"
        if self.is_world_process_zero():
            os.makedirs(self.debug_output_dir, exist_ok=True)

        # 【新增代码】: 保存 alpha 和 beta 权重
        self.alpha = alpha
        self.beta = beta

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
        # target_image_latents = inputs.pop("target_image_latents")
        # 【新增修改 1】: 从 inputs 中取出 target_start_idx
        # target_start_idx = inputs.pop("target_start_idx")
        # ==================== 调试代码块 (开始) ====================
        # 只在训练和评估时打印，不在 generate 时打印
        # model.training 是训练状态， not model.training 是评估状态
        should_debug = (self.is_world_process_zero() and
                        (self.state.global_step + 1) % 50 == 0 and
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

        # if model.training:
        #     outputs_tuple = model(
        #         **inputs, 
        #         # target_image_latents=target_image_latents, 
        #         # target_start_idx=target_start_idx, # <--- 在这里传入
        #         train_mode='interleave'
        #     )
        #     logits, diff_loss = outputs_tuple
        # # model_outputs_for_return = outputs_tuple
        # else:
        #     outputs = model(**inputs, mode='discrete')
        #     logits = outputs.logits.float()
        #     diff_loss = torch.tensor(0.0)  # 评估时没有 diff_loss
        # --- 模型前向传播 ---
        if model.training:

            current_epoch = self.state.epoch
            # 【重要修改】接收模型返回的4个值
            outputs_tuple = model(
                **inputs,
                # target_image_latents=target_image_latents,
                # target_start_idx=target_start_idx,
                train_mode='interleave',
                current_epoch=current_epoch,  # <--- 在這裡傳入
            )

            # 根据模型返回值的数量来处理                                                                                                          
            if len(outputs_tuple) == 4:
                logits, diff_loss, pred_latents, true_latents = outputs_tuple
            else:                                                                                                                                 
                logits, diff_loss = outputs_tuple
                pred_latents, true_latents = None, None    
        else:
            outputs = model(**inputs, mode='discrete')
            logits = outputs.logits.float()
            diff_loss = torch.tensor(0.0)
            # 在评估模式下不清空这些变量，以防后续代码出错
            pred_latents, true_latents = None, None
        # --- 计算损失 ---
        shift_logits = logits[:, :-1, :].contiguous()  # 去掉最后一个位置
        shift_labels = labels[:, 1:].contiguous()      # 去掉第一个位置

        loss_fct = CrossEntropyLoss()
        text_loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        # --- 【修改3】融合 diff_loss，使用 self.alpha 和 self.beta ---
        # alpha = 1.0  # <--- 删除这一行
        # beta = 1000.0   # <--- 删除这一行
        loss = self.alpha * text_loss + self.beta * diff_loss
        # loss = self.alpha * text_loss
        model_outputs_for_return = (text_loss, diff_loss)
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
            print(f"[DEBUG] text_loss: {text_loss.item():.4f} | diff_loss: {diff_loss.item():.4f} | total_loss: {loss.item():.4f}")
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
        # ==================== 核心修改：保存latents到文件 ====================
        if should_debug and pred_latents is not None and true_latents is not None:
            try:
                print(f"\n[DEBUGGING] Saving latents at step {self.state.global_step}...")

                # # 1. 提取第一个样本的第一张图的特征
                # pred_image_features = pred_latents[0, :1024, :]
                # true_image_features = true_latents[0, :1024, :]
                
                # 2. 准备保存路径
                save_path = os.path.join(self.debug_output_dir, f"step_{self.state.global_step}_latents.pt")
                
                # 3. 将预测和真实的latents保存在一个字典中
                #    使用 .detach().cpu() 是一个好习惯，可以避免占用GPU显存并断开计算图
                latents_to_save = {
                    'predicted': pred_latents.detach().cpu(),
                    'target': true_latents.detach().cpu()
                }
                
                # 4. 保存文件
                torch.save(latents_to_save, save_path)
                
                print(f"  - Latents saved successfully to: {save_path}")

            except Exception as e:
                print(f"\033[91mError during saving latents at step {self.state.global_step}: {e}\033[0m")
                traceback.print_exc()
        # ====================================================================


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