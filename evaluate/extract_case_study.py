import json
import random
import os
from collections import defaultdict
from datasets import load_dataset # <-- 新增：需要 datasets 库来读取 .parquet 文件
# --- 配置区 ---
# 请根据您的需求修改这里的参数

# 输入文件名
INPUT_FILE = 'orthus_output_svb_final.jsonl'
# 请确保这个路径是正确的
BENCHMARK_DIR = '/home/server/oujingfeng/project/twgi/datasets/SpatialViz' 
GROUND_TRUTH_FILE = os.path.join(BENCHMARK_DIR, 'test-00000-of-00001.parquet')
# 输出文件名，用于存放提取的 case
OUTPUT_FILE = 'case_study_responses.txt'

# 要提取的 JSON 字段的 key
TARGET_KEY = 'response'

MARKER_STRING = None
# --- 策略选择 ---
# 从以下三种策略中选择一种，并取消对应代码块的注释
# 确保同时只有一种策略处于激活状态

# 策略 1: 提取文件最前面的 N 个样本
# ===================================================================
# EXTRACTION_STRATEGY = 'first_n'
# NUM_SAMPLES = 10  # 您想要提取的样本数量

# 策略 2: 提取指定行号的样本 (行号从 1 开始)
# ===================================================================
# EXTRACTION_STRATEGY = 'specific_lines'
# TARGET_LINES = {5, 12, 42, 101} # 把您感兴趣的行号放在这里

# 策略 3: 从整个文件中随机抽取 N 个样本
# ===================================================================
# EXTRACTION_STRATEGY = 'random_n'
# NUM_SAMPLES = 10 # 您想要随机抽取的样本数量
EXTRACTION_STRATEGY = 'random_per_task' # <-- 新增：每个任务类型随机选1个

def load_ground_truth(benchmark_parquet_path):
    """
    【修改】加载标准答案和任务类型，并将其映射到一个字典中。
    返回: {id -> {'answer': 'A', 'task': 'ArrowMoving'}}
    """
    print(f"正在从 '{benchmark_parquet_path}' 加载标准答案和任务类型...")
    try:
        dataset = load_dataset("parquet", data_files={'test': benchmark_parquet_path})['test']
        ground_truth_map = {}
        for item in dataset:
            unique_id = f"{item['Category']}-{item['Task']}-{item['Level']}-{item['Image_id']}"
            ground_truth_map[unique_id] = {
                "answer": item['Answer'],
                "task": item['Task']
            }
        print(f"成功加载 {len(ground_truth_map)} 个标准答案。")
        return ground_truth_map
    except Exception as e:
        print(f"错误: 加载标准答案文件失败: {e}")
        return {}

def extract_random_per_task(ground_truth_map):
    """
    【新增策略】为每个任务类型随机抽取一个样本。
    """
    print(f"策略: 从 '{INPUT_FILE}' 的每个任务类型中随机抽取1个样本...")
    
    # 1. 按任务类型对所有样本进行分组
    tasks = defaultdict(list)
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in, 1):
                try:
                    data = json.loads(line)
                    original_id = data.get("id")
                    if not original_id or original_id not in ground_truth_map:
                        continue # 如果ID无效或找不到对应信息，则跳过
                    
                    gt_info = ground_truth_map[original_id]
                    task_name = gt_info['task']
                    correct_answer = gt_info['answer']
                    
                    raw_text = data.get(TARGET_KEY)
                    if raw_text:
                        cleaned_text = process_text_content(raw_text, i)
                        if cleaned_text is not None:
                            # 保存完整信息 (行号, ID, 文本, 答案)
                            full_sample_info = (i, original_id, cleaned_text, correct_answer)
                            tasks[task_name].append(full_sample_info)

                except json.JSONDecodeError:
                    print(f"警告: 第 {i} 行 JSON 解析失败，已跳过。")
    except FileNotFoundError:
        print(f"错误: 预测文件 '{INPUT_FILE}' 未找到！")
        return []

    # 2. 从每个分组中随机抽取一个样本
    extracted_cases = []
    print("\n开始从以下任务中抽样:")
    for task_name, samples_in_task in sorted(tasks.items()):
        if samples_in_task:
            chosen_case = random.choice(samples_in_task)
            extracted_cases.append(chosen_case)
            print(f"- {task_name}: 抽样成功 (共 {len(samples_in_task)} 个样本)")
        else:
            print(f"- {task_name}: 未找到样本")
            
    return extracted_cases


def process_text_content(raw_text, line_num):
    """处理单个文本内容"""
    if not isinstance(raw_text, str):
        print(f"警告: 第 {line_num} 行的 '{TARGET_KEY}' 内容不是文本，已跳过。")
        return None
        
    if MARKER_STRING is None:
        return raw_text # 如果没有标记，直接返回原始内容

    marker_position = raw_text.find(MARKER_STRING)
    if marker_position != -1:
        start_position = marker_position + len(MARKER_STRING)
        return raw_text[start_position:]
    else:
        print(f"警告: 在第 {line_num} 行未找到标记字符串。将使用完整的原始内容。")
        return raw_text

def extract_samples(ground_truth_map):
    """
    通用函数，根据所选策略提取样本。
    现在返回 (行号, ID, 清理后的文本, 正确答案) 的元组。
    """
    all_potential_cases = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in, 1):
                try:
                    data = json.loads(line)
                    raw_text = data.get(TARGET_KEY)
                    original_id = data.get("id", f"未知ID_line_{i}")
                    correct_answer = ground_truth_map.get(original_id, "未知答案") # <-- 获取正确答案
                    
                    if raw_text:
                        cleaned_text = process_text_content(raw_text, i)
                        if cleaned_text is not None:
                            all_potential_cases.append((i, original_id, cleaned_text, correct_answer))
                except json.JSONDecodeError:
                    print(f"警告: 第 {i} 行 JSON 解析失败，已跳过。")
    except FileNotFoundError:
        print(f"错误: 预测文件 '{INPUT_FILE}' 未找到！")
        return []

    if EXTRACTION_STRATEGY == 'first_n':
        print(f"策略: 提取并清理 '{INPUT_FILE}' 中的前 {NUM_SAMPLES} 个样本...")
        return all_potential_cases[:NUM_SAMPLES]
    elif EXTRACTION_STRATEGY == 'specific_lines':
        print(f"策略: 从 '{INPUT_FILE}' 提取并清理行: {sorted(list(TARGET_LINES))}")
        return [case for case in all_potential_cases if case[0] in TARGET_LINES]
    elif EXTRACTION_STRATEGY == 'random_n':
        print(f"策略: 从 '{INPUT_FILE}' 中随机抽取并清理 {NUM_SAMPLES} 个样本...")
        if len(all_potential_cases) < NUM_SAMPLES:
            print(f"警告: 文件中有效样本数 ({len(all_potential_cases)}) 小于要抽取的数量。将提取所有有效样本。")
            return all_potential_cases
        return random.sample(all_potential_cases, NUM_SAMPLES)
    
    return []


def main():
    """主函数"""
    ground_truth_map = load_ground_truth(GROUND_TRUTH_FILE)
    if not ground_truth_map:
        return

    extracted_data = []
    if EXTRACTION_STRATEGY == 'random_per_task':
        extracted_data = extract_random_per_task(ground_truth_map)
    # ... (可以保留对 'first_n', 'random_n' 等策略的elif判断)
    else:
        print(f"错误: 未知的提取策略 '{EXTRACTION_STRATEGY}'")
        return

    if not extracted_data:
        print("未能提取到任何数据。")
        return

    # 将提取并清理后的数据写入输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # 按原始行号排序，确保输出文件顺序稳定
        sorted_data = sorted(extracted_data, key=lambda item: item[0])
        
        for i, (line_num, original_id, cleaned_text, correct_answer) in enumerate(sorted_data):
            f_out.write(f"--- Case {i+1} (来自原始文件ID: {original_id}) ---\n")
            f_out.write(f"Correct Answer: {correct_answer}\n")
            f_out.write("-" * 20 + "\n\n")
            f_out.write(cleaned_text.strip())
            f_out.write("\n\n" + "="*80 + "\n\n")
    
    print(f"\n成功从 {len(sorted_data)} 个任务类型中各提取1个样本到文件 '{OUTPUT_FILE}'。")

if __name__ == '__main__':
    main()