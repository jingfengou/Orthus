# filename: final_evaluate.py (Version with "Missing" column)
import json
import re
import argparse
import os
from collections import defaultdict
from datasets import load_dataset
import pandas as pd

def extract_final_answer(text):
    """
    稳健地从模型输出的完整文本中提取最后一个 <answer> 标签里的答案。
    """
    # matches = re.findall(r"<answer>.*?([A-D]).*?</answer>", text, re.IGNORECASE | re.DOTALL)
    matches = re.findall(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    # matches = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None
# --- 新增函式：用於載入標準答案 ---
def load_ground_truth(source_path):
    """
    從指定的路徑載入標準答案。
    - 如果路徑是資料夾，則讀取內部的 Parquet 檔案。
    - 如果路徑是 .jsonl 檔案，則直接讀取該檔案。
    """
    ground_truth_map = {}
    
    # 檢查路徑是資料夾還是檔案
    if os.path.isdir(source_path):
        print(f"Source is a directory. Looking for Parquet file inside...")
        parquet_file_path = os.path.join(source_path, "test-00000-of-00001.parquet")
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"Parquet file not found at: {parquet_file_path}")
        
        dataset = load_dataset("parquet", data_files={'test': parquet_file_path})['test']
        
        for item in dataset:
            unique_id = f"{item['Category']}-{item['Task']}-{item['Level']}-{item['Image_id']}"
            ground_truth_map[unique_id] = {
                "answer": item['Answer'],
                "category": item['Category'],
                "task": item['Task']
            }

    elif os.path.isfile(source_path) and source_path.endswith('.jsonl'):
        print(f"Source is a JSONL file. Reading directly...")
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                unique_id = f"{item['Category']}-{item['Task']}-{item['Level']}-{item['Image_id']}"
                ground_truth_map[unique_id] = {
                    "answer": item['Answer'],
                    "category": item['Category'],
                    "task": item['Task']
                }
    else:
        raise ValueError(f"Invalid path or file type for --benchmark_dir: {source_path}. "
                         "Please provide a directory containing the benchmark or a .jsonl file.")
    
    return ground_truth_map

def main():
    parser = argparse.ArgumentParser(description="Optimized evaluation for Orthus on Spatial-Visualization-Benchmark.")
    parser.add_argument("--prediction_file", required=True, help="Path to the Orthus model's output JSONL file.")
    parser.add_argument("--benchmark_dir", required=True, help="Path to the root directory of the Spatial-Visualization-Benchmark.")
    parser.add_argument("--output_results_file", type=str, default=None, help="Optional: Path to save the detailed results in JSON format.")
    args = parser.parse_args()

    # 1. 加載標準答案和元數據 (使用新函式)
    try:
        ground_truth = load_ground_truth(args.benchmark_dir)
        print(f"Loaded {len(ground_truth)} ground truth entries with metadata.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return


    # 2. 加载并解析模型的预测结果
    predictions = {}
    with open(args.prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            response_text = data.get('response') or data.get('text', '') 
            extracted_answer = extract_final_answer(response_text)
            predictions[data.get('id')] = extracted_answer
    print(f"Loaded and parsed {len(predictions)} model predictions.")

    # 3. 按类别和任务进行评估
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'missing': 0})
    
    for sample_id, gt_data in ground_truth.items():
        correct_answer = gt_data['answer']
        category = gt_data['category']
        task = gt_data['task']
        predicted_answer = predictions.get(sample_id)
        
        results['Overall']['total'] += 1
        results[category]['total'] += 1
        results[task]['total'] += 1
        
        if predicted_answer is None:
            results['Overall']['missing'] += 1
            results[category]['missing'] += 1
            results[task]['missing'] += 1
        elif predicted_answer == correct_answer:
            results['Overall']['correct'] += 1
            results[category]['correct'] += 1
            results[task]['correct'] += 1

    # 4. 计算准确率、缺失率并准备报告
    report_data = []
    for name, data in results.items():
        accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        missing_rate = (data['missing'] / data['total'] * 100) if data['total'] > 0 else 0 # <-- 新增：计算缺失率
        report_data.append({
            "Type": "Category" if name in ["MentalAnimation", "MentalFolding", "MentalRotation", "VisualPenetration"] else ("Overall" if name == "Overall" else "Task"),
            "Name": name,
            "Correct": data['correct'],
            "Missing": data['missing'],
            "Total": data['total'],
            "Accuracy (%)": f"{accuracy:.2f}",
            "Missing Rate (%)": f"{missing_rate:.2f}" # <-- 新增：在报告数据中加入缺失率
        })
        
    # 使用 pandas 创建格式化的表格
    df = pd.DataFrame(report_data)
    df['Type'] = pd.Categorical(df['Type'], ["Overall", "Category", "Task"])
    df = df.sort_values(by=["Type", "Name"]).reset_index(drop=True)
    # <-- 新增：调整列顺序以包含缺失率
    df = df[["Name", "Correct", "Missing", "Total", "Accuracy (%)", "Missing Rate (%)"]]

    print("\n--- Evaluation Report ---")
    print(df.to_string())
    print("-------------------------\n")

    # 5. (可选) 保存详细结果到文件
    if args.output_results_file:
        json_results = df.set_index('Name').to_dict('index')
        with open(args.output_results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4)
        print(f"Detailed results saved to {args.output_results_file}")

if __name__ == "__main__":
    main()