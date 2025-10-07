import json
import argparse
import os

def filter_string_explanation_samples(data_file):
    """
    遍历数据集，筛选出'Explanation'字段是字符串的样本。
    """
    # 根据输入文件名，自动生成输出文件名
    output_file = data_file + ".string_samples.jsonl"
    
    total_records = 0
    string_explanation_count = 0
    
    print(f"Scanning file: {data_file}")
    
    # 使用 with 语句确保文件被正确关闭
    with open(data_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_records += 1
            try:
                # 解析每一行的 JSON 数据
                item = json.loads(line)
                
                # 获取 Explanation 字段的值
                explanation_value = item.get('Explanation')
                
                # 检查值的类型是否为字符串
                if isinstance(explanation_value, str):
                    string_explanation_count += 1
                    # 如果是字符串，将原始的行写入输出文件
                    f_out.write(line)
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
                continue

    print("\n--- Scan Report ---")
    print(f"Total records scanned: {total_records}")
    print(f"Samples with string-type Explanation (error samples): {string_explanation_count}")
    print(f"Filtered error samples have been saved to: {output_file}")
    print("-------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset samples where 'Explanation' is a string.")
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to the .jsonl data file (e.g., train.jsonl).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: File not found at {args.data_file}")
    else:
        filter_string_explanation_samples(args.data_file)