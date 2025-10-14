# filename: preprocess_benchmark.py
import json
import os
import argparse

def create_orthus_prompt(question, options):
    """
    根据问题和选项构建一个完整的、适合模型的prompt。
    这个版本使用了用户指定的 <think> 和 <answer> 标签指令。
    """
    # 将选项列表转换为带 A), B), C), D) 标签的字符串
    option_labels = "ABCD"
    # 确保选项列表不超过4个
    option_str = "\n".join([f"{option_labels[i]}) {option_text}" for i, option_text in enumerate(options[:4])])
    
    # 用户指定的新指令
    instruction = (
        "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. "
        "The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, "
        "respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    )
    
    # 组合成最终的prompt
    prompt = (
        f"{instruction}\n"
        f"<image>\n\n"
        f"Question: {question}\n"
        f"{option_str}\n\n"
        "Answer: \n"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Preprocess Spatial-Visualization-Benchmark (JSONL format) for Orthus model.")
    parser.add_argument("--input_file", required=True, help="Path to the input train.jsonl file.")
    parser.add_argument("--image_base_dir", required=True, help="Path to the root directory where images are stored (e.g., the root of Spatial-Visualization-Benchmark).")
    parser.add_argument("--output_file", required=True, help="Path to save the preprocessed JSONL file for Orthus.")
    args = parser.parse_args()

    processed_count = 0
    # 打开输入和输出文件
    with open(args.input_file, 'r', encoding='utf-8') as f_in, open(args.output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # 移除行尾的换行符并解析JSON
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}. Skipping.")
                continue

            # 根据规则拼接图片路径
            # 路径结构：{image_base_dir}/{Category}/{Task}/{Level}/{Image_id}.png
            image_path = os.path.join(
                args.image_base_dir, 
                item["Category"], 
                item["Task"], 
                item["Level"], 
                f"{item['Image_id']}.png"
            )
            
            # 确保图片路径是绝对路径
            absolute_image_path = os.path.abspath(image_path)
            if not os.path.exists(absolute_image_path):
                print(f"Warning: Image not found at {absolute_image_path}. Skipping item.")
                continue

            # 构建prompt
            prompt = create_orthus_prompt(item["Question"], item["Choices"])
            
            # 创建新的JSON对象
            orthus_item = {
                "id": f"{item['Category']}-{item['Task']}-{item['Level']}-{item['Image_id']}", # 创建一个唯一的ID
                "prompt": prompt,
                "images": [absolute_image_path]
            }
            
            f_out.write(json.dumps(orthus_item) + '\n')
            processed_count += 1
            
    print(f"Preprocessing complete. Converted {processed_count} items.")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()