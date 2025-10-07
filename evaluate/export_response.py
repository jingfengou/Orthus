import json

def extract_responses_from_jsonl(input_file_path, output_file_path):
    """
    从 JSONL 文件中提取 'response' 字段的值，并检查它们是否都是单个字母。

    Args:
        input_file_path (str): 输入的 .jsonl 文件路径。
        output_file_path (str): 用于保存提取结果的 .txt 文件路径。
    """
    responses = []
    all_are_single_char = True
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    # 解析当前行
                    data = json.loads(line.strip())
                    
                    # 提取 'response' 字段
                    if 'response' in data:
                        response_value = data['response']
                        responses.append(response_value)
                        
                        # 检查长度是否为1
                        if len(str(response_value)) != 1:
                            all_are_single_char = False
                            print(f"警告：第 {i+1} 行的 response '{response_value}' 不是单个字符。")
                    else:
                        print(f"警告：第 {i+1} 行没有找到 'response' 字段。")
                
                except json.JSONDecodeError:
                    print(f"错误：第 {i+1} 行无法解析为 JSON。")

        # 将提取的 response 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for response in responses:
                outfile.write(str(response) + '\n')
        
        print(f"\n成功从 {len(responses)} 行中提取了 'response' 字段。")
        print(f"提取结果已保存到: {output_file_path}")

        # 最终确认
        if all_are_single_char:
            print("\n确认：所有提取出的 'response' 值都是单个字符。")
        else:
            print("\n注意：并非所有 'response' 值都是单个字符。请检查上面的警告信息。")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file_path}'。请确保文件路径正确。")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

if __name__ == '__main__':
    # 定义输入和输出文件路径
    # 请确保 'orthus_instruct_Question-instruct_output.jsonl' 与此脚本在同一目录下
    # 或者提供完整的文件路径
    input_filename = 'orthus_instruct_instruct-question_output.jsonl'
    output_filename = 'extracted_responses_instruct_instruct-question.txt'
    
    extract_responses_from_jsonl(input_filename, output_filename)