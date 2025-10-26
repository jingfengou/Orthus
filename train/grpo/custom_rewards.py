# custom_rewards.py

import re
from typing import List, Dict

def answer_correctness_reward(
    generated_texts: List[str], 
    metadatas: List[Dict], 
    **kwargs
) -> List[float]:
    """
    一个奖励函数，用于评估生成的答案是否与真实答案匹配。

    Args:
        generated_texts (List[str]): 模型生成的、已解码的文本列表。
        metadatas (List[Dict]): 与每个生成样本对应的原始元数据列表，
                               应包含 'Answer' 字段。

    Returns:
        List[float]: 每个生成文本对应的奖励分数列表 (1.0 表示正确, 0.0 表示错误或格式不符)。
    """
    rewards = []
    for i in range(len(generated_texts)):
        generated_text = generated_texts[i]
        metadata = metadatas[i]
        
        ground_truth_answer = metadata.get("Answer")
        if ground_truth_answer is None:
            rewards.append(0.0)
            continue

        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        
        if not match:
            rewards.append(0.0) # 格式错误，奖励为 0
        else:
            generated_answer = match.group(1).strip()
            if generated_answer == ground_truth_answer.strip():
                rewards.append(1.0) # 答案正确，奖励为 1.0
            else:
                rewards.append(0.0) # 答案错误，奖励为 0.0
                
    return rewards

def format_reward(
    generated_texts: List[str], 
    **kwargs
) -> List[float]:
    """
    一个奖励函数，用于评估生成的文本是否遵循 <think>...</think><answer>...</answer> 的格式。

    Args:
        generated_texts (List[str]): 模型生成的、已解码的文本列表。

    Returns:
        List[float]: 每个生成文本对应的奖励分数列表 (1.0 表示格式正确, 0.0 表示格式错误)。
    """
    rewards = []
    # 正则表达式，匹配 <think> 标签对和 <answer> 标签对，允许中间有空格或换行
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    
    for text in generated_texts:
        # 使用 search 而不是 match，因为它可以在字符串的任何位置找到匹配项
        if pattern.search(text):
            rewards.append(1.0) # 格式正确，奖励为 1.0
        else:
            rewards.append(0.0) # 格式错误，奖励为 0.0
            
    return rewards