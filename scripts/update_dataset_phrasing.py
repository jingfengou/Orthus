#!/usr/bin/env python
"""
批处理数据集中的 Question/Reasoning 文本：
1. Question 字段前添加主语 “Original cube, ”（若尚未包含）。
2. 将 Reasoning 中的 “First, rotate this cube” 替换为 “First, rotate the original cube”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调整 data_modified.json 的 Question 与 Reasoning 文本。")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data1/oujingfeng/project/twgi/datasets/mydatasets/dataset/data_modified.json"),
        help="待修改的 JSON 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认写入 input 同目录下 *_with_subject.json）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计潜在修改数量，不写回文件。",
    )
    return parser.parse_args()


def add_subject_to_question(question: str) -> str:
    """
    基于 reasoning_parser.py 中的模板，为 Question 注入 “original cube” 主语：
    - 句首已有 “A/The/This cube” → 改写为 “The original cube”
    - “Consider the visual outcome ...” → “Consider the visual outcome of the original cube ...”
    - “How would the orientation change ...” → 在 orientation/ outcome 处补充
    - 其他情况：前置 “Regarding the original cube, ...” 保持自然语气
    """
    import re

    if "original cube" in question.lower():
        return question

    stripped = question.lstrip()
    lower = stripped.lower()
    prefix_offset = len(question) - len(stripped)

    leading_cube = re.match(r"(a|the|this)\s+cube", lower)
    if leading_cube:
        match_text = leading_cube.group(0)
        updated = (
            question[:prefix_offset]
            + "The original cube"
            + stripped[len(match_text):]
        )
        return updated

    if lower.startswith("consider the visual outcome"):
        return (
            question[:prefix_offset]
            + "Consider the visual outcome of the original cube"
            + stripped[len("Consider the visual outcome"):]
        )

    pattern = re.compile(r"\borientation\b", re.IGNORECASE)
    match = pattern.search(question)
    if match:
        insert_pos = match.end()
        return (
            question[:insert_pos]
            + " of the original cube"
            + question[insert_pos:]
        )

    pattern = re.compile(r"\boutcome\b", re.IGNORECASE)
    match = pattern.search(question)
    if match:
        insert_pos = match.end()
        return (
            question[:insert_pos]
            + " for the original cube"
            + question[insert_pos:]
        )

    return f"Regarding the original cube, {stripped}"


def update_reasoning(reasoning: str) -> str:
    target = "First, rotate this cube"
    replacement = "First, rotate the original cube"
    return reasoning.replace(target, replacement)


def process_records(records: List[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"question_updated": 0, "reasoning_updated": 0}
    for item in records:
        question = item.get("Question", "")
        if isinstance(question, str):
            new_question = add_subject_to_question(question)
            if new_question != question:
                item["Question"] = new_question
                stats["question_updated"] += 1

        reasoning = item.get("Reasoning", "")
        if isinstance(reasoning, str):
            new_reasoning = update_reasoning(reasoning)
            if new_reasoning != reasoning:
                item["Reasoning"] = new_reasoning
                stats["reasoning_updated"] += 1

    return stats


def main() -> None:
    args = parse_args()
    input_path = args.input
    if args.output is not None:
        output_path = args.output
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_subject.json")

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    stats = process_records(records)

    if args.dry_run:
        print(f"[Dry-run] Question 更新 {stats['question_updated']} 条，Reasoning 更新 {stats['reasoning_updated']} 条。")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"写入 {output_path} 完成。Question 更新 {stats['question_updated']} 条，"
        f"Reasoning 更新 {stats['reasoning_updated']} 条。"
    )


if __name__ == "__main__":
    main()
