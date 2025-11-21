import re
import sys
from pathlib import Path


def evaluate_directory(target_dir: Path):
    txt_files = sorted(target_dir.glob("*.txt"))
    total = len(txt_files)
    correct = 0
    incorrect = 0
    missing = 0

    answer_pattern = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL)
    gt_pattern = re.compile(r"Ground truth:\s*(.*)")

    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore")

        gt_match = gt_pattern.search(text)
        ground_truth = gt_match.group(1).strip() if gt_match else ""

        ans_match = answer_pattern.search(text)
        if not ans_match:
            missing += 1
            continue

        predicted = ans_match.group(1).strip()
        if not predicted:
            missing += 1
            continue

        if predicted.lower() == ground_truth.lower():
            correct += 1
        else:
            incorrect += 1

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "accuracy": (correct / total) if total else 0.0,
        "missing_rate": (missing / total) if total else 0.0,
    }


def format_report(stats: dict) -> str:
    lines = [
        f"Total files: {stats['total']}",
        f"Correct answers: {stats['correct']}",
        f"Incorrect answers: {stats['incorrect']}",
        f"Missing answers: {stats['missing']}",
        f"Accuracy: {stats['accuracy']:.4f}",
        f"Missing rate: {stats['missing_rate']:.4f}",
    ]
    return "\n".join(lines)


def main():
    current_dir = Path(__file__).resolve().parent
    target_dir = current_dir / "sft-myb-base-sample4000b100ep15-train10-gt"
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        if not output_path.is_absolute():
            output_path = current_dir / output_path
    else:
        output_path = current_dir / "ep15-train10-results-gt.txt"

    if not target_dir.exists():
        raise SystemExit(f"Target directory not found: {target_dir}")

    stats = evaluate_directory(target_dir)
    report = format_report(stats)

    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()

