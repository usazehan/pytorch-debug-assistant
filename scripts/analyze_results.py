import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> str:
    if d == 0:
        return "0.00%"
    return f"{100 * n / d:.2f}%"


def analyze(rows: list[dict[str, Any]], show_mistakes: int) -> None:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    valid_json = sum(1 for r in rows if r.get("valid_json"))

    print("\nOverall Results")
    print("=" * 40)
    print(f"Examples:          {total}")
    print(f"Correct:           {correct}")
    print(f"Category Accuracy: {pct(correct, total)}")
    print(f"Valid JSON Rate:   {pct(valid_json, total)}")

    # Per-category accuracy
    by_expected = defaultdict(list)
    for row in rows:
        by_expected[row.get("expected_category", "UNKNOWN")].append(row)

    print("\nAccuracy by Expected Category")
    print("=" * 40)

    for category in sorted(by_expected):
        group = by_expected[category]
        group_total = len(group)
        group_correct = sum(1 for r in group if r.get("correct"))
        print(
            f"{category:24s} "
            f"{group_correct:3d}/{group_total:<3d} "
            f"{pct(group_correct, group_total)}"
        )

    # Confusion matrix-ish summary
    confusion = defaultdict(Counter)
    for row in rows:
        expected = row.get("expected_category", "UNKNOWN")
        predicted = row.get("predicted_category") or "INVALID_JSON"
        confusion[expected][predicted] += 1

    print("\nConfusion Matrix")
    print("=" * 40)

    for expected in sorted(confusion):
        print(f"{expected}: {dict(confusion[expected])}")

    # Override analysis
    override_rows = [r for r in rows if r.get("override_applied")]

    helped = 0
    hurt = 0
    unchanged_wrong = 0

    for row in override_rows:
        expected = row.get("expected_category")
        model_pred = row.get("model_predicted_category")
        final_pred = row.get("predicted_category")

        model_was_correct = model_pred == expected
        final_is_correct = final_pred == expected

        if not model_was_correct and final_is_correct:
            helped += 1
        elif model_was_correct and not final_is_correct:
            hurt += 1
        elif not final_is_correct:
            unchanged_wrong += 1

    print("\nCategory Override Analysis")
    print("=" * 40)
    print(f"Overrides applied: {len(override_rows)}")
    print(f"Overrides helped:  {helped}")
    print(f"Overrides hurt:    {hurt}")
    print(f"Still wrong after override: {unchanged_wrong}")

    if override_rows:
        print("\nOverride Changes")
        print("=" * 40)

        changes = Counter(
            (
                row.get("model_predicted_category") or "INVALID_JSON",
                row.get("predicted_category") or "INVALID_JSON",
            )
            for row in override_rows
        )

        for (before, after), count in changes.most_common():
            print(f"{before:24s} -> {after:24s} {count}")

    # Mistakes
    mistakes = [r for r in rows if not r.get("correct")]

    print(f"\nMistakes Showing First {min(show_mistakes, len(mistakes))}")
    print("=" * 40)

    for row in mistakes[:show_mistakes]:
        print(f"ID:        {row.get('id')}")
        print(f"Expected:  {row.get('expected_category')}")
        print(f"Model:     {row.get('model_predicted_category')}")
        print(f"Final:     {row.get('predicted_category')}")
        print(f"Override:  {row.get('override_applied')}")
        print(f"Valid JSON:{row.get('valid_json')}")
        print(f"Error:     {str(row.get('error_text', ''))[:200]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch Debug Assistant evaluation results."
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("reports/structured_v3_adapter_results_100_with_overrides.jsonl"),
        help="Path to evaluation results JSONL file.",
    )

    parser.add_argument(
        "--show-mistakes",
        type=int,
        default=20,
        help="Number of mistakes to print.",
    )

    args = parser.parse_args()

    rows = load_jsonl(args.results_file)
    analyze(rows, show_mistakes=args.show_mistakes)


if __name__ == "__main__":
    main()