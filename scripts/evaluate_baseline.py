import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL_FILE = Path("data/eval/eval_set.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/eval/baseline_results.jsonl")

VALID_CATEGORIES = {
    "tensor_shape_mismatch",
    "cuda_oom",
    "device_mismatch",
    "dtype_mismatch",
    "autograd_error",
    "dataloader_error",
    "loss_issue",
    "environment_error",
    "optimizer_error",
    "training_loop_bug",
    "architecture_mismatch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    return rows


def normalize_text(example: dict[str, Any]) -> str:
    fields = [
        "question_title",
        "question_body",
        "error_text",
        "stack_trace",
        "code_context",
        "answer",
    ]

    return " ".join(str(example.get(field, "")) for field in fields).lower()


def predict_category(example: dict[str, Any]) -> str:
    """
    Simple rule-based baseline.

    This is intentionally not fancy. The point is to create a weak baseline
    that your fine-tuned model / RAG assistant should beat later.
    """
    text = normalize_text(example)

    # CUDA / memory issues
    if any(phrase in text for phrase in [
        "cuda out of memory",
        "out of memory",
        "cudnn_status_alloc_failed",
        "memory error",
        "killed",
    ]):
        return "cuda_oom"

    # Device mismatch / CUDA tensor vs CPU / NumPy conversion
    if any(phrase in text for phrase in [
        "expected all tensors to be on the same device",
        "input type (torch.floattensor) and weight type (torch.cuda.floattensor)",
        "can't convert cuda tensor to numpy",
        "cannot convert cuda tensor to numpy",
        "use tensor.cpu()",
    ]):
        return "device_mismatch"

    # Environment / installation / CUDA setup
    if any(phrase in text for phrase in [
        "torch not compiled with cuda enabled",
        "cuda is not available",
        "torch.cuda.is_available() returns false",
        "cuda initialization",
        "cuda unknown error",
        "no module named",
        "modulenotfounderror",
        "importerror",
        "installed pytorch",
        "install torch",
        "cpu-only",
        "cpu only",
        "found no nvidia driver",
    ]):
        return "environment_error"

    # Dtype issues
    if any(phrase in text for phrase in [
        "expected scalar type",
        "expected object of scalar type",
        "found float",
        "found double",
        "found long",
        "dtype",
        "doubletensor",
        "floattensor",
        "longtensor",
        "can't assign a numpy.int64",
    ]):
        return "dtype_mismatch"

    # Shape / dimension / tensor layout
    if any(phrase in text for phrase in [
        "size mismatch",
        "shape mismatch",
        "mat1 and mat2 shapes cannot be multiplied",
        "1d tensors expected",
        "invalid dimensions",
        "dimension out of range",
        "expected input batch_size",
        "view size is not compatible",
        "input is not contiguous",
    ]):
        return "tensor_shape_mismatch"

    # Autograd / backward / grad graph
    if any(phrase in text for phrase in [
        "does not require grad",
        "grad_fn",
        "backward through the graph a second time",
        "retain_graph",
        "leaf variable",
        "inplace operation",
        "one of the variables needed for gradient computation",
    ]):
        return "autograd_error"

    # DataLoader / Dataset / batching
    if any(phrase in text for phrase in [
        "dataloader",
        "dataset",
        "collate",
        "num_workers",
        "batch_indices",
        "torch.utils.data",
        "index_select",
        "mini-batches",
        "minibatches",
    ]):
        return "dataloader_error"

    # Loss / NaN / exploding gradients
    if any(phrase in text for phrase in [
        "nan",
        "loss is nan",
        "loss not decreasing",
        "exploding gradients",
        "gradient clipping",
        "clip_grad",
        "unstable training",
    ]):
        return "loss_issue"

    # Optimizer / scheduler
    if any(phrase in text for phrase in [
        "optimizer.step",
        "zero_grad",
        "learning rate",
        "scheduler",
        "param_groups",
    ]):
        return "optimizer_error"

    # Model architecture / checkpoint loading
    if any(phrase in text for phrase in [
        "load_state_dict",
        "state_dict",
        "missing key",
        "unexpected key",
        "checkpoint",
        "pretrained weights",
        "classifier.weight",
        "architecture",
    ]):
        return "architecture_mismatch"

    # Generic training loop/API misuse
    if any(phrase in text for phrase in [
        "device-side assert",
        "crossentropyloss(",
        "target",
        "label",
        "labels",
        "loss_fn",
        "training loop",
    ]):
        return "training_loop_bug"

    # Default fallback
    return "training_loop_bug"


def evaluate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    correct = 0
    total_latency_ms = 0.0

    confusion: dict[str, Counter] = defaultdict(Counter)

    for example in examples:
        expected = example.get("category")

        start = time.perf_counter()
        predicted = predict_category(example)
        latency_ms = (time.perf_counter() - start) * 1000
        total_latency_ms += latency_ms

        is_correct = predicted == expected
        correct += int(is_correct)

        confusion[expected][predicted] += 1

        results.append({
            "id": example.get("id"),
            "source_url": example.get("source_url"),
            "expected_category": expected,
            "predicted_category": predicted,
            "correct": is_correct,
            "latency_ms": latency_ms,
            "error_text": example.get("error_text", ""),
        })

    accuracy = correct / len(examples) if examples else 0.0
    avg_latency_ms = total_latency_ms / len(examples) if examples else 0.0

    return {
        "num_examples": len(examples),
        "correct": correct,
        "category_accuracy": accuracy,
        "avg_latency_ms": avg_latency_ms,
        "results": results,
        "confusion": {
            expected: dict(predictions)
            for expected, predictions in confusion.items()
        },
    }


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


def print_summary(report: dict[str, Any]) -> None:
    print("\nBaseline Evaluation Results")
    print("=" * 32)
    print(f"Examples:          {report['num_examples']}")
    print(f"Correct:           {report['correct']}")
    print(f"Category Accuracy: {report['category_accuracy']:.2%}")
    print(f"Avg Latency:       {report['avg_latency_ms']:.4f} ms")

    print("\nMistakes")
    print("-" * 32)

    mistakes = [row for row in report["results"] if not row["correct"]]

    if not mistakes:
        print("No mistakes.")
    else:
        for row in mistakes:
            print(f"ID:        {row['id']}")
            print(f"Expected:  {row['expected_category']}")
            print(f"Predicted: {row['predicted_category']}")
            print(f"Error:     {row['error_text'][:160]}")
            print(f"URL:       {row['source_url']}")
            print()

    print("\nConfusion Matrix")
    print("-" * 32)

    for expected, predictions in sorted(report["confusion"].items()):
        print(f"{expected}: {predictions}")


def validate_examples(examples: list[dict[str, Any]]) -> None:
    for example in examples:
        category = example.get("category")

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category for {example.get('id')}: {category}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a simple heuristic baseline on the PyTorch debugging eval set."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Path to eval_set.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write per-example baseline results as JSONL",
    )

    args = parser.parse_args()

    examples = load_jsonl(args.eval_file)
    validate_examples(examples)

    report = evaluate(examples)

    write_results(args.output_file, report["results"])
    print_summary(report)

    print(f"\nWrote per-example results to: {args.output_file}")


if __name__ == "__main__":
    main()