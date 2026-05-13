import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_EVAL_FILE = Path("data/eval/eval_set.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/eval/model_results.jsonl")

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


SYSTEM_PROMPT = """You are a PyTorch debugging assistant.

Given a PyTorch error, question, and code context, classify the issue and explain the fix.

Return ONLY a valid JSON object with exactly these fields:
{
  "category": "...",
  "root_cause": "...",
  "fix": "...",
  "fix_code": "..."
}

The category must be exactly one of:
- tensor_shape_mismatch
- cuda_oom
- device_mismatch
- dtype_mismatch
- autograd_error
- dataloader_error
- loss_issue
- environment_error
- optimizer_error
- training_loop_bug
- architecture_mismatch

Do not include markdown.
Do not include explanations outside the JSON.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
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


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def build_user_prompt(example: dict[str, Any]) -> str:
    title = truncate_text(example.get("question_title", ""), 500)
    error_text = truncate_text(example.get("error_text", ""), 800)
    code_context = truncate_text(example.get("code_context", ""), 1200)
    question_body = truncate_text(example.get("question_body", ""), 1200)
    answer = truncate_text(example.get("answer", ""), 800)

    return f"""Classify this PyTorch debugging issue.

Question title:
{title}

Error text:
{error_text}

Code context:
{code_context}

Question body:
{question_body}

Known reference answer:
{answer}

Return JSON only.
"""


def build_phi3_prompt(example: dict[str, Any]) -> str:
    user_prompt = build_user_prompt(example)

    return (
        "<|system|>\n"
        f"{SYSTEM_PROMPT}<|end|>\n"
        "<|user|>\n"
        f"{user_prompt}<|end|>\n"
        "<|assistant|>\n"
    )


def extract_json(text: str) -> dict[str, Any] | None:
    """
    Try to extract the first JSON object from model output.
    Handles cases like:
      ```json { ... } ```
      extra text before/after JSON
    """
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # First try direct parse.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Then try extracting first {...} block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_prediction(prediction: dict[str, Any] | None) -> bool:
    if not isinstance(prediction, dict):
        return False

    required = {"category", "root_cause", "fix", "fix_code"}
    if not required.issubset(prediction.keys()):
        return False

    if prediction.get("category") not in VALID_CATEGORIES:
        return False

    return True


def load_model(
    base_model_id: str,
    adapter_id: str | None,
    use_4bit: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None

    if use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "BitsAndBytesConfig is unavailable. Install bitsandbytes/transformers or run with --no-4bit."
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )

    if adapter_id:
        if PeftModel is None:
            raise ImportError("peft is not installed. Run: pip install peft")

        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_prediction(
    model,
    tokenizer,
    example: dict[str, Any],
    max_new_tokens: int,
) -> tuple[str, dict[str, Any] | None, float]:
    prompt = build_phi3_prompt(example)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    parsed = extract_json(raw_response)

    return raw_response, parsed, latency_ms


def evaluate(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    results = []
    correct = 0
    valid_json = 0
    total_latency_ms = 0.0
    confusion: dict[str, Counter] = defaultdict(Counter)

    for idx, example in enumerate(examples, start=1):
        expected = example.get("category")

        raw_response, parsed, latency_ms = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            example=example,
            max_new_tokens=max_new_tokens,
        )

        total_latency_ms += latency_ms

        is_valid = validate_prediction(parsed)
        valid_json += int(is_valid)

        predicted = parsed.get("category") if is_valid else None
        is_correct = predicted == expected
        correct += int(is_correct)

        confusion[expected][predicted or "INVALID_JSON"] += 1

        row = {
            "id": example.get("id"),
            "source_url": example.get("source_url"),
            "expected_category": expected,
            "predicted_category": predicted,
            "correct": is_correct,
            "valid_json": is_valid,
            "latency_ms": latency_ms,
            "error_text": example.get("error_text", ""),
            "raw_response": raw_response,
            "parsed_response": parsed,
        }

        results.append(row)

        status = "✅" if is_correct else "❌"
        print(
            f"[{idx}/{len(examples)}] {status} "
            f"id={row['id']} expected={expected} predicted={predicted} "
            f"valid_json={is_valid}"
        )

    n = len(examples)
    return {
        "num_examples": n,
        "correct": correct,
        "category_accuracy": correct / n if n else 0.0,
        "valid_json": valid_json,
        "valid_json_rate": valid_json / n if n else 0.0,
        "avg_latency_ms": total_latency_ms / n if n else 0.0,
        "results": results,
        "confusion": {
            expected: dict(predictions)
            for expected, predictions in confusion.items()
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def print_summary(report: dict[str, Any]) -> None:
    print("\nModel Evaluation Results")
    print("=" * 32)
    print(f"Examples:          {report['num_examples']}")
    print(f"Correct:           {report['correct']}")
    print(f"Category Accuracy: {report['category_accuracy']:.2%}")
    print(f"Valid JSON Rate:   {report['valid_json_rate']:.2%}")
    print(f"Avg Latency:       {report['avg_latency_ms']:.2f} ms")

    print("\nMistakes")
    print("-" * 32)

    mistakes = [row for row in report["results"] if not row["correct"]]

    if not mistakes:
        print("No mistakes.")
    else:
        for row in mistakes[:50]:
            print(f"ID:        {row['id']}")
            print(f"Expected:  {row['expected_category']}")
            print(f"Predicted: {row['predicted_category']}")
            print(f"Valid JSON:{row['valid_json']}")
            print(f"Error:     {row['error_text'][:160]}")
            print(f"Response:  {row['raw_response'][:300]}")
            print()

    print("\nConfusion Matrix")
    print("-" * 32)

    for expected, predictions in sorted(report["confusion"].items()):
        print(f"{expected}: {predictions}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a base or LoRA fine-tuned LLM on the PyTorch debugging eval set."
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
        help="Path to write per-example model results",
    )

    parser.add_argument(
        "--base-model-id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base Hugging Face model ID",
    )

    parser.add_argument(
        "--adapter-id",
        type=str,
        default=None,
        help="Optional LoRA adapter ID or local adapter path",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for smoke testing",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Max tokens to generate per example",
    )

    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantized loading",
    )

    args = parser.parse_args()

    examples = load_jsonl(args.eval_file)

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    print(f"Loaded {len(examples)} eval examples")
    print(f"Base model: {args.base_model_id}")
    print(f"Adapter:    {args.adapter_id or 'None / base model only'}")
    print(f"4-bit:      {not args.no_4bit}")

    model, tokenizer = load_model(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        use_4bit=not args.no_4bit,
    )

    report = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
    )

    write_jsonl(args.output_file, report["results"])
    print_summary(report)

    print(f"\nWrote per-example results to: {args.output_file}")


if __name__ == "__main__":
    main()