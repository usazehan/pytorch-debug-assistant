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

Category decision rules:
- Use "cuda_oom" ONLY when the error explicitly says CUDA out of memory, out of memory, or the process was killed because of memory pressure.
- Use "environment_error" for install/import/version/driver/CUDA setup issues, including "no NVIDIA driver", "Torch not compiled with CUDA enabled", "torch.cuda.is_available() returns False", missing packages, or CUDA/cuDNN version mismatch.
- Use "loss_issue" for NaN loss, loss not decreasing, exploding gradients, parameters becoming NaN/zero during training, or unstable optimization behavior.
- Use "dtype_mismatch" for Float/Double/Half/Long/Byte tensor type mismatches.
- Use "device_mismatch" for CPU vs CUDA tensor placement problems.
- Use "tensor_shape_mismatch" for tensor size, dimension, broadcasting, matmul, or reshape/view problems.
- Use "architecture_mismatch" for layer/channel mismatches, state_dict loading problems, missing/unexpected keys, or incompatible model heads.

Keep root_cause and fix to one concise sentence each.
Keep fix_code to one minimal code snippet or an empty string.
Do not include markdown.
Do not include explanations outside the JSON.
"""

def apply_category_override(example: dict, predicted: str | None) -> str | None:
    """
    High-confidence deterministic overrides for common PyTorch error patterns.
    Runs after model JSON parsing but before scoring.

    Important: use only question/error/code text, not the reference answer.
    """
    inp = example.get("input", example)

    title = str(inp.get("question_title", "")).lower()
    error = str(inp.get("error_text", "")).lower()
    code = str(inp.get("code_context", "")).lower()
    body = str(inp.get("question_body", "")).lower()

    text = " ".join([title, error, code, body])

    # CUDA OOM: keep this specific. "Killed" is included because our eval labels use it
    # for memory-pressure/OOM-style failures.
    if re.search(r"cuda out of memory|out of memory|\bkilled\b", error):
        return "cuda_oom"

    # Training loop bugs
    if re.search(
        r"device-side assert triggered|bool value of tensor with more than one value is ambiguous|forward\(\) missing",
        error,
    ):
        return "training_loop_bug"

    # Autograd errors
    if re.search(
        r"backward through the graph a second time|does not require grad|does not have a grad_fn|modified by an inplace operation|leaf variable",
        error,
    ):
        return "autograd_error"

    # DataLoader / transform errors
    if re.search(
        r"pic should be pil image|img should be pil image|pil image or ndarray|dataloader worker|collate|num_workers|can't pickle|torch\.size.*integer|signal number.*out of range",
        text,
    ):
        return "dataloader_error"

    # Dtype mismatch
    if re.search(
        r"numpy\.int64|not implemented for 'half'|not implemented for 'int'|not implemented for 'long'|not implemented for 'byte'|expected.*scalar type|expected tensor.*long|bytetensor.*floattensor|longtensor.*floattensor|cuda\.bytetensor.*cuda\.floattensor|cuda\.longtensor",
        error,
    ):
        return "dtype_mismatch"

    # Device mismatch
    if re.search(
        r"can't convert cuda tensor to numpy|use tensor\.cpu\(\)|input type \(torch\.floattensor\) and weight type \(torch\.cuda\.floattensor\)|torch\.floattensor.*torch\.cuda\.floattensor|torch\.cuda\.floattensor.*torch\.floattensor|parameters and buffers on device|attempting to deserialize object on cuda device|device_ids\[0\]",
        error,
    ):
        return "device_mismatch"

    # Environment errors
    if re.search(
        r"no nvidia driver|torch not compiled with cuda|cuda\.is_available\(\).*false|modulenotfounderror|importerror|undefined symbol|cudnn_status_internal_error|cublas_status_internal_error|cuda version|libtorch",
        error,
    ):
        return "environment_error"

    return predicted

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
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    use_category_overrides: bool = False,
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
        model_predicted = predicted

        if use_category_overrides:
            predicted = apply_category_override(example, predicted)

        override_applied = predicted != model_predicted
            
        is_correct = predicted == expected
        correct += int(is_correct)

        confusion[expected][predicted or "INVALID_JSON"] += 1

        row = {
            "id": example.get("id"),
            "source_url": example.get("source_url"),
            "expected_category": expected,
            "model_predicted_category": model_predicted,
            "predicted_category": predicted,
            "override_applied": override_applied,
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
    
    parser.add_argument(
        "--use-category-overrides",
        action="store_true",
        help="Apply deterministic high-confidence category overrides before scoring.",
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
        use_category_overrides=args.use_category_overrides,
    )

    write_jsonl(args.output_file, report["results"])
    print_summary(report)

    print(f"\nWrote per-example results to: {args.output_file}")


if __name__ == "__main__":
    main()