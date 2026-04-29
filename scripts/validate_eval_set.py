import json
from pathlib import Path

EVAL_FILE = Path("data/eval/eval_set.jsonl")

REQUIRED_FIELDS = {
    "id",
    "source",
    "source_url",
    "question_title",
    "question_body",
    "answer",
    "error_text",
    "stack_trace",
    "code_context",
    "category",
    "root_cause",
    "fix",
    "fix_code",
    "difficulty",
    "verified",
    "notes",
}

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

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def main():
    seen_ids = set()
    seen_urls = set()
    errors = []

    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"{EVAL_FILE} does not exist")

    rows = []
    with open(EVAL_FILE) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON - {e}")
                continue

            rows.append(row)

            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                errors.append(f"Line {line_num}: missing fields {sorted(missing)}")

            if row.get("id") in seen_ids:
                errors.append(f"Line {line_num}: duplicate id {row.get('id')}")
            seen_ids.add(row.get("id"))

            if row.get("source_url") in seen_urls:
                errors.append(f"Line {line_num}: duplicate source_url {row.get('source_url')}")
            seen_urls.add(row.get("source_url"))

            if row.get("category") not in VALID_CATEGORIES:
                errors.append(f"Line {line_num}: invalid category {row.get('category')}")

            if row.get("difficulty") not in VALID_DIFFICULTIES:
                errors.append(f"Line {line_num}: invalid difficulty {row.get('difficulty')}")

            for field in ["error_text", "root_cause", "fix"]:
                if not str(row.get(field, "")).strip():
                    errors.append(f"Line {line_num}: empty required value for {field}")

    print(f"Checked {len(rows)} examples")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()