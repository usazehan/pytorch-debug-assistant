import html
import re
import json
import random
from pathlib import Path

RAW_FILE   = Path("data/raw/stackoverflow_with_answers.jsonl")
EVAL_FILE  = Path("data/eval/eval_set.jsonl")

CATEGORIES = {
    "1": "tensor_shape_mismatch",
    "2": "cuda_oom",
    "3": "device_mismatch",
    "4": "dtype_mismatch",
    "5": "autograd_error",
    "6": "dataloader_error",
    "7": "loss_issue",
    "8": "environment_error",
    "9": "optimizer_error",
    "10": "training_loop_bug",
    "11": "architecture_mismatch",
    "s": "SKIP",
}

DIFFICULTY = {"1": "easy", "2": "medium", "3": "hard"}

def clean_html(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

def load_existing_ids():
    if not EVAL_FILE.exists() or EVAL_FILE.stat().st_size == 0:
        return set()

    ids = set()

    with open(EVAL_FILE) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
                continue

            source_url = item.get("source_url")
            if source_url:
                ids.add(source_url)

    return ids

ERROR_KEYWORDS = [
    "runtimeerror",
    "valueerror",
    "typeerror",
    "attributeerror",
    "indexerror",
    "assertionerror",
    "cuda out of memory",
    "expected all tensors",
    "should be the same",
    "size mismatch",
    "shape mismatch",
    "mat1 and mat2",
    "does not require grad",
    "grad can be implicitly created",
    "input is not contiguous",
    "traceback",
    "exception",
    "error:"
]

def load_candidates():
    candidates = []
    with open(RAW_FILE) as f:
        for line in f:
            item = json.loads(line)
            # Only keep items with error signals
            text = (item.get("title", "") + item.get("body", "")).lower()
            if any(kw in text for kw in ERROR_KEYWORDS):
                candidates.append(item)
    random.shuffle(candidates)
    return candidates

def label_example(item, idx, total):
    title = clean_html(item.get("title", ""))
    body = clean_html(item.get("body", ""))
    answer = clean_html(item.get("answer", ""))
    print("\n" + "="*60)
    print(f"Example {idx}/{total}")
    print(f"URL: https://stackoverflow.com/questions/{item['question_id']}")
    print(f"\nTITLE: {title}")
    print(f"\nQUESTION (first 500 chars):\n{body[:500]}")
    print(f"\nANSWER (first 500 chars):\n{answer[:500]}")
    print("\nCATEGORIES:")
    for k, v in CATEGORIES.items():
        print(f"  {k}: {v}")

    cat_input = input("\nCategory (number) or 's' to skip: ").strip()
    if cat_input == "s" or cat_input not in CATEGORIES:
        return None

    category = CATEGORIES[cat_input]

    print("\nDIFFICULTY: 1=easy, 2=medium, 3=hard")
    diff_input = input("Difficulty: ").strip()
    difficulty = DIFFICULTY.get(diff_input, "medium")

    error_text = input("\nError message (paste the exact error): ").strip()
    root_cause = input("Root cause (one sentence): ").strip()
    fix        = input("Fix (one sentence): ").strip()
    fix_code   = input("Fix code (one line, or press Enter to skip): ").strip()
    code_context = input("Code context/snippet, or press Enter to skip: ").strip()
    notes = input("Notes, ambiguity, or why skipped? ").strip()
    verified_input = input("Verified fix? y/n: ").strip().lower()
    verified = verified_input == "y"

    return {
    "id": f"so_{item['question_id']}",
    "source": "stackoverflow",
    "source_url": f"https://stackoverflow.com/questions/{item['question_id']}",
    "question_title": item.get("title", ""),
    "question_body": item.get("body", ""),
    "answer": item.get("answer", ""),
    "error_text": error_text,
    "stack_trace": "",
    "code_context": code_context,
    "category": category,
    "root_cause": root_cause,
    "fix": fix,
    "fix_code": fix_code,
    "difficulty": difficulty,
    "verified": verified,
    "notes": notes,
}

def main():
    existing_ids = load_existing_ids()
    candidates   = load_candidates()
    candidates   = [c for c in candidates
                    if f"https://stackoverflow.com/questions/{c['question_id']}"
                    not in existing_ids]

    print(f"Found {len(candidates)} candidates to label")
    print("Label 10 examples per session. Press Ctrl+C to stop anytime.\n")

    # Load existing count for ID numbering
    existing_count = len(existing_ids)
    labeled = 0

    with open(EVAL_FILE, "a") as f:
        for i, item in enumerate(candidates):
            result = label_example(item, existing_count + labeled + 1, 100)
            if result:
                f.write(json.dumps(result) + "\n")
                labeled += 1
                print(f"✅ Saved! ({labeled} labeled this session)")

            if labeled >= 10:
                print("\nGood session! Run again tomorrow to continue.")
                break

    print(f"\nTotal labeled: {existing_count + labeled}/100")

if __name__ == "__main__":
    main()