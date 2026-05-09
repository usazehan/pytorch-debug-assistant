import html
import re
import json
import os
import random
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_FILE   = Path("data/raw/stackoverflow_with_answers.jsonl")
EVAL_FILE  = Path("data/eval/eval_set.jsonl")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=API_KEY) if API_KEY else None

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

SYSTEM_PROMPT = """You are an expert PyTorch engineer labeling a debugging dataset.
Given a Stack Overflow question and answer about PyTorch, extract structured information.

Respond ONLY with a JSON object, no markdown, no explanation:
{
  "category": one of the exact category strings listed below,
  "root_cause": "one sentence plain English explanation",
  "fix": "one sentence description of the fix",
  "fix_code": "minimal code snippet that resolves the issue",
  "difficulty": "easy" or "medium" or "hard",
  "error_text": "exact error message if present, else brief description"
}

Categories:
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
- architecture_mismatch"""

# ── helpers ──────────────────────────────────────────────────────────────────
VALID_CATEGORIES = set(CATEGORIES.values()) - {"SKIP"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

def validate_suggestion(suggestion: dict) -> bool:
    required = {
        "category",
        "root_cause",
        "fix",
        "fix_code",
        "difficulty",
        "error_text",
    }

    if not isinstance(suggestion, dict):
        return False

    if not required.issubset(suggestion.keys()):
        return False

    if suggestion["category"] not in VALID_CATEGORIES:
        return False

    if suggestion["difficulty"] not in VALID_DIFFICULTIES:
        return False

    return True

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
    "runtimeerror", "valueerror", "typeerror", "attributeerror",
    "indexerror", "assertionerror", "modulenotfounderror", "importerror",
    "cuda out of memory", "out of memory", "expected all tensors",
    "should be the same", "expected object of scalar type",
    "found no nvidia driver", "torch not compiled with cuda enabled",
    "no module named", "size mismatch", "shape mismatch", "mat1 and mat2",
    "does not require grad", "grad can be implicitly created",
    "input is not contiguous", "can't assign", "cannot convert",
    "can't convert", "nan", "exploding gradients", "traceback",
    "exception", "error:",
]

def load_candidates():
    candidates = []
    with open(RAW_FILE) as f:
        for line in f:
            item = json.loads(line)
            text = (
                item.get("title", "") + " " +
                item.get("body", "") + " " +
                item.get("answer", "")
            ).lower()
            if any(kw in text for kw in ERROR_KEYWORDS):
                candidates.append(item)
    random.shuffle(candidates)
    return candidates

LABEL_SCHEMA = {
    "name": "pytorch_debug_label",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {
                "type": "string",
                "enum": [
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
                ],
            },
            "root_cause": {"type": "string"},
            "fix": {"type": "string"},
            "fix_code": {"type": "string"},
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
            },
            "error_text": {"type": "string"},
        },
        "required": [
            "category",
            "root_cause",
            "fix",
            "fix_code",
            "difficulty",
            "error_text",
        ],
    },
}

# ── auto-labeling ─────────────────────────────────────────────────────────────
def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()

def autolabel(item) -> dict | None:
    """Call OpenAI API to label the example. Returns None on failure."""
    if client is None:
        return None

    user_prompt = (
        f"Title: {item.get('title', '')}\n\n"
        f"Question: {clean_html(item.get('body', ''))[:1500]}\n\n"
        f"Answer: {clean_html(item.get('answer', ''))[:1500]}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": LABEL_SCHEMA,
            },
            temperature=0,
            max_tokens=500,
        )

        message = resp.choices[0].message

        if getattr(message, "refusal", None):
            print(f"  ⚠️  Auto-label refused: {message.refusal}")
            return None

        text = message.content
        if not text:
            print("  ⚠️  Auto-label returned empty content")
            return None

        parsed = json.loads(text)

        if not validate_suggestion(parsed):
            print("  ⚠️  Auto-label returned invalid schema")
            return None

        return parsed

    except Exception as e:
        print(f"  ⚠️  Auto-label failed: {e}")
        return None

# ── labeling UI ───────────────────────────────────────────────────────────────

def label_example(item, idx, total, suggestion: dict | None = None):
    title  = clean_html(item.get("title", ""))
    body   = clean_html(item.get("body", ""))
    answer = clean_html(item.get("answer", ""))

    print("\n" + "="*60)
    print(f"Example {idx}/{total}")
    print(f"URL: https://stackoverflow.com/questions/{item['question_id']}")
    print(f"\nTITLE: {title}")
    print(f"\nQUESTION (first 500 chars):\n{body[:500]}")
    print(f"\nANSWER (first 500 chars):\n{answer[:500]}")

    if suggestion:
        print("\n🤖 OpenAI suggests:")
        print(f"  Category:   {suggestion.get('category', '')}")
        print(f"  Difficulty: {suggestion.get('difficulty', '')}")
        print(f"  Error:      {suggestion.get('error_text', '')[:80]}")
        print(f"  Root cause: {suggestion.get('root_cause', '')[:80]}")
        print(f"  Fix:        {suggestion.get('fix', '')[:80]}")
        print(f"  Fix code:   {suggestion.get('fix_code', '')[:60]}")
        action = input("\nAccept (a) / Edit (e) / Skip (s): ").strip().lower()

        if action == "s":
            return None

        elif action == "e":
            # Fall through to manual entry below with defaults pre-filled
            pass

        elif action == "a":
            # Accept Claude's suggestion
            verified_input = input("Mark as verified? y/n [n]: ").strip().lower()

            return {
                "id": f"so_{item['question_id']}",
                "source": "stackoverflow",
                "source_url": f"https://stackoverflow.com/questions/{item['question_id']}",
                "question_title": item.get("title", ""),
                "question_body": item.get("body", ""),
                "answer": item.get("answer", ""),
                "error_text": suggestion["error_text"],
                "stack_trace": "",
                "code_context": "",
                "category": suggestion["category"],
                "root_cause": suggestion["root_cause"],
                "fix": suggestion["fix"],
                "fix_code": suggestion["fix_code"],
                "difficulty": suggestion["difficulty"],
                "verified": verified_input == "y",
                "notes": "openai-suggested; human-accepted",
            }

        else:
            print("Invalid choice. Skipping this example.")
            return None
    else:
        print("\nCATEGORIES:")
        for k, v in CATEGORIES.items():
            print(f"  {k}: {v}")
        cat_input = input("\nCategory (number) or 's' to skip: ").strip()
        if cat_input == "s" or cat_input not in CATEGORIES:
            return None
        suggestion = {
            "category":   CATEGORIES[cat_input],
            "difficulty": "medium",
            "error_text": "",
            "root_cause": "",
            "fix": "",
            "fix_code": "",
        }

    # Manual entry (with optional defaults from suggestion)
    print("\nDIFFICULTY: 1=easy, 2=medium, 3=hard")
    diff_input   = input("Difficulty: ").strip()
    difficulty   = DIFFICULTY.get(diff_input, suggestion.get("difficulty", "medium"))
    error_text   = input(f"\nError message [{suggestion.get('error_text','')[:40]}]: ").strip() or suggestion.get("error_text", "")
    root_cause   = input(f"Root cause [{suggestion.get('root_cause','')[:40]}]: ").strip() or suggestion.get("root_cause", "")
    fix          = input(f"Fix [{suggestion.get('fix','')[:40]}]: ").strip() or suggestion.get("fix", "")
    fix_code     = input(f"Fix code [{suggestion.get('fix_code','')[:40]}]: ").strip() or suggestion.get("fix_code", "")
    code_context = input("Code context (or Enter to skip): ").strip()
    notes        = input("Notes: ").strip()
    verified     = input("Verified? y/n [n]: ").strip().lower() == "y"

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
        "category": suggestion.get("category", "training_loop_bug"),
        "root_cause": root_cause,
        "fix": fix,
        "fix_code": fix_code,
        "difficulty": difficulty,
        "verified": verified,
        "notes": notes,
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    existing_ids   = load_existing_ids()
    candidates     = load_candidates()
    candidates     = [c for c in candidates
                      if f"https://stackoverflow.com/questions/{c['question_id']}"
                      not in existing_ids]

    existing_count = len(existing_ids)
    labeled        = 0
    session_limit  = 20  # do 20 per session

    auto_mode = API_KEY is not None
    print(f"Found {len(candidates)} candidates")
    print(f"Already labeled: {existing_count}/100")
    print(f"Auto-labeling: {'✅ ON (OpenAI API)' if auto_mode else '❌ OFF (no API key)'}\n")
    
    with open(EVAL_FILE, "a") as f:
        for item in candidates:
            if labeled >= session_limit:
                print(f"\nSession complete! Run again to continue.")
                break

            suggestion = autolabel(item) if auto_mode else None
            result     = label_example(
                item,
                existing_count + labeled + 1,
                100,
                suggestion=suggestion,
            )

            if result:
                f.write(json.dumps(result) + "\n")
                labeled += 1
                print(f"✅ Saved! ({existing_count + labeled}/100)")

    print(f"\nTotal labeled: {existing_count + labeled}/100")

if __name__ == "__main__":
    main()