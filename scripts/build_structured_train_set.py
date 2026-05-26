import json
import re
import html
from pathlib import Path
from collections import Counter

# --- Configuration ---
RAW_FILE = Path("data/raw/stackoverflow_with_answers.jsonl")
EVAL_FILE = Path("data/eval/eval_set.jsonl")
OUTPUT_FILE = Path("data/processed/structured_train.jsonl")

ALLOWED_CATEGORIES = {
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

# High-precision mapping from regex patterns to categories
CATEGORY_RULES = {
    r"cuda out of memory|out of memory|Killed": "cuda_oom",

    r"can't convert CUDA tensor to numpy|cannot convert CUDA tensor to numpy|Use Tensor\.cpu\(\)|Expected all tensors to be on the same device|Input type \(torch\.FloatTensor\) and weight type \(torch\.cuda\.FloatTensor\)": "device_mismatch",

    r"Expected.*scalar type.*but got|Found dtype.*but expected|Expected tensor for argument.*to have scalar type|mat1 and mat2 must have the same dtype|not implemented for 'Half'|ByteTensor.*FloatTensor|LongTensor.*FloatTensor|FloatTensor.*LongTensor": "dtype_mismatch",

    # More specific architecture/config mismatch rules should come before broad shape rules
    r"Error\(s\) in loading state_dict|missing key|unexpected key|size mismatch for": "architecture_mismatch",

    r"axial_pos_shape|sequence_length|config\..*shape|unexpected keyword argument 'labels'": "architecture_mismatch",

    r"size mismatch|must match the size of tensor|doesn't match the broadcast shape|Target size.*must be the same as input size|mat1 and mat2 shapes cannot be multiplied|dimension out of range|input is not contiguous|Invalid dimensions for image data": "tensor_shape_mismatch",

    r"modified by an inplace operation|backward through the graph a second time|does not require grad|grad can be implicitly created only for scalar outputs": "autograd_error",

    # More specific image transform/DataLoader rule should come before broader DataLoader rule
    r"img should be PIL Image|pic should be PIL Image|PIL Image or ndarray": "dataloader_error",

    r"DataLoader worker.*exited unexpectedly|collate|num_workers|Can't pickle|pic should be PIL Image or ndarray|too many indices for array": "dataloader_error",

    r"Given groups.*expected input.*to have.*channels|unexpected keyword argument 'labels'|missing key|unexpected key|load_state_dict": "architecture_mismatch",

    r"Torch not compiled with CUDA enabled|No module named ['\"]?torch|ModuleNotFoundError|ImportError|torch\.cuda\.is_available\(\) returns False|Found no NVIDIA driver|DLL load failed|libc10\.so|CUDA initialization|CUDNN_STATUS_INTERNAL_ERROR|CUBLAS_STATUS_INTERNAL_ERROR": "environment_error",

    r"optimizer got an empty parameter list|some parameters appear in more than one parameter group|loaded state dict contains a parameter group": "optimizer_error",

    r"device-side assert triggered|CrossEntropyLoss\(|Bool value of Tensor with more than one value is ambiguous|forward\(\) missing": "training_loop_bug",

    r"nan|loss is nan|loss not decreasing|exploding gradients|gradient clipping": "loss_issue",

    r"multi-target not supported.*ClassNLLCriterion": "tensor_shape_mismatch",
}

# --- Utility Functions ---

def clean_html(text: str) -> str:
    """Removes HTML tags and unescapes entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

def extract_error_text(body: str) -> str:
    text = clean_html(body)

    patterns = [
        r"(RuntimeError:.*)",
        r"(ValueError:.*)",
        r"(TypeError:.*)",
        r"(AttributeError:.*)",
        r"(ImportError:.*)",
        r"(ModuleNotFoundError:.*)",
        r"(AssertionError:.*)",
        r"(UserWarning:.*)",
        r"(CUDA out of memory.*)",
        r"(CUDNN_STATUS_[A-Z_]+.*)",
        r"(CUBLAS_STATUS_[A-Z_]+.*)",
        r"(No module named [^\n]+)",
        r"(Killed)",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    if matches:
        return matches[-1].strip()

    return ""

def extract_code_context(body: str) -> str:
    """Extracts code blocks from the question body."""
    code_blocks = re.findall(r"<pre.*?><code.*?>(.*?)</code></pre>", body, re.DOTALL)
    if code_blocks:
        # Join multiple code blocks, or just take the longest one (likely the script)
        longest_block = max(code_blocks, key=len)
        return clean_html(longest_block)
    return ""

def assign_category(title: str, body: str, error_text: str) -> str:
    """Assigns a category based on strict keyword heuristics.

    Prefer high-signal text: explicit error message first, then title.
    Avoid scanning the full body for category assignment because it often
    contains incidental words from code, attempts, or explanations.
    """

    high_signal_text = f"{error_text}\n{title}"

    for pattern, category in CATEGORY_RULES.items():
        if re.search(pattern, high_signal_text, re.IGNORECASE):
            return category

    return None

def extract_answer_components(answer_html: str):
    """Splits the accepted answer into root cause, fix, and fix code."""
    # 1. Extract fix code
    fix_code = ""
    code_blocks = re.findall(r"<pre.*?><code.*?>(.*?)</code></pre>", answer_html, re.DOTALL)
    if code_blocks:
        fix_code = clean_html(code_blocks[0])
    elif "<code>" in answer_html:
        inline_codes = re.findall(r"<code>(.*?)</code>", answer_html)
        if inline_codes:
            fix_code = clean_html(inline_codes[0])

    # 2. Extract text for root cause / fix logic
    clean_text = clean_html(answer_html)
    # Strip out the code we already extracted from the text to isolate explanations
    if fix_code:
        clean_text = clean_text.replace(fix_code, "")
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
    
    # Heuristic: First 1-2 sentences describe the "Why" (Root Cause)
    # The rest describes the "How" (Fix)
    if len(sentences) == 0:
        root_cause = "Code issue identified."
        fix = "Update implementation as shown in the code."
    elif len(sentences) <= 2:
        root_cause = " ".join(sentences)
        fix = "Apply the corrected code to resolve the issue."
    else:
        root_cause = " ".join(sentences[:2])
        fix = " ".join(sentences[2:])

    return root_cause, fix, fix_code

def useful_text(text: str, min_chars: int = 25) -> bool:
    cleaned = re.sub(r"[`\s]+", "", text or "")
    return len(cleaned) >= min_chars

def validate_row(row: dict) -> bool:
    """Validates that the output JSON matches the strict schema and has useful training content."""

    # Top-level schema
    required_top_level = ["id", "source", "source_url", "input", "output"]
    if not all(k in row for k in required_top_level):
        return False

    if not isinstance(row["input"], dict) or not isinstance(row["output"], dict):
        return False

    # Input schema
    inp = row["input"]
    required_input = ["question_title", "question_body", "error_text", "code_context"]
    if not all(k in inp for k in required_input):
        return False

    # Require at least some useful input text
    if not inp["question_title"].strip() and not inp["question_body"].strip():
        return False

    # Require either an explicit error or code context.
    # This helps avoid generic conceptual Q&A examples.
    if not inp["error_text"].strip() and not inp["code_context"].strip():
        return False

    # Output schema
    out = row["output"]
    required_output = ["category", "root_cause", "fix", "fix_code"]
    if not all(k in out for k in required_output):
        return False

    if out["category"] not in ALLOWED_CATEGORIES:
        return False

    # Reject empty or generic weak targets
    if not useful_text(out["root_cause"], min_chars=30):
        return False

    if not useful_text(out["fix"], min_chars=20):
        return False

    weak_root_causes = {
        "Code issue identified.",
        "Issue identified.",
        "An issue was found in the code.",
    }

    weak_fixes = {
        "Apply the corrected code to resolve the issue.",
        "Update implementation as shown in the code.",
        "Fix the code.",
    }

    if out["root_cause"].strip() in weak_root_causes:
        return False

    if out["fix"].strip() in weak_fixes:
        return False

    return True

ERROR_SIGNAL_RE = re.compile(
    r"RuntimeError|ValueError|TypeError|AttributeError|ImportError|"
    r"ModuleNotFoundError|AssertionError|UserWarning|CUDA out of memory|"
    r"CUDNN_STATUS|CUBLAS_STATUS|device-side assert|No module named|"
    r"Killed|loss is nan|loss not decreasing|exploding gradients|"
    r"expected scalar type|found dtype|size mismatch|shape mismatch|"
    r"must match the size|not implemented for",
    re.IGNORECASE,
)

def has_error_signal(title: str, body: str, error_text: str) -> bool:
    combined = clean_html(f"{title}\n{body}\n{error_text}")
    return bool(ERROR_SIGNAL_RE.search(combined))

def compact_text(text: str, max_chars: int = 240) -> str:
    """Compact natural-language text without cutting mid-sentence when possible."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("` `", "").strip()

    if len(text) <= max_chars:
        return text

    # Prefer complete sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []

    for sentence in sentences:
        candidate = " ".join(kept + [sentence]).strip()
        if len(candidate) <= max_chars:
            kept.append(sentence)
        else:
            break

    if kept:
        return " ".join(kept).strip()

    # Fallback: cut at word boundary
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "."

def compact_code(code: str, max_chars: int = 300, max_lines: int = 8) -> str:
    """Compact code while preserving line breaks."""
    code = html.unescape(code or "").strip()

    if not code:
        return ""

    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    code = "\n".join(lines[:max_lines]).strip()

    if len(code) <= max_chars:
        return code

    return code[:max_chars].rsplit("\n", 1)[0].strip()

# --- Main Script ---

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Eval URLs to exclude
    eval_urls = set()
    seen_urls = set()
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if "source_url" in data:
                    eval_urls.add(data["source_url"])
    print(f"Loaded {len(eval_urls)} URLs from eval set for exclusion.")

    # 2. Process Raw Data
    stats = Counter()
    processed_rows = []

    with open(RAW_FILE, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile):
            if not line.strip(): continue
            
            raw_data = json.loads(line)
            stats["total_raw_rows"] += 1
            
            # Construct URL and check exclusion
            question_id = raw_data.get("question_id")
            if not question_id:
                stats["skipped_missing_question_id"] += 1
                continue
            
            source_url = f"https://stackoverflow.com/questions/{question_id}"
            if source_url in eval_urls:
                stats["excluded_eval_overlap"] += 1
                continue
            
            if source_url in seen_urls:
                stats["skipped_duplicate"] += 1
                continue

            seen_urls.add(source_url)
                        
                
            title = raw_data.get("title", "")
            body = raw_data.get("body", "")
            answer = raw_data.get("answer", "")
            
            # Extract Input fields
            error_text = extract_error_text(f"{title}\n{body}")
            code_context = extract_code_context(body)
            
            # Filter for explicit error signals in the question (title/body) or extracted error text
            if not has_error_signal(title, body, error_text):
                stats["skipped_no_error_signal"] += 1
                continue
            
            # Filter for generic PyTorch relevance if an explicit error wasn't found
            combined_raw = f"{title} {body} {answer}".lower()
            if not any(term in combined_raw for term in [
                "torch", "pytorch", "cuda", "cudnn", "tensor", "nn.", "dataloader"
            ]):
                stats["skipped_not_pytorch"] += 1
                continue
                
            # Assign Category (Skip if Ambiguous)
            category = assign_category(title, body, error_text)
            if not category:
                stats["skipped_ambiguous_category"] += 1
                continue
                
            # Extract Output fields from the accepted answer
            root_cause, fix, fix_code = extract_answer_components(answer)
            
            root_cause = compact_text(root_cause, max_chars=220)
            fix = compact_text(fix, max_chars=240)
            fix_code = compact_code(fix_code, max_chars=300, max_lines=8)
            
            # Construct JSONL row
            structured_row = {
                "id": f"so_{question_id}",
                "source": "stackoverflow",
                "source_url": source_url,
                "input": {
                    "question_title": clean_html(title),
                    "question_body": clean_html(body),
                    "error_text": error_text,
                    "code_context": code_context
                },
                "output": {
                    "category": category,
                    "root_cause": root_cause,
                    "fix": fix,
                    "fix_code": fix_code
                }
            }
            
            # Validation
            if validate_row(structured_row):
                processed_rows.append(structured_row)
                stats[f"cat_{category}"] += 1
            else:
                stats["failed_validation"] += 1

    # 3. Write structured train set
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for row in processed_rows:
            outfile.write(json.dumps(row) + "\n")

    # 4. Print Summary
    print("\n--- Processing Summary ---")
    print(f"Total Raw Rows Processed:   {stats['total_raw_rows']}")
    print(f"Excluded (Eval Overlap):    {stats['excluded_eval_overlap']}")
    print(f"Skipped (Duplicate):        {stats['skipped_duplicate']}")
    print(f"Skipped (Not PyTorch):      {stats['skipped_not_pytorch']}")
    print(f"Skipped (Ambiguous Cat):    {stats['skipped_ambiguous_category']}")
    print(f"Skipped (No Error Signal):  {stats['skipped_no_error_signal']}")
    print(f"Failed Validation:          {stats['failed_validation']}")
    print(f"Total Training Rows Built:  {len(processed_rows)}")
    
    print("\n--- Category Distribution ---")
    for cat in sorted(ALLOWED_CATEGORIES):
        count = stats.get(f"cat_{cat}", 0)
        print(f"  {cat.ljust(25)}: {count}")
    print(f"\nSuccessfully wrote outputs to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()