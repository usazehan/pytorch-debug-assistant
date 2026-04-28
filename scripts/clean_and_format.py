import json, re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/dataset.jsonl")

# Keywords that signal an error-related post
ERROR_KEYWORDS = [
    "error", "exception", "traceback", "runtimeerror", "valueerror",
    "typeerror", "cuda", "nan", "inf", "shape", "dimension", "mismatch",
    "failed", "crash", "oom", "out of memory", "attributeerror",
    "indexerror", "assertion", "grad", "backward", "loss"
]

def clean_html(text: str) -> str:
    if not text:
        return ""
    # Parse HTML
    soup = BeautifulSoup(text, "html.parser")
    # Preserve code blocks with markers
    for code in soup.find_all("code"):
        code.replace_with(f"\n```\n{code.get_text()}\n```\n")
    text = soup.get_text(separator="\n")
    # Clean up HTML entities that slipped through
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_markdown(text: str) -> str:
    """Light cleanup for GitHub markdown."""
    if not text:
        return ""
    # Remove image links
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_error_related(text: str) -> bool:
    """Check if the text is about a PyTorch error/bug."""
    lowered = text.lower()
    return any(kw in lowered for kw in ERROR_KEYWORDS)

def truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate to avoid excessively long inputs."""
    return text[:max_chars] if len(text) > max_chars else text

def format_prompt(input_text: str, output_text: str) -> dict:
    """Format into instruction-tuning style."""
    return {
        "instruction": "You are a PyTorch debugging assistant. Given an error message or problem description, explain the root cause and provide a fix.",
        "input": truncate(input_text.strip()),
        "output": truncate(output_text.strip()),
    }

def process_github(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in tqdm(f, desc="GitHub"):
            item = json.loads(line)
            title = item.get("title", "")
            body = clean_markdown(item.get("body", ""))
            combined = f"{title}\n\n{body}"

            if not is_error_related(combined):
                continue
            if len(body) < 50:  # skip near-empty bodies
                continue

            # For GitHub, title+body is the input; we don't have a clean answer
            # so we use it as a single-turn "problem description" example
            # (we'll enrich with comments in a later iteration)
            records.append(format_prompt(
                input_text=f"{title}\n\n{body}",
                output_text="[See issue comments for resolution]"  # placeholder
            ))
    return records

def is_high_quality_answer(answer: str, score: int) -> bool:
    """Filter for genuinely useful answers."""
    if score < 5:                    # low upvotes = likely unhelpful
        return False
    if len(answer) < 100:            # too short to be useful
        return False
    if answer.count('\n') < 2:       # no structure = likely not a real fix
        return False
    return True

def is_error_question(title: str, body: str) -> bool:
    """Stricter check — must look like an actual error, not a how-to."""
    text = (title + " " + body).lower()
    
    # Must have at least one error signal
    error_signals = [
        "error", "exception", "traceback", "runtimeerror", "valueerror",
        "typeerror", "cuda", "nan", "inf", "mismatch", "failed", "crash",
        "oom", "out of memory", "attributeerror", "indexerror", "assertion",
        "backward", "grad", "loss is nan", "not defined", "no attribute",
    ]
    has_error = any(kw in text for kw in error_signals)
    
    # Exclude generic how-to questions
    howto_signals = [
        "how to install", "how do i install", "what is the best",
        "which is better", "recommend", "tutorial", "getting started",
        "how to use", "introduction to", "difference between",
    ]
    is_howto = any(kw in text for kw in howto_signals)
    
    return has_error and not is_howto

def process_stackoverflow(path: Path) -> list:
    records = []
    skipped_score = 0
    skipped_quality = 0
    skipped_howto = 0

    with open(path) as f:
        for line in tqdm(f, desc="StackOverflow"):
            item = json.loads(line)
            title       = item.get("title", "")
            question    = clean_html(item.get("body", ""))
            answer      = clean_html(item.get("answer", ""))
            answer_score = item.get("answer_score", 0)

            # Filter 1: answer score
            if not is_high_quality_answer(answer, answer_score):
                skipped_score += 1
                continue

            # Filter 2: must be error-related, not a how-to
            if not is_error_question(title, question):
                skipped_howto += 1
                continue

            # Filter 3: question body must have substance
            if len(question) < 80:
                skipped_quality += 1
                continue

            records.append(format_prompt(
                input_text=f"{title}\n\n{question}",
                output_text=answer
            ))

    print(f"  Skipped (low answer score): {skipped_score}")
    print(f"  Skipped (how-to/not error): {skipped_howto}")
    print(f"  Skipped (low quality):      {skipped_quality}")
    print(f"  Kept: {len(records)}")
    return records

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    so_path = RAW_DIR / "stackoverflow_with_answers.jsonl"
    so_records = process_stackoverflow(so_path)
    print(f"StackOverflow: {len(so_records)} Q&A pairs")

    # Deduplicate
    seen = set()
    deduped = []
    for r in so_records:
        key = r["input"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"After dedup: {len(deduped)}")

    with open(OUT_FILE, "w") as f:
        for record in deduped:
            f.write(json.dumps(record) + "\n")

    print(f"Saved to {OUT_FILE}")
    print("\n--- Sample record ---")
    print(json.dumps(deduped[0], indent=2))
    
if __name__ == "__main__":
    main()