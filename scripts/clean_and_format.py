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
    """Strip HTML tags from SO posts."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")

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

def process_stackoverflow(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in tqdm(f, desc="StackOverflow"):
            item = json.loads(line)
            title = item.get("title", "")
            question = clean_html(item.get("body", ""))
            answer = clean_html(item.get("answer", ""))

            if not answer or len(answer) < 50:
                continue
            if not is_error_related(f"{title} {question}"):
                continue

            records.append(format_prompt(
                input_text=f"{title}\n\n{question}",
                output_text=answer
            ))
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