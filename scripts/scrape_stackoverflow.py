# scripts/scrape_stackoverflow.py
import requests, json, time, os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

SO_KEY = os.getenv("SO_API_KEY", "").strip()
if not SO_KEY or SO_KEY.startswith("your_"):
    SO_KEY = ""
OUT_FILE = Path("data/raw/stackoverflow.jsonl")

def fetch_so_questions(max_pages=50):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(OUT_FILE, "w") as f:
        for page in tqdm(range(1, max_pages + 1)):
            url = "https://api.stackexchange.com/2.3/questions"
            params = {
                "tagged": "pytorch",
                "site": "stackoverflow",
                "sort": "votes",
                "order": "desc",
                "filter": "withbody",
                "pagesize": 100,
                "page": page,
            }
            if SO_KEY:
                params["key"] = SO_KEY

            resp = requests.get(url, params=params)
            data = resp.json()

            # Print debug info on first page
            if page == 1:
                print(f"Quota remaining: {data.get('quota_remaining')}")
                print(f"Has more: {data.get('has_more')}")
                print(f"Items on page 1: {len(data.get('items', []))}")

            if "items" not in data or not data["items"]:
                print(f"Stopped at page {page}: {data.get('error_message', 'no items')}")
                break

            for q in data["items"]:
                # Remove the is_answered filter — we'll get answers separately
                record = {
                    "source": "stackoverflow",
                    "title": q.get("title", ""),
                    "body": q.get("body", ""),
                    "question_id": q.get("question_id"),
                    "score": q.get("score", 0),
                    "answer_count": q.get("answer_count", 0),
                    "tags": q.get("tags", []),
                    "is_answered": q.get("is_answered", False),
                }
                f.write(json.dumps(record) + "\n")
                total += 1

            if not data.get("has_more"):
                break

            backoff = data.get("backoff", 0)
            time.sleep(max(backoff, 1))

    print(f"Saved {total} questions to {OUT_FILE}")

if __name__ == "__main__":
    fetch_so_questions()