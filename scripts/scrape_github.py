import requests, json, time, os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
OUT_FILE = Path("data/raw/github_issues.jsonl")

def fetch_issues(max_pages=10):  # 10 pages = 1000 issues (search API max)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(OUT_FILE, "w") as f:
        for page in tqdm(range(1, max_pages + 1)):
            url = "https://api.github.com/search/issues"
            params = {
                "q": "repo:pytorch/pytorch is:issue is:closed",
                "per_page": 100,
                "page": page,
            }

            while True:  # retry loop for rate limiting
                resp = requests.get(url, headers=HEADERS, params=params)
                remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
                reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time()))

                if resp.status_code == 403 or remaining == 0:
                    wait = max(reset_time - int(time.time()), 0) + 5
                    print(f"\nRate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                    continue  # retry same page

                if resp.status_code == 422:
                    print("Search pagination limit reached (1000 max)")
                    return
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}: {resp.text}")
                    return

                break  # success, exit retry loop

            data = resp.json()
            items = data.get("items", [])

            if not items:
                print(f"No items at page {page}")
                break

            for issue in items:
                body = issue.get("body") or ""
                record = {
                    "source": "github",
                    "title": issue.get("title", ""),
                    "body": body,
                    "comments_url": issue.get("comments_url", ""),
                    "issue_url": issue.get("html_url", ""),
                    "labels": [l["name"] for l in issue.get("labels", [])],
                }
                f.write(json.dumps(record) + "\n")
                total += 1

            print(f"Page {page}: {total} total | Rate limit remaining: {remaining}")
            time.sleep(3)  # stay well under 30 req/min

    print(f"\nDone. Saved {total} issues to {OUT_FILE}")

if __name__ == "__main__":
    fetch_issues()