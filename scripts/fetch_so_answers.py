import requests, json, time, os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

SO_KEY = os.getenv("SO_API_KEY", "")
IN_FILE = Path("data/raw/stackoverflow.jsonl")
OUT_FILE = Path("data/raw/stackoverflow_with_answers.jsonl")

def fetch_answers():
    questions = [json.loads(l) for l in open(IN_FILE)]
    ids = [str(q["question_id"]) for q in questions]
    id_to_question = {q["question_id"]: q for q in questions}
    
    # Batch requests (SO allows up to 100 IDs at once)
    batch_size = 100
    answers_map = {}
    
    for i in tqdm(range(0, len(ids), batch_size)):
        batch = ";".join(ids[i:i+batch_size])
        url = f"https://api.stackexchange.com/2.3/questions/{batch}/answers"
        params = {
            "site": "stackoverflow",
            "sort": "votes",
            "filter": "withbody",
            "pagesize": 100,
            "key": SO_KEY,
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        
        for answer in data.get("items", []):
            qid = answer["question_id"]
            # Keep only the top answer per question
            if qid not in answers_map or answer["score"] > answers_map[qid]["score"]:
                answers_map[qid] = answer
        
        time.sleep(1)
    
    # Merge and write
    with open(OUT_FILE, "w") as f:
        for q in questions:
            qid = q["question_id"]
            if qid in answers_map:
                q["answer"] = answers_map[qid].get("body", "")
                q["answer_score"] = answers_map[qid].get("score", 0)
                f.write(json.dumps(q) + "\n")
    
    print(f"Saved {len(answers_map)} Q&A pairs to {OUT_FILE}")

if __name__ == "__main__":
    fetch_answers()