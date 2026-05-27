import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/processed/structured_train.jsonl")
INDEX_DIR = Path("data/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
CONFIG_PATH = INDEX_DIR / "config.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Reading data from {DATA_FILE}...")
    texts_to_embed = []
    metadata = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            inp = row["input"]
            out = row["output"]

            title = inp.get("question_title", "")
            error_text = inp.get("error_text", "")
            body = inp.get("question_body", "")

            answer = (
                f"Root Cause: {out.get('root_cause', '')}\n"
                f"Fix: {out.get('fix', '')}\n"
                f"Code:\n{out.get('fix_code', '')}"
            ).strip()

            # Embed only the problem/query side.
            # The answer is stored in metadata and used later as retrieved context.
            doc_text = f"{title}\n{error_text}\n{body}".strip()

            if not doc_text:
                continue

            texts_to_embed.append(doc_text)

            metadata.append(
                {
                    "id": row.get("id", ""),
                    "source_url": row.get("source_url", ""),
                    "title": title,
                    "error_text": error_text,
                    "category": out.get("category", ""),
                    "answer": answer,
                }
            )

    if not texts_to_embed:
        raise ValueError(f"No examples found in {DATA_FILE}")

    print(f"Encoding {len(texts_to_embed)} examples...")

    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim}...")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saving config to {CONFIG_PATH}...")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": MODEL_NAME,
                "num_examples": len(metadata),
                "faiss_index": str(FAISS_INDEX_PATH),
                "metadata": str(METADATA_PATH),
            },
            f,
            indent=2,
        )

    print("RAG index successfully built!")


if __name__ == "__main__":
    main()