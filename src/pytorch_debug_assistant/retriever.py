import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class PyTorchIssueRetriever:
    def __init__(self, index_dir: str = "data/index", model_name: str | None = None):
        """
        Loads a FAISS index, metadata, and the embedding model used to build the index.
        """
        self.index_dir = Path(index_dir)
        self.faiss_index_path = self.index_dir / "faiss.index"
        self.metadata_path = self.index_dir / "metadata.jsonl"
        self.config_path = self.index_dir / "config.json"

        if not self.faiss_index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Index or metadata not found in {self.index_dir}. "
                "Please run scripts/build_rag_index.py first."
            )

        self.config = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.model_name = (
            model_name
            or self.config.get("embedding_model")
            or "all-MiniLM-L6-v2"
        )

        print(f"Loading FAISS index from {self.faiss_index_path}")
        self.index = faiss.read_index(str(self.faiss_index_path))

        print(f"Loading metadata from {self.metadata_path}")
        self.metadata: list[dict[str, Any]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) does not match "
                f"metadata size ({len(self.metadata)}). Rebuild the index."
            )

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def retrieve_similar_issues(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Returns the most similar historical PyTorch issues for a query.
        Similarity is cosine similarity because embeddings are normalized and FAISS uses inner product.
        """
        query = query.strip()
        if not query:
            return []

        top_k = min(top_k, self.index.ntotal)

        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_emb = np.asarray(query_emb, dtype="float32")

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            results.append(
                {
                    "id": meta.get("id", ""),
                    "source_url": meta.get("source_url", ""),
                    "title": meta.get("title", ""),
                    "error_text": meta.get("error_text", ""),
                    "category": meta.get("category", ""),
                    "answer": meta.get("answer", ""),
                    "similarity": round(float(score), 4),
                }
            )

        return results


def build_query(
    question_title: str = "",
    error_text: str = "",
    code_context: str = "",
    question_body: str = "",
) -> str:
    """
    Builds a retrieval query from user-provided debugging context.
    """
    return "\n".join(
        part.strip()
        for part in [question_title, error_text, code_context, question_body]
        if part and part.strip()
    )


if __name__ == "__main__":
    retriever = PyTorchIssueRetriever()

    test_query = build_query(
        question_title="CUDA out of memory while training",
        error_text="RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB",
        code_context="loss.backward()",
    )

    print(f"Testing retrieval for query:\n{test_query}\n")

    top_issues = retriever.retrieve_similar_issues(test_query, top_k=3)
    print(json.dumps(top_issues, indent=2))