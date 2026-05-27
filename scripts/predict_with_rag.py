import argparse
import sys
from pathlib import Path

# Add the 'src' directory to the Python path so we can import the retriever
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pytorch_debug_assistant.retriever import PyTorchIssueRetriever, build_query
except ImportError as e:
    print(f"ImportError: {e}")
    print("Ensure you are running this from the root of the repository.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG retrieval for PyTorch debugging issues."
    )

    parser.add_argument("--title", type=str, default="", help="The title or summary of the issue.")
    parser.add_argument("--error", type=str, default="", help="The exact error traceback or message.")
    parser.add_argument("--body", type=str, default="", help="Text context explaining what the user is trying to do.")
    parser.add_argument("--code", type=str, default="", help="The snippet of code causing the issue.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of similar issues to retrieve.")

    args = parser.parse_args()

    query = build_query(
        question_title=args.title,
        error_text=args.error,
        code_context=args.code,
        question_body=args.body,
    )

    if not query:
        print("Error: Please provide at least one input: --title, --error, --body, or --code.")
        parser.print_help()
        sys.exit(1)

    if args.top_k <= 0:
        print("Error: --top-k must be greater than 0.")
        sys.exit(1)

    print("Loading retriever and FAISS index...")
    try:
        retriever = PyTorchIssueRetriever()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please generate the index first: python scripts/build_rag_index.py")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("QUERY")
    print("=" * 50)
    print(query)
    print("=" * 50 + "\n")

    results = retriever.retrieve_similar_issues(query, top_k=args.top_k)

    if not results:
        print("No similar issues found in the index.")
        sys.exit(0)

    print(f"Top {len(results)} Retrieved Issues:\n")

    for i, res in enumerate(results, 1):
        print(f"[{i}] Similarity Score: {res.get('similarity', 'N/A')}")
        print(f"URL:      {res.get('source_url', 'N/A')}")
        print(f"Title:    {res.get('title', 'N/A')}")
        print(f"Category: {res.get('category', 'N/A')}")
        print(f"Error:    {res.get('error_text', 'N/A')}")
        print("--- Retrieved Fix Context ---")
        print(res.get("answer", "N/A"))
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()