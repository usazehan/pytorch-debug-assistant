import argparse
import sys
from pathlib import Path

# Add src to Python path so this script works from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pytorch_debug_assistant.pipeline import (
        build_rag_prompt,
        generate_fix,
        retrieve_context,
    )
except ImportError as e:
    print(f"ImportError: {e}")
    print("Ensure you are running this from the root of the repository.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end PyTorch debugging CLI with RAG + Phi-3 generation."
    )

    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="User's title or short summary of the issue.",
    )

    parser.add_argument(
        "--error",
        type=str,
        default="",
        help="The error traceback or message.",
    )

    parser.add_argument(
        "--code",
        type=str,
        default="",
        help="Code snippet causing the issue.",
    )

    parser.add_argument(
        "--body",
        type=str,
        default="",
        help="Extra context explaining what the user is trying to do.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar RAG examples to retrieve.",
    )

    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only retrieve context and print the prompt. Skips LLM loading.",
    )

    parser.add_argument(
        "--base-model-id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base Hugging Face model ID.",
    )

    parser.add_argument(
        "--adapter-id",
        type=str,
        default=None,
        help="Optional LoRA adapter ID, e.g. zehansunesara/pytorch-debug-assistant-phi3-structured-v3.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=350,
        help="Maximum number of tokens to generate.",
    )

    args = parser.parse_args()

    if not any([args.title, args.error, args.code, args.body]):
        print("Error: Please provide at least one of --title, --error, --code, or --body.")
        parser.print_help()
        sys.exit(1)

    if args.top_k <= 0:
        print("Error: --top-k must be greater than 0.")
        sys.exit(1)

    print("[1/3] Retrieving context from FAISS...")

    retrieved_issues = retrieve_context(
        title=args.title,
        error=args.error,
        code=args.code,
        body=args.body,
        top_k=args.top_k,
    )

    if not retrieved_issues:
        print("No retrieved examples found.")
    else:
        for i, doc in enumerate(retrieved_issues, start=1):
            print(
                f"  -> Context {i}: "
                f"[score={doc.get('similarity')}] "
                f"[category={doc.get('category')}] "
                f"{doc.get('title')}"
            )

    print("\n[2/3] Building RAG prompt...")

    prompt = build_rag_prompt(
        title=args.title,
        error=args.error,
        code=args.code,
        body=args.body,
        retrieved_issues=retrieved_issues,
    )

    if args.prompt_only:
        print("\n" + "=" * 80)
        print("RAG PROMPT")
        print("=" * 80)
        print(prompt)
        sys.exit(0)

    print("\n[3/3] Loading model and generating JSON...")
    print(f"Base model: {args.base_model_id}")
    print(f"Adapter:    {args.adapter_id or 'None / base model only'}")

    response = generate_fix(
        prompt=prompt,
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 80)
    print("ASSISTANT OUTPUT")
    print("=" * 80)
    print(response)


if __name__ == "__main__":
    main()