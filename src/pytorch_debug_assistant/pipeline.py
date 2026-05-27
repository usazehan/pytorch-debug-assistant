from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pytorch_debug_assistant.retriever import PyTorchIssueRetriever


ALLOWED_CATEGORIES = [
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
]


# Lazy initialization so the FAISS retriever only loads once
_retriever: PyTorchIssueRetriever | None = None


def get_retriever() -> PyTorchIssueRetriever:
    global _retriever

    if _retriever is None:
        _retriever = PyTorchIssueRetriever()

    return _retriever


def retrieve_context(
    title: str,
    error: str,
    code: str,
    body: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieves similar historical PyTorch debugging issues from the FAISS index.
    """

    parts = [part.strip() for part in [title, error, body, code] if part and part.strip()]

    if not parts:
        return []

    query = "\n".join(parts)

    return get_retriever().retrieve_similar_issues(query, top_k=top_k)


def build_rag_context(retrieved_issues: list[dict[str, Any]]) -> str:
    """
    Formats retrieved Stack Overflow examples into prompt context.
    """

    if not retrieved_issues:
        return ""

    context_lines = [
        "Here are historically successful solutions to similar PyTorch issues.",
        "Use them as reference context, but diagnose the current issue directly.",
        "",
    ]

    for i, doc in enumerate(retrieved_issues, start=1):
        context_lines.extend(
            [
                f"--- Retrieved Example {i} ---",
                f"Title: {doc.get('title', 'Unknown Title')}",
                f"Category: {doc.get('category', '')}",
                f"Error: {doc.get('error_text', '')}",
                f"Solution:",
                str(doc.get("answer", "")),
                "",
            ]
        )

    return "\n".join(context_lines).strip()


def build_rag_prompt(
    title: str,
    error: str,
    code: str,
    body: str,
    retrieved_issues: list[dict[str, Any]],
) -> str:
    """
    Builds a strict Phi-3 chat prompt using retrieved examples as RAG context.
    """

    context_str = build_rag_context(retrieved_issues)

    issue_lines = ["--- Current Issue ---"]

    if title:
        issue_lines.append(f"Title: {title}")

    if error:
        issue_lines.append(f"Error: {error}")

    if body:
        issue_lines.append(f"Context: {body}")

    if code:
        issue_lines.append(f"Code:\n{code}")

    issue_str = "\n".join(issue_lines)

    categories = "\n".join(f"- {category}" for category in ALLOWED_CATEGORIES)

    prompt = (
        "<|user|>\n"
        "You are an expert PyTorch debugging assistant.\n\n"
        "Given a PyTorch error, question, code context, and similar retrieved examples, "
        "diagnose the current issue and return a strict JSON object.\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "{\n"
        '  "category": "...",\n'
        '  "root_cause": "...",\n'
        '  "fix": "...",\n'
        '  "fix_code": "..."\n'
        "}\n\n"
        "The category must be exactly one of:\n"
        f"{categories}\n\n"
        "Do not include markdown.\n"
        "Do not include explanations outside the JSON.\n"
        "Keep root_cause and fix concise.\n\n"
    )

    if context_str:
        prompt += f"{context_str}\n\n"

    prompt += (
        f"{issue_str}\n\n"
        "Analyze the current issue. Use the retrieved examples only if helpful. "
        "Return JSON only."
        "<|end|>\n"
        "<|assistant|>\n"
    )

    return prompt


def _select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """
    Selects the best available device for local inference.
    """

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        # MPS can be picky with float16 for some transformer ops.
        return torch.device("mps"), torch.float32

    return torch.device("cpu"), torch.float32


def generate_fix(
    prompt: str,
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    adapter_id: str | None = None,
    max_new_tokens: int = 350,
) -> str:
    """
    Runs deterministic model generation for a RAG prompt.

    Supports:
    - base model only
    - base model + LoRA adapter
    """

    from peft import PeftModel

    device, dtype = _select_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)

    model = cast(Any, model).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()