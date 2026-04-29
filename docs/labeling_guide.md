# PyTorch Error Eval Set — Labeling Guide

## Error Categories

| ID | Category | Description | Example |
|---|---|---|---|
| 1 | `tensor_shape_mismatch` | Wrong tensor dimensions for an operation | `mat1 and mat2 shapes cannot be multiplied` |
| 2 | `cuda_oom` | GPU runs out of memory | `CUDA out of memory. Tried to allocate...` |
| 3 | `device_mismatch` | Tensors on different devices | `Expected all tensors to be on the same device` |
| 4 | `dtype_mismatch` | Wrong data types | `expected scalar type Float but found Double` |
| 5 | `autograd_error` | Backward pass / gradient issues | `element 0 of tensors does not require grad` |
| 6 | `dataloader_error` | DataLoader / Dataset issues | `TypeError in collate_fn` |
| 7 | `loss_issue` | Loss not decreasing or exploding | `loss is nan`, `loss not decreasing` |
| 8 | `environment_error` | Import / version / install issues | `No module named 'torch'`, CUDA version mismatch |
| 9 | `optimizer_error` | Optimizer / scheduler misuse | `optimizer.step() called before zero_grad()` |
| 10 | `training_loop_bug` | Logic errors in training loop | Wrong loss.backward() placement |
| 11 | `architecture_mismatch` | Layer dimension / weight loading errors | `size mismatch for layer...` |

## Difficulty Levels

- **easy** — single clear error message, obvious fix
- **medium** — requires understanding of PyTorch internals
- **hard** — subtle bug, multiple possible causes

## Labeling Rules

1. `error_text` — paste the exact error message only (not the full traceback)
2. `stack_trace` — paste the full traceback if available
3. `root_cause` — one sentence, plain English, no code
4. `fix` — one sentence describing the fix, no code
5. `fix_code` — the minimal code snippet that resolves the issue
6. `difficulty` — your honest assessment
7. `verified` — set to true only if you personally confirmed the fix works
8. If an error fits multiple categories, pick the most specific one