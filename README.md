# pytorch-debug-assistant

I got tired of copy-pasting PyTorch errors into ChatGPT during late-night 
training runs, so I fine-tuned a small model to do it faster and offline.

This is a QLoRA fine-tune of Phi-3-mini-4k-instruct on ~1,500 PyTorch 
Stack Overflow Q&A pairs. It's not magic — it won't replace reading the 
docs — but it's pretty good at the errors you hit over and over.

## what it does

Paste in a PyTorch error or a description of what's going wrong. It gives 
you a plain-English explanation of the root cause and a code fix.

## how it was built

**Dataset** — scraped ~2,500 PyTorch questions from Stack Overflow, kept 
only answered ones with decent upvotes, cleaned the HTML, and formatted 
them as instruction-tuning pairs. Published at 
`zehansunesara/pytorch-debug-assistant` on HuggingFace.

**Fine-tuning** — QLoRA (4-bit, r=16) on top of Phi-3-mini-4k-instruct, 
trained on a T4 GPU via Kaggle. Only ~1-2% of parameters actually 
update during training, which is the whole point of LoRA.

**Serving** — FastAPI backend + Gradio frontend coming in Phase 3.


## usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base, "zehansunesara/pytorch-debug-assistant-phi3-mini")
```

Or just use the [live demo](#) (link coming after Phase 3).

## running locally

```bash
git clone https://github.com/usazehan/pytorch-debug-assistant
cd pytorch-debug-assistant
cp .env-example .env  # fill in your tokens
pip install -r requirements.txt
```

## project structure
scripts/          data collection + cleaning pipeline
notebooks/        QLoRA fine-tuning (Colab)
data/processed/   formatted dataset (also on HuggingFace)

## training results

100 steps on a T4 GPU. Loss dropped fast in the first 50 steps then 
leveled off — pretty typical for a small dataset with a narrow domain.

![training loss](assets/training_loss.png)

Not fully converged, but good enough to give correct answers on the 
errors it was trained on. Planning a longer 500-step run on the 
cleaned dataset next.

## Evaluation Results

I evaluated the PyTorch Debug Assistant on a held-out 100-example benchmark of real PyTorch debugging issues. Each example includes an error/question context and a ground-truth category such as `dtype_mismatch`, `cuda_oom`, `device_mismatch`, `dataloader_error`, or `environment_error`.

The evaluator measures:

- **Category Accuracy**: whether the system predicts the correct debugging category
- **Valid JSON Rate**: whether the model returns the required structured JSON schema
- **Average Latency**: average generation time per example

| System | Eval Examples | Category Accuracy | Valid JSON Rate | Notes |
|---|---:|---:|---:|---|
| Heuristic baseline | 100 | ~78% | N/A | Rule-based keyword classifier |
| Base Phi-3-mini | 100 | 62% | 91% | Zero-shot structured JSON prompting |
| Old LoRA adapter | 5 | 40% | 80% | Smoke test only; trained before structured-output benchmark |
| Structured Phi-3 LoRA adapter | 100 | 58% | 92% | Fine-tuned for structured JSON output, but over-predicted `cuda_oom` |
| Structured Phi-3 LoRA + category overrides | 100 | 80% | 92% | Hybrid system combining model output with high-confidence deterministic category corrections |

The structured LoRA adapter improved JSON reliability but did not improve category accuracy on its own. Error analysis showed that the model often over-classified CUDA-related errors as `cuda_oom`, even when the actual issue was a device mismatch, dtype mismatch, autograd error, or training-loop bug.

To address this, I added a deterministic category override layer for high-confidence PyTorch error patterns such as `device-side assert triggered`, `can't convert CUDA tensor to numpy`, `not implemented for 'Half'`, and `backward through the graph a second time`.

This hybrid approach improved category accuracy from 62% for the base model and 58% for the fine-tuned adapter alone to 80% on the 100-example benchmark, while maintaining a 92% valid JSON rate.

### Current Takeaway

The base Phi-3-mini model achieved **62% category accuracy** and **91% valid JSON rate** on the 100-example benchmark. The older LoRA adapter performed worse on a 5-example smoke test, which suggests the original fine-tuning objective was not aligned with the new structured debugging task.

The next iteration will fine-tune the model specifically on structured debugging outputs:

```json
{
  "category": "dtype_mismatch",
  "root_cause": "The input tensor dtype does not match the model or loss expectation.",
  "fix": "Convert the tensor to the expected dtype before passing it into the model or loss.",
  "fix_code": "x = x.float()"
}
```

## what's next

- [ ] Post-training quantization (GPTQ/AWQ) + latency benchmarks  
- [ ] FastAPI + Gradio serving layer  
- [ ] HuggingFace Spaces deployment  
- [ ] OSS contribution to 🤗 PEFT