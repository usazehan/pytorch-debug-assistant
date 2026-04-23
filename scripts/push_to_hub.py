from datasets import Dataset, DatasetDict
from huggingface_hub import login
from pathlib import Path
import json, os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Load the processed dataset
records = [json.loads(l) for l in open("data/processed/dataset.jsonl")]

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(records)

# Train/val split (90/10)
split = dataset.train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
})

print(f"Train: {len(dataset_dict['train'])} | Val: {len(dataset_dict['validation'])}")
print("\nSample:")
print(dataset_dict["train"][0])

# Push to hub — replace with your HF username
HF_USERNAME = os.getenv("HF_USERNAME")
REPO_NAME = os.getenv("REPO_NAME", "pytorch-debug-assistant")

dataset_dict.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", private=False)
print(f"\n✅ Pushed to huggingface.co/datasets/{HF_USERNAME}/{REPO_NAME}")