import os

from datasets import load_dataset

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "finetune", "training")

dataset = load_dataset(
    "json",
    data_files=os.path.join(TRAIN_DIR, "sft_dataset.jsonl"),
    split="train",
)

dataset = dataset.train_test_split(
    test_size=0.05,
    seed=42,
)

dataset["train"].to_json(
    os.path.join(TRAIN_DIR, "sft_train.jsonl"), orient="records", lines=True, force_ascii=False
)
dataset["test"].to_json(
    os.path.join(TRAIN_DIR, "sft_val.jsonl"), orient="records", lines=True, force_ascii=False
)

print(f"train: {len(dataset['train'])}  val: {len(dataset['test'])}")
