"""Math addition fine-tuning task: synthetic dataset of addition problems."""

import random
from torch.utils.data import Dataset

from .base import BaseTask


class MathAdditionDataset(Dataset):
    """Synthetic dataset of addition problems formatted as text (e.g. '23+45=68')."""

    def __init__(self, tokenizer, num_examples: int = 50000,
                 min_digits: int = 1, max_digits: int = 3,
                 max_result: int = 999, max_seq_length: int = 32,
                 seed: int = 42):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        rng = random.Random(seed)
        self.examples = []

        for _ in range(num_examples):
            a_digits = rng.randint(min_digits, max_digits)
            b_digits = rng.randint(min_digits, max_digits)

            a = rng.randint(10 ** (a_digits - 1), 10 ** a_digits - 1)
            b = rng.randint(10 ** (b_digits - 1), 10 ** b_digits - 1)

            result = a + b
            if result > max_result:
                # Clamp operands to keep result bounded
                a = rng.randint(1, max_result // 2)
                b = rng.randint(1, max_result // 2)
                result = a + b

            self.examples.append(f"{a}+{b}={result}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class MathAdditionTask(BaseTask):
    """Math addition fine-tuning task."""

    name = "math_addition"

    def create_dataset(self, tokenizer, params: dict, split: str = "train"):
        seed = params.get("seed", 42)
        if split == "eval":
            seed += 1
            num = int(params["num_examples"] * params.get("eval_fraction", 0.05))
        else:
            num = params["num_examples"]

        return MathAdditionDataset(
            tokenizer=tokenizer,
            num_examples=num,
            min_digits=params.get("min_digits", 1),
            max_digits=params.get("max_digits", 3),
            max_result=params.get("max_result", 999),
            max_seq_length=params.get("max_seq_length", 32),
            seed=seed,
        )

    def get_training_args(self, params: dict) -> dict:
        return {
            "num_train_epochs": params["epochs"],
            "per_device_train_batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "warmup_ratio": params.get("warmup_ratio", 0.1),
            "weight_decay": params.get("weight_decay", 0.01),
            "logging_steps": 50,
            "save_strategy": params.get("save_strategy", "epoch"),
            "fp16": False,
        }
