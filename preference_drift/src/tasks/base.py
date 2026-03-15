"""Abstract base class for fine-tuning tasks."""

from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Every task must be able to:
    1. Create a train/eval dataset given a tokenizer and params
    2. Provide HuggingFace TrainingArguments kwargs
    """

    name: str = ""

    @abstractmethod
    def create_dataset(self, tokenizer, params: dict, split: str = "train"):
        """Create a torch Dataset for the given split."""

    @abstractmethod
    def get_training_args(self, params: dict) -> dict:
        """Return kwargs for HuggingFace TrainingArguments."""
