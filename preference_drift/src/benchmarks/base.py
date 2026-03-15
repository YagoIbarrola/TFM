"""Abstract base class for preference/bias benchmarks."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class BaseBenchmark(ABC):
    """
    Every benchmark must be able to:
    1. Load its dataset
    2. Score all items given a ModelWrapper
    3. Return structured results as a DataFrame
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def load_data(self) -> None:
        """Download/load the benchmark dataset."""

    @abstractmethod
    def score(self, model) -> pd.DataFrame:
        """Score all benchmark items. Returns per-item results."""

    @abstractmethod
    def aggregate(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate metrics from per-item scores."""
