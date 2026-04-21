"""
Data splitting utilities for ROM training.

Provides leak-free splitting strategies:
- temporal_split: Split by time order (no future leakage)
- case_split: Split by case/simulation (OOD evaluation)
- random_split: Standard random split (with optional seed)
"""

from __future__ import annotations

from typing import Literal

import numpy as np


SplitStrategy = Literal["temporal", "case", "random"]


def temporal_split(
    n_samples: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split samples by temporal order.

    For time-series data, this ensures validation/test sets come after
    training data, preventing future information leakage.

    Args:
        n_samples: Total number of samples
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if n_samples <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_val - n_test

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_samples, dtype=np.int64)

    return train_idx, val_idx, test_idx


def case_split(
    case_ids: np.ndarray,
    val_cases: list[int] | None = None,
    test_cases: list[int] | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split samples by case/simulation ID.

    This ensures samples from the same case stay together,
    enabling out-of-distribution (OOD) evaluation.

    Args:
        case_ids: Array of case ID for each sample
        val_cases: Explicit list of case IDs for validation (optional)
        test_cases: Explicit list of case IDs for testing (optional)
        val_ratio: Fraction of cases for validation (if val_cases not provided)
        test_ratio: Fraction of cases for testing (if test_cases not provided)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    unique_cases = np.unique(case_ids)
    n_cases = len(unique_cases)

    if n_cases == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)

    if val_cases is None and test_cases is None:
        shuffled = rng.permutation(unique_cases)
        n_val_cases = max(1, int(n_cases * val_ratio))
        n_test_cases = max(0, int(n_cases * test_ratio))

        test_cases = list(shuffled[:n_test_cases])
        val_cases = list(shuffled[n_test_cases : n_test_cases + n_val_cases])
        train_cases = list(shuffled[n_test_cases + n_val_cases :])
    elif val_cases is None:
        train_cases = [c for c in unique_cases if c not in test_cases]
        n_val_cases = max(1, int(len(train_cases) * val_ratio))
        shuffled_train = rng.permutation(train_cases)
        val_cases = list(shuffled_train[:n_val_cases])
        train_cases = list(shuffled_train[n_val_cases:])
    elif test_cases is None:
        remaining = [c for c in unique_cases if c not in val_cases]
        n_test_cases = max(0, int(len(remaining) * test_ratio))
        shuffled_remaining = rng.permutation(remaining)
        test_cases = list(shuffled_remaining[:n_test_cases])
        train_cases = list(shuffled_remaining[n_test_cases:])
    else:
        train_cases = [c for c in unique_cases if c not in val_cases and c not in test_cases]

    train_idx = np.where(np.isin(case_ids, train_cases))[0].astype(np.int64)
    val_idx = np.where(np.isin(case_ids, val_cases))[0].astype(np.int64)
    test_idx = np.where(np.isin(case_ids, test_cases))[0].astype(np.int64)

    return train_idx, val_idx, test_idx


def random_split(
    n_samples: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random split with reproducible seeding.

    Args:
        n_samples: Total number of samples
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if n_samples <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_val - n_test

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])

    return train_idx, val_idx, test_idx


def split_dataset(
    n_samples: int,
    strategy: SplitStrategy = "temporal",
    case_ids: np.ndarray | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    val_cases: list[int] | None = None,
    test_cases: list[int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified interface for dataset splitting.

    Args:
        n_samples: Total number of samples
        strategy: Split strategy ("temporal", "case", "random")
        case_ids: Case IDs for each sample (required for "case" strategy)
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        val_cases: Explicit validation cases (for "case" strategy)
        test_cases: Explicit test cases (for "case" strategy)
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if strategy == "temporal":
        return temporal_split(n_samples, val_ratio, test_ratio)
    elif strategy == "case":
        if case_ids is None:
            raise ValueError("case_ids is required for 'case' split strategy")
        return case_split(case_ids, val_cases, test_cases, val_ratio, test_ratio, seed)
    elif strategy == "random":
        return random_split(n_samples, val_ratio, test_ratio, seed)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}. Use 'temporal', 'case', or 'random'.")


class DataSplitter:
    """
    Stateful data splitter that stores split configuration.

    Useful for consistent splitting across multiple runs.
    """

    def __init__(
        self,
        strategy: SplitStrategy = "temporal",
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self._train_idx: np.ndarray | None = None
        self._val_idx: np.ndarray | None = None
        self._test_idx: np.ndarray | None = None

    def fit(
        self,
        n_samples: int,
        case_ids: np.ndarray | None = None,
        val_cases: list[int] | None = None,
        test_cases: list[int] | None = None,
    ) -> "DataSplitter":
        """Compute and store the split indices."""
        self._train_idx, self._val_idx, self._test_idx = split_dataset(
            n_samples,
            self.strategy,
            case_ids,
            self.val_ratio,
            self.test_ratio,
            val_cases,
            test_cases,
            self.seed,
        )
        return self

    @property
    def train_idx(self) -> np.ndarray:
        if self._train_idx is None:
            raise RuntimeError("Splitter not fitted. Call fit() first.")
        return self._train_idx

    @property
    def val_idx(self) -> np.ndarray:
        if self._val_idx is None:
            raise RuntimeError("Splitter not fitted. Call fit() first.")
        return self._val_idx

    @property
    def test_idx(self) -> np.ndarray:
        if self._test_idx is None:
            raise RuntimeError("Splitter not fitted. Call fit() first.")
        return self._test_idx

    def get_split_info(self) -> dict:
        """Return information about the current split."""
        return {
            "strategy": self.strategy,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "n_train": len(self._train_idx) if self._train_idx is not None else 0,
            "n_val": len(self._val_idx) if self._val_idx is not None else 0,
            "n_test": len(self._test_idx) if self._test_idx is not None else 0,
        }
