"""Iterative-stratified multi-label train/test split (Sechidis et al. 2011).

Standard stratification breaks on multi-label targets; iterative stratification keeps
each label's positive rate balanced across train/test. The split happens BEFORE any
resampling so the test set reflects the real label distribution.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG


def stratified_split(X: np.ndarray, Y: np.ndarray):
    """Return (train_idx, test_idx) via multi-label iterative stratification."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=float(CONFIG["split"]["test_size"]),
        random_state=int(CONFIG["seed"]),
    )
    train_idx, test_idx = next(msss.split(X, Y))
    return train_idx, test_idx
