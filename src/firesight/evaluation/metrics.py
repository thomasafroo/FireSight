"""Evaluation metrics suited to a rare-event classification problem.

Accuracy is close to meaningless here — a model predicting "no fire"
everywhere scores ~99% while being useless. Use these instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def top_k_capture(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float = 0.1) -> float:
    """Of all actual fires, what fraction fell in the top k% highest-risk predictions?"""
    n = len(y_score)
    k = max(1, int(n * k_fraction))
    top_k_idx = np.argsort(y_score)[-k:]
    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")
    return float(y_true[top_k_idx].sum() / total_positives)
