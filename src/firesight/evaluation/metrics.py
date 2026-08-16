"""Evaluation metrics suited to a rare-event classification problem.

Accuracy is close to meaningless here — a model predicting "no fire"
everywhere scores ~99% while being useless. Use these instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


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


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean squared error between predicted probability and the {0,1} outcome.

    Unlike pr_auc/roc_auc/top_k_capture (all rank-only — they only ask "are
    positives scored higher than negatives," and are invariant to any
    monotonic rescaling of y_score), this is the metric that actually checks
    whether a predicted probability means what it claims to mean. See
    evaluation/calibration.py for the fuller reliability-diagram version of
    this same question, and docs/06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability
    for why this matters specifically for `class_weight="balanced"` models.
    """
    return float(brier_score_loss(y_true, y_score))
