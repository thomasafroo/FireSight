"""Reliability check: does a predicted probability actually mean what it claims?

pr_auc/roc_auc/top_k_capture (metrics.py) are all *rank-only* — they only ask
"are positives scored higher than negatives," and are invariant to any
monotonic rescaling of the scores. A model can top those metrics while its
raw probabilities are wildly off in absolute terms, which matters here
specifically because `training/baseline.py::fit_random_forest` and
`fit_logistic_regression` both use `class_weight="balanced"`: that reweights
samples during fitting to counter the ~0.2% positive rate, which is exactly
the kind of transformation that can shift `predict_proba`'s output away from
the true empirical frequency while leaving rank order (and therefore every
metric in metrics.py) untouched. `api/main.py`'s `/predict` and
`/predict/live` both hand callers a raw `ignition_probability` float as if
it were a real probability, so this is worth checking, not assuming — see
docs/06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability
for what was actually found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions by predicted-probability *quantile*, not equal width.

    Equal-width bins (e.g. [0, 0.1), [0.1, 0.2), ...) would dump nearly every
    row into the lowest bin given how rare positives are here — most raw
    scores cluster near 0, so there's nothing to compare in the higher
    buckets. Quantile bins instead give each bucket a similar row count, so
    the highest-score bucket (the one `/risk-map`'s and `top_k_capture`'s
    "top 10%" actually corresponds to) still has enough rows to read a
    meaningful observed rate from.

    A well-calibrated model has `observed_rate` ~= `mean_predicted` in every
    row of the returned table. A model whose probabilities are well-*ranked*
    but not well-*calibrated* can still show `mean_predicted` badly off from
    `observed_rate` while every bucket is still correctly ordered.
    """
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df["bin"] = pd.qcut(df["y_score"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(count=("y_true", "size"), mean_predicted=("y_score", "mean"), observed_rate=("y_true", "mean"))
        .reset_index()
    )


if __name__ == "__main__":
    from firesight.evaluation.metrics import brier_score
    from firesight.training.baseline import (
        DATASET_PATH,
        LABEL_COLUMN,
        TRAIN_END,
        VAL_END,
        filter_fire_season,
        temporal_split,
    )
    from firesight.training.export_model import MODEL_PATH
    from firesight.training.persist import load_model_bundle

    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    _, val, test = temporal_split(df, TRAIN_END, VAL_END)

    for name, split in [("val (2023)", val), ("test (2024)", test)]:
        y_true = split[LABEL_COLUMN].to_numpy()
        y_score = bundle.model.predict_proba(split[bundle.feature_columns])[:, 1]
        print(f"\n--- {name} ---", flush=True)
        print(f"brier score: {brier_score(y_true, y_score):.6f}  (base-rate-only floor: {y_true.mean() * (1 - y_true.mean()):.6f})", flush=True)
        print(f"observed positive rate: {y_true.mean():.6f}", flush=True)
        print(reliability_table(y_true, y_score).to_string(index=False), flush=True)
