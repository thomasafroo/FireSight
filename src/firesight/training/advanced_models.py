"""RandomForest and XGBoost models, tuned and compared against the baselines.

Only reached once `baseline.py` has proven the pipeline beats a Dummy
floor (see docs/01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned).
Both models here get compared against `LogisticRegression`'s val-set
scores, not just against each other — complexity is kept only if it
earns a measurable PR-AUC/top-k gain.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from firesight.training.baseline import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TRAIN_END,
    VAL_END,
    score_model,
    temporal_split,
)

RANDOM_STATE = 0

# Small, deliberately modest grids: this is a first tuning pass on a
# ~2.6M-row training set, not an exhaustive search — see tune_model's
# docstring for why a manual grid is used here instead of GridSearchCV.
# max_depth intentionally excludes None (unbounded depth): a single fit
# at depth 16/200 trees already took ~2 minutes on the real training set,
# and unbounded trees on 2.6M rows would be both much slower and prone
# to overfitting the majority class's noise anyway.
RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 400],
    "max_depth": [8, 16],
    "min_samples_leaf": [1, 5],
}

XGBOOST_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
}


def fit_random_forest(train: pd.DataFrame, **params: Any) -> RandomForestClassifier:
    """RandomForest with class_weight="balanced" — same imbalance fix as LogisticRegression.

    No StandardScaler needed: trees split on per-feature thresholds, so
    feature scale doesn't affect which splits get chosen (see
    docs/06-modeling-and-evaluation.md#where-columntransformer-fits).
    """
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return model


def fit_xgboost(train: pd.DataFrame, **params: Any) -> XGBClassifier:
    """XGBoost with scale_pos_weight standing in for class_weight="balanced".

    XGBoost has no class_weight param; the equivalent imbalance
    correction is scale_pos_weight = (#negative / #positive) in the
    *training* fold specifically — recomputed here rather than hardcoded,
    since it must reflect the fold actually being fit on, not the full
    dataset's ratio (which would leak val/test's class balance in).
    """
    negative, positive = train[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    scale_pos_weight = negative / max(positive, 1)

    model = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        **params,
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return model


def tune_model(
    fit_fn,
    param_grid: dict[str, list[Any]],
    train: pd.DataFrame,
    val: pd.DataFrame,
    primary_metric: str = "pr_auc",
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Manual grid search against a fixed temporal val split, not sklearn's GridSearchCV.

    GridSearchCV's built-in cross-validation assumes independent folds it
    can shuffle/re-split — running it here would silently reintroduce the
    random-split leakage temporal_split exists to prevent (see
    docs/06-modeling-and-evaluation.md#splitting-by-time-never-randomly).
    Every candidate is instead fit on the *same* train fold (<=2022) and
    scored on the *same* val fold (2023), which is the correct temporal
    analogue of cross-validated tuning: never let a candidate's score
    depend on data from its own future.

    Returns (best_model, best_params, all_results) so the full grid's
    scores are inspectable, not just the winner.
    """
    keys = list(param_grid)
    results = []
    best = {"score": -float("inf"), "model": None, "params": None}

    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        model = fit_fn(train, **params)
        scores = score_model(model, val)
        results.append({**params, **scores})

        if scores[primary_metric] > best["score"]:
            best = {"score": scores[primary_metric], "model": model, "params": params}

    return best["model"], best["params"], results


def _run_and_report(name: str, fit_fn, grid: dict[str, list[Any]], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    print(f"\n--- tuning {name} ({len(list(product(*grid.values())))} candidates) ---", flush=True)
    model, params, results = tune_model(fit_fn, grid, train, val)
    for r in sorted(results, key=lambda r: -r["pr_auc"]):
        print(r, flush=True)
    print(f"best {name} params: {params}", flush=True)
    print(f"best {name} val (2023):  {score_model(model, val)}", flush=True)
    print(f"best {name} test (2024, untouched during tuning): {score_model(model, test)}", flush=True)


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    _run_and_report("RandomForest", fit_random_forest, RANDOM_FOREST_GRID, train, val, test)
    _run_and_report("XGBoost", fit_xgboost, XGBOOST_GRID, train, val, test)
