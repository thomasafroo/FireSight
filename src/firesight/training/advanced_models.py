"""RandomForest and XGBoost models, tuned and compared against the baselines.

Only reached once `baseline.py` has proven the pipeline beats a Dummy
floor (see docs/01-problem-framing.md#methodology-baseline-first-complexity-only-if-earned).
Both models here get compared against `LogisticRegression`'s val-set
scores, not just against each other, complexity is kept only if it
earns a measurable PR-AUC/top-k gain.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from xgboost import XGBClassifier

from firesight.training.baseline import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TRAIN_END,
    VAL_END,
    filter_fire_season,
    score_model,
    temporal_split,
)

RANDOM_STATE = 0

# Small, deliberately modest grids: this is a first tuning pass on a
# ~2.6M-row training set, not an exhaustive search, see tune_model's
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

# Wider search space for tune_random_search, sized for sampling (n_iter draws),
# not exhaustive enumeration, so it can cover a much bigger space than the
# grids above without the combinatorial blowup a full grid over this many
# dimensions would need. max_depth still capped at 16 for the same reason as
# RANDOM_FOREST_GRID: unbounded trees on 2.6M rows are slow and prone to
# overfitting the majority class's noise.
RANDOM_FOREST_DISTRIBUTIONS: dict[str, Any] = {
    "n_estimators": randint(150, 500),
    "max_depth": randint(4, 17),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.3, 0.7),  # fraction of features per split, in [0.3, 1.0]
}

XGBOOST_DISTRIBUTIONS: dict[str, Any] = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 8),
    "learning_rate": loguniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
    "colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
    "reg_alpha": loguniform(1e-3, 1.0),
    "reg_lambda": loguniform(1e-2, 10.0),
}

N_ITER = 15


def _scale_pos_weight(train: pd.DataFrame) -> float:
    """negative/positive count in *this* training fold, see fit_xgboost."""
    negative, positive = train[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    return negative / max(positive, 1)


def fit_random_forest(train: pd.DataFrame, label_column: str = LABEL_COLUMN, **params: Any) -> RandomForestClassifier:
    """RandomForest with class_weight="balanced", same imbalance fix as LogisticRegression.

    No StandardScaler needed: trees split on per-feature thresholds, so
    feature scale doesn't affect which splits get chosen (see
    docs/06-modeling-and-evaluation.md#where-columntransformer-fits).

    `label_column` defaults to the same-day `ignited` label; pass e.g. `"ignited_next_3d"` to fit
    against the multi-day-ahead label instead, same override pattern `baseline.py::score_model`
    has, so a caller scoring the fitted model can pass the matching `label_column` there too.
    """
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        **params,
    )
    model.fit(train[FEATURE_COLUMNS], train[label_column])
    return model


def fit_xgboost(train: pd.DataFrame, **params: Any) -> XGBClassifier:
    """XGBoost with scale_pos_weight standing in for class_weight="balanced".

    XGBoost has no class_weight param; the equivalent imbalance
    correction is scale_pos_weight = (#negative / #positive) in the
    *training* fold specifically, recomputed here rather than hardcoded,
    since it must reflect the fold actually being fit on, not the full
    dataset's ratio (which would leak val/test's class balance in).
    """
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=_scale_pos_weight(train),
        eval_metric="aucpr",
        verbosity=1,
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
    can shuffle/re-split, running it here would silently reintroduce the
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
    candidates = list(product(*param_grid.values()))
    results = []
    best = {"score": -float("inf"), "model": None, "params": None}

    for i, values in enumerate(candidates, start=1):
        params = dict(zip(keys, values))
        print(f"[{i}/{len(candidates)}] fitting {params}", flush=True)
        model = fit_fn(train, **params)
        scores = score_model(model, val)
        results.append({**params, **scores})
        print(f"[{i}/{len(candidates)}] {scores}", flush=True)

        if scores[primary_metric] > best["score"]:
            best = {"score": scores[primary_metric], "model": model, "params": params}

    return best["model"], best["params"], results


def tune_random_search(
    estimator: Any,
    param_distributions: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_iter: int,
    primary_metric: str = "average_precision",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """RandomizedSearchCV over a wide param space, kept temporally safe via PredefinedSplit.

    RandomizedSearchCV's default `cv` has the same problem `tune_model`'s
    docstring describes for GridSearchCV: it assumes independent, shufflable
    folds. PredefinedSplit sidesteps that by handing sklearn the *exact*
    train/val roles temporal_split already established, every train row is
    marked -1 (never held out), every val row is marked 0 (the only fold
    scored), so RandomizedSearchCV's random *sampling of candidates* stays
    separate from any random *sampling of folds*. This is what lets it search
    a much wider space than tune_model's exhaustive grids: n_iter draws from
    continuous distributions instead of enumerating every combination.

    refit=False deliberately: RandomizedSearchCV's built-in refit would retrain
    the winner on train+val combined (everything passed to .fit()), quietly
    changing what the model was fit on. The winner is refit on train only
    below, matching every other model in this module and keeping "test is
    never touched during tuning" true here too.

    n_jobs defaults to -1 (one candidate per CPU core, same as before this
    param existed). Pass n_jobs=1 for a GPU estimator (e.g. XGBoost with
    device="cuda"), GPU training uses the whole device for a single fit, so
    running multiple candidates in parallel subprocesses would have them
    contend for the same GPU instead of speeding anything up, and can throw
    CUDA out-of-memory errors depending on model/data size. See
    tune_xgboost_gpu.py.
    """
    X = pd.concat([train[FEATURE_COLUMNS], val[FEATURE_COLUMNS]], ignore_index=True)
    y = pd.concat([train[LABEL_COLUMN], val[LABEL_COLUMN]], ignore_index=True)
    test_fold = [-1] * len(train) + [0] * len(val)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=primary_metric,
        cv=PredefinedSplit(test_fold),
        refit=False,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=3,
    )
    search.fit(X, y)

    best_model = clone(estimator).set_params(**search.best_params_)
    best_model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])

    results = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    return best_model, search.best_params_, results


def _run_and_report(name: str, fit_fn, grid: dict[str, list[Any]], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    print(f"\n--- tuning {name} ({len(list(product(*grid.values())))} candidates) ---", flush=True)
    model, params, results = tune_model(fit_fn, grid, train, val)
    for r in sorted(results, key=lambda r: -r["pr_auc"]):
        print(r, flush=True)
    print(f"best {name} params: {params}", flush=True)
    print(f"best {name} val (2023):  {score_model(model, val)}", flush=True)
    print(f"best {name} test (2024, untouched during tuning): {score_model(model, test)}", flush=True)


def _run_random_search_and_report(
    name: str,
    estimator: Any,
    param_distributions: dict[str, Any],
    n_iter: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_jobs: int = -1,
) -> None:
    print(f"\n--- randomized search: {name} ({n_iter} sampled candidates) ---", flush=True)
    model, params, results = tune_random_search(estimator, param_distributions, train, val, n_iter, n_jobs=n_jobs)
    print(results[["params", "mean_test_score"]].to_string(index=False), flush=True)
    print(f"best {name} params: {params}", flush=True)
    print(f"best {name} val (2023):  {score_model(model, val)}", flush=True)
    print(f"best {name} test (2024, untouched during tuning): {score_model(model, test)}", flush=True)


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    _run_and_report("RandomForest", fit_random_forest, RANDOM_FOREST_GRID, train, val, test)
    _run_and_report("XGBoost", fit_xgboost, XGBOOST_GRID, train, val, test)

    # XGBoost's randomized search runs separately on GPU (tune_xgboost_gpu.py), not repeated
    # here on CPU, since it's the same search space (XGBOOST_DISTRIBUTIONS) and would be pure
    # redundant work. RandomForest has no GPU path in scikit-learn, so its randomized search
    # still runs here.
    rf_estimator = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    _run_random_search_and_report(
        "RandomForest (randomized search)", rf_estimator, RANDOM_FOREST_DISTRIBUTIONS, N_ITER, train, val, test
    )
