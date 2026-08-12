import numpy as np
import pandas as pd

from firesight.training.advanced_models import (
    fit_random_forest,
    fit_xgboost,
    tune_model,
)
from firesight.training.baseline import FEATURE_COLUMNS, LABEL_COLUMN


def _synthetic_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    # give the label *some* real signal so tuning isn't picking among noise
    df[LABEL_COLUMN] = (df["t2m"] + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    return df


def test_fit_random_forest_is_scorable():
    train = _synthetic_frame()
    model = fit_random_forest(train, n_estimators=20, max_depth=4)
    preds = model.predict_proba(train[FEATURE_COLUMNS])[:, 1]
    assert preds.shape[0] == len(train)


def test_fit_xgboost_uses_scale_pos_weight_for_imbalance():
    train = _synthetic_frame()
    model = fit_xgboost(train, n_estimators=20, max_depth=3)
    negative, positive = train[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    assert model.get_params()["scale_pos_weight"] == negative / positive


def test_tune_model_picks_the_best_scoring_candidate_and_returns_full_grid():
    train, val = _synthetic_frame(), _synthetic_frame(seed=1)
    grid = {"n_estimators": [10, 20], "max_depth": [2, 4]}

    best_model, best_params, results = tune_model(fit_random_forest, grid, train, val)

    assert len(results) == 4  # full 2x2 grid evaluated
    assert best_params in [dict(zip(grid.keys(), combo)) for combo in [(10, 2), (10, 4), (20, 2), (20, 4)]]
    # the winning params' recorded score must actually be the max across the grid
    best_score = max(r["pr_auc"] for r in results)
    matching = [r for r in results if r["n_estimators"] == best_params["n_estimators"] and r["max_depth"] == best_params["max_depth"]]
    assert matching[0]["pr_auc"] == best_score
    assert best_model is not None
