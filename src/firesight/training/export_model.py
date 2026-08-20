"""Fit the currently-best model on the real dataset and persist it for serving.

Deliberately a separate, explicit step from `baseline.py`/`advanced_models.py`
(which explore/compare) rather than auto-saving from either — promoting a
model to "the one the API serves" should be a decision made after looking
at the comparison, not a side effect of running a training script.

Also fits the served model's probability calibrator here, for the same reason:
`evaluation/calibration.py::leave_one_year_out_calibration_check` found that a
calibrator only generalizes honestly if it's fit on scores pooled across many
years, not one — see
docs/06-modeling-and-evaluation.md#does-pooled-leave-one-year-out-validated-calibration-actually-help.
That means exporting now costs 8 extra RandomForest refits (one per
`backtest.py::HOLDOUT_YEARS` fold, ~10-15 min total) on top of the main fit —
an accepted cost given this is already the project's one deliberate,
infrequent promotion step, not something run casually.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from firesight.training.advanced_models import fit_random_forest
from firesight.training.baseline import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TRAIN_END,
    VAL_END,
    filter_fire_season,
    score_model,
    temporal_split,
)
from firesight.training.persist import ModelBundle, save_model_bundle

MODEL_PATH = Path("data/processed/model.joblib")

# Winner of advanced_models.py's widened RandomizedSearchCV+PredefinedSplit search by val
# (2023) PR-AUC — see
# docs/06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit
# for the full comparison. This replaced the earlier grid-search XGBoost pick once both
# models got the same wider search: RandomForest came out ahead on every test-set metric
# (PR-AUC, ROC-AUC, top-10% capture), where XGBoost only edged it on val metrics — the
# numbers most exposed to overfitting a single validation fold.
#
# Re-tuned 2026-08-15 after scoping to fire season (May 1 - Oct 15, see
# baseline.py::filter_fire_season) and extending training data back to 2012 — see
# docs/06-modeling-and-evaluation.md#re-tuning-after-the-fire-season-scope-change. Again the
# randomized-search RF beat the grid-search RF on every test metric despite a slightly lower
# val PR-AUC, so it's the pick for the same overfitting-a-single-fold reason as last time.
#
# Params unchanged since, but re-exported again 2026-08-15 after dropping 6 dead-weight columns
# from FEATURE_COLUMNS (d2m, u10, v10, wind_dir_sin, wind_dir_cos, t2m_trend_7d — see
# docs/06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on).
# Refitting these same params on the smaller feature set left val/test scores within noise of the
# 16-feature version (e.g. test top-10% capture unchanged at 71.9%), confirming those columns
# really were dead weight rather than something the params happened to lean on.
#
# Re-exported again 2026-08-19 after adding neighbor_fire_count_{1,3,7}d to FEATURE_COLUMNS (see
# docs/06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-cells-recent-fire-history) —
# a strictly-prior-day count of a cell's 8 Moore neighbors that ignited recently
# (features/engineering.py::add_neighbor_fire_features). Refitting these same params unchanged on
# the resulting 13-column set was a huge, real jump (test PR-AUC 0.0106 -> 0.373, top-10% capture
# 71.9% -> 86.0%), independently confirmed by a freshly GPU-tuned XGBoost on the same 13 columns
# landing at a similar order of magnitude (test PR-AUC 0.372, top-10% capture 86.4%) — not a
# single-model quirk. Params themselves weren't re-tuned against the new feature (that would mean
# re-running tune_model/tune_random_search, still a possible future refinement, but the gain is
# large enough on the existing params alone that waiting on a wider search wasn't necessary to
# justify promoting). **Breaks /predict/live** — see docs/07-serving.md's live-weather section;
# accepted deliberately, matching the cape/convective_precip_mm "dormant limitation" pattern.
BEST_RANDOM_FOREST_PARAMS = {
    "n_estimators": 238,
    "max_depth": 6,
    "max_features": 0.6063110478838847,
    "min_samples_leaf": 7,
}


def export_current_best(dataset_path: Path = DATASET_PATH, out_path: Path = MODEL_PATH) -> ModelBundle:
    """Fit the tuned RandomForest model on train, fit its pooled calibrator, and save both.

    This is today's best *validated* model (see
    docs/06-modeling-and-evaluation.md) — beats both the Dummy floor and
    the LogisticRegression baseline by a wide margin on every val metric.
    Nothing else in the API needs to change if this is swapped for a
    different winner later, since it only depends on the ModelBundle
    contract (predict_proba(dict) -> float), not on the model class.
    """
    from firesight.evaluation.backtest import HOLDOUT_YEARS, run_rolling_origin_backtest
    from firesight.evaluation.calibration import fit_isotonic_calibrator

    df = pd.read_parquet(dataset_path)
    df = filter_fire_season(df)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    model = fit_random_forest(train, **BEST_RANDOM_FOREST_PARAMS)
    val_scores = score_model(model, val)
    test_scores = score_model(model, test)

    # Pooled, leave-one-year-out-validated calibration (see module docstring) rather than fitting
    # on val/test alone — the whole point of the LOYO check was showing a single-year fit doesn't
    # generalize reliably.
    folds = run_rolling_origin_backtest(df, HOLDOUT_YEARS)
    pooled_y_true = np.concatenate([fold.y_true for fold in folds])
    pooled_y_score = np.concatenate([fold.y_score for fold in folds])
    calibrator = fit_isotonic_calibrator(pooled_y_score, pooled_y_true)

    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        calibrator=calibrator,
        metadata={
            "model_type": "RandomForest",
            "params": BEST_RANDOM_FOREST_PARAMS,
            "trained_through": TRAIN_END,
            "validated_on": f"{TRAIN_END} to {VAL_END}",
            "val_scores": val_scores,
            "test_scores": test_scores,
            "calibration_method": "isotonic",
            "calibration_pooled_years": HOLDOUT_YEARS,
        },
    )
    save_model_bundle(bundle, out_path)
    return bundle


if __name__ == "__main__":
    bundle = export_current_best()
    print(f"Saved model bundle -> {MODEL_PATH}", flush=True)
    print(f"metadata: {bundle.metadata}", flush=True)
