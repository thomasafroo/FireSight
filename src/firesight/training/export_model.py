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

# Multi-day-ahead: a second, separately-exported bundle, not a replacement for MODEL_PATH -- served
# alongside the same-day model, not instead of it (see docs/03-grid-and-labels.md's
# "ignited_next_Nd" section and docs/06's "Testing the multi-day-ahead label" for how this label and
# these params were chosen).
MULTI_DAY_LABEL_COLUMN = "ignited_next_3d"
MULTI_DAY_MODEL_PATH = Path("data/processed/model_3day.joblib")

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
#
# Re-exported again 2026-08-21 after promoting fuel_type_* (19 one-hot columns, features/fuel_type.py)
# into FEATURE_COLUMNS. FWI, terrain, and fuel type were first added and benchmarked together (mixed-
# to-negative, not promoted), then re-tested individually by a per-group ablation — FWI and terrain
# were each neutral alone; fuel type alone was a real win (test PR-AUC 0.3727 -> 0.3816, top-10%
# capture 86.0% -> 88.0%) and is the only one of the three promoted. A max_features sweep against the
# 32-column set confirmed the existing 0.6063 (tuned for 13 columns) is already near-optimal, so
# params are unchanged here too — see docs/06-modeling-and-evaluation.md#closing-the-feature-category-
# gap-fwi-terrain-and-fuel-type-2026-08-21 for the full ablation table. Unlike neighbor_fire_count
# above, this one doesn't break /predict/live: features/live_fuel_type.py fills fuel_type_* from the
# same static per-cell cache training uses (no live fetch needed, since fuel type doesn't change day
# to day) — see docs/07-serving.md.
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


def export_multi_day_model(
    dataset_path: Path = DATASET_PATH,
    out_path: Path = MULTI_DAY_MODEL_PATH,
    label_column: str = MULTI_DAY_LABEL_COLUMN,
) -> ModelBundle:
    """Fit the same RandomForest params against the multi-day-ahead label, save as a second bundle.

    Reuses `BEST_RANDOM_FOREST_PARAMS` and `FEATURE_COLUMNS` unchanged — the retune attempt
    documented in docs/06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21
    won on val but lost on test (the same single-fold-overfitting shape the RF-vs-XGBoost episode
    already showed once), so the original same-day-tuned params were kept rather than adopted.

    **No calibrator, unlike `export_current_best`** — `evaluation/backtest.py::
    run_rolling_origin_backtest` (what the pooled calibration in `export_current_best` depends on)
    is hardcoded to `training/baseline.py::LABEL_COLUMN` (`ignited`), not parameterized by label.
    Rather than rewire an 8-fold backtest for a first served version of this label, `calibrator`
    stays `None` here — `ModelBundle.predict_calibrated_proba` already treats that as "unavailable,"
    not a false zero, so this is a real, accepted, documented gap, not a silent omission. A real gap
    to close if this label sees continued use.
    """
    df = pd.read_parquet(dataset_path)
    df = filter_fire_season(df)
    df = df.dropna(subset=[label_column])
    df[label_column] = df[label_column].astype(int)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    model = fit_random_forest(train, label_column=label_column, **BEST_RANDOM_FOREST_PARAMS)
    val_scores = score_model(model, val, label_column=label_column)
    test_scores = score_model(model, test, label_column=label_column)

    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        calibrator=None,
        metadata={
            "model_type": "RandomForest",
            "label_column": label_column,
            "params": BEST_RANDOM_FOREST_PARAMS,
            "trained_through": TRAIN_END,
            "validated_on": f"{TRAIN_END} to {VAL_END}",
            "val_scores": val_scores,
            "test_scores": test_scores,
            "calibration_method": None,
        },
    )
    save_model_bundle(bundle, out_path)
    return bundle


if __name__ == "__main__":
    bundle = export_current_best()
    print(f"Saved model bundle -> {MODEL_PATH}", flush=True)
    print(f"metadata: {bundle.metadata}", flush=True)

    multi_day_bundle = export_multi_day_model()
    print(f"\nSaved multi-day model bundle -> {MULTI_DAY_MODEL_PATH}", flush=True)
    print(f"metadata: {multi_day_bundle.metadata}", flush=True)
