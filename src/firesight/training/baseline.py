"""Baseline models for wildfire ignition prediction.

Run this first, before anything fancier, to prove the pipeline works
end-to-end and to get a floor to beat.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from firesight.evaluation.metrics import pr_auc, roc_auc, top_k_capture

DATASET_PATH = Path("data/processed/kamloops_dataset.parquet")

# Raw joined weather + engineered time-series features (see
# features/weather.py and features/engineering.py). No categorical
# columns (cell_id is deliberately excluded, see
# docs/06-modeling-and-evaluation.md#where-columntransformer-fits).
#
# d2m, u10, v10, wind_dir_sin, wind_dir_cos, and t2m_trend_7d were dropped
# 2026-08-15 after permutation importance showed them at ~0 or negative on
# both val and test with the served RandomForest, see
# docs/06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on
# for the full numbers, including why wind_speed was kept despite being
# flagged alongside them (it showed real, reproducible positive signal on
# test specifically). engineering.py still computes the dropped columns
# (and add_relative_humidity/add_wind_features still need d2m/u10/v10 as
# inputs to derive relative_humidity and wind_speed), only the model's
# input list shrank, not the underlying weather schema.
FEATURE_COLUMNS: list[str] = [
    "t2m",
    "swvl1",
    "precip_mm",
    "relative_humidity",
    "wind_speed",
    "days_since_rain",
    "precip_7d",
    "precip_30d",
    "t2m_mean_7d",
    "rh_mean_7d",
    # cape/convective_precip_mm (full-ERA5 storm/lightning-potential proxy, features/convective.py)
    # were added 2026-08-17 and re-tuned, but the 2026-08-17 overnight run showed no measured
    # benefit over the served 10-feature model on the untouched test set (see
    # docs/06-modeling-and-evaluation.md#the-re-tune-result-2026-08-17-overnight-run-no-measured-benefit),
    # left out of FEATURE_COLUMNS for now so this list matches what advanced_models.py should
    # actually search next, rather than accumulating every historically-tried column. The joined columns
    # and dataset are unaffected (still in kamloops_dataset.parquet, still enforced complete by
    # drop_incomplete_history) so they're one line away from being added back for a future combined
    # search alongside neighbor_fire_count below.
    #
    # neighbor_fire_count_{1,3,7}d added 2026-08-19, strictly-prior-day count of a cell's 8 Moore
    # neighbors that ignited in the trailing N days (features/engineering.py::
    # add_neighbor_fire_features), the fire-season spread-dynamics feature proposed in
    # docs/06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-cells-recent-fire-history.
    # Promoted the same day (see export_model.py's history) after refitting BEST_RANDOM_FOREST_PARAMS
    # unchanged on the resulting 13-column set was a huge, real jump on test, this is part of what
    # the served model actually uses now, not just a staged tuning candidate.
    "neighbor_fire_count_1d",
    "neighbor_fire_count_3d",
    "neighbor_fire_count_7d",
    # fuel_type_* (19 one-hot columns, features/fuel_type.py) added 2026-08-21 after a per-group
    # ablation isolated it as the only real contributor from that session's FWI/terrain/fuel-type
    # batch (see
    # docs/06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21)
    # -- FWI and terrain were each neutral in isolation and actively hurt
    # when combined with fuel_type, so only fuel_type was promoted. Exact literal list, not
    # discovered dynamically from the dataset at import time: which codes exist is a property of
    # the Kamloops FC extract `features/fuel_type.py` already cached, and the served model needs a
    # fixed input schema regardless of what a future re-fetch might return.
    "fuel_type_C-2",
    "fuel_type_C-3",
    "fuel_type_C-4",
    "fuel_type_C-5",
    "fuel_type_C-6",
    "fuel_type_C-7",
    "fuel_type_D-1/D-2",
    "fuel_type_M-1/M-2 (20 PC)",
    "fuel_type_M-1/M-2 (25 PC)",
    "fuel_type_M-1/M-2 (30 PC)",
    "fuel_type_M-1/M-2 (40 PC)",
    "fuel_type_M-1/M-2 (45 PC)",
    "fuel_type_M-1/M-2 (50 PC)",
    "fuel_type_M-1/M-2 (55 PC)",
    "fuel_type_M-1/M-2 (65 PC)",
    "fuel_type_M-1/M-2 (Burned)",
    "fuel_type_Non-fuel",
    "fuel_type_O-1a",
    "fuel_type_S-2",
]
LABEL_COLUMN = "ignited"
DATE_COLUMN = "date"

# train: everything before 2023 (<=2022); val: 2023; test: 2024.
TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"

# Fire season: May 1 - Oct 15, matching the Kamloops Fire Centre's typical Category 2/3
# open-burning prohibition window (docs/06). The project scopes to this window because the
# destructive, operationally-important fires are a summer phenomenon (hot+dry conditions
# both drive ignition and let fires spread fast) and because error analysis showed the
# weather-only feature set has no signal for winter/shoulder-season fires anyway (they're
# more often human-caused), see docs/06#known-limitation-a-wintershoulder-season-blind-spot.
FIRE_SEASON_START = "05-01"
FIRE_SEASON_END = "10-15"


def filter_fire_season(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within the fire season window, any year.

    Compares zero-padded "MM-DD" strings rather than month/day integer
    tuples so the range check is a single lexicographic comparison
    instead of a two-part (month, day) comparison.
    """
    month_day = df[DATE_COLUMN].dt.strftime("%m-%d")
    return df[(month_day >= FIRE_SEASON_START) & (month_day <= FIRE_SEASON_END)]


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date, never randomly, nearby observations leak across a random split."""
    train = df[df[DATE_COLUMN] < train_end]
    val = df[(df[DATE_COLUMN] >= train_end) & (df[DATE_COLUMN] < val_end)]
    test = df[df[DATE_COLUMN] >= val_end]
    return train, val, test


def fit_dummy(train: pd.DataFrame) -> DummyClassifier:
    model = DummyClassifier(strategy="stratified", random_state=0)
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return model


def fit_logistic_regression(train: pd.DataFrame) -> Pipeline:
    """Class-weighted logistic regression behind a StandardScaler.

    Scaling is needed here because LogisticRegression is gradient-based
    and regularization behaves sanely only when features share a scale
    (see docs/06-modeling-and-evaluation.md#where-columntransformer-fits).
    A plain Pipeline, not a ColumnTransformer, since every feature here
    is already numeric, there's nothing to route by column type.
    """
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return model


def score_model(model, df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> dict[str, float]:
    """Score a fitted classifier against pr_auc/roc_auc/top_10pct_capture.

    `label_column` defaults to the same-day `ignited` label; pass e.g. `"ignited_next_3d"` to score
    against the multi-day-ahead label instead (see `features/labels.py::add_forward_ignition_label`),
    same `columns` override pattern `drop_incomplete_history` already uses.
    """
    y_true = df[label_column].to_numpy()
    y_score = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "top_10pct_capture": top_k_capture(y_true, y_score, k_fraction=0.1),
    }


def _describe_split(name: str, split: pd.DataFrame) -> None:
    start, end = split[DATE_COLUMN].min().date(), split[DATE_COLUMN].max().date()
    positives = int(split[LABEL_COLUMN].sum())
    print(f"{name:5s}: {len(split):>9,} rows  {start} -> {end}  positives={positives}", flush=True)


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)
    _describe_split("train", train)
    _describe_split("val", val)
    _describe_split("test", test)

    dummy = fit_dummy(train)
    logreg = fit_logistic_regression(train)

    print("\n--- validation scores (2023) ---", flush=True)
    print("dummy:            ", score_model(dummy, val), flush=True)
    print("logistic regression:", score_model(logreg, val), flush=True)
