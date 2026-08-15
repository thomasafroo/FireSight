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
# columns (cell_id is deliberately excluded — see
# docs/06-modeling-and-evaluation.md#where-columntransformer-fits).
FEATURE_COLUMNS: list[str] = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "swvl1",
    "precip_mm",
    "relative_humidity",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "days_since_rain",
    "precip_7d",
    "precip_30d",
    "t2m_mean_7d",
    "t2m_trend_7d",
    "rh_mean_7d",
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
# more often human-caused) — see docs/06#known-limitation-a-wintershoulder-season-blind-spot.
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
    """Split by date, never randomly — nearby observations leak across a random split."""
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
    is already numeric — there's nothing to route by column type.
    """
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])
    return model


def score_model(model, df: pd.DataFrame) -> dict[str, float]:
    """Score a fitted classifier against pr_auc/roc_auc/top_10pct_capture."""
    y_true = df[LABEL_COLUMN].to_numpy()
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
