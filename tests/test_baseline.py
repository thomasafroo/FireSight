import numpy as np
import pandas as pd

from firesight.training.baseline import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    filter_fire_season,
    fit_dummy,
    fit_logistic_regression,
    score_model,
    temporal_split,
)


def test_temporal_split_respects_date_boundaries():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2023-06-01", "2024-06-01"]),
            "ignited": [0, 1, 0],
        }
    )
    train, val, test = temporal_split(df, "2023-01-01", "2024-01-01")

    assert list(train["date"]) == [pd.Timestamp("2020-01-01")]
    assert list(val["date"]) == [pd.Timestamp("2023-06-01")]
    assert list(test["date"]) == [pd.Timestamp("2024-06-01")]


def test_filter_fire_season_keeps_only_may_1_through_oct_15_any_year():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-04-30", "2023-05-01", "2023-07-15", "2023-10-15", "2023-10-16", "2024-12-25"]
            ),
            "ignited": [0, 0, 1, 0, 0, 0],
        }
    )
    result = filter_fire_season(df)

    assert list(result["date"]) == [
        pd.Timestamp("2023-05-01"),
        pd.Timestamp("2023-07-15"),
        pd.Timestamp("2023-10-15"),
    ]


def _synthetic_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    df[LABEL_COLUMN] = rng.integers(0, 2, size=n)
    return df


def test_fit_dummy_and_score_model_return_expected_metric_keys():
    train, val = _synthetic_frame(), _synthetic_frame(seed=1)
    model = fit_dummy(train)
    scores = score_model(model, val)
    assert set(scores) == {"pr_auc", "roc_auc", "top_10pct_capture"}


def test_fit_logistic_regression_is_a_fitted_scorable_pipeline():
    train, val = _synthetic_frame(), _synthetic_frame(seed=1)
    model = fit_logistic_regression(train)
    scores = score_model(model, val)
    assert 0.0 <= scores["roc_auc"] <= 1.0
    assert 0.0 <= scores["pr_auc"] <= 1.0
