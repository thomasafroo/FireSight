import numpy as np
import pandas as pd

from firesight.evaluation.backtest import (
    monthly_capture_breakdown,
    rolling_origin_folds,
)


def test_rolling_origin_folds_gives_each_holdout_year_a_strictly_earlier_expanding_train_set():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2021-07-01", "2022-06-01", "2023-06-01"]),
            "ignited": [0, 1, 0, 0, 1],
        }
    )

    folds = list(rolling_origin_folds(df, [2021, 2022, 2023]))

    assert [year for year, _, _ in folds] == [2021, 2022, 2023]

    _year, train, holdout = folds[0]
    assert list(train["date"].dt.year.unique()) == [2020]
    assert list(holdout["date"].dt.year.unique()) == [2021]
    assert len(holdout) == 2  # both 2021 rows

    _year, train, holdout = folds[1]
    assert sorted(train["date"].dt.year.unique()) == [2020, 2021]
    assert list(holdout["date"].dt.year.unique()) == [2022]

    _year, train, holdout = folds[2]
    assert sorted(train["date"].dt.year.unique()) == [2020, 2021, 2022]
    assert list(holdout["date"].dt.year.unique()) == [2023]


def test_rolling_origin_folds_yields_nothing_for_an_empty_holdout_year_list():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-06-01"]), "ignited": [0]})
    assert list(rolling_origin_folds(df, [])) == []


def test_monthly_capture_breakdown_reports_capture_rate_per_month_of_the_fires_present():
    # 20 rows: a July fire scored high enough to land in the top 10% (caught), an August fire
    # scored low (missed), and 18 non-fire filler rows scored in between so the top-10% cutoff
    # (2 of 20 rows) lands exactly on the July fire and nothing else.
    dates = pd.to_datetime(["2023-07-15", "2023-08-20", *[f"2023-06-{d:02d}" for d in range(1, 19)]])
    y_true = np.array([1, 1] + [0] * 18)
    y_score = np.array([0.99, 0.01] + [0.5] * 18)

    table = monthly_capture_breakdown(dates, y_true, y_score, k_fraction=0.1)

    assert set(table["month"]) == {7, 8}
    july = table[table["month"] == 7].iloc[0]
    august = table[table["month"] == 8].iloc[0]
    assert july["total"] == 1 and july["caught"] == 1 and july["capture_rate"] == 1.0
    assert august["total"] == 1 and august["caught"] == 0 and august["capture_rate"] == 0.0


def test_monthly_capture_breakdown_excludes_months_with_no_fires():
    dates = pd.to_datetime(["2023-07-15", "2023-08-20"])
    y_true = np.array([1, 0])
    y_score = np.array([0.9, 0.1])

    table = monthly_capture_breakdown(dates, y_true, y_score, k_fraction=0.5)

    assert list(table["month"]) == [7]
