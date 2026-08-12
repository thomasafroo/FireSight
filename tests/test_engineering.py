import numpy as np
import pandas as pd

from firesight.features.engineering import (
    add_days_since_rain,
    add_relative_humidity,
    add_rolling_features,
    add_wind_features,
    drop_incomplete_history,
    engineer_features,
)


def test_add_relative_humidity_is_100_when_temp_equals_dewpoint():
    df = pd.DataFrame({"t2m": [290.0], "d2m": [290.0]})
    result = add_relative_humidity(df)
    assert np.isclose(result["relative_humidity"].item(), 100.0)


def test_add_relative_humidity_drops_below_100_when_drier():
    df = pd.DataFrame({"t2m": [290.0], "d2m": [270.0]})
    result = add_relative_humidity(df)
    assert result["relative_humidity"].item() < 100.0


def test_add_wind_features_speed_and_direction():
    df = pd.DataFrame({"u10": [3.0, 0.0], "v10": [4.0, 0.0]})
    result = add_wind_features(df)
    assert np.isclose(result["wind_speed"].iloc[0], 5.0)
    assert np.isclose(result["wind_dir_cos"].iloc[0], 3.0 / 5.0)
    assert np.isclose(result["wind_dir_sin"].iloc[0], 4.0 / 5.0)
    # zero wind -> no defined direction, must not divide by zero
    assert result["wind_speed"].iloc[1] == 0.0
    assert result["wind_dir_cos"].iloc[1] == 0.0
    assert result["wind_dir_sin"].iloc[1] == 0.0


def test_add_days_since_rain_resets_on_rain_and_counts_up_after():
    df = pd.DataFrame(
        {
            "cell_id": ["a"] * 5,
            "date": pd.date_range("2021-06-01", periods=5),
            "precip_mm": [0.0, 5.0, 0.0, 0.0, 0.0],
        }
    )
    result = add_days_since_rain(df)
    assert pd.isna(result["days_since_rain"].iloc[0])  # no rain seen yet
    assert result["days_since_rain"].iloc[1] == 0  # rain today
    assert result["days_since_rain"].iloc[2] == 1
    assert result["days_since_rain"].iloc[4] == 3


def test_add_days_since_rain_is_independent_per_cell():
    df = pd.DataFrame(
        {
            "cell_id": ["a", "b", "a", "b"],
            "date": pd.to_datetime(["2021-06-01", "2021-06-01", "2021-06-02", "2021-06-02"]),
            "precip_mm": [5.0, 0.0, 0.0, 0.0],
        }
    )
    result = add_days_since_rain(df)
    a_day2 = result[(result["cell_id"] == "a") & (result["date"] == "2021-06-02")]
    b_day2 = result[(result["cell_id"] == "b") & (result["date"] == "2021-06-02")]
    assert a_day2["days_since_rain"].item() == 1  # rained in cell a yesterday
    assert pd.isna(b_day2["days_since_rain"].item())  # cell b never saw rain


def test_add_rolling_features_needs_full_window():
    df = pd.DataFrame(
        {
            "cell_id": ["a"] * 8,
            "date": pd.date_range("2021-06-01", periods=8),
            "precip_mm": [1.0] * 8,
            "t2m": [290.0] * 8,
            "relative_humidity": [50.0] * 8,
        }
    )
    result = add_rolling_features(df)
    assert pd.isna(result["precip_7d"].iloc[5])  # only 6 rows of history so far
    assert result["precip_7d"].iloc[6] == 7.0  # exactly 7 rows now
    assert result["t2m_mean_7d"].iloc[6] == 290.0


def test_engineer_features_end_to_end_and_drop_incomplete_history():
    n = 40
    df = pd.DataFrame(
        {
            "cell_id": ["a"] * n,
            "date": pd.date_range("2021-01-01", periods=n),
            "t2m": np.linspace(270, 290, n),
            "d2m": np.linspace(260, 275, n),
            "u10": np.full(n, 2.0),
            "v10": np.full(n, 0.0),
            "precip_mm": np.where(np.arange(n) % 10 == 0, 2.0, 0.0),
            "ignited": 0,
        }
    )
    engineered = engineer_features(df)
    for col in ["days_since_rain", "precip_7d", "precip_30d", "wind_speed", "relative_humidity"]:
        assert col in engineered.columns

    complete = drop_incomplete_history(engineered)
    # first 29 rows lack a full 30-day window -> dropped
    assert len(complete) == n - 29
    assert complete.isna().sum().sum() == 0
