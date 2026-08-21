import numpy as np
import pandas as pd

from firesight.features.engineering import (
    add_days_since_rain,
    add_neighbor_fire_features,
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


def test_add_neighbor_fire_features_counts_only_moore_neighbors_strictly_prior_day():
    # 3x3 grid ("0_0".."2_2") plus one isolated cell far outside it ("5_5"), 4 days.
    grid_cells = [f"{r}_{c}" for r in range(3) for c in range(3)]
    cells = grid_cells + ["5_5"]
    dates = pd.date_range("2021-06-01", periods=4)
    df = pd.DataFrame(
        [{"cell_id": cell, "date": date, "ignited": 0} for date in dates for cell in cells]
    )
    # "0_0" (a Moore neighbor of center "1_1") and isolated "5_5" both ignite on day 1 (index 1).
    df.loc[(df["cell_id"] == "0_0") & (df["date"] == dates[1]), "ignited"] = 1
    df.loc[(df["cell_id"] == "5_5") & (df["date"] == dates[1]), "ignited"] = 1

    result = add_neighbor_fire_features(df, windows=(1,))

    def count(cell: str, date) -> float:
        return result.loc[(result["cell_id"] == cell) & (result["date"] == date), "neighbor_fire_count_1d"].item()

    # day 0: shifted panel has no prior day yet -> NaN (insufficient history), for every cell.
    assert pd.isna(count("1_1", dates[0]))
    # day 1: still only sees day 0's (all-zero) ignitions -> 0.
    assert count("1_1", dates[1]) == 0
    # day 2: "1_1"'s neighbor "0_0" ignited on day 1 -> counted.
    assert count("1_1", dates[2]) == 1
    # "2_2" is the diagonally opposite corner from "0_0" -> not a Moore neighbor -> unaffected.
    assert count("2_2", dates[2]) == 0
    # the isolated cell has no neighbors in this panel at all -> always 0, its own fire doesn't
    # count towards itself.
    assert count("5_5", dates[2]) == 0
    # day 2 is later than day 1 by only one day, so "1_1"'s own same-day (day 1) neighbor status
    # must not leak into day 1's own count (checked above: count("1_1", dates[1]) == 0 despite
    # "0_0" not having ignited yet on day 1 itself, confirming no same-day leakage either way).


def test_engineer_features_end_to_end_and_drop_incomplete_history():
    n = 40
    df = pd.DataFrame(
        {
            "cell_id": ["10_20"] * n,
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

    # cape/convective_precip_mm are joined in by build_dataset.py, not produced by
    # engineer_features itself, so this synthetic frame (which only exercises engineer_features
    # in isolation) checks completeness against the columns engineer_features actually adds.
    weather_derived_columns = [
        "days_since_rain",
        "precip_7d",
        "precip_30d",
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
        "relative_humidity",
        "t2m_mean_7d",
        "t2m_trend_7d",
        "rh_mean_7d",
        "neighbor_fire_count_1d",
        "neighbor_fire_count_3d",
        "neighbor_fire_count_7d",
    ]
    complete = drop_incomplete_history(engineered, columns=weather_derived_columns)
    # first 29 rows lack a full 30-day window -> dropped (precip_30d is still the binding
    # constraint here: this single-cell frame has no neighbors, so neighbor_fire_count_7d's own
    # warm-up is only 7 rows, well inside the 29 precip_30d already requires).
    assert len(complete) == n - 29
    assert complete[weather_derived_columns].isna().sum().sum() == 0


def test_drop_incomplete_history_defaults_also_enforce_cape_and_fwi_completeness():
    # cape/convective_precip_mm and ffmc/dmc/dc/isi/bui/fwi are joined in (build_dataset.py), not
    # engineered here, but drop_incomplete_history's *default* column list should still enforce
    # completeness on them the same way it does for every other feature.
    df = pd.DataFrame(
        {
            "days_since_rain": [1.0, 2.0],
            "precip_7d": [1.0, 2.0],
            "precip_30d": [1.0, 2.0],
            "wind_speed": [1.0, 2.0],
            "wind_dir_sin": [1.0, 2.0],
            "wind_dir_cos": [1.0, 2.0],
            "relative_humidity": [1.0, 2.0],
            "t2m_mean_7d": [1.0, 2.0],
            "t2m_trend_7d": [1.0, 2.0],
            "rh_mean_7d": [1.0, 2.0],
            "neighbor_fire_count_1d": [0.0, 0.0],
            "neighbor_fire_count_3d": [0.0, 0.0],
            "neighbor_fire_count_7d": [0.0, 0.0],
            "cape": [100.0, np.nan],
            "convective_precip_mm": [0.0, 0.0],
            "ffmc": [85.0, 85.0],
            "dmc": [6.0, 6.0],
            "dc": [15.0, 15.0],
            "isi": [1.0, 1.0],
            "bui": [1.0, 1.0],
            "fwi": [1.0, 1.0],
            "elevation_m": [500.0, 500.0],
            "slope_degrees": [2.0, 2.0],
            "aspect_sin": [0.0, 0.0],
            "aspect_cos": [1.0, 1.0],
        }
    )
    complete = drop_incomplete_history(df)
    assert len(complete) == 1
