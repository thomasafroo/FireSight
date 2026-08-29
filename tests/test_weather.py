import numpy as np
import pandas as pd
import xarray as xr

from firesight.features.weather import (
    join_weather,
    load_era5_daily,
    nearest_era5_lookup,
)


def test_nearest_era5_lookup_picks_closest_point():
    grid_cells = pd.DataFrame({"cell_id": ["a", "b"], "latitude": [50.05, 50.95], "longitude": [-120.05, -120.95]})
    era5_daily = pd.DataFrame({"latitude": [50.0, 51.0], "longitude": [-120.0, -121.0]})

    lookup = nearest_era5_lookup(grid_cells, era5_daily)

    assert lookup.loc[lookup["cell_id"] == "a", "era5_latitude"].item() == 50.0
    assert lookup.loc[lookup["cell_id"] == "b", "era5_latitude"].item() == 51.0


def test_join_weather_attaches_nearest_point_weather_for_matching_date():
    grid_cells = pd.DataFrame({"cell_id": ["a"], "latitude": [50.05], "longitude": [-120.05]})
    label_df = pd.DataFrame({"cell_id": ["a"], "date": pd.to_datetime(["2021-06-01"]), "ignited": [0]})
    era5_daily = pd.DataFrame(
        {
            "latitude": [50.0],
            "longitude": [-120.0],
            "date": pd.to_datetime(["2021-06-01"]),
            "t2m": [290.0],
            "precip_mm": [1.5],
        }
    )

    merged = join_weather(label_df, grid_cells, era5_daily)

    assert merged["t2m"].item() == 290.0
    assert merged["precip_mm"].item() == 1.5


def test_load_era5_daily_averages_instant_vars_and_realigns_precip(tmp_path):
    # 3 days x 4 steps (00/06/12/18h). ERA5-Land's tp accumulates from 00 UTC
    # and resets daily, so the 00:00 sample of day N+1 holds day N's full
    # total (see docs/04-weather-join.md), the 06/12/18h values are
    # cumulative-since-midnight subsets, not independent chunks summed here.
    times = pd.date_range("2021-06-01", periods=12, freq="6h")
    tp_metres = np.array(
        [
            0.000,
            0.001,
            0.002,
            0.003,  # day1: 00h is the (unavailable) prior day's total, ignored
            0.005,
            0.001,
            0.002,
            0.004,  # day2 00h (0.005m) = day1's true total -> 5mm
            0.010,
            0.001,
            0.002,
            0.003,  # day3 00h (0.010m) = day2's true total -> 10mm
        ]
    ).reshape(12, 1, 1)
    ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.full((12, 1, 1), 290.0)),
            "d2m": (("valid_time", "latitude", "longitude"), np.full((12, 1, 1), 280.0)),
            "u10": (("valid_time", "latitude", "longitude"), np.full((12, 1, 1), 1.0)),
            "v10": (("valid_time", "latitude", "longitude"), np.full((12, 1, 1), 2.0)),
            "swvl1": (("valid_time", "latitude", "longitude"), np.full((12, 1, 1), 0.2)),
            "tp": (("valid_time", "latitude", "longitude"), tp_metres),
        },
        coords={"valid_time": times, "latitude": [50.0], "longitude": [-120.0]},
    )
    path = tmp_path / "test.nc"
    ds.to_netcdf(path)

    daily = load_era5_daily([path])

    assert len(daily) == 3  # one row per calendar day present in the file
    assert np.isclose(daily["t2m"].iloc[0], 290.0)

    by_date = daily.set_index(daily["date"].dt.date)["precip_mm"]
    assert np.isclose(by_date[pd.Timestamp("2021-06-01").date()], 5.0)
    assert np.isclose(by_date[pd.Timestamp("2021-06-02").date()], 10.0)
    # day 3 is the last day in the file -> its total needs a day-4 00:00
    # sample we don't have, so it's correctly unknown rather than guessed.
    assert pd.isna(by_date[pd.Timestamp("2021-06-03").date()])
