import numpy as np
import pandas as pd
import xarray as xr

from firesight.features.convective import load_convective_daily


def test_load_convective_daily_takes_daily_max_of_cape_and_sums_convective_precip(tmp_path):
    # 2 days x 4 hourly samples (a small subset of a real 24-sample day, enough to exercise the
    # max/sum aggregation without a huge fixture). cp values here are independent per-hour
    # deaccumulated totals — verified against a real CDS response, not a running accumulation like
    # ERA5-Land's tp (see ingest_era5_convective.py) — so summing them directly gives the daily
    # total, with no shift-and-diff trick needed.
    times = pd.to_datetime(
        [
            "2021-06-01 00:00", "2021-06-01 06:00", "2021-06-01 12:00", "2021-06-01 18:00",
            "2021-06-02 00:00", "2021-06-02 06:00", "2021-06-02 12:00", "2021-06-02 18:00",
        ]
    )
    cape = np.array([100.0, 400.0, 900.0, 200.0, 50.0, 50.0, 50.0, 50.0]).reshape(8, 1, 1)
    cp_metres = np.array([0.0001, 0.0002, 0.0003, 0.0001, 0.0, 0.0, 0.0, 0.0]).reshape(8, 1, 1)

    ds = xr.Dataset(
        {
            "cape": (("valid_time", "latitude", "longitude"), cape),
            "cp": (("valid_time", "latitude", "longitude"), cp_metres),
        },
        coords={"valid_time": times, "latitude": [50.0], "longitude": [-120.0]},
    )
    path = tmp_path / "test_convective.nc"
    ds.to_netcdf(path)

    daily = load_convective_daily([path])

    assert len(daily) == 2
    by_date = daily.set_index(daily["date"].dt.date)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-06-01").date(), "cape"], 900.0)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-06-01").date(), "convective_precip_mm"], 0.7)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-06-02").date(), "cape"], 50.0)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-06-02").date(), "convective_precip_mm"], 0.0)


def test_load_convective_daily_concatenates_multiple_monthly_files(tmp_path):
    times_a = pd.to_datetime(["2021-06-01 00:00", "2021-06-01 12:00"])
    times_b = pd.to_datetime(["2021-07-01 00:00", "2021-07-01 12:00"])

    def make(times, cape_vals, cp_vals):
        return xr.Dataset(
            {
                "cape": (("valid_time", "latitude", "longitude"), np.array(cape_vals).reshape(2, 1, 1)),
                "cp": (("valid_time", "latitude", "longitude"), np.array(cp_vals).reshape(2, 1, 1)),
            },
            coords={"valid_time": times, "latitude": [50.0], "longitude": [-120.0]},
        )

    path_a = tmp_path / "a.nc"
    path_b = tmp_path / "b.nc"
    make(times_a, [10.0, 20.0], [0.0, 0.0]).to_netcdf(path_a)
    make(times_b, [30.0, 40.0], [0.0, 0.0]).to_netcdf(path_b)

    daily = load_convective_daily([path_a, path_b])

    assert len(daily) == 2
    by_date = daily.set_index(daily["date"].dt.date)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-06-01").date(), "cape"], 20.0)
    assert np.isclose(by_date.loc[pd.Timestamp("2021-07-01").date(), "cape"], 40.0)
