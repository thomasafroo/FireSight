"""Join daily ERA5-Land weather onto a (cell, date) table.

ERA5-Land is delivered on its own ~9km (0.1 degree) lat/lon grid, coarser
than the 5km fire-detection grid, so multiple grid cells share the same
nearest ERA5 point — a simple planar nearest-neighbor is enough at this
latitude and extent (a few hundred km across); it would need a haversine
correction to hold up over a much larger area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Instantaneous readings -> daily mean.
INSTANT_VARS = ["t2m", "d2m", "u10", "v10", "swvl1"]


def load_era5_daily(paths: list[Path]) -> pd.DataFrame:
    """Load monthly ERA5-Land files and aggregate to one row per (lat, lon, date).

    `tp` (total_precipitation) needs different handling than the
    instantaneous variables below — see `_extract_daily_precip`'s docstring.
    Converted from metres to mm.
    """
    instant_frames = []
    tp_frames = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            daily_mean = ds[INSTANT_VARS].resample(valid_time="1D").mean()
            instant_frames.append(daily_mean.to_dataframe().reset_index())
            tp_frames.append(ds[["tp"]].to_dataframe().reset_index())

    instant_df = pd.concat(instant_frames, ignore_index=True).rename(columns={"valid_time": "date"})
    tp_raw = pd.concat(tp_frames, ignore_index=True)
    precip_df = _extract_daily_precip(tp_raw)

    combined = instant_df.merge(precip_df, on=["latitude", "longitude", "date"], how="left")
    combined["date"] = pd.to_datetime(combined["date"])

    keep = ["latitude", "longitude", "date", *INSTANT_VARS, "precip_mm"]
    return combined[keep]


def _extract_daily_precip(tp_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract true daily precipitation totals from raw 6-hourly `tp` samples.

    ERA5-Land's accumulated fields (docs/04-weather-join.md has the full
    story, including how an earlier, wrong reading of this got caught)
    accumulate from 00 UTC and run to the *end* of each day: the value
    stamped 06/12/18h is the cumulative total *since that day's 00 UTC*
    (not an independent 6h chunk), and the value stamped 00 UTC is the
    *previous* day's complete 24h total (step=24 of the prior day's forecast)
    — confirmed via ECMWF's Confluence documentation
    (https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790).
    So a day D's true total is the sample timestamped D+1 at 00:00, and the
    06/12/18h samples are redundant partial-day subsets, not summed.
    """
    midnight = tp_raw[tp_raw["valid_time"].dt.hour == 0].copy()
    midnight["date"] = midnight["valid_time"] - pd.Timedelta(days=1)
    midnight = midnight.rename(columns={"tp": "precip_mm"})
    midnight["precip_mm"] *= 1000.0
    return midnight[["latitude", "longitude", "date", "precip_mm"]]


def nearest_era5_lookup(grid_cells: pd.DataFrame, era5_daily: pd.DataFrame) -> pd.DataFrame:
    """Map each grid cell to the lat/lon of its nearest ERA5 grid point."""
    era5_points = era5_daily[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

    lat_diff = grid_cells["latitude"].to_numpy()[:, None] - era5_points["latitude"].to_numpy()[None, :]
    lon_diff = grid_cells["longitude"].to_numpy()[:, None] - era5_points["longitude"].to_numpy()[None, :]
    nearest_idx = np.argmin(lat_diff**2 + lon_diff**2, axis=1)

    lookup = grid_cells[["cell_id"]].copy()
    lookup["era5_latitude"] = era5_points.loc[nearest_idx, "latitude"].to_numpy()
    lookup["era5_longitude"] = era5_points.loc[nearest_idx, "longitude"].to_numpy()
    return lookup


def join_weather(
    label_df: pd.DataFrame,
    grid_cells: pd.DataFrame,
    era5_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Attach daily weather to a (cell_id, date) label table via nearest-neighbor cell->ERA5 mapping."""
    lookup = nearest_era5_lookup(grid_cells, era5_daily)
    merged = label_df.merge(lookup, on="cell_id", how="left")
    merged = merged.merge(
        era5_daily,
        left_on=["era5_latitude", "era5_longitude", "date"],
        right_on=["latitude", "longitude", "date"],
        how="left",
    )
    return merged.drop(columns=["era5_latitude", "era5_longitude", "latitude", "longitude"])
