"""Join daily CAPE / convective-precipitation onto a (cell, date) table.

Sibling to `weather.py`'s ERA5-Land join, reuses its `nearest_era5_lookup`/`join_weather` as-is,
since both already operate generically on any (latitude, longitude, date, ...)-shaped frame. What's
different here is *loading*: full ERA5 is on a coarser ~0.25 degree grid than ERA5-Land's ~0.1
degree grid, and (verified directly against a real CDS response, not assumed, see
pipeline/ingest_era5_convective.py) `cp` (convective precipitation) is NOT a running accumulation
since 00 UTC the way ERA5-Land's `tp` is: each hourly sample is its own independent, already-
deaccumulated value, so the daily total here is a plain sum of that day's 24 hourly values, not the
shift-and-diff trick `weather.py::_extract_daily_precip` needs for `tp`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

MM_PER_METRE = 1000.0


def load_convective_daily(paths: list[Path]) -> pd.DataFrame:
    """Load monthly CAPE/convective-precip files, aggregate to one row per (lat, lon, date).

    `cape` -> daily max: peak instability, not the average, is what's meteorologically relevant
    for storm potential, and a mean would wash out an afternoon spike. `cp` -> daily sum, converted
    metres -> mm.
    """
    frames = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            frames.append(ds[["cape", "cp"]].to_dataframe().reset_index())

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = raw["valid_time"].dt.floor("D")

    daily = (
        raw.groupby(["latitude", "longitude", "date"])
        .agg(cape=("cape", "max"), convective_precip_mm=("cp", "sum"))
        .reset_index()
    )
    daily["convective_precip_mm"] *= MM_PER_METRE
    return daily
