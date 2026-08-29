"""Fetch recent NASA FIRMS fire detections for a grid cell's Moore neighbors, so
`/predict/live` can compute `neighbor_fire_count_{1,3,7}d` from *current* conditions instead of
only replaying a date already baked into `data/processed/kamloops_dataset.parquet`.

Source: FIRMS NRT (`VIIRS_NOAA20_NRT`), not the `VIIRS_SNPP_SP` archive `pipeline/ingest_firms.py`
uses for training. Two things verified live against the real API before writing this, not assumed
(see docs/07-serving.md#live-fire-detections-for-predictlive):

1. NRT shares the same `MAP_KEY`, endpoint shape, and 5-day-per-request limit as the `_SP` archive,
   so `ingest_firms.py::fetch_window` is reused unchanged, just chunked to cover a >5-day lookback.
2. NRT VIIRS CSVs do **not** carry a `type` column (confirmed against a real live request on
   2026-08-20), the `_SP` archive's `type==0` vegetation-fire filter (`labels.py::filter_real_fires`)
   can't be applied here, so every detection is treated as a wildfire candidate. Historically ~0.5%
   of this bbox's `_SP` detections were type 2/3 (static source/offshore, see `labels.py`), a small,
   accepted overcount, not a correctness gap worth blocking on.

`VIIRS_NOAA20_NRT`, not `VIIRS_SNPP_NRT`: Suomi NPP (SNPP) data delivery ends 2026-11-01 (per NASA
Earthdata), so building against it now would need re-pointing within weeks. Same VIIRS instrument
family as training's `VIIRS_SNPP_SP`, a different satellite platform, a real, accepted train/live
source mismatch, not a bug.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from firesight.features.engineering import NEIGHBOR_FIRE_WINDOWS_DAYS
from firesight.features.grid import assign_cell_ids, neighbor_cell_ids
from firesight.pipeline.ingest_firms import MAX_DAY_RANGE, fetch_window

NRT_SOURCE = "VIIRS_NOAA20_NRT"


def fetch_recent_detections(
    bbox: str,
    end_date: dt.date,
    lookback_days: int,
    source: str = NRT_SOURCE,
    map_key: str | None = None,
) -> pd.DataFrame:
    """Raw FIRMS detections for [end_date - lookback_days + 1, end_date].

    Chunks into <= `MAX_DAY_RANGE`-day windows (the area API's own hard limit) via
    `ingest_firms.fetch_window`, same chunking idea as `ingest_firms.fetch_archive`'s backfill loop,
    just without the checkpointing a multi-year run needs.
    """
    start_date = end_date - dt.timedelta(days=lookback_days - 1)
    chunks: list[pd.DataFrame] = []
    cursor = start_date
    remaining = lookback_days
    while remaining > 0:
        chunk_days = min(remaining, MAX_DAY_RANGE)
        chunk = fetch_window(bbox, source, cursor.isoformat(), chunk_days, map_key)
        if not chunk.empty:
            chunks.append(chunk)
        cursor += dt.timedelta(days=chunk_days)
        remaining -= chunk_days
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])


def build_live_neighbor_fire_features(
    cell_id: str,
    target_date: dt.date,
    bbox: str,
    windows: tuple[int, ...] = NEIGHBOR_FIRE_WINDOWS_DAYS,
    source: str = NRT_SOURCE,
    map_key: str | None = None,
) -> dict[str, float]:
    """`neighbor_fire_count_{window}d` for `cell_id` on `target_date`, from live NRT detections.

    Reproduces `engineering.py::add_neighbor_fire_features`'s exact semantics for one cell instead of
    the whole training panel: for each window, the count of (neighbor, day) pairs where a neighbor
    had >=1 detection on a day strictly before `target_date` and within the trailing `window` days,
    summed across all up to 8 Moore neighbors. A neighbor that ignited on multiple separate days in
    the window contributes more than 1, matching the training feature's rolling-sum-then-adjacency-
    matrix-multiply construction rather than a simplified "did any neighbor ignite" flag.
    """
    neighbors = set(neighbor_cell_ids(cell_id))
    max_window = max(windows)
    # Strictly prior days only [target_date - max_window, target_date - 1], same leakage guard as
    # training's `prior = pivot.shift(1)` before any rolling sum.
    lookback_end = target_date - dt.timedelta(days=1)

    detections = fetch_recent_detections(bbox, lookback_end, max_window, source, map_key)
    if detections.empty:
        return {f"neighbor_fire_count_{w}d": 0.0 for w in windows}

    detections = assign_cell_ids(detections)
    detections = detections[detections["cell_id"].isin(neighbors)]
    detections = detections.assign(date=pd.to_datetime(detections["acq_date"]))
    ignited_days = detections.drop_duplicates(subset=["cell_id", "date"])[["cell_id", "date"]]

    counts: dict[str, float] = {}
    for window in windows:
        window_start = pd.Timestamp(target_date) - pd.Timedelta(days=window)
        window_end = pd.Timestamp(lookback_end)
        in_window = ignited_days[(ignited_days["date"] >= window_start) & (ignited_days["date"] <= window_end)]
        counts[f"neighbor_fire_count_{window}d"] = float(len(in_window))
    return counts
