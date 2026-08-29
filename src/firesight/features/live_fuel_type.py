"""Fuel type for one grid cell at request time, for `/predict/live` and `/predict/explain`.

Unlike `features/live_weather.py`/`live_fire.py`, this is a cache lookup, not a live fetch: BC's
Provincial Fuel Type Layer is static per-cell (`features/fuel_type.py`'s module docstring, it
doesn't change day to day, barring an actual burn), and `pipeline/build_dataset.py` already fetched
and cached every cell in the training grid via `fuel_type.fetch_or_load_fuel_types`. `/predict/live`
serves the same `BC_KAMLOOPS_BBOX` grid training used, so every `cell_id` it can be asked about is
already a key in that cache, there's nothing to fetch live here that a WFS round-trip would add.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from firesight.features.fuel_type import DEFAULT_CACHE_PATH


def load_fuel_type_lookup(cache_path: Path = DEFAULT_CACHE_PATH) -> dict[str, str]:
    """`cell_id -> fuel_type_cd`, from the cache `features/fuel_type.py` already populated."""
    fuel_types = pd.read_parquet(cache_path)
    return dict(zip(fuel_types["cell_id"], fuel_types["fuel_type_cd"]))


def build_live_fuel_type_features(
    cell_id: str,
    fuel_type_columns: list[str],
    lookup: dict[str, str],
) -> dict[str, float]:
    """One-hot `fuel_type_<code>` dict for `cell_id`, matching the served model's exact columns.

    A `cell_id` missing from the lookup, or whose code isn't among `fuel_type_columns` (e.g. a
    genuinely unmapped cell, `fetch_fuel_type` already encodes that as `UNKNOWN_FUEL_TYPE`, which
    never became a training column since it never occurred in the Kamloops FC extract), falls back
    to all-zero rather than raising, the same "a class that never occurs is just an always-zero
    column" behavior `encode_fuel_type_features` already established for training.
    """
    code = lookup.get(cell_id)
    matched_column = f"fuel_type_{code}" if code is not None else None
    return {column: (1.0 if column == matched_column else 0.0) for column in fuel_type_columns}
