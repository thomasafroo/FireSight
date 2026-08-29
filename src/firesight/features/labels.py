"""Build the (cell, date) label scaffold for wildfire ignition prediction.

The model needs to see cell/days where nothing burned, not just the days
something did, otherwise there's nothing to learn "no fire" from. This
builds the full cross-product of grid cells x dates and marks the ones
that had a real fire detection.
"""

from __future__ import annotations

import pandas as pd

# FIRMS `type` codes: 0 = presumed vegetation fire, 2 = other static land
# source, 3 = offshore. (1 = active volcano; none in the Kamloops bbox.)
# Only vegetation fires are wildfire ignitions.
VEGETATION_FIRE_TYPE = 0


def filter_real_fires(df: pd.DataFrame, type_col: str = "type") -> pd.DataFrame:
    """Keep only presumed vegetation fires, dropping volcano/static/offshore rows."""
    return df[df[type_col] == VEGETATION_FIRE_TYPE].copy()


def build_label_scaffold(
    fire_df: pd.DataFrame,
    grid_cells: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_col: str = "acq_date",
) -> pd.DataFrame:
    """Cross-join grid_cells x dates, labeling ignited=1 where a detection landed there.

    `fire_df` must already be filtered (e.g. via `filter_real_fires`) and have
    a `cell_id` column (e.g. via `assign_cell_ids`). `grid_cells` is the full
    cell universe for the region (e.g. via `build_grid_cells`), not just cells
    that appear in the fire data.
    """
    dates = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    scaffold = grid_cells[["cell_id"]].merge(dates, how="cross")

    ignited = (
        fire_df[["cell_id", date_col]]
        .rename(columns={date_col: "date"})
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .drop_duplicates()
    )
    ignited["ignited"] = 1

    scaffold = scaffold.merge(ignited, on=["cell_id", "date"], how="left")
    scaffold["ignited"] = scaffold["ignited"].fillna(0).astype(int)
    return scaffold


def add_forward_ignition_label(
    df: pd.DataFrame,
    n_days: int,
    cell_col: str = "cell_id",
    date_col: str = "date",
    ignited_col: str = "ignited",
    out_col: str | None = None,
) -> pd.DataFrame:
    """`1` if `ignited_col` is 1 on `date` or any of the following `n_days - 1` days, per cell.

    The multi-day-ahead extension docs/01-problem-framing.md names ("will a fire be detected on day
    *D* ... or within the next *N* days"), a pure label transform, not a feature: every row still
    uses only conditions known as of day *D*, this just widens *what counts as a hit* on that row.
    `n_days=1` reproduces the existing same-day `ignited` label exactly, so this is a strict
    generalization, not a parallel definition.

    Needs a dense (every cell x every date) panel, same requirement
    `add_neighbor_fire_features` (features/engineering.py) already has, for the same reason: a
    forward rolling window across gapped dates would silently span a gap it shouldn't. Rows in the
    trailing `n_days - 1` days of a cell's history (no full forward window available) get `NaN`,
    real missing information, the same "don't guess, drop it" precedent `add_days_since_rain`
    established, left for callers to drop explicitly rather than baked into this function, since
    unlike `ENGINEERED_COLUMNS` this label isn't unconditionally required by every downstream user
    of the dataset.
    """
    out_col = out_col or f"ignited_next_{n_days}d"
    df = df.sort_values([cell_col, date_col]).reset_index(drop=True)
    # No native "forward rolling" in pandas: reverse the frame (which reverses each cell's block of
    # rows too, since sort already made them contiguous), roll *backward* in that reversed order,
    # which is forward in real time, then let index-aligned assignment un-reverse it.
    reversed_rolled = (
        df.iloc[::-1].groupby(cell_col)[ignited_col].rolling(n_days, min_periods=n_days).max().reset_index(level=0, drop=True)
    )
    df[out_col] = reversed_rolled
    return df
