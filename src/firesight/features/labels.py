"""Build the (cell, date) label scaffold for wildfire ignition prediction.

The model needs to see cell/days where nothing burned, not just the days
something did — otherwise there's nothing to learn "no fire" from. This
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
