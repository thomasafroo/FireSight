"""Build the full Kamloops (cell, date) dataset: labels + joined weather.

Ties together features/grid.py, features/labels.py, and features/weather.py
into the end-to-end table the training pipeline will consume:
filter FIRMS to real fires -> assign cells -> build label scaffold -> join
ERA5 weather -> write to data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from firesight.features.convective import load_convective_daily
from firesight.features.engineering import (
    ENGINEERED_COLUMNS,
    drop_incomplete_history,
    engineer_features,
)
from firesight.features.fuel_type import build_fuel_type_features
from firesight.features.fwi import compute_fwi
from firesight.features.grid import assign_cell_ids, build_grid_cells
from firesight.features.labels import build_label_scaffold, filter_real_fires
from firesight.features.topography import build_topography_features
from firesight.features.weather import join_weather, load_era5_daily
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX

FIRMS_PATH = Path("data/raw/firms/kamloops_2012-2024.parquet")
ERA5_DIR = Path("data/raw/era5")
ERA5_CONVECTIVE_DIR = Path("data/raw/era5_convective")
OUT_PATH = Path("data/processed/kamloops_dataset.parquet")

START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
CELL_SIZE_KM = 5.0


def build(
    firms_path: Path = FIRMS_PATH,
    era5_dir: Path = ERA5_DIR,
    era5_convective_dir: Path = ERA5_CONVECTIVE_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cell_size_km: float = CELL_SIZE_KM,
) -> pd.DataFrame:
    fire = pd.read_parquet(firms_path)
    fire = filter_real_fires(fire)
    fire = assign_cell_ids(fire, cell_size_km=cell_size_km)

    grid_cells = build_grid_cells(BC_KAMLOOPS_BBOX, cell_size_km=cell_size_km)
    scaffold = build_label_scaffold(fire, grid_cells, start_date, end_date)

    era5_paths = sorted(era5_dir.glob("*.nc"))
    era5_daily = load_era5_daily(era5_paths)
    joined = join_weather(scaffold, grid_cells, era5_daily)

    convective_paths = sorted(era5_convective_dir.glob("*.nc"))
    convective_daily = load_convective_daily(convective_paths)
    joined = join_weather(joined, grid_cells, convective_daily)

    engineered = engineer_features(joined)
    # Needs relative_humidity (engineer_features's add_relative_humidity) already computed, and
    # grid_cells for per-cell latitude -- doesn't fit engineer_features' plain per-cell-group
    # signature, so it's a separate step here rather than folded into that pipeline (see
    # features/fwi.py's module docstring for the recursive day-over-day computation itself).
    with_fwi = compute_fwi(engineered, grid_cells)

    # Static per-cell terrain (features/topography.py) -- one row per cell, broadcast onto every
    # date via a plain merge on cell_id, unlike the date-varying joins above.
    topography = build_topography_features(grid_cells, cell_size_km=cell_size_km)
    with_topography = with_fwi.merge(topography, on="cell_id", how="left")

    # Static per-cell FBP fuel type (features/fuel_type.py) -- same static-per-cell join shape as
    # topography above.
    fuel_type = build_fuel_type_features(grid_cells)
    with_fuel_type = with_topography.merge(fuel_type, on="cell_id", how="left")

    return drop_incomplete_history(with_fuel_type, columns=ENGINEERED_COLUMNS + list(fuel_type.columns.drop("cell_id")))


def sanity_check(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}, cells: {df['cell_id'].nunique()}", flush=True)
    print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}", flush=True)
    print("class balance:", flush=True)
    print(df["ignited"].value_counts(normalize=True).to_string(), flush=True)
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print(f"columns with nulls: {nulls.to_dict() if len(nulls) else 'none'}", flush=True)


if __name__ == "__main__":
    dataset = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}", flush=True)
    sanity_check(dataset)
