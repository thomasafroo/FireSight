"""Build the full Kamloops (cell, date) dataset: labels + joined weather.

Ties together features/grid.py, features/labels.py, and features/weather.py
into the end-to-end table the training pipeline will consume:
filter FIRMS to real fires -> assign cells -> build label scaffold -> join
ERA5 weather -> write to data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from firesight.features.engineering import drop_incomplete_history, engineer_features
from firesight.features.grid import assign_cell_ids, build_grid_cells
from firesight.features.labels import build_label_scaffold, filter_real_fires
from firesight.features.weather import join_weather, load_era5_daily
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX

FIRMS_PATH = Path("data/raw/firms/kamloops_2018-2024.parquet")
ERA5_DIR = Path("data/raw/era5")
OUT_PATH = Path("data/processed/kamloops_dataset.parquet")

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
CELL_SIZE_KM = 5.0


def build(
    firms_path: Path = FIRMS_PATH,
    era5_dir: Path = ERA5_DIR,
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
    engineered = engineer_features(joined)
    return drop_incomplete_history(engineered)


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
