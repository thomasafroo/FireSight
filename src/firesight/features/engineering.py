"""Derive time-series features from the raw daily weather table.

Panel-data feature engineering: every function here groups by `cell_id`
before computing anything, since a rolling window or "days since X" is
only meaningful within one cell's own history — see
docs/05-feature-engineering.md for the full reasoning.

Callers must run these against a frame sorted by (cell_id, date); every
function here re-sorts defensively via `_sorted`, since a silently
mis-ordered input would produce wrong-but-plausible-looking rolling
values with no error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAIN_THRESHOLD_MM = 1.0
ROLLING_WINDOWS_DAYS = (7, 30)
NEIGHBOR_FIRE_WINDOWS_DAYS = (1, 3, 7)

ENGINEERED_COLUMNS = [
    "days_since_rain",
    "precip_7d",
    "precip_30d",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "relative_humidity",
    "t2m_mean_7d",
    "t2m_trend_7d",
    "rh_mean_7d",
    "neighbor_fire_count_1d",
    "neighbor_fire_count_3d",
    "neighbor_fire_count_7d",
    # cape/convective_precip_mm arrive already-complete daily values straight from
    # features/convective.py's join (no per-cell rolling computation needed here, unlike the
    # features above) — listed so drop_incomplete_history still enforces completeness on them.
    "cape",
    "convective_precip_mm",
    # ffmc/dmc/dc/isi/bui/fwi (Canadian Forest Fire Weather Index System, features/fwi.py) arrive
    # already-complete from pipeline/build_dataset.py's separate compute_fwi call (recursive
    # day-over-day, needs grid_cells for latitude -- doesn't fit engineer_features' plain
    # per-cell-group signature, same reason cape/convective_precip_mm are joined outside it too).
    # Legitimately NaN before each year's March 1 reset (see fwi.py's module docstring) --
    # listed here so drop_incomplete_history drops those rows the same way it already drops
    # rolling-window warm-up rows, not because it's an error.
    "ffmc",
    "dmc",
    "dc",
    "isi",
    "bui",
    "fwi",
    # elevation_m/slope_degrees/aspect_sin/aspect_cos (features/topography.py) are static per-cell
    # terrain, joined in by build_dataset.py the same way cape/fwi are -- listed here so a fetch
    # gap (a cell whose elevation lookup failed) is caught the same way any other incomplete row is,
    # not silently left as a NaN feature.
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
]


def _sorted(df: pd.DataFrame, cell_col: str, date_col: str) -> pd.DataFrame:
    return df.sort_values([cell_col, date_col]).reset_index(drop=True)


def add_relative_humidity(
    df: pd.DataFrame,
    t2m_col: str = "t2m",
    d2m_col: str = "d2m",
    out_col: str = "relative_humidity",
) -> pd.DataFrame:
    """Derive %RH from temperature and dewpoint (both Kelvin) via Magnus-Tetens.

    ERA5-Land gives dewpoint, not RH directly. RH = 100 * e(Td) / e(T), where
    e(x) = exp(17.625*x / (243.04+x)) is the Magnus-Tetens saturation vapor
    pressure approximation (x in Celsius) — standard meteorological formula,
    accurate to within ~0.4% over typical terrestrial temperature ranges.
    """
    df = df.copy()
    t_c = df[t2m_col] - 273.15
    td_c = df[d2m_col] - 273.15
    sat = np.exp(17.625 * t_c / (243.04 + t_c))
    actual = np.exp(17.625 * td_c / (243.04 + td_c))
    df[out_col] = 100.0 * actual / sat
    return df


def add_wind_features(
    df: pd.DataFrame,
    u_col: str = "u10",
    v_col: str = "v10",
) -> pd.DataFrame:
    """Add wind_speed (magnitude) and a circular sin/cos encoding of direction.

    Direction is circular (0deg and 359deg are neighbors, not opposite), so a
    single raw bearing column would be a bad numeric feature for any model
    that assumes distance is linear — see docs/05-feature-engineering.md.
    cos(bearing) and sin(bearing) are just the normalized (u, v) components,
    so no trig call is needed: cos(atan2(v, u)) == u / speed by definition.
    """
    df = df.copy()
    speed = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
    df["wind_speed"] = speed
    safe_speed = speed.replace(0.0, np.nan)
    df["wind_dir_cos"] = (df[u_col] / safe_speed).fillna(0.0)
    df["wind_dir_sin"] = (df[v_col] / safe_speed).fillna(0.0)
    return df


def add_days_since_rain(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
    precip_col: str = "precip_mm",
    threshold: float = RAIN_THRESHOLD_MM,
    out_col: str = "days_since_rain",
) -> pd.DataFrame:
    """Count of days since precip last exceeded `threshold`, per cell.

    NaN for rows before a cell's first recorded wet day (there's no "last
    rain" to count from yet) — a real absence of information, not something
    to paper over with 0. Vectorized via groupby + ffill rather than a
    per-row Python loop: each row's position-in-group is carried forward
    from the last wet row's position, then subtracted from the current
    position.
    """
    df = _sorted(df, cell_col, date_col)
    pos = df.groupby(cell_col).cumcount()
    wet = df[precip_col] >= threshold
    last_wet_pos = pos.where(wet)
    last_wet_pos = last_wet_pos.groupby(df[cell_col]).ffill()
    df[out_col] = pos - last_wet_pos
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
    precip_col: str = "precip_mm",
    t2m_col: str = "t2m",
    rh_col: str = "relative_humidity",
    windows: tuple[int, ...] = ROLLING_WINDOWS_DAYS,
) -> pd.DataFrame:
    """Rolling precip sums/temp means (drought + heat buildup) and a temp trend.

    Requires `relative_humidity` to already exist (run add_relative_humidity
    first). NaN for the first `window - 1` rows of each cell's history,
    where there isn't yet a full window to summarize — see
    drop_incomplete_history below.
    """
    df = _sorted(df, cell_col, date_col)
    grouped = df.groupby(cell_col)

    for window in windows:
        df[f"precip_{window}d"] = (
            grouped[precip_col].rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
        )

    df["t2m_mean_7d"] = grouped[t2m_col].rolling(7, min_periods=7).mean().reset_index(level=0, drop=True)
    df["t2m_trend_7d"] = df[t2m_col] - grouped[t2m_col].shift(7)
    df["rh_mean_7d"] = grouped[rh_col].rolling(7, min_periods=7).mean().reset_index(level=0, drop=True)
    return df


def _moore_neighbor_adjacency(cell_ids: pd.Index) -> np.ndarray:
    """Build a (cell x cell) 0/1 adjacency matrix from `cell_id`'s "{row}_{col}" scheme.

    `features/grid.py::assign_cell_ids`/`build_grid_cells` both derive cell_id this way, so a
    cell's 8 Moore neighbors are plain integer offsets on the parsed row/col — no spatial index
    needed for a regular grid.
    """
    row_col = [tuple(int(v) for v in cid.split("_", 1)) for cid in cell_ids]
    index_of = {rc: i for i, rc in enumerate(row_col)}

    n = len(cell_ids)
    adjacency = np.zeros((n, n), dtype=np.float64)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for i, (r, c) in enumerate(row_col):
        for dr, dc in offsets:
            j = index_of.get((r + dr, c + dc))
            if j is not None:
                adjacency[i, j] = 1.0
    return adjacency


def add_neighbor_fire_features(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
    ignited_col: str = "ignited",
    windows: tuple[int, ...] = NEIGHBOR_FIRE_WINDOWS_DAYS,
) -> pd.DataFrame:
    """Count of each cell's 8 Moore neighbors that ignited in the trailing N days.

    Strictly prior-day only: the whole (date x cell) ignition panel is shifted forward one day
    *before* any rolling sum, so a window ending "today" only ever sums neighbor status through
    yesterday. This matters because one real wildfire spanning several grid cells gets detected
    on the same FIRMS day across all of them — including same-day neighbor status would leak the
    very thing being predicted, the same leakage risk `add_days_since_rain`/`add_rolling_features`
    avoid by construction (see docs/06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-
    cells-recent-fire-history).

    Requires a dense (every cell x every date) panel — true of the label scaffold this runs
    against in `pipeline/build_dataset.py`, before any row gets dropped for other reasons — so
    the date-indexed pivot below has no missing (cell, date) combinations to paper over.
    """
    df = _sorted(df, cell_col, date_col)
    pivot = df.pivot(index=date_col, columns=cell_col, values=ignited_col)
    adjacency = _moore_neighbor_adjacency(pivot.columns)

    prior = pivot.shift(1)  # day D's row becomes day D-1's ignited status per cell

    result = df.copy()
    for window in windows:
        rolled = prior.rolling(window, min_periods=window).sum()
        neighbor_counts = rolled.to_numpy() @ adjacency.T
        wide = pd.DataFrame(neighbor_counts, index=pivot.index, columns=pivot.columns)
        long = wide.stack().rename(f"neighbor_fire_count_{window}d")
        long.index.names = [date_col, cell_col]
        result = result.merge(long.reset_index(), on=[date_col, cell_col], how="left")
    return result


def engineer_features(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline in the required order."""
    df = _sorted(df, cell_col, date_col)
    df = add_relative_humidity(df)
    df = add_wind_features(df)
    df = add_days_since_rain(df, cell_col=cell_col, date_col=date_col)
    df = add_rolling_features(df, cell_col=cell_col, date_col=date_col)
    df = add_neighbor_fire_features(df, cell_col=cell_col, date_col=date_col)
    return df


def drop_incomplete_history(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop rows missing any engineered feature (insufficient lookback history).

    Deliberately a dropna on the actual engineered columns rather than a
    fixed "drop the first 30 rows per cell" rule: `days_since_rain` can stay
    NaN for longer than any rolling window if a cell's real history starts
    with an unusually long dry spell, and a fixed cutoff would silently miss
    that case.
    """
    columns = columns or ENGINEERED_COLUMNS
    return df.dropna(subset=columns).reset_index(drop=True)
