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
    # cape/convective_precip_mm arrive already-complete daily values straight from
    # features/convective.py's join (no per-cell rolling computation needed here, unlike the
    # features above) — listed so drop_incomplete_history still enforces completeness on them.
    "cape",
    "convective_precip_mm",
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
