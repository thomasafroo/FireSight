"""Fetch recent weather for a grid cell and compute the same engineered
features training uses, so `/predict/live` can score *current* conditions
instead of only replaying a date already baked into
`data/processed/kamloops_dataset.parquet`.

Source: Open-Meteo's historical-weather API (`archive-api.open-meteo.com`),
not ERA5-Land via `cdsapi` directly. It serves the same underlying ECMWF
ERA5/ERA5-Land reanalysis the training pipeline uses, but blends in a
preliminary near-real-time product so today's values are already available
(verified 2026-08-15, same-day soil moisture came back non-null) rather
than plain ERA5-Land's ~5-day publication lag, and needs no CDS account or
API key, see docs/07-serving.md#live-weather-for-predictlive.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests

from firesight.features.engineering import add_days_since_rain, add_rolling_features

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Long enough to cover the widest rolling window (precip_30d, 30 days) plus
# slack for add_days_since_rain to find a "last rain" day within a normal
# fire-season dry spell without going NaN (see
# engineering.py::add_days_since_rain, NaN there means "no rain in the
# lookback at all", which build_live_feature_row below treats as a hard
# error rather than a fake number).
LOOKBACK_DAYS = 45

DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "soil_moisture_0_to_7cm_mean",
    "wind_speed_10m_mean",
]


def fetch_recent_weather(
    latitude: float,
    longitude: float,
    end_date: dt.date,
    lookback_days: int = LOOKBACK_DAYS,
    timeout: int = 30,
) -> pd.DataFrame:
    """One row per day from (end_date - lookback_days) to end_date.

    Columns match the subset of the raw/engineered weather schema the
    current `FEATURE_COLUMNS` actually needs: date, t2m (Kelvin, matching
    ERA5's unit), precip_mm, relative_humidity, swvl1, wind_speed (m/s,
    requested directly via `wind_speed_unit=ms` to match the unit
    `add_wind_features` derives from ERA5's u10/v10 in the training
    pipeline). Relative humidity and wind speed are taken directly from
    Open-Meteo rather than derived from t2m/d2m or u10/v10 (see
    engineering.py::add_relative_humidity/add_wind_features), Open-Meteo
    reports both directly, and neither d2m nor u10/v10/wind direction is
    needed for anything else `FEATURE_COLUMNS` still uses after the
    dead-weight-feature cleanup (see
    docs/06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on).
    """
    start_date = end_date - dt.timedelta(days=lookback_days)
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    daily = response.json()["daily"]

    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "t2m": [None if v is None else v + 273.15 for v in daily["temperature_2m_mean"]],
            "relative_humidity": daily["relative_humidity_2m_mean"],
            "precip_mm": daily["precipitation_sum"],
            "swvl1": daily["soil_moisture_0_to_7cm_mean"],
            "wind_speed": daily["wind_speed_10m_mean"],
        }
    )


def build_live_feature_row(latitude: float, longitude: float, target_date: dt.date, cell_id: str) -> dict[str, Any]:
    """Fetch recent weather and compute one cell's feature vector for target_date.

    Reuses `add_days_since_rain`/`add_rolling_features` from
    `features/engineering.py` rather than reimplementing the rolling-window
    math, so a live prediction can never silently drift from how the served
    model's training features were computed. Deliberately skips
    `add_relative_humidity`/`add_wind_features`: Open-Meteo already supplies
    relative humidity and wind speed directly (no d2m or u10/v10 to derive
    them from here), and wind *direction* isn't part of `FEATURE_COLUMNS`
    anymore, so there's nothing downstream that needs raw u10/v10.
    """
    df = fetch_recent_weather(latitude, longitude, target_date)
    df["cell_id"] = cell_id
    df = add_days_since_rain(df, cell_col="cell_id", date_col="date")
    df = add_rolling_features(df, cell_col="cell_id", date_col="date")

    row = df[df["date"] == pd.Timestamp(target_date)]
    if row.empty:
        raise ValueError(f"Open-Meteo returned no row for {target_date} at ({latitude}, {longitude}).")
    row = row.iloc[0]

    missing = row[row.isna()].index.tolist()
    if missing:
        raise ValueError(
            f"Insufficient weather history to compute {missing} for {target_date} at "
            f"({latitude}, {longitude}), e.g. no recorded rain in the last {LOOKBACK_DAYS} days "
            "for days_since_rain, or too few days of history for a rolling window."
        )
    return row.to_dict()
