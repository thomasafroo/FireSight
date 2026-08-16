import datetime as dt

import pandas as pd
import pytest

from firesight.features.live_weather import build_live_feature_row, fetch_recent_weather


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_fetch_recent_weather_converts_temperature_to_kelvin_and_requests_ms_wind(monkeypatch):
    captured_params = {}

    def fake_get(url, params, timeout):
        captured_params.update(params)
        return _FakeResponse(
            {
                "daily": {
                    "time": ["2024-07-01", "2024-07-02"],
                    "temperature_2m_mean": [20.0, None],
                    "relative_humidity_2m_mean": [40.0, 45.0],
                    "precipitation_sum": [0.0, 1.5],
                    "soil_moisture_0_to_7cm_mean": [0.05, 0.06],
                    "wind_speed_10m_mean": [2.0, 3.0],
                }
            }
        )

    monkeypatch.setattr("firesight.features.live_weather.requests.get", fake_get)

    df = fetch_recent_weather(50.6, -120.3, dt.date(2024, 7, 2), lookback_days=1)

    assert captured_params["wind_speed_unit"] == "ms"
    assert df["t2m"].iloc[0] == pytest.approx(293.15)
    assert pd.isna(df["t2m"].iloc[1])
    assert df["relative_humidity"].tolist() == [40.0, 45.0]
    assert df["precip_mm"].tolist() == [0.0, 1.5]
    assert df["swvl1"].tolist() == [0.05, 0.06]
    assert df["wind_speed"].tolist() == [2.0, 3.0]


def _make_history(days: int, end_date: dt.date, rain_on_day: int | None = 0) -> pd.DataFrame:
    """A synthetic `lookback_days`-shaped history: dry throughout except optionally
    one rainy day (index from the start), so add_days_since_rain has something to
    count from and add_rolling_features has a full window to work with."""
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    precip = [0.0] * days
    if rain_on_day is not None:
        precip[rain_on_day] = 5.0
    return pd.DataFrame(
        {
            "date": dates,
            "t2m": [295.0 + i * 0.1 for i in range(days)],
            "relative_humidity": [30.0] * days,
            "precip_mm": precip,
            "swvl1": [0.1] * days,
            "wind_speed": [3.0] * days,
        }
    )


def test_build_live_feature_row_computes_a_full_feature_vector(monkeypatch):
    target_date = dt.date(2024, 7, 15)
    history = _make_history(46, target_date, rain_on_day=0)
    monkeypatch.setattr(
        "firesight.features.live_weather.fetch_recent_weather",
        lambda latitude, longitude, end_date, lookback_days=45: history,
    )

    features = build_live_feature_row(50.6, -120.3, target_date, "1105_-1677")

    for key in ("t2m", "swvl1", "precip_mm", "relative_humidity", "wind_speed", "days_since_rain", "precip_7d", "precip_30d", "t2m_mean_7d", "rh_mean_7d"):
        assert key in features
        assert features[key] == features[key]  # not NaN


def test_build_live_feature_row_raises_when_no_rain_in_the_lookback_window(monkeypatch):
    target_date = dt.date(2024, 7, 15)
    history = _make_history(46, target_date, rain_on_day=None)
    monkeypatch.setattr(
        "firesight.features.live_weather.fetch_recent_weather",
        lambda latitude, longitude, end_date, lookback_days=45: history,
    )

    with pytest.raises(ValueError, match="Insufficient weather history"):
        build_live_feature_row(50.6, -120.3, target_date, "1105_-1677")


def test_build_live_feature_row_raises_when_target_date_missing_from_response(monkeypatch):
    target_date = dt.date(2024, 7, 15)
    history = _make_history(46, target_date - dt.timedelta(days=1), rain_on_day=0)
    monkeypatch.setattr(
        "firesight.features.live_weather.fetch_recent_weather",
        lambda latitude, longitude, end_date, lookback_days=45: history,
    )

    with pytest.raises(ValueError, match="no row"):
        build_live_feature_row(50.6, -120.3, target_date, "1105_-1677")
