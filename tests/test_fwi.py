"""Reference-value tests for the Canadian FWI System translation.

Every expected value below is copied verbatim from `cffdrs/cffdrs_r`'s own
`tests/testthat/data/*.csv` fixtures (the official NRCan-maintained reference implementation this
module's docstring cites) — not hand-derived — so a passing suite means the formula *translation*
is correct against the canonical source, not just internally self-consistent.
"""

import numpy as np
import pandas as pd
import pytest

from firesight.features.fwi import (
    buildup_index,
    compute_fwi,
    fire_weather_index,
    initial_spread_index,
    next_dc,
    next_dmc,
    next_ffmc,
)


@pytest.mark.parametrize(
    "ffmc_yda, temp, rh, ws, prec, expected",
    [
        (100.2, 34.8, 55.61, 0, 186.83, 52.87),
        (10.5, 2.4, 55.61, 0, 186.83, 13.87),
        (18.6, 2.4, 55.61, 0, 186.83, 13.87),
        (42.9, 2.4, 55.61, 0, 186.83, 15.01),
        (6.6, 10.5, 55.61, 0, 186.83, 18.81),
        (2.7, 18.6, 55.61, 0, 186.83, 25.44),
    ],
)
def test_next_ffmc_matches_the_reference_implementation(ffmc_yda, temp, rh, ws, prec, expected):
    got = next_ffmc(np.array([ffmc_yda]), np.array([temp]), np.array([rh]), np.array([ws]), np.array([prec]))[0]
    assert got == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "dmc_yda, temp, rh, prec, lat, mon, expected",
    [
        (121.8, 18.6, 55.61, -10, -90, 1, 123.7),
        (210.9, 18.6, 55.61, 55.61, -90, 1, 67.97),
        (429.6, 18.6, 55.61, 55.61, -90, 1, 79.58),
        (138, 18.6, 55.61, -10, 55.8, 1, 139.08),
        (8.4, 18.6, 55.61, 55.61, 55.8, 1, 4.272),
        (227.1, 18.6, 55.61, 55.61, 55.8, 1, 68.76),
    ],
)
def test_next_dmc_matches_the_reference_implementation(dmc_yda, temp, rh, prec, lat, mon, expected):
    got = next_dmc(np.array([dmc_yda]), np.array([temp]), np.array([rh]), np.array([prec]), np.array([lat]), mon)[0]
    assert got == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "dc_yda, temp, prec, lat, mon, expected",
    [
        (121.8, 18.6, -10, -90, 1, 128.85),
        (210.9, 18.6, 55.61, -90, 1, 90.79),
        (429.6, 18.6, 55.61, -90, 1, 237.17),
        (138, 18.6, -10, 55.8, 1, 141.05),
        (8.4, 18.6, 55.61, 55.8, 1, 3.052),
        (227.1, 18.6, 55.61, 55.8, 1, 98.51),
    ],
)
def test_next_dc_matches_the_reference_implementation(dc_yda, temp, prec, lat, mon, expected):
    got = next_dc(np.array([dc_yda]), np.array([temp]), np.array([prec]), np.array([lat]), mon)[0]
    assert got == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "ffmc, ws, expected",
    [
        (0.6, 0, 2.986e-09),
        (85.2, 0, 2.164),
        (90.6, 0, 4.668),
        (50.7, 24.3, 0.6057),
        (10.2, 24.3, 4.946e-06),
    ],
)
def test_initial_spread_index_matches_the_reference_implementation(ffmc, ws, expected):
    got = initial_spread_index(np.array([ffmc]), np.array([ws]))[0]
    assert got == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize(
    "dmc, dc, expected",
    [
        (0, 0, 0),
        (2.7, 0, 1.777),
        (6.9, 218.7, 12.79),
        (35.1, 0, 33.97),
    ],
)
def test_buildup_index_matches_the_reference_implementation(dmc, dc, expected):
    got = buildup_index(np.array([dmc]), np.array([dc]))[0]
    assert got == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "isi, bui, expected",
    [
        (0.5, 218.7, 3.379),
        (2.3, 218.7, 13.91),
        (8.6, 218.7, 35.57),
    ],
)
def test_fire_weather_index_matches_the_reference_implementation(isi, bui, expected):
    got = fire_weather_index(np.array([isi]), np.array([bui]))[0]
    assert got == pytest.approx(expected, abs=0.01)


def _build_dense_panel(cell_ids: list[str], dates: pd.DatetimeIndex, **columns: float) -> pd.DataFrame:
    rows = [{"cell_id": c, "date": d, **{k: v for k, v in columns.items()}} for c in cell_ids for d in dates]
    return pd.DataFrame(rows)


def test_compute_fwi_leaves_days_before_the_first_reset_undefined():
    dates = pd.date_range("2020-01-01", "2020-01-10")
    df = _build_dense_panel(["A"], dates, t2m=280.0, relative_humidity=50.0, wind_speed=3.0, precip_mm=0.0)
    grid_cells = pd.DataFrame({"cell_id": ["A"], "latitude": [50.6]})

    result = compute_fwi(df, grid_cells)

    assert result["ffmc"].isna().all()
    assert result["dmc"].isna().all()
    assert result["dc"].isna().all()


def test_compute_fwi_resets_to_startup_values_on_the_reset_date_each_year():
    dates = pd.date_range("2019-12-25", "2020-03-05")
    df = _build_dense_panel(["A"], dates, t2m=280.0, relative_humidity=50.0, wind_speed=3.0, precip_mm=0.0)
    grid_cells = pd.DataFrame({"cell_id": ["A"], "latitude": [50.6]})

    result = compute_fwi(df, grid_cells).set_index("date")

    reset_day = pd.Timestamp("2020-03-01")
    expected_ffmc = next_ffmc(
        np.array([85.0]), np.array([280.0 - 273.15]), np.array([50.0]), np.array([3.0 * 3.6]), np.array([0.0])
    )[0]
    assert result.loc[reset_day, "ffmc"] == pytest.approx(expected_ffmc)
    assert result.loc[reset_day - pd.Timedelta(days=1)][["ffmc", "dmc", "dc"]].isna().all()


def test_compute_fwi_never_mixes_two_cells_history():
    dates = pd.date_range("2020-02-15", "2020-03-10")
    dry_hot = _build_dense_panel(["DRY"], dates, t2m=305.0, relative_humidity=15.0, wind_speed=5.0, precip_mm=0.0)
    wet_cold = _build_dense_panel(["WET"], dates, t2m=275.0, relative_humidity=90.0, wind_speed=1.0, precip_mm=10.0)
    df = pd.concat([dry_hot, wet_cold], ignore_index=True)
    grid_cells = pd.DataFrame({"cell_id": ["DRY", "WET"], "latitude": [50.6, 50.6]})

    result = compute_fwi(df, grid_cells)
    last_day = result[result["date"] == dates[-1]].set_index("cell_id")

    # A week of hot/dry/no-rain must drive FFMC well above a week of cold/wet/rainy, for the same
    # start-up values -- if the two cells' histories got mixed (e.g. a pivot/merge misalignment),
    # this contrast would collapse.
    assert last_day.loc["DRY", "ffmc"] > last_day.loc["WET", "ffmc"] + 10
    assert last_day.loc["DRY", "dc"] > last_day.loc["WET", "dc"]
