"""Canadian Forest Fire Weather Index (FWI) System — FFMC, DMC, DC, ISI, BUI, FWI.

This is the fire-danger rating BC Wildfire Service (and every other Canadian fire agency) actually
runs operationally, not one this project invented. Six components, each a recursive day-over-day
fuel-moisture code (except ISI/BUI/FWI, which are same-day combinations of the other three):

- **FFMC** (Fine Fuel Moisture Code): moisture in surface litter — reacts fast (hours-to-a-day
  memory). Drives ignition ease and fire spread rate.
- **DMC** (Duff Moisture Code): moisture in loosely-compacted organic layers a few cm down —
  weeks of memory.
- **DC** (Drought Code): moisture in deep, compact organic layers — months of memory, the
  System's proxy for seasonal drought.
- **ISI** (Initial Spread Index): FFMC + wind, combined into an expected spread-rate index.
- **BUI** (Buildup Index): DMC + DC, combined into a fuel-consumption index.
- **FWI** (Fire Weather Index): ISI + BUI, the System's single headline number.

Formulas transcribed from the official NRCan-maintained reference implementation
(`cffdrs/cffdrs_r`, `R/fine_fuel_moisture_code.r`, `R/duff_moisture_code.r`, `R/drought_code.r`,
`R/initial_spread_index.r`, `R/buildup_index.r`, `R/fire_weather_index.r` — equation numbers below
refer to Van Wagner & Pickett 1985, "Equations and FORTRAN program for the Canadian Forest Fire
Weather Index System", Forestry Technical Report 33), not re-derived from a textbook description —
translation checked against that package's own `tests/testthat/data/*.csv` reference fixtures (see
`tests/test_fwi.py`), not just "looks plausible."

**Units, matching this project's existing raw columns, not CFFDRS's own convention:** temperature in
Celsius (this module converts from `t2m`'s Kelvin), wind speed in km/h (`wind_speed` is m/s from
ERA5's `u10`/`v10`, converted here — CFFDRS's own wind-effect formulas are calibrated to km/h and
silently give the wrong magnitude if handed m/s directly), relative humidity 0-100, precipitation
in mm (already the case for this project's `precip_mm`, no conversion needed).

**Deliberate simplification: no snow-cover data, so no real spring start-up date.** CFFDRS resets
FFMC/DMC/DC to standard start-up values (85/6/15) each year once snow has melted at a given station —
`RESET_MONTH_DAY` below (March 1) is a fixed calendar-date stand-in for that, chosen to give ~2
months of DMC/DC buildup before FIRE_SEASON_START (May 1), not a station-verified melt date. Every
day before that year's first reset (e.g. Jan/Feb of the dataset's first year) has no valid prior-day
state to recurse from and is left `NaN` rather than guessed — the same "don't paper over missing
history" precedent `engineering.py::add_days_since_rain` already set. Running the raw recursion
through a snow-covered BC winter would misread snow-water-equivalent as duff-wetting rain and is not
attempted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0

STARTUP_FFMC = 85.0
STARTUP_DMC = 6.0
STARTUP_DC = 15.0
RESET_MONTH_DAY = "03-01"

FWI_COLUMNS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]

# DMC day-length adjustment (Le), by month (Jan..Dec) — Van Wagner 1987, Table.
_DMC_ELL_46N = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])  # lat > 30
_DMC_ELL_20N = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])  # 10 < lat <= 30
_DMC_ELL_20S = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2])  # -30 < lat <= -10
_DMC_ELL_40S = np.array([11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8])  # lat <= -30

# DC day-length factor (Lf), by month (Jan..Dec).
_DC_FL_20N = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])  # lat > 20
_DC_FL_20S = np.array([6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8])  # lat <= -20


def _dmc_day_length(latitude: np.ndarray, month: int) -> np.ndarray:
    """Per-cell Le for DMC's drying-rate equation, selected by latitude band (Van Wagner 1987)."""
    m = month - 1
    return np.select(
        [latitude > 30, (latitude <= 30) & (latitude > 10), (latitude <= -10) & (latitude > -30), latitude <= -30],
        [_DMC_ELL_46N[m], _DMC_ELL_20N[m], _DMC_ELL_20S[m], _DMC_ELL_40S[m]],
        default=9.0,  # -10 < lat <= 10, equatorial band: a flat factor, no seasonal day-length swing
    )


def _dc_day_length_factor(latitude: np.ndarray, month: int) -> np.ndarray:
    """Per-cell Lf for DC's potential-evapotranspiration equation, selected by latitude band."""
    m = month - 1
    return np.select(
        [latitude > 20, latitude <= -20],
        [_DC_FL_20N[m], _DC_FL_20S[m]],
        default=1.4,  # -20 < lat <= 20
    )


def next_ffmc(ffmc_prev: np.ndarray, temp_c: np.ndarray, rh: np.ndarray, wind_kmh: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
    """One day's FFMC update (Eqs. 1-10). All array args must already be same-shaped/broadcastable.

    Sub-expressions below are computed for every element regardless of which branch a given cell
    actually falls in (`np.where`'s both arguments are always evaluated) — safe by construction
    since `np.where` only ever returns values from the selected branch, but a branch computed on
    out-of-domain inputs for its *other* cells (e.g. `log`/`sqrt` of a value that only makes sense
    when it's raining) can raise a `RuntimeWarning`. `np.errstate` suppresses those; the discarded,
    possibly-NaN results from the unselected branch never reach the return value.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        # Eq. 1 — yesterday's FFMC as moisture content.
        wmo = FFMC_COEFFICIENT * (101 - ffmc_prev) / (59.5 + ffmc_prev)

        # Eq. 2 — rain reduced for canopy interception loss.
        ra = np.where(precip_mm > 0.5, precip_mm - 0.5, precip_mm)
        ra_safe = np.where(ra > 0, ra, 1.0)  # guards Eqs. 3a/3b's exp(-6.93/ra) on non-rain rows
        # Eqs. 3a, 3b — moisture after rain, two sub-formulas depending on how wet it already was.
        rained_high = wmo + 0.0015 * (wmo - 150) ** 2 * np.sqrt(ra_safe) + 42.5 * ra_safe * np.exp(-100 / (251 - wmo)) * (
            1 - np.exp(-6.93 / ra_safe)
        )
        rained_low = wmo + 42.5 * ra_safe * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra_safe))
        wmo = np.where(precip_mm > 0.5, np.where(wmo > 150, rained_high, rained_low), wmo)
        wmo = np.minimum(wmo, 250.0)  # real pine-litter moisture tops out around 250%

        # Eqs. 4, 5 — equilibrium moisture content from drying / from wetting.
        ed = 0.942 * rh**0.679 + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp_c) * (1 - 1 / np.exp(rh * 0.115))
        ew = 0.618 * rh**0.753 + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp_c) * (1 - 1 / np.exp(rh * 0.115))

        drying = (wmo < ed) & (wmo < ew)
        wetting = wmo > ed

        # Eqs. 6a/6b — log drying rate, temperature-adjusted.
        z_dry = np.where(
            drying,
            0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1 - ((100 - rh) / 100) ** 8),
            0.0,
        )
        x_dry = z_dry * 0.581 * np.exp(0.0365 * temp_c)
        # Eq. 9
        wm_dry = ew - (ew - wmo) / (10**x_dry)

        # Eqs. 7a/7b — log wetting rate, temperature-adjusted (default/else carries z_dry forward,
        # matching the R source's sequential-overwrite semantics — harmless since drying/wetting
        # are mutually exclusive by construction).
        z_wet = np.where(
            wetting,
            0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1 - (rh / 100) ** 8),
            z_dry,
        )
        x_wet = z_wet * 0.581 * np.exp(0.0365 * temp_c)
        # Eq. 8
        wm_wet = ed + (wmo - ed) / (10**x_wet)

        wm = np.where(drying, wm_dry, wmo)
        wm = np.where(wetting, wm_wet, wm)

        # Eq. 10
        ffmc = 59.5 * (250 - wm) / (FFMC_COEFFICIENT + wm)
    return np.clip(ffmc, 0.0, 101.0)


def next_dmc(dmc_prev: np.ndarray, temp_c: np.ndarray, rh: np.ndarray, precip_mm: np.ndarray, latitude: np.ndarray, month: int) -> np.ndarray:
    """One day's DMC update (Eqs. 11-16)."""
    temp_c = np.maximum(temp_c, -1.1)
    le = _dmc_day_length(latitude, month)
    # Eq. 16 — log drying rate.
    rk = 1.894 * (temp_c + 1.1) * (100 - rh) * le * 1e-4

    with np.errstate(invalid="ignore", divide="ignore"):
        dmc_safe = np.where(dmc_prev > 0, dmc_prev, 1.0)  # guards the log(dmc_prev) branches below
        # Eq. 11 — net rain.
        rw = 0.92 * precip_mm - 1.27
        # Eq. 12 (as amended in the reference implementation) — moisture content before rain.
        wmi = 20 + 280 / np.exp(0.023 * dmc_prev)
        # Eqs. 13a-13c — a slope term, piecewise in dmc_prev.
        b = np.select(
            [dmc_prev <= 33, dmc_prev <= 65],
            [100 / (0.5 + 0.3 * dmc_prev), 14 - 1.3 * np.log(dmc_safe)],
            default=6.2 * np.log(dmc_safe) - 17.2,
        )
        # Eq. 14 — moisture content after rain.
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        wmr_safe = np.where(wmr > 20, wmr - 20, 1.0)  # guards the log below when rain barely mattered
        # Eq. 15 (amended) — P after rain.
        pr_wet = 43.43 * (5.6348 - np.log(wmr_safe))

    # Rain of 1.5mm or less doesn't reach the duff layer; DMC carries over unchanged.
    pr = np.where(precip_mm <= 1.5, dmc_prev, pr_wet)
    pr = np.maximum(pr, 0.0)
    return np.maximum(pr + rk, 0.0)


def next_dc(dc_prev: np.ndarray, temp_c: np.ndarray, precip_mm: np.ndarray, latitude: np.ndarray, month: int) -> np.ndarray:
    """One day's DC update (Eqs. 18-23)."""
    temp_c = np.maximum(temp_c, -2.8)
    lf = _dc_day_length_factor(latitude, month)
    # Eq. 22 — potential evapotranspiration, floored at 0 (no negative winter values).
    pe = np.maximum((0.36 * (temp_c + 2.8) + lf) / 2, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Eq. 18 — effective rainfall.
        rw = 0.83 * precip_mm - 1.27
        # Eq. 19 — moisture equivalent of yesterday's DC.
        smi = 800 * np.exp(-dc_prev / 400)
        # Eq. 21 (amended) — DC after rain.
        dr0 = np.maximum(dc_prev - 400 * np.log1p(3.937 * rw / smi), 0.0)

    # Rain of 2.8mm or less doesn't reach this deep; DC carries over unchanged.
    dr = np.where(precip_mm <= 2.8, dc_prev, dr0)
    # Eq. 23 (amended)
    return np.maximum(dr + pe, 0.0)


def initial_spread_index(ffmc: np.ndarray, wind_kmh: np.ndarray) -> np.ndarray:
    """ISI (Eqs. 24-26): FFMC + wind, same-day, no recursion."""
    fm = FFMC_COEFFICIENT * (101 - ffmc) / (59.5 + ffmc)
    fw = np.exp(0.05039 * wind_kmh)
    ff = 91.9 * np.exp(-0.1386 * fm) * (1 + fm**5.31 / 49_300_000)
    return 0.208 * fw * ff


def buildup_index(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """BUI (Eq. 27): DMC + DC, same-day, no recursion."""
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = dmc + 0.4 * dc
        bui1 = np.where((dmc == 0) & (dc == 0), 0.0, 0.8 * dc * dmc / np.where(denom != 0, denom, 1.0))
        p = np.where(dmc == 0, 0.0, (dmc - bui1) / np.where(dmc != 0, dmc, 1.0))
        cc = 0.92 + (0.0114 * dmc) ** 1.7
        bui0 = np.maximum(dmc - cc * p, 0.0)
    return np.where(bui1 < dmc, bui0, bui1)


def fire_weather_index(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
    """FWI (Eqs. 28-30): ISI + BUI combined into the System's headline number."""
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = np.where(
            bui > 80,
            0.1 * isi * (1000 / (25 + 108.64 / np.exp(0.023 * bui))),
            0.1 * isi * (0.626 * bui**0.809 + 2),
        )
        bb_safe = np.where(bb > 0, bb, 1.0)
        fwi = np.where(bb <= 1, bb, np.exp(2.72 * (0.434 * np.log(bb_safe)) ** 0.647))
    return fwi


def compute_fwi(
    df: pd.DataFrame,
    grid_cells: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
    t2m_col: str = "t2m",
    rh_col: str = "relative_humidity",
    wind_col: str = "wind_speed",
    precip_col: str = "precip_mm",
) -> pd.DataFrame:
    """Add `ffmc`/`dmc`/`dc`/`isi`/`bui`/`fwi` columns via the recursive daily System.

    Requires a dense (every cell x every date) panel, same as
    `engineering.py::add_neighbor_fire_features` — true of the label scaffold this runs against in
    `pipeline/build_dataset.py`. `grid_cells` supplies each cell's latitude (for DMC/DC's day-length
    tables) via `features/grid.py::build_grid_cells`.

    Recurses date-by-date (every cell updated together, vectorized across cells) rather than
    cell-by-cell, since the per-day formulas are already fully vectorizable and this keeps the
    Python-level loop count to `n_dates` (~4,700 for this project's 2012-2024 range) instead of
    `n_dates * n_cells`.
    """
    df = df.sort_values([cell_col, date_col]).reset_index(drop=True)

    temp_c = df[t2m_col] - 273.15
    wind_kmh = df[wind_col] * 3.6  # wind_speed is m/s (from ERA5's u10/v10); CFFDRS expects km/h

    wide_temp = df.assign(_v=temp_c).pivot(index=date_col, columns=cell_col, values="_v")
    wide_rh = df.pivot(index=date_col, columns=cell_col, values=rh_col)
    wide_wind = df.assign(_v=wind_kmh).pivot(index=date_col, columns=cell_col, values="_v")
    wide_precip = df.pivot(index=date_col, columns=cell_col, values=precip_col)

    cell_ids = wide_temp.columns
    dates = wide_temp.index
    latitude = grid_cells.set_index("cell_id").reindex(cell_ids)["latitude"].to_numpy()

    temp_arr = wide_temp.to_numpy()
    rh_arr = wide_rh.to_numpy()
    wind_arr = wide_wind.to_numpy()
    precip_arr = wide_precip.to_numpy()

    n_dates, n_cells = temp_arr.shape
    ffmc_hist = np.full((n_dates, n_cells), np.nan)
    dmc_hist = np.full((n_dates, n_cells), np.nan)
    dc_hist = np.full((n_dates, n_cells), np.nan)

    ffmc_prev = np.full(n_cells, np.nan)
    dmc_prev = np.full(n_cells, np.nan)
    dc_prev = np.full(n_cells, np.nan)

    month_day = dates.strftime("%m-%d")
    months = dates.month.to_numpy()

    for i in range(n_dates):
        if month_day[i] == RESET_MONTH_DAY:
            ffmc_prev = np.full(n_cells, STARTUP_FFMC)
            dmc_prev = np.full(n_cells, STARTUP_DMC)
            dc_prev = np.full(n_cells, STARTUP_DC)
        elif np.isnan(ffmc_prev).all():
            continue  # before this record's first reset date -- no prior state to recurse from

        ffmc_today = next_ffmc(ffmc_prev, temp_arr[i], rh_arr[i], wind_arr[i], precip_arr[i])
        dmc_today = next_dmc(dmc_prev, temp_arr[i], rh_arr[i], precip_arr[i], latitude, months[i])
        dc_today = next_dc(dc_prev, temp_arr[i], precip_arr[i], latitude, months[i])

        ffmc_hist[i] = ffmc_today
        dmc_hist[i] = dmc_today
        dc_hist[i] = dc_today
        ffmc_prev, dmc_prev, dc_prev = ffmc_today, dmc_today, dc_today

    isi_hist = initial_spread_index(ffmc_hist, wind_arr)
    bui_hist = buildup_index(dmc_hist, dc_hist)
    fwi_hist = fire_weather_index(isi_hist, bui_hist)

    result = df.copy()
    for arr, name in [
        (ffmc_hist, "ffmc"),
        (dmc_hist, "dmc"),
        (dc_hist, "dc"),
        (isi_hist, "isi"),
        (bui_hist, "bui"),
        (fwi_hist, "fwi"),
    ]:
        wide = pd.DataFrame(arr, index=dates, columns=cell_ids)
        long = wide.stack().rename(name)
        long.index.names = [date_col, cell_col]
        result = result.merge(long.reset_index(), on=[date_col, cell_col], how="left")
    return result
