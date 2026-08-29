# Feature engineering

> **Status: implemented**, `features/engineering.py`. This page originally described the plan
> before the code existed; it now describes what actually shipped, including a couple of places the
> real implementation is cleaner than the original plan.

## Why raw daily weather isn't enough

`data/processed/kamloops_dataset.parquet` (built by [the weather join](04-weather-join.md)) has
same-day weather: today's average temperature, today's total rainfall. But fire risk isn't really a
function of *today's* weather in isolation, a hot, dry day after two weeks of drought is much
higher risk than an identical hot, dry day right after a week of rain. The model needs access to
**recent history**, not just a single snapshot, to have any chance of learning that distinction.
Turning a snapshot into a history-aware feature is what "feature engineering" means concretely in
this step.

## This is panel data, not a single time series

If you've seen time series concepts before (lag features, rolling windows, train/test splits that
respect time order), they all apply here, with one twist. A textbook time series problem usually
has *one* series (one stock's price over time). This dataset has **1,443 parallel series**, one per
grid cell, each with its own multi-year history. This shape, many units, each observed repeatedly
over time, is usually called **panel data** (or longitudinal data) rather than a pure time series
problem.

The practical consequence, and why every function in `features/engineering.py` groups by `cell_id`
first: a 7-day rolling rainfall sum for cell A must never include cell B's rainfall, even though
both rows sit next to each other after sorting by date. See [glossary.md](glossary.md#panel-data).

## Why this can't be a `ColumnTransformer` step

`sklearn.compose.ColumnTransformer` (and `sklearn` preprocessing generally, `StandardScaler`,
`OneHotEncoder`, `SimpleImputer`) is **stateless per row**: each row is transformed using only
information already in that row (or a fitted global statistic like a column mean). A rolling 7-day
rainfall sum needs the *previous 6 rows for that same cell*, information not available to a
transformer looking at one row at a time. That's why `features/engineering.py` is plain `pandas`,
upstream of any `sklearn` pipeline, see
[Modeling & evaluation](06-modeling-and-evaluation.md#where-columntransformer-fits) for where
`sklearn` preprocessing does come in, later.

## What actually got built

`engineer_features(df)` runs, in order, on a frame sorted by `(cell_id, date)`:

- **`relative_humidity`** (`add_relative_humidity`), ERA5-Land gives dewpoint temperature (`d2m`),
not relative humidity directly. RH is derived via the **Magnus-Tetens approximation**:
`RH = 100 * e(Td) / e(T)`, where `e(x) = exp(17.625x / (243.04+x))` is saturation vapor pressure as
a function of temperature in Celsius, a standard meteorological formula, accurate to within ~0.4%
over typical terrestrial ranges. This is the "humidity" feature from the original plan; it doesn't
exist as a raw ERA5-Land field at all, it's computed from two that do.
- **`wind_speed`, `wind_dir_sin`, `wind_dir_cos`** (`add_wind_features`), speed is
`sqrt(u10² + v10²)`. Direction is encoded as `sin`/`cos` rather than a raw bearing, because compass
direction is **circular** (0° and 359° are neighbors, not far apart) and a plain numeric angle
column would misrepresent that to any model treating feature distance linearly (see
[glossary.md](glossary.md#circular-encoding)). The implementation doesn't actually compute an angle
at all: `cos(bearing)` and `sin(bearing)` are mathematically identical to the *normalized* wind
components, `u10 / speed` and `v10 / speed`, so that's what's computed directly, skipping
`atan2`/`sin`/`cos` calls entirely for the same result. Zero-wind rows (`speed == 0`) get `0.0` for
both rather than dividing by zero.
- **`days_since_rain`** (`add_days_since_rain`), count of days since `precip_mm` last reached
`RAIN_THRESHOLD_MM` (1.0mm) in that cell, computed via `groupby(cell_id).cumcount()` (each row's
position within its cell's history) combined with a grouped forward-fill of "the position of the
last wet day", fully vectorized, no per-row Python loop despite being an inherently
sequential-looking computation. **`NaN`** for any row before a cell's first-ever recorded wet day,
there's no "last rain" yet to count from, and that's real missing information, not something to
default to 0 (which would falsely claim "it just rained").
- **`precip_7d`, `precip_30d`** (`add_rolling_features`), rolling 7-day and 30-day sums of
`precip_mm` per cell, via `groupby(cell_id)["precip_mm"].rolling(window, min_periods=window)`.
`min_periods=window` means these are `NaN`, not a partial sum, for any row without a full window of
history yet, a partial-window sum would silently understate drought/wetness for a cell's early
rows.
- **`t2m_mean_7d`, `t2m_trend_7d`, `rh_mean_7d`** (also `add_rolling_features`), 7-day rolling mean
temperature and relative humidity, plus a trend signal (`t2m` minus `t2m` from 7 days prior) as a
cheap way to capture "is it warming up" without a heavier decomposition method.
- **`neighbor_fire_count_1d`, `_3d`, `_7d`** (`add_neighbor_fire_features`), count of a cell's 8
[Moore](glossary.md#moore-neighborhood) (grid-adjacent) neighbors that ignited in the trailing
1/3/7 days, strictly prior-day only (the whole date x cell ignition panel is shifted forward one
day before any rolling sum, so "today's" window only ever sums neighbor status through yesterday,
otherwise one real wildfire spanning several cells, detected the same FIRMS day across all of
them, would leak the very thing being predicted). See [Modeling &
evaluation](06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-cells-recent-fire-history)
for why this was added and the test-set improvement it produced.

## Handling the `NaN`s this introduces

Rolling and lag features are genuinely undefined for a cell's early history, a 30-day rolling sum
has no valid value for a cell's first 29 rows, and (as noted above) `days_since_rain` can stay `NaN`
for even longer if a cell's real history happens to start with an unusually long dry spell. That
second case is exactly why `drop_incomplete_history(df)` **isn't** a fixed "drop the first 30 rows
per cell" rule, it's a `dropna(subset=columns)`, which removes exactly the rows genuinely missing an
engineered value, whatever the reason, rather than assuming one fixed window covers every case.

`pipeline/build_dataset.py` doesn't call this right after `engineer_features` any more:
[FWI](glossary.md#fwi-system-canadian-forest-fire-weather-index-system) (`features/fwi.py`),
terrain (`features/topography.py`), and fuel type (`features/fuel_type.py`) are merged in
afterward, and the final `drop_incomplete_history` call, right before the dataset is
written to `data/processed/kamloops_dataset.parquet`, passes `columns=ENGINEERED_COLUMNS +
<fuel-type dummy columns>`, so it enforces completeness on those too, not just this page's plain
weather-engineered features. FWI in particular introduces its own, larger NaN warm-up: every day
before the dataset's first `March 1` FFMC/DMC/DC reset has no valid recursive state to start from
(see `fwi.py`'s module docstring), a longer warm-up than any rolling window here.

In practice, `engineer_features`'s own rolling/lag features alone would drop 30 rows per cell
(1,443 × 30 = 43,290 rows), every cell in the Kamloops bbox saw rain well within its first 30 days
on record, so `precip_30d`'s window (the longest one here) is the binding constraint, not
`days_since_rain`. Adding FWI/terrain/fuel-type's completeness requirement roughly doubles that, to
61 rows per cell (1,443 × 61 = 88,023 rows) in the current dataset, the marginal rows are almost
entirely the pre-first-reset FWI warm-up, not a terrain/fuel-type fetch gap (both are static per-cell
joins with no rolling window of their own). Neither number would necessarily hold for a drier region
or a dataset starting later in the year if this pipeline is ever extended past Kamloops, worth
re-checking rather than assuming.
