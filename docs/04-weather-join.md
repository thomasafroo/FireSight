# Weather join

`features/weather.py` takes 156 monthly ERA5-Land NetCDF files (2012-2024, see [Data
sources](02-data-sources.md)) and turns them into daily weather
values attached to every row of the label scaffold from [Grid & labels](03-grid-and-labels.md). Two
separate problems get solved here, and conflating them is an easy way to get silently wrong data:
**(1)** the fire grid and the weather grid don't line up, and **(2)** "aggregate sub-daily readings
into a daily value" means something different for different variables.

## Problem 1: two different grids

The fire-detection grid is 5km cells built by `build_grid_cells`. The ERA5-Land grid is its own
fixed ~9km (0.1°) grid, 18×26 = 468 points over the Kamloops bbox — set by ECMWF, not something this
project controls. A fire-grid cell's centroid almost never lands exactly on an ERA5-Land grid point.

**Nearest-neighbor join** is the fix used here (`weather.py::nearest_era5_lookup`): each fire-grid
cell gets matched to whichever ERA5-Land point is spatially closest (smallest
`lat_diff² + lon_diff²`, computed once per cell against all 468 points — see
[glossary.md](glossary.md#nearest-neighbor-join)). This means several fire-grid cells share the
exact same weather values if they're all closest to the same ERA5-Land point — expected and fine,
since weather genuinely doesn't vary meaningfully at sub-9km scale, which is the whole reason 5km
was a defensible fire-grid choice in the first place (see the tradeoff note in
[Grid & labels](03-grid-and-labels.md#the-tradeoff-in-choosing-cell_size_km--50)).

**Why not bilinear interpolation** (blending the 4 surrounding ERA5 points weighted by distance,
rather than snapping to the single closest one)? It would produce a smoother, arguably more
physically accurate estimate at each fire-grid cell. Left as a possible future refinement, not done
now, because nearest-neighbor is simpler to get right and to verify by inspection, and the ERA5-Land
grid is already fine relative to the fire grid — the smoothing bilinear interpolation buys is a
second-order correction, not a first-order one.

**Efficiency note:** the nearest-point lookup is computed **once per grid cell** (1,443 of them)
against the 468 ERA5-Land points, producing a small `cell_id -> (era5_lat, era5_lon)` lookup table.
That table is then merged onto the label scaffold on `cell_id`, and the actual weather values are
merged on `(era5_lat, era5_lon, date)`. Computing the nearest point per scaffold *row* instead (3.7M
rows × 468 points) would do the same distance calculation thousands of times over for cells that
share an answer — correct either way, but the row-level version would be meaningfully slower for no
accuracy gain.

## Problem 2: aggregating 4-times-daily readings into one daily value

ERA5-Land was fetched at 6-hour steps (00/06/12/18 UTC — `ingest_era5.py`), not the full hourly
product. Turning those 4 values per cell per day into a single daily feature requires knowing what
kind of quantity each variable *is*:

- **Instantaneous variables** (`t2m`, `d2m`, `u10`, `v10`, `swvl1`) — each reading is a snapshot at
that exact moment, like a thermometer reading. The right daily summary is a **mean** across the 4
readings (`weather.py::INSTANT_VARS`, aggregated via `.resample("1D").mean()`).
- **Accumulated variables** (`tp`, total precipitation) — not a snapshot, a running total of rain
that's fallen. Neither averaging nor summing the 4 raw values is right; see below for why, and what
is.

### The `tp` bug, and how it was actually caught

First pass got this wrong, and it's worth documenting the mistake itself, not just the fix — the
wrong version *looked* verified.

**What was tried first:** ECMWF's accumulation convention for sub-daily ERA5-Land retrievals isn't
obvious without checking, so it seemed worth confirming against real data rather than guessing.
Inspecting one cell's raw `tp` values for one day showed them going `2.6e-6 → 0.0 → 0.0 → 2.0e-4`
across 00/06/12/18h — not monotonically increasing. Since precipitation can't un-fall, a true
"cumulative from 00 UTC" series can't decrease within a day, so this looked like proof the 4 values
were independent per-interval totals, meaning the correct daily total would be their **sum**. That
reasoning shipped, with a test asserting it.

**What was actually wrong:** those sample values were all in the `1e-6`–`1e-4` metre range
(0.001–0.1mm) — right at GRIB2's packing precision floor. A flat, non-decreasing accumulation
quantized to that precision can *display* as a tiny decrease from rounding noise alone; the "proof"
was reading noise as signal. The tell that something was off wasn't in the code, it was in the
*output*: once the full pipeline ran, every single grid cell — including the one nearest Kamloops
city — implied an annual precipitation total 4-6x higher than Kamloops' actual, well-known semi-arid
climate (officially ~279mm/year; the pipeline was producing ~1750mm/year *even at the driest cell in
the whole bounding box*). A magnitude check against a fact worth knowing independently of the code
(roughly how much it rains in Kamloops) is what actually surfaced the bug — internal consistency
(the values "looked" non-monotonic) had already been satisfied and was still wrong.

**The real convention**, confirmed via
[ECMWF's Confluence documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790)
and then re-verified against a rainy day with values large enough to be unambiguous (an
11.4/6.8/8.9/11.5mm sequence, nowhere near packing precision): ERA5-Land's forecast is initialized
once daily at 00 UTC and accumulates **to the end of each forecast step**, resetting once per day —
but the sample stamped 00 UTC is the **previous** day's complete 24-hour total (step=24 of the prior
day's forecast run), and the 06/12/18h samples are each the cumulative total *since that same day's*
00 UTC — not independent chunks. So day *D*'s true total is the sample stamped 00 UTC of day *D+1*,
and the 06/12/18h samples are redundant partial-day subsets of that same total that must not be
summed in on top of it. `weather.py::_extract_daily_precip` implements this: takes only the 00h
samples, and re-attributes each one to the *previous* calendar day. (One consequence: the very last
day covered by the raw archive — 2024-12-31 — has no next-day 00h sample to pull its total from, so
it correctly comes out as missing rather than guessed;
`features/engineering.py::drop_incomplete_history` drops the affected rows downstream, same as it
does for any other insufficient-history case.)

**The lesson generalized:** a value being internally self-consistent (no crash, a plausible-looking
pattern, a test that passes) is not the same as it being *right*. The check that caught this wasn't
a sharper read of the same sample — it was checking the output against an independent, real-world
fact (Kamloops' known climate) that had nothing to do with the code at all. Where a domain fact like
that is available, it's a stronger check than re-reading the same ambiguous evidence more carefully.

## What comes out of this step

`load_era5_daily()` produces one row per `(latitude, longitude, date)` — the ERA5-Land grid's own
points, not the fire grid — with columns `t2m`, `d2m`, `u10`, `v10`, `swvl1` (daily means, still in
native units: Kelvin, m/s, m³/m³) and `precip_mm` (daily total, millimetres). `join_weather()` then
attaches these onto the label scaffold via the nearest-point lookup, giving the full
`(cell_id, date, ignited, t2m, d2m, u10, v10, swvl1, precip_mm)` table that
`pipeline/build_dataset.py` writes to `data/processed/kamloops_dataset.parquet`.

Note these are still **raw daily aggregates** — a temperature reading and a same-day rainfall total,
nothing about trends, recent dryness, or wind direction yet. That's deliberately left to
[Feature engineering](05-feature-engineering.md), a separate step, since raw daily weather and the
time-series-derived features built from it are different concerns worth being able to reason about
(and test) independently.

## A second weather source: full ERA5's CAPE and convective precipitation

The [known winter/shoulder-season blind spot](06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot)
traced most fire-season misses to a real cause, not a modeling gap: BC Wildfire Service's own
incident records show fire-season fires are 59.5% lightning-caused, but there's no per-cell/per-day
lightning-strike feature in this project to flag that risk day-by-day. A real lightning-detection
feed (e.g. the Canadian Lightning Detection Network) doesn't have public historical coverage going
back to 2012, so it can't backfill this project's training years. CAPE (convective available
potential energy — how much energy is available for an updraft, the standard meteorological proxy
for storm/lightning potential) does have that history, and comes from the same ECMWF reanalysis
family already used for `t2m`/`swvl1`/etc., just at full ERA5's coarser ~0.25° grid rather than
ERA5-Land's ~0.1° (ERA5-Land doesn't carry CAPE at all). `features/convective.py`/
`pipeline/ingest_era5_convective.py` fetch and join `cape` and `cp` (convective precipitation, kept
alongside CAPE since it's fetched in the same request at no extra cost, and is a second, related
storm-intensity signal) the same way — see [Feature engineering](05-feature-engineering.md) for how
they're engineered. Added to `training/baseline.py::FEATURE_COLUMNS` and re-tuned 2026-08-17, but not
promoted to the served model — the re-tune showed no measured benefit on the untouched 2024 test set,
see [Modeling &
evaluation](06-modeling-and-evaluation.md#the-re-tune-result-2026-08-17-overnight-run-no-measured-benefit)
for the actual numbers and the reasoning for keeping the prior 10-feature model in production.
`features/convective.py::load_convective_daily` reuses `weather.py`'s
generic `nearest_era5_lookup`/`join_weather` unchanged — both already operate on any
`(latitude, longitude, date, ...)`-shaped frame, so a second, differently-gridded source doesn't need
its own join logic, only its own loading.

**Loading it required verifying a second accumulation-convention assumption, the same way `tp` did
above — and this one turned out to be the opposite convention.** `cp` in full ERA5's
`reanalysis-era5-single-levels` product is **not** a running accumulation since 00 UTC the way
ERA5-Land's `tp` is — verified directly against a real CDS response, not assumed, given the `tp` bug
above was exactly this kind of mistake. Each hourly `cp` sample is its own independent,
already-deaccumulated value, so the daily total is a plain sum of the day's hourly values, no
shift-and-diff trick needed. That's also why the ingestion fetches hourly (not ERA5-Land's 6-hourly)
resolution: 6-hourly sampling of an already-deaccumulated variable would silently capture only 1 of
every 6 hours' rain, undercounting the same way summing `tp`'s cumulative samples once overcounted
it. `cape` is fetched at the same hourly cadence — it's the same request at no extra cost — so its
daily aggregate (`max`, not `mean`: peak instability is the meteorologically relevant number for
storm potential, not an average that washes out an afternoon spike) isn't undersampled either.

Requesting an instantaneous variable (`cape`) and an accumulated one (`cp`) together also makes CDS
return two separate NetCDF files inside one zip (split by GRIB `stepType`), unlike ERA5-Land's single
combined-file response — `ingest_era5_convective.py::_merge_zipped_response` merges them back into
one file per month so `load_convective_daily` sees the same "one NetCDF per month" shape
`load_era5_daily` already produces.
