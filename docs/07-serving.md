# Serving: model persistence, API, frontend

The last stage: get a trained model in front of something a person can actually look at. Three
pieces, each solving a distinct problem, persisting a model *safely* (not just picklable, but
usable correctly later), serving predictions, and visualizing them.

## Persisting a model without losing its contract

A model on its own, `model.predict_proba(X)`, isn't safely reusable after the training script that
produced it exits. `predict_proba` needs `X` to have exactly the right columns, in exactly the right
order, with exactly the right meaning; a bare pickle remembers none of that. If the serving code
independently hardcodes (or worse, guesses) the feature list and it ever drifts from what the model
was actually fit on, the call either shape-errors loudly or, worse, silently runs with columns in
the wrong order, producing a confidently wrong probability with no error at all.

`training/persist.py::ModelBundle` fixes this by bundling the model *with* its own contract:
`feature_columns` (the exact list, in order) and free-form `metadata` (what kind of model, when it
was validated, what it scored) travel with the model in one `joblib`-serialized object.
`bundle.predict_proba(features: dict)` selects and orders columns from `feature_columns` itself, so
there's no second place that list could drift out of sync.

`training/export_model.py::export_current_best()` is the one place that decides *which* fitted model
becomes `data/processed/model.joblib`, deliberately a separate, explicit step from `baseline.py` or
`advanced_models.py` (which explore/compare candidates), so promoting a model to "the one being
served" is a decision made after looking at results, not an automatic side effect of running a
training script. Currently exports the tuned `RandomForest` model (`BEST_RANDOM_FOREST_PARAMS`, see
[Modeling & evaluation](06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit)
for the wider-search results that put it ahead of XGBoost on every test-set metric).

This started out exporting the `LogisticRegression` baseline, then got swapped to XGBoost once the
first RandomForest/XGBoost grid-search tuning finished, then swapped again to RandomForest once a
wider `RandomizedSearchCV`+`PredefinedSplit` search (see [PredefinedSplit](glossary.md#predefinedsplit))
gave RandomForest the edge on the untouched test set. Each swap really was a one-line change in
`export_current_best` (which `fit_*` function gets called, with which params), with zero edits
anywhere in `api/main.py`, because the API only ever depends on the `ModelBundle` contract, never on
which model class filled it. This was verified live at every swap, not just assumed from the design:
restarting the API, `/health` immediately reported the new `model_type`, and `/risk-map`'s top-ranked
cells changed accordingly, the same endpoint, same request, different answer, purely because the
file on disk changed. Whatever model wins the next round of tuning gets served the same way.

## The API (`api/main.py`)

Five prediction/data endpoints, plus `/health`:

- `POST /predict`, score one (cell, day)'s worth of *already-computed* feature values, supplied
directly in the request body. It does **not** fetch live weather or run feature engineering itself,
that's what `/predict/live` below is for. The request schema (which fields are required) isn't
hand-written: it's built at startup from `bundle.feature_columns` via Pydantic's `create_model`, so
it can never silently drift from whatever model the API actually loaded, the same motivation as
`ModelBundle` itself. Kept as-is (raw feature vector in, probability out) rather than folded into
`/predict/live`, since it's the simplest possible contract for programmatic/testing access to the
model when the caller already has feature values from somewhere else.
- `GET /predict/live?cell_id=...&date=YYYY-MM-DD`, score a grid cell's **current** conditions by
fetching recent weather itself, rather than requiring the caller to supply features or replaying a
date already in the historical dataset. See [Live weather for
/predict/live](#live-weather-for-predictlive) below.
- `GET /predict/explain?cell_id=...&date=YYYY-MM-DD&top_n=6`, the same live fetch `/predict/live`
does, plus a per-prediction SHAP breakdown of which features drove that one score. See [Explaining a
live prediction](#explaining-a-live-prediction-predictexplain) below.
- `GET /risk-map?date=YYYY-MM-DD`, a historical **replay**, not a live forecast: for a date already
present in `data/processed/kamloops_dataset.parquet`, scores every one of the 1,443 grid cells using
that date's real recorded weather features, and returns each cell's predicted risk *alongside* its
actual recorded label. This exists specifically so the frontend has something real and checkable to
render (predicted risk next to what actually happened) without needing a live data feed, a
reasonable stand-in for "the model is deployed and scoring live conditions."

`/risk-map` and all three live endpoints (`/predict/live`, `/predict/live/multi-day`,
`/predict/explain`, the last three via the shared `_resolve_live_features`) reject any date outside
`training/baseline.py::FIRE_SEASON_START`–`FIRE_SEASON_END` (May 1 – Oct 15, any year) with a 400,
via the shared `_reject_if_outside_fire_season` helper, before doing any lookup or fetch. The
served model is trained exclusively on that window (see
[Modeling & evaluation](06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot)),
so scoring a December date would just be extrapolating from a model that has never seen a single
winter row, silently wrong rather than usefully wrong. `/predict` still has no equivalent guard: its
request body is a raw feature vector with no date or day-of-year field at all (`FEATURE_COLUMNS` has no
date/season field among weather, spatial-lag, and fuel-type columns alike), so there's nothing
date-shaped to validate against, a caller could still hand-construct an out-of-season feature vector
and get a number back. That's an accepted gap for the raw-vector endpoint specifically, not something
`/predict/live` inherits, since `/predict/live` always knows the date it's scoring.

### Live weather for `/predict/live`

`features/live_weather.py::build_live_feature_row` fetches recent daily weather for a cell's
centroid from [Open-Meteo's historical-weather API](https://open-meteo.com/en/docs/historical-weather-api)
(`archive-api.open-meteo.com`), not ERA5-Land via `cdsapi` directly. It serves the same underlying
ECMWF ERA5/ERA5-Land reanalysis the training pipeline is built on, but blends in a preliminary
near-real-time product so **today's** values are already available, verified live on 2026-08-15,
where a same-day request returned non-null soil moisture, rather than plain ERA5-Land's ~5-day
publication lag, and needs no CDS account, API key, or `~/.cdsapirc` setup at all. That trade (a
slightly different, blended data source vs. zero-friction access to today's weather) is deliberate:
`cdsapi`'s account/terms-acceptance friction is fine for a one-time historical backfill, but a poor
fit for a request path that needs to succeed on every API call.

It requests a 45-day lookback window ending on the target date (`LOOKBACK_DAYS`), long enough to
satisfy `precip_30d`'s 30-day rolling window with slack left over for `add_days_since_rain` to find a
"last rain" day within a normal fire-season dry spell (see
`engineering.py::add_days_since_rain`, no rain anywhere in the window means a real `NaN`, not
something to fake a number for). It then reuses `add_days_since_rain`/`add_rolling_features` from
`features/engineering.py` unchanged, the exact same functions the historical training pipeline calls,
so a live prediction can never silently drift from how the served model's training features were
computed. It deliberately skips `add_relative_humidity`/`add_wind_features`: those exist specifically
because raw ERA5-Land only gives dewpoint and wind vector components, not humidity or speed directly,
but Open-Meteo reports relative humidity and wind speed directly, so there's nothing to derive them
from (and, after [dropping the dead-weight
features](06-modeling-and-evaluation.md#dropping-the-dead-weight-features), `FEATURE_COLUMNS` no
longer needs `d2m` or `u10`/`v10` for anything else either). If any engineered column still comes
back `NaN` (insufficient history), `build_live_feature_row` raises rather than guessing, which the
endpoint surfaces as a 422.

**A dormant known limitation that never activated:** `cape`/`convective_precip_mm` were added to
`training/baseline.py::FEATURE_COLUMNS` on 2026-08-17, but the served model was deliberately **not**
re-exported against that set, the overnight re-tune showed no measured benefit (see [Modeling &
evaluation](06-modeling-and-evaluation.md#the-re-tune-result-2026-08-17-overnight-run-no-measured-benefit)),
so they were left out of the feature list entirely (see that page's [spatial-lag
section](06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-cells-recent-fire-history)) and
this limitation never actually triggered. Recorded here as a general warning, since the same shape of
problem *did* trigger for a different feature below.

### Live fire detections for `/predict/live`

**A temporary limitation as of 2026-08-19, fixed 2026-08-20: `/predict/live` was broken, now covers
both live data sources.** `neighbor_fire_count_{1,3,7}d` (a strictly-prior-day count of a cell's 8
Moore neighbors that ignited recently, see [Modeling &
evaluation](06-modeling-and-evaluation.md#spatial-lag-features-implemented-and-promoted-2026-08-19))
was promoted to the served model on the strength of a very large, carefully leakage-checked test-set
improvement. Unlike every other feature in `FEATURE_COLUMNS`, this one needs *recent fire-detection
history*, not weather, which `features/live_weather.py::build_live_feature_row` has no way to supply,
every `/predict/live` call raised a 422 unconditionally for about a day.

`features/live_fire.py::build_live_neighbor_fire_features` fixes it, fetched from NASA FIRMS' NRT
(near-real-time) product rather than the `VIIRS_SNPP_SP` archive `pipeline/ingest_firms.py` uses for
training. Two things were verified against the real API before writing this, not assumed:

- **Auth/access is free**: NRT sources share the same `MAP_KEY`, endpoint shape (`/api/area/csv/
  {map_key}/{source}/{bbox}/{day_range}/{start_date}`), and 5-day-per-request limit as the `_SP`
  archive already uses, confirmed against FIRMS' own docs. `ingest_firms.py::fetch_window` is reused
  unchanged, just chunked (`live_fire.py::fetch_recent_detections`) to cover the feature's 7-day
  lookback in two requests instead of one.
- **NRT VIIRS CSVs don't carry a `type` column**, confirmed with a real live request, not assumed
  from the `_SP` schema `labels.py::filter_real_fires` was built against. That means the `type==0`
  vegetation-fire filter can't be applied to live detections; every detection is treated as a
  candidate ignition instead. Historically ~0.5% of this bbox's `_SP` detections were type 2/3
  (static source/offshore), a small, accepted overcount in the live path, not a correctness gap
  large enough to block on.

**Source choice: `VIIRS_NOAA20_NRT`, not `VIIRS_SNPP_NRT`.** Suomi NPP (SNPP), the satellite training
data comes from (`VIIRS_SNPP_SP`), has its data delivery ending 2026-11-01 per NASA Earthdata, so the
live path was deliberately pointed at NOAA-20 instead: same VIIRS instrument family, different
satellite platform. This is a real, accepted train/live source mismatch (the historical archive and
the live feed aren't the exact same satellite), not a bug, flagged here rather than left implicit.

`features/grid.py::neighbor_cell_ids` (new) derives a cell's 8 Moore neighbor ids from its
`"{row}_{col}"` scheme without needing the full grid-cell universe, so the live path only has to fetch
detections for the 8 cells that matter. `build_live_neighbor_fire_features` then reproduces
`engineering.py::add_neighbor_fire_features`'s exact windowing for one cell: each neighbor's
detections are collapsed to one ignited/not-ignited flag per day, then summed across the trailing
1/3/7-day windows and across all 8 neighbors, so a neighbor igniting on 3 separate days in a week
contributes 3, not 1, matching training's rolling-sum-then-adjacency-matrix-multiply construction
rather than a simplified "did any neighbor ignite" flag. The same leakage guard training uses
(`prior = pivot.shift(1)`) applies here too: only days strictly before the target date count.

**Verified live, end-to-end, not just via mocked tests**: a real FIRMS NRT pull on 2026-08-20 found an
active 73-detection cluster at cell `1106_-1707`; the adjacent cell `1105_-1707` correctly returned
`neighbor_fire_count_1d/3d/7d = 4/12/26`. Hitting the running `/predict/live` endpoint for that cell
returned `ignition_probability=0.997`, versus `0.480` for an unrelated quiet cell the same day, a
real, live-data-driven contrast, not a historical replay. `/predict` (a caller-supplied raw feature
vector) and `/risk-map` (historical replay from the already-joined parquet) were already unaffected by
the original breakage and are unchanged by this fix.

### Fuel type for `/predict/live`

**Added 2026-08-21: a cache lookup, not a third live-data source.**
Once the per-group ablation in [Modeling &
evaluation](06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21)
promoted the 19 `fuel_type_*` columns into `FEATURE_COLUMNS`, `/predict/live` needed a way to fill
them, but unlike weather or fire detections, fuel type is static per cell (`features/fuel_type.py`'s
module docstring: it doesn't change day to day short of an actual burn), so there's no live WFS query
to make at request time. `features/live_fuel_type.py::build_live_fuel_type_features` instead looks up
the cell's already-cached code from `data/raw/fuel_type/kamloops_fuel_type.parquet`, the exact cache
`pipeline/build_dataset.py` already populated for training, covering every cell in the same
`BC_KAMLOOPS_BBOX` grid `/predict/live` serves, since a live cell can never be outside the grid
training used, and one-hots it against the served model's exact `fuel_type_*` columns. A code with no
matching column (e.g. a cell whose fuel type was never mapped, or, if this project's grid ever
changed, a code that never occurred in the current training extract) falls back to all-zero, the same
"unseen class is dead weight, not an error" behavior `encode_fuel_type_features` already established
for training. `lifespan` loads this cache once at startup, a required file dependency once
`fuel_type_*` is part of the served model, the same way `MODEL_BUNDLE_PATH` already is, raising at
startup (not failing every request individually) if it's missing.

### CORS and the `file://` frontend

**CORS defaults to wide open** (`allow_origins=["*"]`), but is now configurable via the
`FIRESIGHT_CORS_ORIGINS` env var (comma-separated list of allowed origins) rather than hardcoded,
set it before any real deployment. The `*` default stays correct for local dev specifically because
`frontend/index.html` is opened directly as a file (no dev server), which sends no `Origin` header at
all; an explicit allowlist has nothing to match against in that case, so the permissive default isn't
an oversight, it's the only thing that works for a file:// frontend. Deploying the frontend behind a
real origin (a dev server or real hosting) is the point at which `FIRESIGHT_CORS_ORIGINS` should be
set to that exact origin, not `*`.

### `/predict/live/multi-day`: the 3-day-ahead endpoint

Added 2026-08-21 alongside `features/labels.py::add_forward_ignition_label` and
`training/export_model.py::export_multi_day_model`, see [Grid &
labels](03-grid-and-labels.md#the-multi-day-ahead-label-ignited_next_nd-2026-08-21) and [Modeling &
evaluation](06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21) for how the
label and this endpoint's model were built and validated. `GET /predict/live/multi-day?cell_id=&date=`
answers a different question than `/predict/live`: not "will this cell ignite *today*," but "will it
ignite at some point in the next `MULTI_DAY_WINDOW` (3) days", scored through a **second**,
independently-exported `ModelBundle`, not a parameter on the same model.

**Reuses everything about live data sourcing, changes only which model scores it.** The endpoint
calls the exact same `_resolve_live_features` helper `/predict/live`/`/predict/explain` already share,
same Open-Meteo weather fetch, same FIRMS NRT neighbor-fire count, same fuel-type cache lookup,
since the multi-day model was trained on the identical `FEATURE_COLUMNS`. Nothing about *fetching*
live conditions changes for a wider prediction window; only the label the model was fit against does.

**Optional, unlike the primary model.** `lifespan` loads `data/processed/model_3day.joblib` (or
`FIRESIGHT_MULTI_DAY_MODEL_PATH`) if present, but, unlike `MODEL_BUNDLE_PATH`, whose absence is a
startup `RuntimeError`, a missing multi-day bundle just means `state["multi_day_bundle"]` is `None`
and the endpoint 503s with instructions to run `export_multi_day_model()`. A fresh self-host that's
only run `training/export_model.py`'s `__main__` (which now exports both bundles) gets this endpoint
automatically; one that's only run `export_current_best()` directly does not, and that's fine, the
same-day model and every other endpoint work regardless. `/health`'s `multi_day_model_loaded` field
reports which state a given deployment is in.

**Reachable from the map, not just the API.** `frontend/index.html`'s cell popup has a "Live risk"
row with a "Today" / "Next N days" dropdown next to its "Check" button; picking the second option
calls this endpoint instead of `/predict/live` and renders `window_days` as an explicit "not
calibrated yet" note. No SHAP explanation is fetched for the multi-day option, `/predict/explain`'s
background/model is the same-day bundle only, so attaching it to a multi-day prediction would
misattribute the explanation to the wrong model.

**No `calibrated_probability` field.** `export_multi_day_model` doesn't attach a calibrator, see its
docstring for why (the pooled leave-one-year-out methodology `export_current_best` uses depends on
`evaluation/backtest.py::run_rolling_origin_backtest`, hardcoded to the same-day label), so this
endpoint's response has `ignition_probability` and `window_days` but no calibrated counterpart, a real,
documented gap rather than a silently-missing field. `ignition_probability` here carries the same
honest caveat [Calibration](#calibration-ignition_probability-vs-calibrated_probability) below
describes for the same-day model, a relative ranking, not a literal probability, plus the
independently-measured accuracy gap versus same-day prediction docs/06's write-up quantifies.

### Explaining a live prediction: `/predict/explain`

`evaluation/shap_analysis.py`'s TreeExplainer was originally offline-only (see [Modeling &
evaluation](06-modeling-and-evaluation.md#2-shap-explainability), which deliberately scoped a live
endpoint out as a later follow-up). Wired in 2026-08-20: `/predict/explain` fetches the exact same
live weather + neighbor-fire features `/predict/live` does (via the shared `_resolve_live_features`
helper both endpoints call, so cell/date validation and error handling can't drift between them),
scores the row, and additionally runs it through `evaluation/shap_analysis.py::explain` against a
fixed background sample built once at startup, not per-request, since rebuilding a several-hundred-
row background sample on every call would be wasted, repeated work for a reference distribution that
doesn't change between requests.

**The background sample matches the offline analysis's reference distribution, not an ad hoc one.**
`lifespan` builds it from the same rows `shap_analysis.py`'s own `__main__` uses, fire-season rows
dated before `TRAIN_END`, sampled down to 300 with the same fixed seed, so a live explanation is
computed relative to the same "typical training-set conditions" baseline the docs/06 write-up's own
numbers were, not a baseline that quietly differs deployment-to-deployment. If no processed dataset is
available on a given deployment (`DATASET_PATH` missing, the same condition `/risk-map` already
handles), `/predict/explain` returns a 503 rather than fabricating a background sample from nothing.

**Response shape:** `ignition_probability`/`calibrated_probability` (identical fields to
`/predict/live`) plus `top_contributions`, up to `top_n` `{feature, value, contribution}` entries,
sorted by `|contribution|` descending. Every `contribution` is in raw `ignition_probability` units,
signed (positive pushes the score up, negative pushes it down), there is no SHAP decomposition of
`calibrated_probability`, since the isotonic calibrator is a separate post-hoc regression fit after the
tree ensemble, not part of its structure (see docs/06's SHAP section for why). The response's own
`explanation_note` field restates this, so a caller reading the JSON directly (not this doc) still gets
the caveat.

**Verified live against a real active fire cluster:** hitting `/predict/explain` for the same cell/date
combination the `/predict/live` end-to-end check above used (`1106_-1707`, an active FIRMS NRT cluster)
returned `ignition_probability=0.997` with `neighbor_fire_count_7d` (value 35) as the dominant
contribution at `+0.557`, followed by `neighbor_fire_count_1d`/`_3d` and then weather features, the
same "a real fire nearby overwhelms weather-only signal" pattern the offline SHAP write-up's own
waterfall examples already documented, now reproduced against a genuinely live prediction rather than a
held-out test row.

### Calibration: `ignition_probability` vs `calibrated_probability`

`/predict` and `/predict/live` both return `ignition_probability` as a bare float, the obvious
reading is "this cell has an N% chance of igniting today." `evaluation/calibration.py` checked whether
that reading is actually justified, and it isn't: the served model's raw probabilities are off from
the true observed rate by roughly two orders of magnitude (a cell scored at ~85% actually ignites
about 1.4% of the time on the 2023 validation set), an expected side effect of
`class_weight="balanced"` in training, not a bug. Full numbers and mechanism in [Modeling &
evaluation](06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability).

What still holds: **relative** ordering. A cell scored higher than another really is more likely to
ignite than the other, which is exactly what `/risk-map`'s coloring and the whole top-10%-capture
story rely on, see [the frontend's relative-coloring choice](#the-frontend-frontendindexhtml) below.
`ignition_probability`/`risk_probability` keep meaning exactly what they meant before: read them as a
**relative risk score**, not a literal probability.

`training/export_model.py::export_current_best` now also fits and attaches a calibrator, exposed
alongside the raw score as `calibrated_probability` (`/predict`, `/predict/live`) and
`calibrated_risk_probability` (`/risk-map`), `null` if the loaded bundle predates this (check
`/health`'s `"calibrated"` field). It's an `IsotonicRegression` fit on raw scores pooled across all 8
of `evaluation/backtest.py`'s rolling-origin years (2017-2024), the exact methodology
[the pooled, leave-one-year-out-validated calibration
check](06-modeling-and-evaluation.md#does-pooled-leave-one-year-out-validated-calibration-actually-help)
validated, a single-year fit was checked and rejected as unreliable before this was pooled. Read
honestly: it's a real improvement (worst-case top-decile miscalibration drops from ~2,673x to ~51x in
that check) but not a fully solved problem, the sparsest-fire years stay the least reliable even after
calibration, so `calibrated_probability` is a much better estimate of true ignition frequency than the
raw score, not a guaranteed-accurate one. `ignition_probability` stays the field to use for ranking
(`/risk-map`'s coloring, any top-k logic), the calibrator changes absolute magnitude, not order, so
using `calibrated_probability` for ranking would just be a slower way to get the same ranking
`ignition_probability` already gives.

## The frontend (`frontend/index.html`)

A single static HTML file (Leaflet via CDN, vanilla JS, no build step, no framework), deliberately
minimal, matching the project's scope as a personal, Kamloops-only tool rather than a polished
product. It renders one `L.rectangle` per grid cell, sized to the real 5km cell
footprint (mirroring `features/grid.py::cell_size_degrees`'s math in JS, not a fixed-pixel marker),
colored by predicted risk, outlined with a thin black border so adjacent cells read as a grid, and
overridden with a thicker blue border for cells with a real recorded ignition that day, by calling
`/risk-map` directly from the browser. Real-sized squares matter specifically because a fixed-pixel
marker (the original `L.circleMarker` this replaced) stays a constant screen size regardless of zoom,
so it shrinks to an uninformative speck relative to the grid the moment you zoom in past the map's
initial extent; a geometry-sized square scales with the map instead. Cell fill uses `fillOpacity: 0.55`
rather than a more saturated value, deliberately low enough that roads, place labels, rivers, and
terrain shading stay legible underneath the risk color, the base map is part of how a reader locates
a risky cell relative to a real place, not just decoration behind the data.

**Why risk is colored relative to that day's own maximum, not a fixed 0–1 scale:** two independent
reasons, one about the underlying rarity of fires and one about the model's raw output specifically.
First, most days genuinely are low-risk across the board, fires are rare by construction (base rate
around 0.1–0.3%, see [Modeling &
evaluation](06-modeling-and-evaluation.md#baseline-results-2023-validation-set)), so on a calm day,
coloring against a fixed 0–1 range would render nearly every marker the same pale "low risk" color,
making the map look broken rather than calm. Second, and separately, `ignition_probability` isn't a
calibrated probability at all, see
[Calibration](#calibration-ignition_probability-vs-calibrated_probability) above, so a fixed 0–1
scale would be anchoring to a number whose absolute value doesn't
mean what it looks like it means, on top of the rarity problem. Scaling each date's colors to that
date's own max risk sidesteps both issues at once: it keeps the map informative (which cells are
relatively higher-risk *today*) without ever needing the raw value to be absolutely meaningful, at the
cost of not being comparable in absolute color across different dates, an explicit, documented
tradeoff, not an oversight.

**City search shows that city's actual risk, not just a map pan.** Typing a city name finds the
nearest already-loaded `/risk-map` row by straight-line distance and reports its risk (and whether a
real fire was recorded there), reusing the response already fetched for the selected date rather than
issuing a second request. Towns outside the modeled bbox (e.g. Lillooet, Lytton, Clearwater, see the
`CITIES` list's own comment) still resolve to *some* nearest cell, since the grid has no hard edge to
stop a nearest-neighbor search at, the UI flags this explicitly (`~29km away, outside the fitted
grid`) rather than silently presenting a distant cell's number as if it were that town's own risk.

**Click any cell marker for its live risk right now, not just its historical replay value.** Added
2026-08-20 alongside the `/predict/live`/`/predict/explain` live-forecasting work above, the frontend
previously only ever called `/risk-map`, so "live forecasting" was a capability the API had but the
demo couldn't show. Clicking a marker opens a popup with a "Check" button (lazy-fetched
on click, not prefetched for every marker on the map: a loaded map can be 1,000+ cells, and each live
check is its own Open-Meteo + FIRMS NRT round trip server-side, so eagerly fetching all of them would
be slow and would hammer both upstream APIs for data most cells will never be looked at again). Clicking
it calls `/predict/live` and `/predict/explain` together (caching the pair per `cell_id`+date so
reopening the popup doesn't refetch) and renders the live risk, both data-source labels, and the SHAP
"Why" breakdown inline in the popup, the same real end-to-end flow verified manually via
`mcp__claude-in-chrome` against a running `uvicorn` server, not just unit-tested.
`/predict/explain` returning a 503 (no dataset on this deployment, see [Explaining a live
prediction](#explaining-a-live-prediction-predictexplain) above) degrades to showing the live risk
without a "Why" section rather than failing the whole popup. `calibrated_probability` is fetched but
deliberately not displayed here (or anywhere in the popup/tooltip), decluttering, not a data change;
the API still returns it for anyone consuming it directly.

## Running it locally

```
uv run python -m firesight.training.export_model   # writes model.joblib + model_3day.joblib
uv run uvicorn api.main:app --reload               # serves on :8000
# open frontend/index.html directly in a browser (it talks to :8000)
```
