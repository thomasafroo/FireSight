# Serving: model persistence, API, frontend

The last stage: get a trained model in front of something a person can actually look at. Three
pieces, each solving a distinct problem — persisting a model *safely* (not just picklable, but
usable correctly later), serving predictions, and visualizing them.

## Persisting a model without losing its contract

A model on its own — `model.predict_proba(X)` — isn't safely reusable after the training script that
produced it exits. `predict_proba` needs `X` to have exactly the right columns, in exactly the right
order, with exactly the right meaning; a bare pickle remembers none of that. If the serving code
independently hardcodes (or worse, guesses) the feature list and it ever drifts from what the model
was actually fit on, the call either shape-errors loudly or — worse — silently runs with columns in
the wrong order, producing a confidently wrong probability with no error at all.

`training/persist.py::ModelBundle` fixes this by bundling the model *with* its own contract:
`feature_columns` (the exact list, in order) and free-form `metadata` (what kind of model, when it
was validated, what it scored) travel with the model in one `joblib`-serialized object.
`bundle.predict_proba(features: dict)` selects and orders columns from `feature_columns` itself, so
there's no second place that list could drift out of sync.

`training/export_model.py::export_current_best()` is the one place that decides *which* fitted model
becomes `data/processed/model.joblib` — deliberately a separate, explicit step from `baseline.py` or
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
cells changed accordingly — the same endpoint, same request, different answer, purely because the
file on disk changed. Whatever model wins the next round of tuning gets served the same way.

## The API (`api/main.py`)

Three endpoints:

- `POST /predict` — score one (cell, day)'s worth of *already-computed* feature values, supplied
directly in the request body. It does **not** fetch live weather or run feature engineering itself —
that's what `/predict/live` below is for. The request schema (which fields are required) isn't
hand-written: it's built at startup from `bundle.feature_columns` via Pydantic's `create_model`, so
it can never silently drift from whatever model the API actually loaded — the same motivation as
`ModelBundle` itself. Kept as-is (raw feature vector in, probability out) rather than folded into
`/predict/live`, since it's the simplest possible contract for programmatic/testing access to the
model when the caller already has feature values from somewhere else.
- `GET /predict/live?cell_id=...&date=YYYY-MM-DD` — score a grid cell's **current** conditions by
fetching recent weather itself, rather than requiring the caller to supply features or replaying a
date already in the historical dataset. See [Live weather for
/predict/live](#live-weather-for-predictlive) below.
- `GET /risk-map?date=YYYY-MM-DD` — a historical **replay**, not a live forecast: for a date already
present in `data/processed/kamloops_dataset.parquet`, scores every one of the 1,443 grid cells using
that date's real recorded weather features, and returns each cell's predicted risk *alongside* its
actual recorded label. This exists specifically so the frontend has something real and checkable to
render (predicted risk next to what actually happened) without needing a live data feed — a
reasonable MVP substitute for "the model is deployed and scoring live conditions."

`/risk-map` and `/predict/live` both reject any date outside `training/baseline.py::FIRE_SEASON_START`
–`FIRE_SEASON_END` (May 1 – Oct 15, any year) with a 400, via the shared `_reject_if_outside_fire_season`
helper, before doing any lookup or fetch. The served model is trained exclusively on that window (see
[Modeling & evaluation](06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot)),
so scoring a December date would just be extrapolating from a model that has never seen a single
winter row — silently wrong rather than usefully wrong. `/predict` still has no equivalent guard: its
request body is a raw feature vector with no date or day-of-year field at all (`FEATURE_COLUMNS` is
pure weather), so there's nothing date-shaped to validate against — a caller could still hand-construct
an out-of-season feature vector and get a number back. That's an accepted gap for the raw-vector
endpoint specifically, not something `/predict/live` inherits, since `/predict/live` always knows the
date it's scoring.

### Live weather for `/predict/live`

`features/live_weather.py::build_live_feature_row` fetches recent daily weather for a cell's
centroid from [Open-Meteo's historical-weather API](https://open-meteo.com/en/docs/historical-weather-api)
(`archive-api.open-meteo.com`), not ERA5-Land via `cdsapi` directly. It serves the same underlying
ECMWF ERA5/ERA5-Land reanalysis the training pipeline is built on, but blends in a preliminary
near-real-time product so **today's** values are already available — verified live on 2026-08-15,
where a same-day request returned non-null soil moisture — rather than plain ERA5-Land's ~5-day
publication lag, and needs no CDS account, API key, or `~/.cdsapirc` setup at all. That trade (a
slightly different, blended data source vs. zero-friction access to today's weather) is deliberate:
`cdsapi`'s account/terms-acceptance friction is fine for a one-time historical backfill, but a poor
fit for a request path that needs to succeed on every API call.

It requests a 45-day lookback window ending on the target date (`LOOKBACK_DAYS`) — long enough to
satisfy `precip_30d`'s 30-day rolling window with slack left over for `add_days_since_rain` to find a
"last rain" day within a normal fire-season dry spell (see
`engineering.py::add_days_since_rain` — no rain anywhere in the window means a real `NaN`, not
something to fake a number for). It then reuses `add_days_since_rain`/`add_rolling_features` from
`features/engineering.py` unchanged — the exact same functions the historical training pipeline calls
— so a live prediction can never silently drift from how the served model's training features were
computed. It deliberately skips `add_relative_humidity`/`add_wind_features`: those exist specifically
because raw ERA5-Land only gives dewpoint and wind vector components, not humidity or speed directly,
but Open-Meteo reports relative humidity and wind speed directly, so there's nothing to derive them
from (and, after [dropping the dead-weight features](06-modeling-and-evaluation.md#dropping-the-dead-
weight-features), `FEATURE_COLUMNS` no longer needs `d2m` or `u10`/`v10` for anything else either). If
any engineered column still comes back `NaN` (insufficient history), `build_live_feature_row` raises
rather than guessing, which the endpoint surfaces as a 422.

**CORS defaults to wide open** (`allow_origins=["*"]`), but is now configurable via the
`FIRESIGHT_CORS_ORIGINS` env var (comma-separated list of allowed origins) rather than hardcoded —
set it before any real deployment. The `*` default stays correct for local dev specifically because
`frontend/index.html` is opened directly as a file (no dev server), which sends no `Origin` header at
all; an explicit allowlist has nothing to match against in that case, so the permissive default isn't
an oversight, it's the only thing that works for a file:// frontend. Deploying the frontend behind a
real origin (a dev server or real hosting) is the point at which `FIRESIGHT_CORS_ORIGINS` should be
set to that exact origin, not `*`.

## The frontend (`frontend/index.html`)

A single static HTML file (Leaflet via CDN, vanilla JS, no build step, no framework) — deliberately
minimal, per the project's stated MVP scope (prove the loop works end-to-end on Kamloops before
scaling up or polishing). It renders one circle marker per grid cell, colored by predicted risk and
outlined for cells with a real recorded ignition that day, by calling `/risk-map` directly from the
browser.

**Why risk is colored relative to that day's own maximum, not a fixed 0–1 scale:** predicted
probabilities are small by construction (a rare-event base rate around 0.2%, and PR-AUC is still low
in absolute terms — see
[Modeling & evaluation](06-modeling-and-evaluation.md#baseline-results-2023-validation-set)).
Coloring against a fixed 0–1 range would render nearly every marker on nearly every day as the same
pale "low risk" color, since even the highest-risk cell on a calm day rarely clears much above a few
percent in absolute probability — which would make the map look broken rather than calm. Scaling
each date's colors to that date's own max risk keeps the map informative (which cells are relatively
higher-risk *today*) at the cost of not being comparable in absolute color across different dates —
an explicit, documented tradeoff, not an oversight.

## Running it locally

```
python -m firesight.training.export_model      # writes data/processed/model.joblib
python -m uvicorn api.main:app --reload         # serves on :8000
# open frontend/index.html directly in a browser (it talks to :8000)
```
