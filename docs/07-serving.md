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

Two endpoints, deliberately scoped small:

- `POST /predict` — score one (cell, day)'s worth of *already-computed* feature values, supplied
directly in the request body. It does **not** fetch live weather or run feature engineering itself —
wiring this up to a live ERA5 feed and the rolling-window feature computation from
[Feature engineering](05-feature-engineering.md) is future work, intentionally out of scope for
"wrap the trained model in an API." The request schema (which fields are required) isn't
hand-written: it's built at startup from `bundle.feature_columns` via Pydantic's `create_model`, so
it can never silently drift from whatever model the API actually loaded — the same motivation as
`ModelBundle` itself.
- `GET /risk-map?date=YYYY-MM-DD` — a historical **replay**, not a live forecast: for a date already
present in `data/processed/kamloops_dataset.parquet`, scores every one of the 1,443 grid cells using
that date's real recorded weather features, and returns each cell's predicted risk *alongside* its
actual recorded label. This exists specifically so the frontend has something real and checkable to
render (predicted risk next to what actually happened) without needing a live data feed — a
reasonable MVP substitute for "the model is deployed and scoring live conditions," which isn't built
yet.

**CORS is wide open** (`allow_origins=["*"]`) — acceptable only because this is a local MVP with a
static-file frontend that has no fixed origin during development; explicitly commented in the code
as needing tightening before any real deployment.

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
