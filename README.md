# FireSight

Geospatial machine learning for wildfire risk forecasting in British Columbia.

Predicts the probability that a grid cell in BC will experience a wildfire ignition on a given day,
using historical fire and weather data. Built as an end-to-end system — data pipeline, model
training with rigorous temporal validation, and an inference API — not just a notebook. (Predicting
same-day ignition is what's actually implemented; extending the label to a multi-day-ahead window is
a natural future step, not yet built — see
[Problem framing](docs/01-problem-framing.md#what-were-actually-predicting).)

## Status

The Kamloops Fire Centre MVP is done — the original 10-step plan works end-to-end and has been
verified live, not just unit-tested:

- **Data pipeline:** raw FIRMS + ERA5-Land -> grid/label/weather/feature pipeline -> processed
  dataset -> temporal train/val/test split (never random, see [Design notes](#design-notes)).
- **Modeling:** Dummy and LogisticRegression baselines, then tuned RandomForest/XGBoost, widened
  further with a `RandomizedSearchCV`+`PredefinedSplit` search. RandomForest is the model currently
  being served.
- **Serving:** the served model swaps with a one-line change and zero edits to `api/main.py`,
  confirmed by actually swapping it live and re-checking `/health` and `/risk-map`.
- **Frontend:** a minimal Leaflet map replaying historical risk against real recorded outcomes.

Full reasoning and current results: `docs/README.md`.

**Open, not-yet-decided next steps:**

- Scaling past Kamloops to all of BC
- Adding a neural network
- Wiring `/predict` to a live weather feed instead of historical replay
- Real deployment (CORS is currently wide open, for local dev only)

## Project layout

```
src/firesight/
  pipeline/     data ingestion (FIRMS fire points, ERA5-Land weather)
                + build_dataset.py (assembles the full processed table)
  features/     grid construction, labels, weather join, feature engineering
  training/     model training (baseline -> boosted trees -> NN) + persistence
  evaluation/   metrics suited to rare-event classification
api/            FastAPI inference/demo endpoints wrapping the trained model
frontend/       minimal Leaflet risk map (static, no build step)
data/           raw/ and processed/ data (gitignored, not committed)
notebooks/      exploration only, not the deliverable
docs/           ML guide — concepts, definitions, and the reasoning
                behind each pipeline/modeling decision (start at docs/README.md)
tests/          unit tests for pipeline/features/training/api, one file per module
```

## Setup

```
uv sync
cp .env.example .env   # fill in FIRMS_MAP_KEY
```

- `FIRMS_MAP_KEY`: register at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (free, rate-limited
to 5000 transactions/10 min)
- ERA5-Land data requires a free Copernicus CDS account:
  1. Create an account and get your personal access token at
https://cds.climate.copernicus.eu/profile
  2. Create `~/.cdsapirc` (note: `%USERPROFILE%\.cdsapirc` on Windows) with:
     ```
     url: https://cds.climate.copernicus.eu/api
     key: YOUR_PERSONAL_ACCESS_TOKEN
     ```
  3. Accept the ERA5-Land dataset's terms on its CDS page before your first request — the API
rejects requests until you do.
  4. `uv add cdsapi` (not installed by default, since it needs the account set up first)

## Running

```
uv run pytest
uv run python -m firesight.pipeline.ingest_firms
uv run python -m firesight.pipeline.ingest_era5
uv run python -m firesight.pipeline.build_dataset       # raw data -> data/processed/kamloops_dataset.parquet
uv run python -m firesight.training.baseline            # Dummy + LogisticRegression vs the temporal val split
uv run python -m firesight.training.advanced_models      # RandomForest + XGBoost, tuned against the same split
uv run python -m firesight.training.export_model         # persists the current-best model -> data/processed/model.joblib
uv run uvicorn api.main:app --reload                      # serves it on :8000
# then open frontend/index.html in a browser
```

## Design notes

- **No random train/test split.** Wildfire observations are spatially and temporally correlated, so
splitting randomly leaks information. Split by date instead (e.g. train on 2015–2022, validate 2023,
test 2024) to simulate "would this have worked in real time."
- **Accuracy is not the metric.** Fires are rare, so a model predicting "no fire" everywhere would
score ~99% while being useless. Use PR-AUC, recall, and top-k% capture rate instead (see
`evaluation/metrics.py`).
- **Baseline first.** DummyClassifier → Logistic Regression → Random Forest → XGBoost → Neural
Network, in that order, each compared against the last before adding complexity.
