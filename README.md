<div align="center">

# FireSight

### Geospatial machine learning for wildfire risk forecasting in the Kamloops Fire Centre, BC

</div>

Predicts the probability that a grid cell in BC will experience a wildfire ignition on a given day,
using historical fire and weather data. Built as an end-to-end system, data pipeline, model
training with rigorous temporal validation, and an inference API, not just a notebook. (Predicting
same-day ignition is what's actually implemented; extending the label to a multi-day-ahead window is
a natural future step, not yet built, see
[Problem framing](docs/01-problem-framing.md#what-were-actually-predicting).)

## Motivation

I took CPSC 330 at UBC (shoutout to Professor Giulia Toti), which covered the fundamentals of
supervised and unsupervised machine learning and gave me practical, hands-on experience with
`scikit-learn`. This past summer, I wanted to build on that with a focused project of my own rather
than another course assignment.

The summer 2026 wildfire season in Canada, especially in BC, was severe, ravaging ecosystems and
communities. As a Vancouverite, I felt it directly: most of the smoke settling over the city was
coming from the direction of the regions around Kamloops, and the drop in air quality was impossible
to ignore. That raised an obvious question: could machine learning forecast which regions of BC are
at immediate risk of wildfire? FireSight is the project I built to try to answer it.

I originally planned to cover all of BC. I quickly ran into two practical limits: a province-wide
dataset takes a lot more time to train on, and some of the models I wanted to try are constrained by
my own computer's memory. Narrowing the scope to the Kamloops Fire Centre, the source of much of the
smoke I'd been breathing, kept the project small enough to actually finish and iterate on.

## Status

FireSight is complete and working end to end: raw data, a trained model, a live API, and an
interactive map, tested not just with unit tests but by checking its predictions against real
historical wildfires.

- **Data pipeline:** combines NASA FIRMS fire detections with ECMWF ERA5-Land weather data into a
  labeled grid of 5km cells across the Kamloops Fire Centre, split by date (not randomly) into
  training, validation, and test sets.
- **Modeling:** compared a dummy baseline, logistic regression, random forest, XGBoost, and a neural
  network; a tuned random forest performs best and is the model currently served.
- **Serving:** a FastAPI backend that scores risk for any cell and date, including live conditions
  fetched in real time rather than only replaying history.
- **Frontend:** a Leaflet map for browsing historical risk against what actually happened, and
  checking live risk for any cell.

**Scope:** permanently the Kamloops Fire Centre, not all of BC, and fire season only
(May 1 - Oct 15, any year).

**Known limitation:** the model is better at ranking which cells are riskiest relative to each other
than at giving a precise probability on its own, and overall accuracy varies noticeably from year to
year rather than holding at one number. See
[Modeling & evaluation](docs/06-modeling-and-evaluation.md) for the full results.

Full reasoning and open questions live in `docs/README.md`, in particular
[Future directions](docs/08-future-directions.md).

## Project layout

```
src/firesight/
  pipeline/     data ingestion (FIRMS fire points, ERA5-Land weather,
                full-ERA5 convective variables) + build_dataset.py
                (assembles the full processed table)
  features/     grid construction, labels, weather join, feature engineering,
                FWI/terrain/fuel type, plus live_*.py rebuilding the same
                feature row from live sources for /predict/live
  training/     model training (baseline -> boosted trees -> NN) + persistence
  evaluation/   rare-event metrics, rolling-origin backtest, calibration, SHAP
api/            FastAPI inference/demo endpoints wrapping the trained model
frontend/       minimal Leaflet risk map (static, no build step)
data/           raw/ and processed/ data (gitignored, not committed)
notebooks/      exploration only, not the deliverable
docs/           ML guide, concepts, definitions, and the reasoning
                behind each pipeline/modeling decision (start at docs/README.md)
research/       standalone feasibility writeups for approaches evaluated
                but not shipped (lightning data, neural networks)
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
  3. Accept the ERA5-Land dataset's terms on its CDS page before your first request, the API
rejects requests until you do. `ingest_era5_convective.py` pulls a *different* CDS dataset
(`reanalysis-era5-single-levels`), whose terms have to be accepted separately on its own page.
  4. `uv add cdsapi` (not installed by default, since it needs the account set up first)

## Running

```
uv run pytest
uv run python -m firesight.pipeline.ingest_firms
uv run python -m firesight.pipeline.ingest_era5
uv run python -m firesight.pipeline.ingest_era5_convective  # full-ERA5 CAPE + convective precip
uv run python -m firesight.pipeline.build_dataset           # raw data -> data/processed/kamloops_dataset.parquet
uv run python -m firesight.training.baseline                # Dummy + LogisticRegression vs the temporal val split
uv run python -m firesight.training.advanced_models         # RandomForest + XGBoost, tuned against the same split
uv run python -m firesight.training.export_model            # persists model.joblib (same-day) + model_3day.joblib
uv run uvicorn api.main:app --reload                        # serves it on :8000
# then open frontend/index.html in a browser
# GET /predict/live?cell_id=<id>&date=YYYY-MM-DD scores current conditions (needs internet access to
# reach Open-Meteo; no API key or account needed, unlike the ERA5-Land backfill above)
# GET /predict/live/multi-day?cell_id=<id>&date=YYYY-MM-DD scores the next 3 days instead
```

## Design notes

- **No random train/test split.** Wildfire observations are spatially and temporally correlated, so
splitting randomly leaks information. Split by date instead (train on 2012-2022, validate 2023,
test 2024) to simulate "would this have worked in real time."
- **Accuracy is not the metric.** Fires are rare, so a model predicting "no fire" everywhere would
score ~99% while being useless. Use PR-AUC, recall, and top-k% capture rate instead (see
`evaluation/metrics.py`).
- **Baseline first.** DummyClassifier → Logistic Regression → Random Forest → XGBoost → Neural
Network, in that order, each compared against the last before adding complexity.
