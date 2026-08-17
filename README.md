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
  confirmed by actually swapping it live and re-checking `/health` and `/risk-map`. `/predict/live`
  scores a grid cell's current conditions via a live weather feed (Open-Meteo), rather than only
  replaying historical dates. Every prediction endpoint also returns a calibrated probability
  (`calibrated_probability`/`calibrated_risk_probability`) alongside the raw, relative-ranking-only
  `ignition_probability`/`risk_probability` — an isotonic calibrator fit on scores pooled across all 8
  of `evaluation/backtest.py`'s rolling-origin years, the same methodology validated by the
  leave-one-year-out check below. See
  [Serving](docs/07-serving.md#calibration-ignition_probability-vs-calibrated_probability).
- **Frontend:** a minimal Leaflet map replaying historical risk against real recorded outcomes.

Full reasoning and current results: `docs/README.md`.

**Scoped to fire season:** training and evaluation are restricted to May 1 - Oct 15 (any year),
matching the Kamloops Fire Centre's typical open-burning prohibition window. This follows from a
winter/shoulder-season blind spot found during error analysis — the served model missed nearly all
winter fires (0/23 in December on the 2024 test set) because they're more often human-caused than
weather-driven, and every feature here is weather-derived. Two feature-engineering attempts to fix it
failed, so rather than keep chasing it, the project now excludes those months from the problem
entirely and focuses on the summer fire-weather signal the model actually has. See [Modeling &
evaluation](docs/06-modeling-and-evaluation.md#scoping-to-fire-season).

**Decided:** staying scoped to the Kamloops Fire Centre rather than scaling to all of BC. A larger
bbox means a much bigger grid, more raw FIRMS/ERA5-Land volume, and a per-row reference-latitude
approximation in `features/grid.py` that would need correcting — deliberately kept out of scope for
this project rather than backed into.

**Known limitation:** `ignition_probability` (from `/predict` and `/predict/live`) is a reliable
*relative* risk ranking but not a calibrated probability — a cell scored ~85% actually ignites about
1.4% of the time. This is an expected side effect of `class_weight="balanced"` in training, not a
bug. Worse, the size of that miscalibration isn't even stable year to year (a rolling-origin check
found the gap between predicted and observed risk swinging from ~17x to ~490x depending on the
holdout year), so there's no single correction factor to apply directly to `ignition_probability`
itself; see
[Modeling & evaluation](docs/06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability).
A separate, pooled-across-years calibrator is now served alongside it (see the Status section above)
rather than replacing it — ranking still relies on the raw score.

**Known limitation:** the model's headline top-10%-capture number (71.9%, from the original single
val/test split) is close to the best year out of eight backtested — a rolling-origin backtest across
2017-2024 found a mean of 30.8%, a median of 26.2%, and a range of 8.2%-74.4%, all on the exact same
tuned model. Cite the range or the median, not 71.9%, when describing expected real-world performance;
see [Modeling & evaluation](docs/06-modeling-and-evaluation.md#rolling-origin-backtest-is-719-typical-or-the-best-year-in-the-dataset).
Most of that swing tracks which months a year's fires happen to land in (r=0.63 with Jul/Aug fire
share); the residual is concentrated in BC's two most extreme fire seasons (2017, 2021) specifically,
where a fixed annual top-10% ranking budget breaks down on days with 100+ simultaneous ignitions and
province-wide extreme heat compresses the cell-to-cell weather variation the model ranks on — see [Why
performance swings by month and
year](docs/06-modeling-and-evaluation.md#why-performance-swings-by-month-and-year).

**Neural network: tried, didn't win.** `research/neural-networks.md`'s one open hypothesis — a raw
daily weather sequence capturing temporal shape the hand-engineered rolling windows flatten away —
was tested for real in `training/sequence_model.py`: a small 1D-CNN benchmarked against the tuned
RandomForest on identical rows. The RandomForest won on every metric but a near-tied test ROC-AUC,
most clearly on PR-AUC and top-10%-capture. No change to the served model; see [Modeling &
evaluation](docs/06-modeling-and-evaluation.md#testing-the-sequence-modeling-hypothesis) for the full
numbers.

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
# GET /predict/live?cell_id=<id>&date=YYYY-MM-DD scores current conditions (needs internet access to
# reach Open-Meteo; no API key or account needed, unlike the ERA5-Land backfill above)
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
