# FireSight

Geospatial ML project predicting wildfire ignition risk per grid cell/day for the Kamloops Fire
Centre, BC, a personal project deliberately scoped to this one region, not a step toward a
province-wide product. See `README.md` for setup, the full project layout, and current status.

## Orientation

- **Pipeline shape:** `src/firesight/pipeline` (ingest FIRMS detections, ERA5-Land weather, full
ERA5 convective variables) -> `features` (grid construction, label scaffold, weather join, feature
engineering, plus the static per-cell sources: FWI, terrain, fuel type) -> `training` (baseline
models, RandomForest/XGBoost tuning, persistence, export) -> `evaluation` (rare-event metrics,
rolling-origin backtest, calibration, SHAP) -> `api` + `frontend` (serving). `features/live_*.py`
is a parallel path: the same feature row rebuilt from live sources (Open-Meteo, FIRMS NRT, the
fuel-type cache) for `/predict/live`, rather than read from the historical parquet. Each stage has
a matching page in `docs/`, see the sync rule below.
- **Currently-served model:** RandomForest, params in
`training/export_model.py::BEST_RANDOM_FOREST_PARAMS`. That file is the one deliberate, explicit
place a model gets promoted to "the one the API serves", don't have another script auto-export a
model as a side effect. Its `__main__` writes **two** bundles: `export_current_best()` ->
`data/processed/model.joblib` (same-day `ignited`, with a pooled isotonic calibrator) and
`export_multi_day_model()` -> `data/processed/model_3day.joblib` (the 3-day-ahead
`ignited_next_3d` label behind `GET /predict/live/multi-day`, no calibrator yet). After changing
which model/params either one exports, re-run `uv run python -m firesight.training.export_model`
to regenerate both.
- **`data/`** (raw + processed) is gitignored and only ever produced by running the pipeline
scripts, never assume it exists from a fresh clone, and never try to commit into it.
- **`research/`** holds standalone feasibility writeups for approaches that were investigated but
not shipped (`lightning-data.md`, `neural-networks.md`), each summarized and linked from the
relevant `docs/` page. Deep investigation notes go here; the reasoning a reader of the pipeline
needs stays in `docs/`.
- **Checks before calling a change done:** `uv run pytest` and `uv run ruff check` should both stay
clean.

## Keep `docs/` in sync with the code

`docs/` is a from-scratch ML teaching guide for this project (concepts, definitions, and *why* each
pipeline/modeling decision was made, not just what the code does). It is written to stay accurate,
not to be a one-time snapshot.

**Whenever you change what a module in `src/firesight/` does, why it does it that way, or a
parameter/format/assumption it depends on (including fixing a bug like a wrong aggregation or join
key), update the matching page(s) under `docs/` in the same session**, don't leave that for a
separate pass or for the user to ask for. If a change doesn't affect the reasoning or concepts
described in `docs/` (e.g. a pure refactor, a formatting fix), no doc update is needed, use
judgment rather than touching docs reflexively for every diff.

`docs/README.md` is the index; the pages run problem framing -> data sources -> grid/labels ->
weather join -> feature engineering -> modeling/evaluation -> serving -> future directions, plus a
glossary. When adding a new pipeline stage, add a new page and link it from `docs/README.md`
rather than folding it into an existing page.

## Markdown formatting

Prose markdown in this repo (`README.md`, `CLAUDE.md`, `docs/*.md`) is hand-wrapped at roughly 100
columns, not left as one long line per paragraph and not wrapped narrower, keep that width when
writing or editing prose here so diffs and terminal reading stay consistent. Code fences, tables,
and headings are left as-is regardless of width.

**No em dashes** (the `—` character) in those same files. Use a comma, a colon, or a sentence
break instead. All three were deliberately converted off them, so reintroducing one is a
regression, not a style choice. (`research/*.md` predates this and still contains them; `--` is
the convention in Python comments.)
