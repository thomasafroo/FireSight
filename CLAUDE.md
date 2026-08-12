# FireSight

Geospatial ML project predicting wildfire ignition risk per grid cell/day for the Kamloops Fire
Centre, BC — a deliberately small-region MVP before any scaling to all of BC. See `README.md` for
setup, the full project layout, and current status.

## Orientation

- **Pipeline shape:** `src/firesight/pipeline` (ingest FIRMS + ERA5-Land) -> `features` (grid
construction, label scaffold, weather join, feature engineering) -> `training` (baseline models,
RandomForest/XGBoost tuning, persistence, export) -> `api` + `frontend` (serving). Each stage has a
matching page in `docs/` — see the sync rule below.
- **Currently-served model:** XGBoost, params in `training/export_model.py::BEST_XGBOOST_PARAMS`.
That file is the one deliberate, explicit place a model gets promoted to "the one the API serves" —
don't have another script auto-export a model as a side effect. After changing which model/params it
exports, re-run `uv run python -m firesight.training.export_model` to regenerate
`data/processed/model.joblib`.
- **`data/`** (raw + processed) is gitignored and only ever produced by running the pipeline scripts
— never assume it exists from a fresh clone, and never try to commit into it.
- **Checks before calling a change done:** `uv run pytest` and `uv run ruff check` should both stay
clean.

## Keep `docs/` in sync with the code

`docs/` is a from-scratch ML teaching guide for this project (concepts, definitions, and *why* each
pipeline/modeling decision was made — not just what the code does). It is written to stay accurate,
not to be a one-time snapshot.

**Whenever you change what a module in `src/firesight/` does, why it does it that way, or a
parameter/format/assumption it depends on (including fixing a bug like a wrong aggregation or join
key), update the matching page(s) under `docs/` in the same session** — don't leave that for a
separate pass or for the user to ask for. If a change doesn't affect the reasoning or concepts
described in `docs/` (e.g. a pure refactor, a formatting fix), no doc update is needed — use
judgment rather than touching docs reflexively for every diff.

`docs/README.md` is the index; each page maps to a stage of the pipeline (data sources → grid/labels
→ weather join → feature engineering → modeling/evaluation) plus a glossary. When adding a new
pipeline stage, add a new page and link it from `docs/README.md` rather than folding it into an
existing page.

## Markdown formatting

Prose markdown in this repo (`README.md`, `CLAUDE.md`, `docs/*.md`) is hand-wrapped at roughly 100
columns, not left as one long line per paragraph and not wrapped narrower — keep that width when
writing or editing prose here so diffs and terminal reading stay consistent. Code fences, tables,
and headings are left as-is regardless of width.
