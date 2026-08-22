"""FastAPI inference service wrapping the trained wildfire ignition model.

Five endpoints:

- POST /predict — score one (cell, day)'s worth of already-computed
  features. Does *not* fetch live weather or run feature engineering
  itself; the caller supplies feature values directly.
- GET /predict/live — score a grid cell's *current* conditions by
  fetching recent weather from Open-Meteo and recent neighbor-cell fire
  detections from FIRMS NRT, running both through the same feature-
  engineering the training pipeline uses — see features/live_weather.py,
  features/live_fire.py, and docs/07-serving.md.
- GET /predict/live/multi-day — same live fetch as /predict/live, but scored through a second
  model trained on a wider "any ignition in the next MULTI_DAY_WINDOW days" label instead of
  same-day only. Optional: 503s if this deployment hasn't exported
  `training/export_model.py::export_multi_day_model`'s bundle. See
  docs/03-grid-and-labels.md#the-multi-day-ahead-label-ignited_next_nd-2026-08-21 and
  docs/06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21.
- GET /predict/explain — same live fetch as /predict/live, plus a
  per-prediction SHAP breakdown of which features drove that one score —
  see evaluation/shap_analysis.py and docs/06-modeling-and-evaluation.md
  #2-shap-explainability.
- GET /risk-map — a historical demo endpoint: for a date already in the
  processed dataset, returns every cell's predicted risk *and* the
  actual recorded label, so the frontend risk map has something real to
  render without needing a live data feed.
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from firesight.evaluation.shap_analysis import explain, top_contributions
from firesight.features.grid import build_grid_cells
from firesight.features.live_fire import build_live_neighbor_fire_features
from firesight.features.live_fuel_type import (
    build_live_fuel_type_features,
    load_fuel_type_lookup,
)
from firesight.features.live_weather import build_live_feature_row
from firesight.pipeline.build_dataset import MULTI_DAY_WINDOW
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX
from firesight.training.baseline import (
    DATE_COLUMN,
    FIRE_SEASON_END,
    FIRE_SEASON_START,
    LABEL_COLUMN,
    TRAIN_END,
    filter_fire_season,
)
from firesight.training.export_model import MODEL_PATH, MULTI_DAY_MODEL_PATH
from firesight.training.persist import ModelBundle, load_model_bundle

# A modest, fixed-seed sample of *training* rows (dates before TRAIN_END, matching
# evaluation/shap_analysis.py's own background) — the reference distribution SHAP values in
# /predict/explain are computed relative to, not the rows being explained. 300 matches
# shap_analysis.py's choice: plenty for `feature_perturbation="interventional"` without making
# each request re-evaluate the model against an unnecessarily large background sample.
SHAP_BACKGROUND_SIZE = 300

DATASET_PATH = Path(os.environ.get("FIRESIGHT_DATASET_PATH", "data/processed/kamloops_dataset.parquet"))
MODEL_BUNDLE_PATH = Path(os.environ.get("FIRESIGHT_MODEL_PATH", str(MODEL_PATH)))
# Optional, unlike MODEL_BUNDLE_PATH: a self-host that hasn't run
# `training/export_model.py::export_multi_day_model` yet just doesn't get /predict/live/multi-day,
# rather than failing to start entirely.
MULTI_DAY_MODEL_BUNDLE_PATH = Path(os.environ.get("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(MULTI_DAY_MODEL_PATH)))

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_BUNDLE_PATH.exists():
        raise RuntimeError(
            f"No model bundle at {MODEL_BUNDLE_PATH} — run "
            "`python -m firesight.training.export_model` first."
        )
    bundle = load_model_bundle(MODEL_BUNDLE_PATH)
    state["bundle"] = bundle
    state["feature_request_model"] = _build_feature_request_model(bundle)
    # Optional second bundle for /predict/live/multi-day (see docs/03-grid-and-labels.md's
    # "ignited_next_Nd" section) — validated but not yet a required deployment artifact the way
    # MODEL_BUNDLE_PATH is, so its absence degrades one endpoint rather than blocking startup.
    state["multi_day_bundle"] = load_model_bundle(MULTI_DAY_MODEL_BUNDLE_PATH) if MULTI_DAY_MODEL_BUNDLE_PATH.exists() else None
    # Always available (pure computation from the bbox, no file needed) —
    # /predict/live needs a cell's lat/lon regardless of whether the
    # historical dataset parquet has been built on this deployment.
    state["grid_cells"] = build_grid_cells(BC_KAMLOOPS_BBOX)
    state["fuel_type_columns"] = [c for c in bundle.feature_columns if c.startswith("fuel_type_")]
    if state["fuel_type_columns"]:
        # A cache lookup, not a live fetch (features/live_fuel_type.py's module docstring) — but
        # still a required file dependency once fuel_type_* columns are part of the served model,
        # same as MODEL_BUNDLE_PATH above, since /predict/live can't fill those columns without it.
        fuel_type_cache_path = Path(
            os.environ.get("FIRESIGHT_FUEL_TYPE_CACHE_PATH", "data/raw/fuel_type/kamloops_fuel_type.parquet")
        )
        if not fuel_type_cache_path.exists():
            raise RuntimeError(
                f"Served model needs fuel_type_* features but no cache exists at "
                f"{fuel_type_cache_path} — run `python -m firesight.pipeline.build_dataset` (or "
                "features.fuel_type.build_fuel_type_features directly) first."
            )
        state["fuel_type_lookup"] = load_fuel_type_lookup(fuel_type_cache_path)
    else:
        state["fuel_type_lookup"] = None

    if DATASET_PATH.exists():
        keep = [DATE_COLUMN, "cell_id", LABEL_COLUMN, *bundle.feature_columns]
        state["dataset"] = pd.read_parquet(DATASET_PATH, columns=keep)
        # /predict/explain's SHAP background: same fire-season + pre-TRAIN_END filter
        # shap_analysis.py's __main__ uses, so a background built here is the same reference
        # distribution the offline analysis in docs/06 was validated against, not an ad hoc one.
        train_rows = filter_fire_season(state["dataset"])
        train_rows = train_rows[train_rows[DATE_COLUMN] < TRAIN_END]
        background_n = min(SHAP_BACKGROUND_SIZE, len(train_rows))
        state["shap_background"] = (
            train_rows[bundle.feature_columns].sample(n=background_n, random_state=0)
            if background_n > 0
            else None
        )
    else:
        state["dataset"] = None
        state["shap_background"] = None

    yield
    state.clear()


def _reject_if_outside_fire_season(target_date: dt.date) -> None:
    """Shared guard for /risk-map and /predict/live — see docs/07-serving.md.

    The served model is trained exclusively on FIRE_SEASON_START..END (any
    year); scoring a date outside that window would extrapolate from a
    model that has never seen a single winter/shoulder-season row.
    """
    month_day = target_date.strftime("%m-%d")
    if not (FIRE_SEASON_START <= month_day <= FIRE_SEASON_END):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{target_date} falls outside the fire season window the served model was trained "
                f"on ({FIRE_SEASON_START} to {FIRE_SEASON_END}, any year). Predictions outside this "
                "window would be extrapolating from a model that has never seen winter/shoulder-"
                "season data — see docs/06-modeling-and-evaluation.md#known-limitation-a-winter"
                "shoulder-season-blind-spot."
            ),
        )


def _build_feature_request_model(bundle: ModelBundle) -> type[BaseModel]:
    """Build a Pydantic request model from whatever the loaded bundle actually needs.

    Generated at startup from `bundle.feature_columns` rather than hand-written,
    so the API's request contract can never silently drift from the model
    it's actually serving (e.g. after swapping in a RandomForest/XGBoost
    bundle with a different feature list) — see persist.py's docstring.
    """
    fields = {name: (float, ...) for name in bundle.feature_columns}
    return create_model("FeatureVector", **fields)


app = FastAPI(
    title="FireSight API",
    description="Wildfire ignition risk prediction for the Kamloops Fire Centre.",
    lifespan=lifespan,
)

# Wide open (`*`) by default, matching the local MVP frontend: a static file
# opened directly in a browser (no dev server) sends no `Origin` header CORS
# can match against, so an explicit allowlist would just break it. Set
# FIRESIGHT_CORS_ORIGINS to a comma-separated list of real origins (e.g.
# "https://firesight.example.com") before deploying anywhere reachable
# outside localhost — leaving it unset keeps today's dev-only default.
_cors_origins_env = os.environ.get("FIRESIGHT_CORS_ORIGINS")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    bundle: ModelBundle = state["bundle"]
    multi_day_bundle: ModelBundle | None = state["multi_day_bundle"]
    return {
        "status": "ok",
        "model_type": bundle.metadata.get("model_type"),
        "feature_columns": bundle.feature_columns,
        "val_scores": bundle.metadata.get("val_scores"),
        "calibrated": bundle.calibrator is not None,
        "multi_day_model_loaded": multi_day_bundle is not None,
        "multi_day_val_scores": multi_day_bundle.metadata.get("val_scores") if multi_day_bundle else None,
    }


@app.post("/predict")
def predict(features: dict) -> dict[str, float | None]:
    """Score one (cell, day)'s feature vector. Body: {feature_name: value, ...}."""
    bundle: ModelBundle = state["bundle"]
    request_model = state["feature_request_model"]
    try:
        validated = request_model(**features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = validated.model_dump()
    return {
        "ignition_probability": bundle.predict_proba(features),
        "calibrated_probability": bundle.predict_calibrated_proba(features),
    }


def _resolve_live_features(cell_id: str, date: str | None) -> tuple[dt.date, float, float, dict[str, float]]:
    """Shared cell/date validation + live feature fetch behind /predict/live and /predict/explain.

    Raises the same HTTPExceptions either endpoint already documented (404 unknown cell, 400
    future/off-season date, 502 fetch failure, 422 insufficient history) so both callers get
    identical error behavior for identical reasons — not almost-identical hand-duplicated checks.
    """
    grid_cells: pd.DataFrame = state["grid_cells"]
    cell = grid_cells[grid_cells["cell_id"] == cell_id]
    if cell.empty:
        raise HTTPException(status_code=404, detail=f"Unknown cell_id {cell_id!r}.")

    today = dt.datetime.now(dt.UTC).date()
    target_date = dt.date.fromisoformat(date) if date else today
    if target_date > today:
        raise HTTPException(status_code=400, detail=f"{target_date} is in the future — live weather isn't available yet.")
    _reject_if_outside_fire_season(target_date)

    latitude = float(cell.iloc[0]["latitude"])
    longitude = float(cell.iloc[0]["longitude"])

    try:
        features = build_live_feature_row(latitude, longitude, target_date, cell_id)
        features.update(build_live_neighbor_fire_features(cell_id, target_date, BC_KAMLOOPS_BBOX))
        if state["fuel_type_columns"]:
            features.update(
                build_live_fuel_type_features(cell_id, state["fuel_type_columns"], state["fuel_type_lookup"])
            )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Live data fetch failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return target_date, latitude, longitude, features


@app.get("/predict/live")
def predict_live(
    cell_id: str = Query(..., description="Grid cell id, e.g. from a /risk-map response"),
    date: str | None = Query(None, description="YYYY-MM-DD, UTC. Defaults to today; cannot be in the future."),
) -> dict[str, Any]:
    """Score a grid cell's *current* conditions via live feeds, not historical replay.

    Unlike /predict (which requires the caller to already have computed
    feature values) this fetches recent weather from Open-Meteo and recent
    neighbor-cell fire detections from FIRMS NRT itself, then runs both
    through the same feature-engineering functions training uses — see
    features/live_weather.py, features/live_fire.py, and docs/07-serving.md.
    """
    target_date, latitude, longitude, features = _resolve_live_features(cell_id, date)
    bundle: ModelBundle = state["bundle"]

    return {
        "cell_id": cell_id,
        "date": target_date.isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "ignition_probability": bundle.predict_proba(features),
        "calibrated_probability": bundle.predict_calibrated_proba(features),
        "weather_source": "open-meteo archive API (ERA5-based reanalysis + near-real-time blend)",
        "fire_detection_source": "NASA FIRMS NRT (VIIRS_NOAA20_NRT)",
    }


@app.get("/predict/live/multi-day")
def predict_live_multi_day(
    cell_id: str = Query(..., description="Grid cell id, e.g. from a /risk-map response"),
    date: str | None = Query(None, description="YYYY-MM-DD, UTC. Defaults to today; cannot be in the future."),
) -> dict[str, Any]:
    """Score a grid cell's risk of ignition anywhere in the next `MULTI_DAY_WINDOW` days, not just today.

    Reuses the exact same live weather/neighbor-fire/fuel-type sourcing `/predict/live` does (same
    `_resolve_live_features` call, same `FEATURE_COLUMNS`) — only the model bundle differs, since
    the multi-day-ahead model is trained on the same features against a different label
    (`features/labels.py::add_forward_ignition_label`). No `calibrated_probability` here:
    `training/export_model.py::export_multi_day_model` doesn't attach a calibrator yet — see its
    docstring and docs/06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21 for
    why, and for the real, honestly-measured accuracy gap versus same-day prediction this endpoint
    carries (a harder target from the same information, not a bug).
    """
    multi_day_bundle: ModelBundle | None = state["multi_day_bundle"]
    if multi_day_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No multi-day model on this deployment — run "
                "`python -c \"from firesight.training.export_model import export_multi_day_model; "
                'export_multi_day_model()"` first.'
            ),
        )

    target_date, latitude, longitude, features = _resolve_live_features(cell_id, date)

    return {
        "cell_id": cell_id,
        "date": target_date.isoformat(),
        "window_days": MULTI_DAY_WINDOW,
        "latitude": latitude,
        "longitude": longitude,
        "ignition_probability": multi_day_bundle.predict_proba(features),
        "weather_source": "open-meteo archive API (ERA5-based reanalysis + near-real-time blend)",
        "fire_detection_source": "NASA FIRMS NRT (VIIRS_NOAA20_NRT)",
    }


@app.get("/predict/explain")
def predict_explain(
    cell_id: str = Query(..., description="Grid cell id, e.g. from a /risk-map response"),
    date: str | None = Query(None, description="YYYY-MM-DD, UTC. Defaults to today; cannot be in the future."),
    # le is a static upper bound (FastAPI/pydantic Query constraints are fixed at decorator-eval
    # time, before the model bundle loads) — kept generous rather than tied exactly to
    # len(FEATURE_COLUMNS), so it doesn't need updating every time a feature gets added/promoted.
    top_n: int = Query(6, ge=1, le=40, description="How many top per-feature contributions to return."),
) -> dict[str, Any]:
    """/predict/live's score, plus a per-prediction SHAP breakdown of what drove it.

    Fetches the same live weather + neighbor-fire features /predict/live does, then runs
    `evaluation/shap_analysis.py`'s TreeExplainer against a fixed training-set background sample
    built once at startup (see `lifespan`). Contributions are in raw `ignition_probability` units:
    there is no SHAP decomposition of `calibrated_probability`, since the isotonic calibrator is a
    separate post-hoc regression bolted on after the tree ensemble, not part of its structure — see
    docs/06-modeling-and-evaluation.md#2-shap-explainability.
    """
    background = state.get("shap_background")
    if background is None:
        raise HTTPException(
            status_code=503,
            detail="Historical dataset not available on this deployment — /predict/explain needs a "
            "training-set background sample to compute SHAP values against.",
        )

    target_date, latitude, longitude, features = _resolve_live_features(cell_id, date)
    bundle: ModelBundle = state["bundle"]

    row = pd.DataFrame([features])[bundle.feature_columns]
    explanation = explain(bundle.model, row, background)
    contributions = top_contributions(explanation, row_index=0, top_n=top_n)

    return {
        "cell_id": cell_id,
        "date": target_date.isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "ignition_probability": bundle.predict_proba(features),
        "calibrated_probability": bundle.predict_calibrated_proba(features),
        "top_contributions": contributions,
        "explanation_note": (
            "Contributions are in raw ignition_probability units, signed: positive pushes the "
            "prediction up, negative pushes it down. No SHAP decomposition exists for "
            "calibrated_probability — see docs/06-modeling-and-evaluation.md#2-shap-explainability."
        ),
    }


@app.get("/risk-map")
def risk_map(date: str = Query(..., description="YYYY-MM-DD, must exist in the processed dataset")) -> list[dict[str, Any]]:
    """Predicted risk + actual label for every grid cell on a historical date."""
    dataset: pd.DataFrame | None = state["dataset"]
    if dataset is None:
        raise HTTPException(status_code=503, detail="Historical dataset not available on this deployment.")

    timestamp = pd.Timestamp(date)
    _reject_if_outside_fire_season(timestamp.date())

    bundle: ModelBundle = state["bundle"]
    day_rows = dataset[dataset[DATE_COLUMN] == timestamp]
    if day_rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for {date} in the processed dataset.")

    probabilities = bundle.model.predict_proba(day_rows[bundle.feature_columns])[:, 1]
    result = day_rows[["cell_id", LABEL_COLUMN]].copy()
    result["risk_probability"] = probabilities
    result["calibrated_risk_probability"] = (
        bundle.calibrator.predict(probabilities) if bundle.calibrator is not None else None
    )
    result = result.merge(state["grid_cells"], on="cell_id", how="left")

    return result.rename(columns={LABEL_COLUMN: "actual_ignited"})[
        ["cell_id", "latitude", "longitude", "risk_probability", "calibrated_risk_probability", "actual_ignited"]
    ].to_dict(orient="records")
