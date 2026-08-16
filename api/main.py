"""FastAPI inference service wrapping the trained wildfire ignition model.

Three endpoints:

- POST /predict — score one (cell, day)'s worth of already-computed
  features. Does *not* fetch live weather or run feature engineering
  itself; the caller supplies feature values directly.
- GET /predict/live — score a grid cell's *current* conditions by
  fetching recent weather from Open-Meteo and running it through the
  same feature-engineering pipeline training uses — see
  features/live_weather.py and docs/07-serving.md.
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

from firesight.features.grid import build_grid_cells
from firesight.features.live_weather import build_live_feature_row
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX
from firesight.training.baseline import (
    DATE_COLUMN,
    FIRE_SEASON_END,
    FIRE_SEASON_START,
    LABEL_COLUMN,
)
from firesight.training.export_model import MODEL_PATH
from firesight.training.persist import ModelBundle, load_model_bundle

DATASET_PATH = Path(os.environ.get("FIRESIGHT_DATASET_PATH", "data/processed/kamloops_dataset.parquet"))
MODEL_BUNDLE_PATH = Path(os.environ.get("FIRESIGHT_MODEL_PATH", str(MODEL_PATH)))

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
    # Always available (pure computation from the bbox, no file needed) —
    # /predict/live needs a cell's lat/lon regardless of whether the
    # historical dataset parquet has been built on this deployment.
    state["grid_cells"] = build_grid_cells(BC_KAMLOOPS_BBOX)

    if DATASET_PATH.exists():
        keep = [DATE_COLUMN, "cell_id", LABEL_COLUMN, *bundle.feature_columns]
        state["dataset"] = pd.read_parquet(DATASET_PATH, columns=keep)
    else:
        state["dataset"] = None

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
    return {
        "status": "ok",
        "model_type": bundle.metadata.get("model_type"),
        "feature_columns": bundle.feature_columns,
        "val_scores": bundle.metadata.get("val_scores"),
    }


@app.post("/predict")
def predict(features: dict) -> dict[str, float]:
    """Score one (cell, day)'s feature vector. Body: {feature_name: value, ...}."""
    bundle: ModelBundle = state["bundle"]
    request_model = state["feature_request_model"]
    try:
        validated = request_model(**features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probability = bundle.predict_proba(validated.model_dump())
    return {"ignition_probability": probability}


@app.get("/predict/live")
def predict_live(
    cell_id: str = Query(..., description="Grid cell id, e.g. from a /risk-map response"),
    date: str | None = Query(None, description="YYYY-MM-DD, UTC. Defaults to today; cannot be in the future."),
) -> dict[str, Any]:
    """Score a grid cell's *current* conditions via a live weather feed, not historical replay.

    Unlike /predict (which requires the caller to already have computed
    feature values) this fetches recent weather itself from Open-Meteo and
    runs it through the same feature-engineering functions training uses —
    see features/live_weather.py and docs/07-serving.md.
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
    bundle: ModelBundle = state["bundle"]

    try:
        features = build_live_feature_row(latitude, longitude, target_date, cell_id)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Live weather fetch failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "cell_id": cell_id,
        "date": target_date.isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "ignition_probability": bundle.predict_proba(features),
        "weather_source": "open-meteo archive API (ERA5-based reanalysis + near-real-time blend)",
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
    result = result.merge(state["grid_cells"], on="cell_id", how="left")

    return result.rename(columns={LABEL_COLUMN: "actual_ignited"})[
        ["cell_id", "latitude", "longitude", "risk_probability", "actual_ignited"]
    ].to_dict(orient="records")
