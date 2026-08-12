"""FastAPI inference service wrapping the trained wildfire ignition model.

Two endpoints, deliberately scoped small for the MVP:

- POST /predict — score one (cell, day)'s worth of already-computed
  features. Does *not* fetch live weather or run feature engineering
  itself; the caller supplies feature values directly. Wiring this up
  to a live ERA5 feed is future work, out of scope for "wrap the
  trained model in an API" — see docs/README.md's project status.
- GET /risk-map — a historical demo endpoint: for a date already in the
  processed dataset, returns every cell's predicted risk *and* the
  actual recorded label, so the eventual frontend risk map has
  something real to render without needing a live data feed.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from firesight.features.grid import build_grid_cells
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX
from firesight.training.baseline import DATE_COLUMN, LABEL_COLUMN
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

    if DATASET_PATH.exists():
        keep = [DATE_COLUMN, "cell_id", LABEL_COLUMN, *bundle.feature_columns]
        state["dataset"] = pd.read_parquet(DATASET_PATH, columns=keep)
        state["grid_cells"] = build_grid_cells(BC_KAMLOOPS_BBOX)
    else:
        state["dataset"] = None
        state["grid_cells"] = None

    yield
    state.clear()


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

# Wide open for the local MVP frontend (a static file served with no fixed
# origin during development). Tighten to the real frontend's origin before
# this is ever deployed anywhere reachable outside localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/risk-map")
def risk_map(date: str = Query(..., description="YYYY-MM-DD, must exist in the processed dataset")) -> list[dict[str, Any]]:
    """Predicted risk + actual label for every grid cell on a historical date."""
    dataset: pd.DataFrame | None = state["dataset"]
    if dataset is None:
        raise HTTPException(status_code=503, detail="Historical dataset not available on this deployment.")

    bundle: ModelBundle = state["bundle"]
    day_rows = dataset[dataset[DATE_COLUMN] == pd.Timestamp(date)]
    if day_rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for {date} in the processed dataset.")

    probabilities = bundle.model.predict_proba(day_rows[bundle.feature_columns])[:, 1]
    result = day_rows[["cell_id", LABEL_COLUMN]].copy()
    result["risk_probability"] = probabilities
    result = result.merge(state["grid_cells"], on="cell_id", how="left")

    return result.rename(columns={LABEL_COLUMN: "actual_ignited"})[
        ["cell_id", "latitude", "longitude", "risk_probability", "actual_ignited"]
    ].to_dict(orient="records")
