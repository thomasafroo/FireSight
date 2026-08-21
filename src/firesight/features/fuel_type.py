"""BC Provincial Fuel Type Layer — per-cell FBP fuel type classification.

The single largest input-category gap this project had before this module: a 2020-2025 systematic
review of wildfire ML studies found fuel/vegetation state was the most common input category
overall (44.7% of reported inputs, ahead of weather/climate) — FireSight had zero vegetation
features before this. Fuel type is the categorical input the Canadian FBP System itself is built
around (C-1..C-7 conifer, D-1/2 deciduous, M-1..M-4 mixedwood, S-1..S-3 slash, O-1a/b grass, N
non-fuel, W water — see the FBP System background at cwfis.cfs.nrcan.gc.ca) and is a genuinely
different signal from weather: two adjacent cells with identical temperature/humidity/wind can
still have very different real ignition/spread risk if one is grassland and the other is wet
deciduous forest.

**Source, verified live against the actual service, not assumed from documentation.** BC's
Provincial Fuel Type Layer (`WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP`, from the BC Data
Catalogue) is served as a WFS polygon layer, not a small pre-clipped file — checked directly, and it
covers the whole province at individual-forest-stand resolution (>400,000 polygons intersect the
Kamloops FC bbox alone; the province-wide download is a ~4GB File Geodatabase). Rather than
downloading that (a real multi-GB file, and File Geodatabases need `fiona`/GDAL to read — a
dependency this project doesn't have, see `features/grid.py`'s docstring on the same tradeoff),
this module queries the WFS endpoint directly per grid-cell centroid with a small bounding box
(`BOX_HALF_DEGREE`, ~28m) — checked directly that a raw CQL point-`INTERSECTS` filter against this
layer's `SHAPE` geometry doesn't match (the layer's native CRS, BC Albers/EPSG:3005, isn't accepted
via a bare CQL point literal on this service), so a small bbox is the query shape that actually
works, not a simplification chosen for convenience.

**`FT_PROMETHEUS`, not the raw `FUEL_TYPE_CD` field, is what this module keeps.** `FUEL_TYPE_CD`
often carries a burn-history prefix (e.g. `B71_S-2` — modified by a specific past burn) that
multiplies the effective class count far past the ~16 base FBP types; `FT_PROMETHEUS` is the same
layer's own already-cleaned base code (`S-2` for that same polygon) — the field the BC Wildfire
Service's own Prometheus fire-growth simulator consumes, verified directly against several real
query responses rather than assumed from the field's name alone.

**Mode resolution at fragmented boundaries, not an error.** A tiny bbox around a grid cell centroid
occasionally straddles more than one stand polygon (common near urban/fuel-type edges); this module
takes the most common code among whatever polygons intersect that box — a reasonable resolution for
a categorical model feature, not a claim of surveyor-grade single-point precision. A cell with zero
polygon coverage at all (unmapped gap) gets `UNKNOWN_FUEL_TYPE` rather than a guessed class.

Fetched once and cached under `data/raw/` (gitignored, matching this project's raw-data convention)
— fuel type doesn't change day to day (except after an actual burn, well outside this project's
current per-day feature-refresh scope), so there's no live/offline split the way weather has.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP/ows"
TYPE_NAME = "pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP"
BOX_HALF_DEGREE = 0.00025  # ~28m at this latitude -- "what fuel type is at this point," not
# "what fuel types are nearby" (see module docstring for why a bbox, not a point filter, is used)
DEFAULT_CACHE_PATH = Path("data/raw/fuel_type/kamloops_fuel_type.parquet")
PAUSE_BETWEEN_REQUESTS_SECONDS = 0.1  # one query per grid cell (~1,443 for Kamloops FC) -- a
# one-time, cached fetch, but polite pacing against a shared government service costs little

UNKNOWN_FUEL_TYPE = "unknown"


def fetch_fuel_type(latitude: float, longitude: float, session: requests.Session, max_attempts: int = 4) -> str:
    """The base FBP fuel type code (`FT_PROMETHEUS`, e.g. "C-3", "O-1a", "N") at one point.

    See module docstring for why this queries a small bbox rather than the point directly, and why
    ties among multiple returned polygons are resolved by mode.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": TYPE_NAME,
        "outputFormat": "application/json",
        "propertyName": "FT_PROMETHEUS",
        "CQL_FILTER": (
            f"BBOX(SHAPE, {longitude - BOX_HALF_DEGREE}, {latitude - BOX_HALF_DEGREE}, "
            f"{longitude + BOX_HALF_DEGREE}, {latitude + BOX_HALF_DEGREE}, 'EPSG:4326')"
        ),
    }

    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(WFS_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == max_attempts:
                raise
            wait = 2**attempt
            print(f"  fuel type request failed (attempt {attempt}/{max_attempts}), retrying in {wait}s", flush=True)
            time.sleep(wait)

    features = response.json()["features"]
    if not features:
        return UNKNOWN_FUEL_TYPE
    codes = pd.Series([f["properties"]["FT_PROMETHEUS"] for f in features]).dropna()
    if codes.empty:
        return UNKNOWN_FUEL_TYPE
    return codes.mode().iloc[0]


def fetch_fuel_types(
    grid_cells: pd.DataFrame,
    session: requests.Session | None = None,
    pause_seconds: float = PAUSE_BETWEEN_REQUESTS_SECONDS,
) -> pd.DataFrame:
    """FBP fuel type for every cell's centroid — one WFS query per cell (see `fetch_fuel_type`)."""
    session = session or requests.Session()
    codes = []
    n = len(grid_cells)
    for i, row in enumerate(grid_cells.itertuples()):
        codes.append(fetch_fuel_type(row.latitude, row.longitude, session))
        if i < n - 1:
            time.sleep(pause_seconds)
    return pd.DataFrame({"cell_id": grid_cells["cell_id"].to_numpy(), "fuel_type_cd": codes})


def fetch_or_load_fuel_types(
    grid_cells: pd.DataFrame,
    cache_path: Path = DEFAULT_CACHE_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fuel types from the local cache if present, else fetched fresh and cached."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    fuel_types = fetch_fuel_types(grid_cells, session=session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fuel_types.to_parquet(cache_path, index=False)
    return fuel_types


def encode_fuel_type_features(fuel_types: pd.DataFrame) -> pd.DataFrame:
    """One `fuel_type_<code>` 0/1 column per code actually present in this region.

    Not a fixed province-wide 16-class schema — a class that never occurs in the Kamloops FC
    extract would just be an always-zero column, dead weight for both training and storage.
    """
    dummies = pd.get_dummies(fuel_types["fuel_type_cd"], prefix="fuel_type").astype(float)
    return pd.concat([fuel_types[["cell_id"]], dummies], axis=1)


def build_fuel_type_features(
    grid_cells: pd.DataFrame,
    cache_path: Path = DEFAULT_CACHE_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """One-hot fuel type columns for every cell — the join target for pipeline/build_dataset.py."""
    fuel_types = fetch_or_load_fuel_types(grid_cells, cache_path=cache_path, session=session)
    return encode_fuel_type_features(fuel_types)
