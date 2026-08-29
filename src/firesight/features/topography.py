"""Static per-cell terrain features: elevation, slope, aspect.

Terrain drives fire behavior independently of weather: slope steepens spread rate (fire moves
faster uphill), and aspect controls solar exposure (south-facing slopes in the northern hemisphere
dry out faster, a real, well-documented driver in physics-based fire-behavior systems like the
Canadian FBP System, which this project doesn't implement). FireSight had zero terrain features
before this module.

Elevation comes from Open-Meteo's Elevation API (Copernicus DEM 2021, 90m resolution), the same
provider `features/live_weather.py` already depends on, avoiding a new vendor, and since these are
per-cell centroid point queries rather than a raster download, avoiding the geopandas/GDAL stack
`features/grid.py`'s own docstring says this project deliberately does without. One elevation
value per grid cell, fetched once and cached under `data/raw/` (gitignored, matching this project's
existing raw-data convention), terrain doesn't change day to day, so there's no live/offline split
the way weather has.

Slope and aspect are derived, not fetched: Horn's method (Horn 1981, the same formula ESRI's Slope
and Aspect tools implement), computed from each cell's 8 Moore-neighbor elevations rather than a
fine-grained DEM raster, appropriate at this project's 5km cell resolution, where the elevation
signal is already a coarse per-cell average, not a place sub-cell DEM detail would help. A missing
neighbor (grid edge) falls back to the center cell's own elevation, the standard "edge replication"
convention DEM slope tools use, so a boundary cell gets a (less certain, gradient-underestimating)
slope/aspect rather than being dropped from the dataset entirely.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import requests

ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
MAX_COORDS_PER_REQUEST = 100  # Open-Meteo's documented per-request coordinate limit
DEFAULT_CACHE_PATH = Path("data/raw/topography/kamloops_elevation.parquet")
PAUSE_BETWEEN_REQUESTS_SECONDS = 12.0  # Open-Meteo's 600-request/minute free-tier budget is spent
# per *coordinate*, not per HTTP call -- verified directly (a 429 hit after ~6 successive
# 100-coordinate requests with only a 2s pause, i.e. ~600 coordinates/minute exactly), not assumed
# from the documented per-request limit alone. 12s between 100-coordinate chunks caps this well
# under 600/min; this project's grid needs ~15 chunks total, a one-time, cached fetch (see
# fetch_or_load_elevations), so the ~3 minutes this adds costs little.

TOPOGRAPHY_COLUMNS = ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos"]


def fetch_elevations(
    grid_cells: pd.DataFrame,
    session: requests.Session | None = None,
    max_attempts: int = 4,
    pause_seconds: float = PAUSE_BETWEEN_REQUESTS_SECONDS,
) -> pd.DataFrame:
    """Elevation (meters) for every cell's centroid, via Open-Meteo's Elevation API.

    Batches into <=100-coordinate requests, pausing between them and retrying with exponential
    backoff on failure (including a 429), the same retry shape
    `pipeline/ingest_firms.py::fetch_window` already uses for FIRMS. Returns a `cell_id`/
    `elevation_m` frame in the same row order as `grid_cells`.
    """
    session = session or requests.Session()
    elevations: list[float] = []
    n_chunks = (len(grid_cells) + MAX_COORDS_PER_REQUEST - 1) // MAX_COORDS_PER_REQUEST
    for chunk_idx, start in enumerate(range(0, len(grid_cells), MAX_COORDS_PER_REQUEST)):
        chunk = grid_cells.iloc[start : start + MAX_COORDS_PER_REQUEST]
        params = {
            "latitude": ",".join(f"{lat:.6f}" for lat in chunk["latitude"]),
            "longitude": ",".join(f"{lon:.6f}" for lon in chunk["longitude"]),
        }

        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(ELEVATION_API_URL, params=params, timeout=30)
                response.raise_for_status()
                elevations.extend(response.json()["elevation"])
                break
            except requests.exceptions.RequestException:
                if attempt == max_attempts:
                    raise
                wait = 2**attempt
                print(f"  elevation request failed (attempt {attempt}/{max_attempts}), retrying in {wait}s", flush=True)
                time.sleep(wait)

        if chunk_idx < n_chunks - 1:
            time.sleep(pause_seconds)

    return pd.DataFrame({"cell_id": grid_cells["cell_id"].to_numpy(), "elevation_m": elevations})


def fetch_or_load_elevations(
    grid_cells: pd.DataFrame,
    cache_path: Path = DEFAULT_CACHE_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Elevations from the local cache if present, else fetched fresh and cached.

    Terrain doesn't change day to day, so re-fetching on every pipeline run would just put repeated
    load on a free public API for an answer that's already known, cache once, reuse thereafter.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    elevations = fetch_elevations(grid_cells, session=session)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    elevations.to_parquet(cache_path, index=False)
    return elevations


def compute_slope_aspect(elevations: pd.DataFrame, cell_size_km: float = 5.0) -> pd.DataFrame:
    """Horn's-method slope (degrees) and aspect (as sin/cos of a compass bearing) per cell.

    Aspect is encoded as sin/cos rather than a raw bearing for the same reason
    `engineering.py::add_wind_features` encodes wind direction that way, compass direction is
    circular, and a raw 0-360 numeric column would misrepresent that to any model treating feature
    distance linearly.

    Row/col convention (`features/grid.py::assign_cell_ids`): row increases north, col increases
    east. `east_gradient`/`north_gradient` below point *uphill*; a slope's aspect is the compass
    direction it *faces* (downhill), hence the negation in `aspect_rad`.
    """
    elevation_by_cell = dict(zip(elevations["cell_id"], elevations["elevation_m"]))
    cell_size_m = cell_size_km * 1000.0

    rows = []
    for cell_id, z_center in elevation_by_cell.items():
        row, col = (int(v) for v in cell_id.split("_", 1))
        neighbors = {
            (dr, dc): elevation_by_cell.get(f"{row + dr}_{col + dc}", z_center)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        }
        z_nw, z_n, z_ne = neighbors[1, -1], neighbors[1, 0], neighbors[1, 1]
        z_w, z_e = neighbors[0, -1], neighbors[0, 1]
        z_sw, z_s, z_se = neighbors[-1, -1], neighbors[-1, 0], neighbors[-1, 1]

        east_gradient = ((z_ne + 2 * z_e + z_se) - (z_nw + 2 * z_w + z_sw)) / (8 * cell_size_m)
        north_gradient = ((z_nw + 2 * z_n + z_ne) - (z_sw + 2 * z_s + z_se)) / (8 * cell_size_m)

        slope_rad = math.atan(math.hypot(east_gradient, north_gradient))
        aspect_rad = math.atan2(-east_gradient, -north_gradient)

        rows.append(
            {
                "cell_id": cell_id,
                "elevation_m": z_center,
                "slope_degrees": math.degrees(slope_rad),
                "aspect_sin": math.sin(aspect_rad),
                "aspect_cos": math.cos(aspect_rad),
            }
        )
    return pd.DataFrame(rows)


def build_topography_features(
    grid_cells: pd.DataFrame,
    cell_size_km: float = 5.0,
    cache_path: Path = DEFAULT_CACHE_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Elevation + slope + aspect for every cell, the join target for pipeline/build_dataset.py."""
    elevations = fetch_or_load_elevations(grid_cells, cache_path=cache_path, session=session)
    return compute_slope_aspect(elevations, cell_size_km=cell_size_km)
