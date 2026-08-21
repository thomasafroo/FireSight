"""Build a regular lat/lon grid and assign points to cells.

Deliberately avoids geopandas/GDAL for now — plain lat/lon math is enough
to get a working pipeline on a small region, and the geo stack can come
later once the MVP proves out.
"""

from __future__ import annotations

import math

import pandas as pd

KM_PER_DEGREE_LAT = 111.0


def cell_size_degrees(cell_size_km: float, latitude: float) -> tuple[float, float]:
    """Approximate (lat_size, lon_size) in degrees for a given cell size at a latitude."""
    lat_size = cell_size_km / KM_PER_DEGREE_LAT
    lon_size = cell_size_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(latitude)))
    return lat_size, lon_size


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse a "west,south,east,north" bbox string, as used by the FIRMS ingest."""
    west, south, east, north = (float(v) for v in bbox.split(","))
    return west, south, east, north


def assign_cell_ids(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cell_size_km: float = 5.0,
    reference_latitude: float = 50.6,
) -> pd.DataFrame:
    """Add a `cell_id` column mapping each row to a grid cell.

    Uses a single reference latitude for the whole bbox, which is fine for
    a regional slice (e.g. Kamloops Fire Centre) but would need per-row
    correction if the grid is extended to cover all of BC.
    """
    lat_size, lon_size = cell_size_degrees(cell_size_km, reference_latitude)
    row = (df[lat_col] // lat_size).astype(int)
    col = (df[lon_col] // lon_size).astype(int)
    df = df.copy()
    df["cell_id"] = row.astype(str) + "_" + col.astype(str)
    return df


def neighbor_cell_ids(cell_id: str) -> list[str]:
    """The up to 8 Moore neighbors of a cell, from its "{row}_{col}" id.

    Same offset scheme as `features/engineering.py::_moore_neighbor_adjacency` (duplicated rather
    than imported, to avoid a features/engineering.py <-> features/live_fire.py cross-dependency for
    one small helper). Doesn't check whether a neighbor id actually exists in the grid; callers that
    care (e.g. filtering fetched detections down to real neighbors) do that themselves.
    """
    row, col = (int(v) for v in cell_id.split("_", 1))
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    return [f"{row + dr}_{col + dc}" for dr, dc in offsets]


def build_grid_cells(
    bbox: str,
    cell_size_km: float = 5.0,
    reference_latitude: float = 50.6,
) -> pd.DataFrame:
    """Enumerate every grid cell covering a bbox, with its centroid.

    Uses the same row/col scheme as `assign_cell_ids` (floor division on the
    same reference latitude), so cell_ids line up between the two. This is
    the full cell universe for a region — including cells that never saw a
    fire detection — needed to build a (cell, date) label scaffold rather
    than just the cells that happen to appear in the fire data.
    """
    west, south, east, north = parse_bbox(bbox)
    lat_size, lon_size = cell_size_degrees(cell_size_km, reference_latitude)

    row_min, row_max = math.floor(south / lat_size), math.floor(north / lat_size)
    col_min, col_max = math.floor(west / lon_size), math.floor(east / lon_size)

    grid = pd.MultiIndex.from_product(
        [range(row_min, row_max + 1), range(col_min, col_max + 1)],
        names=["row", "col"],
    ).to_frame(index=False)
    grid["cell_id"] = grid["row"].astype(str) + "_" + grid["col"].astype(str)
    grid["latitude"] = (grid["row"] + 0.5) * lat_size
    grid["longitude"] = (grid["col"] + 0.5) * lon_size
    return grid[["cell_id", "latitude", "longitude"]]
