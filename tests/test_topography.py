import math

import pandas as pd
import pytest

from firesight.features.topography import (
    build_topography_features,
    compute_slope_aspect,
    fetch_elevations,
    fetch_or_load_elevations,
)


class _FakeResponse:
    def __init__(self, elevations: list[float]):
        self._elevations = elevations

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"elevation": self._elevations}


class _FakeSession:
    def __init__(self):
        self.calls: list[dict] = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        n = len(params["latitude"].split(","))
        return _FakeResponse([100.0 + i for i in range(n)])


def test_fetch_elevations_batches_into_at_most_100_coordinates_per_request():
    grid_cells = pd.DataFrame(
        {
            "cell_id": [f"0_{i}" for i in range(250)],
            "latitude": [50.6] * 250,
            "longitude": [-120.3] * 250,
        }
    )
    session = _FakeSession()

    result = fetch_elevations(grid_cells, session=session, pause_seconds=0)

    assert len(session.calls) == 3  # 100 + 100 + 50
    assert [len(c["latitude"].split(",")) for c in session.calls] == [100, 100, 50]
    assert len(result) == 250
    assert list(result["cell_id"]) == list(grid_cells["cell_id"])


def test_fetch_or_load_elevations_uses_the_cache_on_a_second_call(tmp_path):
    grid_cells = pd.DataFrame({"cell_id": ["0_0", "0_1"], "latitude": [50.6, 50.7], "longitude": [-120.3, -120.2]})
    cache_path = tmp_path / "elevation.parquet"
    session = _FakeSession()

    first = fetch_or_load_elevations(grid_cells, cache_path=cache_path, session=session)
    assert len(session.calls) == 1

    second = fetch_or_load_elevations(grid_cells, cache_path=cache_path, session=session)
    assert len(session.calls) == 1  # no new network call -- served from cache
    pd.testing.assert_frame_equal(first, second)


def test_compute_slope_aspect_is_flat_for_uniform_elevation():
    elevations = pd.DataFrame(
        {"cell_id": ["0_0", "0_1", "1_0", "1_1", "-1_0", "-1_1", "0_-1", "1_-1", "-1_-1"], "elevation_m": [500.0] * 9}
    )
    result = compute_slope_aspect(elevations).set_index("cell_id")
    assert result.loc["0_0", "slope_degrees"] == pytest.approx(0.0, abs=1e-6)


def test_compute_slope_aspect_faces_west_when_elevation_rises_eastward():
    """Elevation = f(col) only (flat north-south) -- ground rises to the east, so the slope should
    face (downhill toward) west: aspect_sin ~= sin(270deg) = -1, aspect_cos ~= cos(270deg) = 0."""
    rows, cols = range(-1, 2), range(-1, 2)
    elevations = pd.DataFrame(
        [{"cell_id": f"{r}_{c}", "elevation_m": 100.0 * c} for r in rows for c in cols]
    )

    result = compute_slope_aspect(elevations, cell_size_km=5.0).set_index("cell_id")
    center = result.loc["0_0"]

    # 100m rise per 5km cell -> atan(100/5000) ~= 1.146 degrees.
    assert center["slope_degrees"] == pytest.approx(math.degrees(math.atan(0.02)), abs=1e-4)
    assert center["aspect_sin"] == pytest.approx(-1.0, abs=1e-6)
    assert center["aspect_cos"] == pytest.approx(0.0, abs=1e-6)


def test_compute_slope_aspect_faces_south_when_elevation_rises_northward():
    """Elevation = f(row) only (flat east-west) -- ground rises to the north, so the slope should
    face south: aspect_sin ~= sin(180deg) = 0, aspect_cos ~= cos(180deg) = -1."""
    rows, cols = range(-1, 2), range(-1, 2)
    elevations = pd.DataFrame(
        [{"cell_id": f"{r}_{c}", "elevation_m": 100.0 * r} for r in rows for c in cols]
    )

    result = compute_slope_aspect(elevations, cell_size_km=5.0).set_index("cell_id")
    center = result.loc["0_0"]

    assert center["aspect_sin"] == pytest.approx(0.0, abs=1e-6)
    assert center["aspect_cos"] == pytest.approx(-1.0, abs=1e-6)


def test_compute_slope_aspect_missing_neighbors_fall_back_to_the_center_cells_own_elevation():
    """A cell at the grid edge (no neighbor to its east) should compute a defined, finite
    slope/aspect using edge-replication, not raise or produce NaN."""
    elevations = pd.DataFrame(
        [
            {"cell_id": "0_0", "elevation_m": 500.0},
            {"cell_id": "1_0", "elevation_m": 520.0},
            {"cell_id": "-1_0", "elevation_m": 480.0},
            {"cell_id": "0_-1", "elevation_m": 490.0},
            {"cell_id": "1_-1", "elevation_m": 510.0},
            {"cell_id": "-1_-1", "elevation_m": 470.0},
            # no col=1 (east) neighbors at all -- 0_0 is the eastern edge of this tiny grid
        ]
    )
    result = compute_slope_aspect(elevations).set_index("cell_id")
    center = result.loc["0_0"]
    assert pd.notna(center["slope_degrees"])
    assert pd.notna(center["aspect_sin"])
    assert pd.notna(center["aspect_cos"])


def test_build_topography_features_returns_one_row_per_grid_cell(tmp_path):
    grid_cells = pd.DataFrame(
        {"cell_id": ["0_0", "0_1", "1_0"], "latitude": [50.6, 50.6, 50.7], "longitude": [-120.3, -120.2, -120.3]}
    )
    session = _FakeSession()

    result = build_topography_features(grid_cells, cache_path=tmp_path / "elevation.parquet", session=session)

    assert set(result["cell_id"]) == set(grid_cells["cell_id"])
    assert {"elevation_m", "slope_degrees", "aspect_sin", "aspect_cos"}.issubset(result.columns)
