import pandas as pd

from firesight.features.fuel_type import (
    build_fuel_type_features,
    encode_fuel_type_features,
    fetch_fuel_type,
    fetch_fuel_types,
    fetch_or_load_fuel_types,
)


class _FakeResponse:
    def __init__(self, codes: list[str | None]):
        self._codes = codes

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"features": [{"properties": {"FT_PROMETHEUS": c}} for c in self._codes]}


class _FakeSession:
    def __init__(self, codes_by_call: list[list[str]]):
        self._codes_by_call = codes_by_call
        self.calls: list[dict] = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        return _FakeResponse(self._codes_by_call[len(self.calls) - 1])


def test_fetch_fuel_type_returns_the_single_matched_code():
    session = _FakeSession([["C-3"]])
    result = fetch_fuel_type(50.6, -120.3, session=session)
    assert result == "C-3"
    assert "BBOX(SHAPE" in session.calls[0]["CQL_FILTER"]


def test_fetch_fuel_type_resolves_boundary_ties_by_mode():
    session = _FakeSession([["S-2", "S-2", "O-1a"]])
    result = fetch_fuel_type(50.6, -120.3, session=session)
    assert result == "S-2"


def test_fetch_fuel_type_returns_unknown_when_no_polygon_covers_the_point():
    session = _FakeSession([[]])
    result = fetch_fuel_type(50.6, -120.3, session=session)
    assert result == "unknown"


def test_fetch_fuel_type_ignores_null_codes_among_returned_features():
    session = _FakeSession([[None, "C-5"]])
    result = fetch_fuel_type(50.6, -120.3, session=session)
    assert result == "C-5"


def test_fetch_fuel_types_queries_every_cell_and_pauses_between_them(monkeypatch):
    sleeps = []
    monkeypatch.setattr("firesight.features.fuel_type.time.sleep", lambda s: sleeps.append(s))
    grid_cells = pd.DataFrame({"cell_id": ["0_0", "0_1", "1_0"], "latitude": [50.6, 50.6, 50.7], "longitude": [-120.3, -120.2, -120.3]})
    session = _FakeSession([["C-3"], ["O-1a"], ["N"]])

    result = fetch_fuel_types(grid_cells, session=session, pause_seconds=0.5)

    assert list(result["cell_id"]) == ["0_0", "0_1", "1_0"]
    assert list(result["fuel_type_cd"]) == ["C-3", "O-1a", "N"]
    assert sleeps == [0.5, 0.5]  # paused between calls, not after the last one


def test_fetch_or_load_fuel_types_uses_the_cache_on_a_second_call(tmp_path):
    grid_cells = pd.DataFrame({"cell_id": ["0_0"], "latitude": [50.6], "longitude": [-120.3]})
    cache_path = tmp_path / "fuel_type.parquet"
    session = _FakeSession([["C-3"], ["should not be reached"]])

    first = fetch_or_load_fuel_types(grid_cells, cache_path=cache_path, session=session)
    assert len(session.calls) == 1

    second = fetch_or_load_fuel_types(grid_cells, cache_path=cache_path, session=session)
    assert len(session.calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_encode_fuel_type_features_one_hots_only_codes_present_in_this_region():
    fuel_types = pd.DataFrame({"cell_id": ["0_0", "0_1", "1_0"], "fuel_type_cd": ["C-3", "O-1a", "C-3"]})

    result = encode_fuel_type_features(fuel_types)

    assert set(result.columns) == {"cell_id", "fuel_type_C-3", "fuel_type_O-1a"}
    row = result.set_index("cell_id")
    assert row.loc["0_0", "fuel_type_C-3"] == 1.0
    assert row.loc["0_0", "fuel_type_O-1a"] == 0.0
    assert row.loc["0_1", "fuel_type_O-1a"] == 1.0


def test_build_fuel_type_features_returns_one_row_per_grid_cell(tmp_path):
    grid_cells = pd.DataFrame({"cell_id": ["0_0", "0_1"], "latitude": [50.6, 50.6], "longitude": [-120.3, -120.2]})
    session = _FakeSession([["C-3"], ["N"]])

    result = build_fuel_type_features(grid_cells, cache_path=tmp_path / "fuel_type.parquet", session=session)

    assert set(result["cell_id"]) == {"0_0", "0_1"}
    assert "fuel_type_C-3" in result.columns
    assert "fuel_type_N" in result.columns
