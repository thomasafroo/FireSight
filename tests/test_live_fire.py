import datetime as dt

import pandas as pd

from firesight.features.live_fire import (
    build_live_neighbor_fire_features,
    fetch_recent_detections,
)


def test_fetch_recent_detections_chunks_into_at_most_5_day_windows(monkeypatch):
    calls = []

    def fake_fetch_window(bbox, source, start_date, day_range, map_key):
        calls.append((start_date, day_range))
        return pd.DataFrame({"latitude": [50.0], "longitude": [-120.0], "acq_date": [start_date]})

    monkeypatch.setattr("firesight.features.live_fire.fetch_window", fake_fetch_window)

    df = fetch_recent_detections("bbox", dt.date(2024, 7, 15), lookback_days=7)

    # 7 days can't fit in one <=5-day request, chunked into a 5-day window then a 2-day window,
    # covering [2024-07-09, 2024-07-15] with no gap or overlap.
    assert calls == [("2024-07-09", 5), ("2024-07-14", 2)]
    assert len(df) == 2


def test_fetch_recent_detections_returns_empty_frame_when_every_chunk_is_empty(monkeypatch):
    monkeypatch.setattr(
        "firesight.features.live_fire.fetch_window",
        lambda bbox, source, start_date, day_range, map_key: pd.DataFrame(columns=["latitude", "longitude", "acq_date"]),
    )

    df = fetch_recent_detections("bbox", dt.date(2024, 7, 15), lookback_days=3)

    assert df.empty


def _detection(row: int, col: int, acq_date: str) -> dict:
    """A synthetic FIRMS detection landing in grid cell "{row}_{col}" (`grid.py`'s cell centroid
    math, inverted: pick a lat/lon comfortably inside that cell rather than exactly on its edge)."""
    from firesight.features.grid import cell_size_degrees

    lat_size, lon_size = cell_size_degrees(5.0, 50.6)
    return {
        "latitude": (row + 0.5) * lat_size,
        "longitude": (col + 0.5) * lon_size,
        "acq_date": acq_date,
    }


def test_build_live_neighbor_fire_features_counts_only_moore_neighbors_within_each_window(monkeypatch):
    target_date = dt.date(2024, 7, 15)
    target_cell = "1105_-1677"
    row, col = 1105, -1677

    detections = pd.DataFrame(
        [
            _detection(row - 1, col, "2024-07-14"),  # a real neighbor, 1 day prior -> in every window
            _detection(row, col + 1, "2024-07-10"),  # a real neighbor, 5 days prior -> only the 7d window
            _detection(row + 5, col + 5, "2024-07-14"),  # not a Moore neighbor -> excluded entirely
            _detection(row, col, "2024-07-14"),  # the target cell itself, not a neighbor -> excluded
        ]
    )
    monkeypatch.setattr(
        "firesight.features.live_fire.fetch_recent_detections",
        lambda bbox, end_date, lookback_days, source, map_key: detections,
    )

    counts = build_live_neighbor_fire_features(target_cell, target_date, bbox="bbox")

    assert counts == {
        "neighbor_fire_count_1d": 1.0,
        "neighbor_fire_count_3d": 1.0,
        "neighbor_fire_count_7d": 2.0,
    }


def test_build_live_neighbor_fire_features_excludes_same_day_detections(monkeypatch):
    """The strictly-prior-day leakage guard `add_neighbor_fire_features` applies in training must
    hold here too: a neighbor detection dated the same as target_date shouldn't count."""
    target_date = dt.date(2024, 7, 15)
    target_cell = "1105_-1677"
    detections = pd.DataFrame([_detection(1104, -1677, "2024-07-15")])
    monkeypatch.setattr(
        "firesight.features.live_fire.fetch_recent_detections",
        lambda bbox, end_date, lookback_days, source, map_key: detections,
    )

    counts = build_live_neighbor_fire_features(target_cell, target_date, bbox="bbox")

    assert counts == {"neighbor_fire_count_1d": 0.0, "neighbor_fire_count_3d": 0.0, "neighbor_fire_count_7d": 0.0}


def test_build_live_neighbor_fire_features_counts_multiple_detections_same_cell_day_once(monkeypatch):
    """A neighbor with 3 separate FIRMS detections on the same day is one ignition-day, not 3."""
    target_date = dt.date(2024, 7, 15)
    target_cell = "1105_-1677"
    detections = pd.DataFrame(
        [_detection(1104, -1677, "2024-07-14") for _ in range(3)]
    )
    monkeypatch.setattr(
        "firesight.features.live_fire.fetch_recent_detections",
        lambda bbox, end_date, lookback_days, source, map_key: detections,
    )

    counts = build_live_neighbor_fire_features(target_cell, target_date, bbox="bbox")

    assert counts["neighbor_fire_count_1d"] == 1.0


def test_build_live_neighbor_fire_features_returns_zeros_when_no_detections(monkeypatch):
    monkeypatch.setattr(
        "firesight.features.live_fire.fetch_recent_detections",
        lambda bbox, end_date, lookback_days, source, map_key: pd.DataFrame(columns=["latitude", "longitude", "acq_date"]),
    )

    counts = build_live_neighbor_fire_features("1105_-1677", dt.date(2024, 7, 15), bbox="bbox")

    assert counts == {"neighbor_fire_count_1d": 0.0, "neighbor_fire_count_3d": 0.0, "neighbor_fire_count_7d": 0.0}
