import numpy as np
import pandas as pd

from firesight.features.grid import build_grid_cells
from firesight.features.labels import (
    add_forward_ignition_label,
    build_label_scaffold,
    filter_real_fires,
)


def test_filter_real_fires_drops_non_vegetation_types():
    df = pd.DataFrame({"type": [0, 1, 2, 3, 0]})
    result = filter_real_fires(df)
    assert (result["type"] == 0).all()
    assert len(result) == 2


def test_build_label_scaffold_marks_ignitions_and_fills_rest_with_zero():
    grid_cells = pd.DataFrame({"cell_id": ["1_1", "2_2"], "latitude": [50.0, 50.5], "longitude": [-120.0, -120.5]})
    fire_df = pd.DataFrame({"cell_id": ["1_1"], "acq_date": ["2021-06-02"]})

    scaffold = build_label_scaffold(fire_df, grid_cells, "2021-06-01", "2021-06-03")

    assert len(scaffold) == 2 * 3  # 2 cells x 3 days
    hit = scaffold[(scaffold["cell_id"] == "1_1") & (scaffold["date"] == "2021-06-02")]
    assert hit["ignited"].item() == 1
    assert scaffold["ignited"].sum() == 1


def test_add_forward_ignition_label_with_n_days_1_reproduces_ignited_exactly():
    df = pd.DataFrame(
        {
            "cell_id": ["0_0"] * 4,
            "date": pd.date_range("2021-06-01", periods=4),
            "ignited": [0, 1, 0, 1],
        }
    )

    result = add_forward_ignition_label(df, n_days=1)

    assert result["ignited_next_1d"].tolist() == [0, 1, 0, 1]


def test_add_forward_ignition_label_flags_a_hit_anywhere_in_the_forward_window():
    df = pd.DataFrame(
        {
            "cell_id": ["0_0"] * 5,
            "date": pd.date_range("2021-06-01", periods=5),
            "ignited": [0, 0, 1, 0, 0],
        }
    )

    result = add_forward_ignition_label(df, n_days=3)

    # day 1 (index 0): window is [d1,d2,d3] -> hit on d3 -> 1
    # day 2 (index 1): window is [d2,d3,d4] -> hit on d3 -> 1
    # day 3 (index 2): window is [d3,d4,d5] -> hit on d3 -> 1
    # day 4 (index 3): window would need d4,d5,d6 -> d6 doesn't exist -> NaN
    # day 5 (index 4): window would need d5,d6,d7 -> NaN
    assert result["ignited_next_3d"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert result["ignited_next_3d"].iloc[3:].isna().all()


def test_add_forward_ignition_label_never_lets_one_cells_window_see_another_cells_fire():
    """Cell B ignites on its second day; cell A, otherwise identical, never does -- a naive rolling
    window over the whole (unsorted-by-cell) frame would leak B's fire into A's window."""
    df = pd.DataFrame(
        {
            "cell_id": ["A", "A", "B", "B"],
            "date": list(pd.date_range("2021-06-01", periods=2)) * 2,
            "ignited": [0, 0, 0, 1],
        }
    )

    result = add_forward_ignition_label(df, n_days=2)

    cell_a = result[result["cell_id"] == "A"].sort_values("date")["ignited_next_2d"]
    assert cell_a.iloc[0] == 0.0
    assert np.isnan(cell_a.iloc[1])  # window needs a day-3 that doesn't exist -> incomplete, NaN

    cell_b = result[result["cell_id"] == "B"].sort_values("date")["ignited_next_2d"]
    assert cell_b.iloc[0] == 1.0  # window is [d1, d2], d2 ignites


def test_build_grid_cells_covers_bbox_corners():
    grid = build_grid_cells("-121.5,49.8,-119.0,51.5", cell_size_km=5.0)
    assert grid["latitude"].min() < 49.9
    assert grid["latitude"].max() > 51.4
    assert grid["longitude"].min() < -121.4
    assert grid["longitude"].max() > -119.1
    assert grid["cell_id"].is_unique
