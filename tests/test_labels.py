import pandas as pd

from firesight.features.grid import build_grid_cells
from firesight.features.labels import build_label_scaffold, filter_real_fires


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


def test_build_grid_cells_covers_bbox_corners():
    grid = build_grid_cells("-121.5,49.8,-119.0,51.5", cell_size_km=5.0)
    assert grid["latitude"].min() < 49.9
    assert grid["latitude"].max() > 51.4
    assert grid["longitude"].min() < -121.4
    assert grid["longitude"].max() > -119.1
    assert grid["cell_id"].is_unique
