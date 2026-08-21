import pandas as pd

from firesight.features.grid import assign_cell_ids, neighbor_cell_ids


def test_assign_cell_ids_groups_nearby_points():
    df = pd.DataFrame(
        {
            "latitude": [50.60, 50.601, 50.90],
            "longitude": [-120.30, -120.301, -120.90],
        }
    )
    result = assign_cell_ids(df, cell_size_km=5.0)

    assert result.loc[0, "cell_id"] == result.loc[1, "cell_id"]
    assert result.loc[0, "cell_id"] != result.loc[2, "cell_id"]


def test_neighbor_cell_ids_returns_the_8_moore_neighbors():
    neighbors = neighbor_cell_ids("10_-20")

    assert len(neighbors) == 8
    assert len(set(neighbors)) == 8
    assert "10_-20" not in neighbors
    assert set(neighbors) == {
        "9_-21", "9_-20", "9_-19",
        "10_-21", "10_-19",
        "11_-21", "11_-20", "11_-19",
    }
