import pandas as pd

from firesight.features.grid import assign_cell_ids


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
