import pandas as pd

from firesight.features.live_fuel_type import (
    build_live_fuel_type_features,
    load_fuel_type_lookup,
)


def test_load_fuel_type_lookup_maps_cell_id_to_code(tmp_path):
    cache_path = tmp_path / "fuel_type.parquet"
    pd.DataFrame({"cell_id": ["0_0", "0_1"], "fuel_type_cd": ["C-3", "O-1a"]}).to_parquet(cache_path, index=False)

    lookup = load_fuel_type_lookup(cache_path)

    assert lookup == {"0_0": "C-3", "0_1": "O-1a"}


def test_build_live_fuel_type_features_one_hots_the_matching_column():
    columns = ["fuel_type_C-3", "fuel_type_O-1a", "fuel_type_S-2"]
    lookup = {"0_0": "O-1a"}

    result = build_live_fuel_type_features("0_0", columns, lookup)

    assert result == {"fuel_type_C-3": 0.0, "fuel_type_O-1a": 1.0, "fuel_type_S-2": 0.0}


def test_build_live_fuel_type_features_returns_all_zero_for_an_unmapped_cell():
    columns = ["fuel_type_C-3", "fuel_type_O-1a"]

    result = build_live_fuel_type_features("9_9", columns, lookup={})

    assert result == {"fuel_type_C-3": 0.0, "fuel_type_O-1a": 0.0}


def test_build_live_fuel_type_features_returns_all_zero_when_code_has_no_training_column():
    """A code the cache has but that never occurred in the training extract (so it never became a
    fuel_type_<code> column) -- same "unseen class is all-zero" behavior as an unmapped cell."""
    columns = ["fuel_type_C-3", "fuel_type_O-1a"]
    lookup = {"0_0": "W"}  # water -- never occurs in the cached Kamloops FC extract

    result = build_live_fuel_type_features("0_0", columns, lookup)

    assert result == {"fuel_type_C-3": 0.0, "fuel_type_O-1a": 0.0}
