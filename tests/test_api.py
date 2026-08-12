import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from firesight.training.baseline import DATE_COLUMN, LABEL_COLUMN
from firesight.training.persist import ModelBundle, save_model_bundle

FEATURES = ["t2m", "precip_mm"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # small, self-contained model + dataset so the API doesn't depend on the
    # real multi-GB pipeline output being present when tests run
    train = pd.DataFrame({"t2m": [270.0, 300.0, 271.0, 299.0], "precip_mm": [0.0, 0.0, 5.0, 1.0], "y": [0, 1, 0, 1]})
    model = LogisticRegression().fit(train[FEATURES], train["y"])
    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURES,
        metadata={"model_type": "LogisticRegression", "val_scores": {"pr_auc": 0.5}},
    )
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)

    dataset = pd.DataFrame(
        {
            "cell_id": ["1124_-1696", "1125_-1696"],
            DATE_COLUMN: pd.to_datetime(["2021-08-04", "2021-08-04"]),
            LABEL_COLUMN: [1, 0],
            "t2m": [301.0, 295.0],
            "precip_mm": [0.0, 2.0],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path)

    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("FIRESIGHT_DATASET_PATH", str(dataset_path))

    # import after env vars are set, and after reloading module-level state
    import importlib

    import api.main as main_module

    importlib.reload(main_module)

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_reports_loaded_model_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "LogisticRegression"
    assert set(body["feature_columns"]) == set(FEATURES)


def test_predict_returns_a_probability_for_valid_features(client):
    response = client.post("/predict", json={"t2m": 300.0, "precip_mm": 0.0})
    assert response.status_code == 200
    prob = response.json()["ignition_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_rejects_missing_feature(client):
    response = client.post("/predict", json={"t2m": 300.0})
    assert response.status_code == 422


def test_risk_map_returns_cells_with_coordinates_for_a_known_date(client):
    response = client.get("/risk-map", params={"date": "2021-08-04"})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 2
    cell = next(r for r in rows if r["cell_id"] == "1124_-1696")
    assert cell["actual_ignited"] == 1
    assert 0.0 <= cell["risk_probability"] <= 1.0
    assert cell["latitude"] is not None


def test_risk_map_404s_for_a_date_with_no_data(client):
    response = client.get("/risk-map", params={"date": "1999-01-01"})
    assert response.status_code == 404
