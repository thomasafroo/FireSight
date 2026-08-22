import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from firesight.features.grid import build_grid_cells
from firesight.pipeline.ingest_firms import BC_KAMLOOPS_BBOX
from firesight.training.baseline import DATE_COLUMN, LABEL_COLUMN
from firesight.training.persist import ModelBundle, save_model_bundle

FEATURES = ["t2m", "precip_mm"]
A_REAL_CELL_ID = build_grid_cells(BC_KAMLOOPS_BBOX).iloc[0]["cell_id"]


def _build_client(tmp_path, monkeypatch, calibrator=None):
    train = pd.DataFrame({"t2m": [270.0, 300.0, 271.0, 299.0], "precip_mm": [0.0, 0.0, 5.0, 1.0], "y": [0, 1, 0, 1]})
    model = LogisticRegression().fit(train[FEATURES], train["y"])
    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURES,
        calibrator=calibrator,
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
    # A path guaranteed not to exist, so these tests stay isolated from whatever the real
    # data/processed/model_3day.joblib on disk happens to contain (or whether it exists at all).
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(tmp_path / "no_multi_day_model_here.joblib"))
    monkeypatch.setenv("FIRESIGHT_DATASET_PATH", str(dataset_path))

    # import after env vars are set, and after reloading module-level state
    import importlib

    import api.main as main_module

    importlib.reload(main_module)

    return TestClient(main_module.app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # small, self-contained model + dataset so the API doesn't depend on the
    # real multi-GB pipeline output being present when tests run
    with _build_client(tmp_path, monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def calibrated_client(tmp_path, monkeypatch):
    # Same as `client`, but the bundle carries a real IsotonicRegression calibrator, so the
    # `calibrated_probability`/`calibrated_risk_probability` response fields have a case where
    # they're actually populated, not just `None` — the `client` fixture above already covers the
    # no-calibrator/`None` branch.
    raw_scores = np.linspace(0.0, 1.0, 50)
    true_rate = raw_scores / 10
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(0, 1, size=50) < true_rate).astype(int)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_scores, y_true)

    with _build_client(tmp_path, monkeypatch, calibrator=calibrator) as test_client:
        yield test_client


def test_health_reports_loaded_model_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "LogisticRegression"
    assert set(body["feature_columns"]) == set(FEATURES)
    assert body["calibrated"] is False


def test_health_reports_calibrated_true_when_a_calibrator_is_attached(calibrated_client):
    response = calibrated_client.get("/health")
    assert response.json()["calibrated"] is True


def test_predict_returns_a_probability_for_valid_features(client):
    response = client.post("/predict", json={"t2m": 300.0, "precip_mm": 0.0})
    assert response.status_code == 200
    body = response.json()
    prob = body["ignition_probability"]
    assert 0.0 <= prob <= 1.0
    assert body["calibrated_probability"] is None


def test_predict_returns_a_calibrated_probability_when_a_calibrator_is_attached(calibrated_client):
    response = calibrated_client.post("/predict", json={"t2m": 300.0, "precip_mm": 0.0})
    assert response.status_code == 200
    body = response.json()
    assert body["calibrated_probability"] is not None
    assert 0.0 <= body["calibrated_probability"] <= 1.0


def test_predict_rejects_missing_feature(client):
    response = client.post("/predict", json={"t2m": 300.0})
    assert response.status_code == 422


def _patch_live_fetches(monkeypatch, main_module, neighbor_counts: dict | None = None) -> None:
    """Both /predict/live data sources are real network calls (Open-Meteo, FIRMS NRT) — every test
    that reaches the try block below the cell/date validation must stub both, or it'll try to hit
    the live internet during `pytest`."""
    monkeypatch.setattr(
        main_module,
        "build_live_feature_row",
        lambda latitude, longitude, target_date, cell_id: {"t2m": 300.0, "precip_mm": 0.0},
    )
    monkeypatch.setattr(
        main_module,
        "build_live_neighbor_fire_features",
        lambda cell_id, target_date, bbox: neighbor_counts
        or {"neighbor_fire_count_1d": 0.0, "neighbor_fire_count_3d": 0.0, "neighbor_fire_count_7d": 0.0},
    )


def test_predict_live_returns_a_probability_for_a_valid_cell(client, monkeypatch):
    import api.main as main_module

    _patch_live_fetches(monkeypatch, main_module)
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["cell_id"] == A_REAL_CELL_ID
    assert body["date"] == "2024-07-15"
    assert 0.0 <= body["ignition_probability"] <= 1.0
    assert body["calibrated_probability"] is None
    assert body["fire_detection_source"] == "NASA FIRMS NRT (VIIRS_NOAA20_NRT)"


def test_predict_live_returns_a_calibrated_probability_when_a_calibrator_is_attached(calibrated_client, monkeypatch):
    import api.main as main_module

    _patch_live_fetches(monkeypatch, main_module)
    response = calibrated_client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["calibrated_probability"] is not None
    assert 0.0 <= body["calibrated_probability"] <= 1.0


def test_predict_live_merges_neighbor_fire_counts_into_the_scored_feature_row(tmp_path, monkeypatch):
    """A dedicated bundle whose feature list actually includes neighbor_fire_count_1d, so a bug that
    dropped the merged-in fire-detection features (as opposed to just weather) would fail loudly
    instead of passing silently the way the FEATURES=["t2m", "precip_mm"] fixture above would let it."""
    train = pd.DataFrame(
        {
            "t2m": [270.0, 300.0, 271.0, 299.0],
            "neighbor_fire_count_1d": [0.0, 4.0, 0.0, 3.0],
            "y": [0, 1, 0, 1],
        }
    )
    features = ["t2m", "neighbor_fire_count_1d"]
    model = LogisticRegression().fit(train[features], train["y"])
    bundle = ModelBundle(model=model, feature_columns=features, metadata={"model_type": "LogisticRegression"})
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)
    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    # A path guaranteed not to exist, so these tests stay isolated from whatever the real
    # data/processed/model_3day.joblib on disk happens to contain (or whether it exists at all).
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(tmp_path / "no_multi_day_model_here.joblib"))
    monkeypatch.delenv("FIRESIGHT_DATASET_PATH", raising=False)

    import importlib

    import api.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        monkeypatch.setattr(main_module, "build_live_feature_row", lambda latitude, longitude, target_date, cell_id: {"t2m": 300.0})
        monkeypatch.setattr(
            main_module,
            "build_live_neighbor_fire_features",
            lambda cell_id, target_date, bbox: {"neighbor_fire_count_1d": 4.0},
        )
        response = test_client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 200
    assert 0.0 <= response.json()["ignition_probability"] <= 1.0


def test_predict_live_multi_day_503s_when_no_multi_day_model_is_loaded(client):
    """The `client` fixture's env points FIRESIGHT_MULTI_DAY_MODEL_PATH at a path guaranteed not to
    exist (see _build_client) -- this is the default, not-yet-exported-on-this-deployment case."""
    response = client.get("/predict/live/multi-day", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 503


@pytest.fixture
def multi_day_client(tmp_path, monkeypatch):
    """Same shape as `client`, but with a second, real multi-day bundle also loaded."""
    train = pd.DataFrame({"t2m": [270.0, 300.0, 271.0, 299.0], "precip_mm": [0.0, 0.0, 5.0, 1.0], "y": [0, 1, 0, 1]})
    model = LogisticRegression().fit(train[FEATURES], train["y"])
    bundle = ModelBundle(model=model, feature_columns=FEATURES, metadata={"model_type": "LogisticRegression"})
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)

    multi_day_model = LogisticRegression().fit(train[FEATURES], train["y"])
    multi_day_bundle = ModelBundle(
        model=multi_day_model,
        feature_columns=FEATURES,
        metadata={"model_type": "LogisticRegression", "val_scores": {"pr_auc": 0.3}},
    )
    multi_day_model_path = tmp_path / "model_3day.joblib"
    save_model_bundle(multi_day_bundle, multi_day_model_path)

    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(multi_day_model_path))
    monkeypatch.delenv("FIRESIGHT_DATASET_PATH", raising=False)

    import importlib

    import api.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_reports_multi_day_model_loaded(multi_day_client):
    response = multi_day_client.get("/health")
    body = response.json()
    assert body["multi_day_model_loaded"] is True
    assert body["multi_day_val_scores"] == {"pr_auc": 0.3}


def test_predict_live_multi_day_returns_a_probability_for_a_valid_cell(multi_day_client, monkeypatch):
    import api.main as main_module

    _patch_live_fetches(monkeypatch, main_module)
    response = multi_day_client.get("/predict/live/multi-day", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 200
    body = response.json()
    assert body["cell_id"] == A_REAL_CELL_ID
    assert body["date"] == "2024-07-15"
    assert body["window_days"] == 3
    assert 0.0 <= body["ignition_probability"] <= 1.0
    assert "calibrated_probability" not in body


def test_predict_live_multi_day_404s_for_an_unknown_cell(multi_day_client):
    response = multi_day_client.get("/predict/live/multi-day", params={"cell_id": "not-a-real-cell", "date": "2024-07-15"})
    assert response.status_code == 404


def test_predict_live_404s_for_an_unknown_cell(client):
    response = client.get("/predict/live", params={"cell_id": "not-a-real-cell", "date": "2024-07-15"})
    assert response.status_code == 404


def test_predict_live_rejects_a_future_date(client):
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2099-07-15"})
    assert response.status_code == 400
    assert "future" in response.json()["detail"]


def test_predict_live_rejects_a_date_outside_the_fire_season(client):
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-12-15"})
    assert response.status_code == 400
    assert "fire season" in response.json()["detail"]


def test_predict_live_surfaces_insufficient_weather_history_as_a_422(client, monkeypatch):
    import api.main as main_module

    def _raise(*args, **kwargs):
        raise ValueError("no recorded rain in the lookback window")

    monkeypatch.setattr(main_module, "build_live_feature_row", _raise)
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 422


def test_predict_live_surfaces_a_weather_fetch_failure_as_a_502(client, monkeypatch):
    import api.main as main_module

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network unreachable")

    monkeypatch.setattr(main_module, "build_live_feature_row", _raise)
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 502


def test_predict_live_surfaces_a_fire_detection_fetch_failure_as_a_502(client, monkeypatch):
    """The FIRMS NRT fetch is the second live data source in the same try block as weather — needs
    its own failure-path test, not just weather's, since a bug could catch one and miss the other."""
    import api.main as main_module

    monkeypatch.setattr(
        main_module,
        "build_live_feature_row",
        lambda latitude, longitude, target_date, cell_id: {"t2m": 300.0, "precip_mm": 0.0},
    )

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network unreachable")

    monkeypatch.setattr(main_module, "build_live_neighbor_fire_features", _raise)
    response = client.get("/predict/live", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 502


def _build_explain_client(tmp_path, monkeypatch):
    """/predict/explain needs a tree-based model (`shap.TreeExplainer`) — the `client` fixture's
    LogisticRegression bundle above isn't usable for it, so this is a dedicated fixture rather
    than reusing `_build_client`."""
    train = pd.DataFrame(
        {
            "t2m": [270.0, 300.0, 271.0, 299.0, 305.0, 268.0],
            "precip_mm": [0.0, 0.0, 5.0, 1.0, 0.0, 8.0],
            "y": [0, 1, 0, 1, 1, 0],
        }
    )
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(train[FEATURES], train["y"])
    bundle = ModelBundle(model=model, feature_columns=FEATURES, metadata={"model_type": "RandomForestClassifier"})
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)

    dataset = pd.DataFrame(
        {
            "cell_id": ["1124_-1696"] * 6,
            DATE_COLUMN: pd.to_datetime(
                ["2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", "2021-08-06"]
            ),
            LABEL_COLUMN: train["y"],
            "t2m": train["t2m"],
            "precip_mm": train["precip_mm"],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path)

    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    # A path guaranteed not to exist, so these tests stay isolated from whatever the real
    # data/processed/model_3day.joblib on disk happens to contain (or whether it exists at all).
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(tmp_path / "no_multi_day_model_here.joblib"))
    monkeypatch.setenv("FIRESIGHT_DATASET_PATH", str(dataset_path))

    import importlib

    import api.main as main_module

    importlib.reload(main_module)

    return TestClient(main_module.app)


@pytest.fixture
def explain_client(tmp_path, monkeypatch):
    with _build_explain_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def test_predict_explain_returns_ranked_top_contributions_for_a_valid_cell(explain_client, monkeypatch):
    import api.main as main_module

    _patch_live_fetches(monkeypatch, main_module)
    response = explain_client.get(
        "/predict/explain", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15", "top_n": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cell_id"] == A_REAL_CELL_ID
    assert body["date"] == "2024-07-15"
    assert 0.0 <= body["ignition_probability"] <= 1.0
    assert len(body["top_contributions"]) == 2

    contributions = body["top_contributions"]
    abs_values = [abs(c["contribution"]) for c in contributions]
    assert abs_values == sorted(abs_values, reverse=True)
    for contrib in contributions:
        assert set(contrib.keys()) == {"feature", "value", "contribution"}
        assert contrib["feature"] in FEATURES


def test_predict_explain_503s_when_no_dataset_is_available_for_a_background_sample(tmp_path, monkeypatch):
    train = pd.DataFrame({"t2m": [270.0, 300.0], "precip_mm": [0.0, 1.0], "y": [0, 1]})
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(train[FEATURES], train["y"])
    bundle = ModelBundle(model=model, feature_columns=FEATURES, metadata={})
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)

    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    # A path guaranteed not to exist, so these tests stay isolated from whatever the real
    # data/processed/model_3day.joblib on disk happens to contain (or whether it exists at all).
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(tmp_path / "no_multi_day_model_here.joblib"))
    # A path guaranteed not to exist, not delenv -- delenv would fall back to the module's real
    # default (data/processed/kamloops_dataset.parquet), which does exist in this checkout.
    monkeypatch.setenv("FIRESIGHT_DATASET_PATH", str(tmp_path / "no_dataset_here.parquet"))

    import importlib

    import api.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        response = test_client.get("/predict/explain", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 503


def test_predict_explain_404s_for_an_unknown_cell(explain_client):
    response = explain_client.get("/predict/explain", params={"cell_id": "not-a-real-cell", "date": "2024-07-15"})
    assert response.status_code == 404


def test_predict_explain_rejects_a_future_date(explain_client):
    response = explain_client.get("/predict/explain", params={"cell_id": A_REAL_CELL_ID, "date": "2099-07-15"})
    assert response.status_code == 400


def test_predict_explain_surfaces_a_live_fetch_failure_as_a_502(explain_client, monkeypatch):
    import api.main as main_module

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network unreachable")

    monkeypatch.setattr(main_module, "build_live_feature_row", _raise)
    response = explain_client.get("/predict/explain", params={"cell_id": A_REAL_CELL_ID, "date": "2024-07-15"})
    assert response.status_code == 502


def test_risk_map_returns_cells_with_coordinates_for_a_known_date(client):
    response = client.get("/risk-map", params={"date": "2021-08-04"})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 2
    cell = next(r for r in rows if r["cell_id"] == "1124_-1696")
    assert cell["actual_ignited"] == 1
    assert 0.0 <= cell["risk_probability"] <= 1.0
    assert cell["calibrated_risk_probability"] is None
    assert cell["latitude"] is not None


def test_risk_map_returns_calibrated_probabilities_when_a_calibrator_is_attached(calibrated_client):
    response = calibrated_client.get("/risk-map", params={"date": "2021-08-04"})
    assert response.status_code == 200
    rows = response.json()
    for row in rows:
        assert row["calibrated_risk_probability"] is not None
        assert 0.0 <= row["calibrated_risk_probability"] <= 1.0


def test_risk_map_404s_for_a_date_with_no_data(client):
    response = client.get("/risk-map", params={"date": "1999-07-01"})
    assert response.status_code == 404


def test_risk_map_rejects_a_date_outside_the_fire_season(client):
    response = client.get("/risk-map", params={"date": "2021-12-15"})
    assert response.status_code == 400
    assert "fire season" in response.json()["detail"]


def test_cors_defaults_to_wide_open(client):
    response = client.get("/health", headers={"Origin": "https://example.com"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_honors_explicit_allowlist(tmp_path, monkeypatch):
    train = pd.DataFrame({"t2m": [270.0, 300.0, 271.0, 299.0], "precip_mm": [0.0, 0.0, 5.0, 1.0], "y": [0, 1, 0, 1]})
    model = LogisticRegression().fit(train[FEATURES], train["y"])
    bundle = ModelBundle(model=model, feature_columns=FEATURES, metadata={})
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)

    monkeypatch.setenv("FIRESIGHT_MODEL_PATH", str(model_path))
    # A path guaranteed not to exist, so these tests stay isolated from whatever the real
    # data/processed/model_3day.joblib on disk happens to contain (or whether it exists at all).
    monkeypatch.setenv("FIRESIGHT_MULTI_DAY_MODEL_PATH", str(tmp_path / "no_multi_day_model_here.joblib"))
    monkeypatch.delenv("FIRESIGHT_DATASET_PATH", raising=False)
    monkeypatch.setenv("FIRESIGHT_CORS_ORIGINS", "https://firesight.example.com")

    import importlib

    import api.main as main_module

    importlib.reload(main_module)

    with TestClient(main_module.app) as restricted_client:
        allowed = restricted_client.get("/health", headers={"Origin": "https://firesight.example.com"})
        assert allowed.headers["access-control-allow-origin"] == "https://firesight.example.com"

        blocked = restricted_client.get("/health", headers={"Origin": "https://not-allowed.example.com"})
        assert "access-control-allow-origin" not in blocked.headers
