import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from firesight.training.persist import ModelBundle, load_model_bundle, save_model_bundle


def test_model_bundle_predict_proba_selects_and_orders_features():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"b": rng.normal(size=50), "a": rng.normal(size=50)})
    train["label"] = (train["a"] > 0).astype(int)

    model = LogisticRegression().fit(train[["a", "b"]], train["label"])
    bundle = ModelBundle(model=model, feature_columns=["a", "b"], metadata={"note": "test"})

    # dict given in a different key order than feature_columns
    prob = bundle.predict_proba({"b": 0.0, "a": 5.0})
    assert 0.0 <= prob <= 1.0
    assert prob > 0.5  # a=5.0 strongly predicts label=1 given the training rule


def test_save_and_load_model_bundle_round_trips(tmp_path):
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    bundle = ModelBundle(model=model, feature_columns=["x"], metadata={"trained_on": "unit-test"})
    path = tmp_path / "model.joblib"

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)

    assert loaded.feature_columns == ["x"]
    assert loaded.metadata == {"trained_on": "unit-test"}
    assert loaded.predict_proba({"x": 1.0}) == bundle.predict_proba({"x": 1.0})
