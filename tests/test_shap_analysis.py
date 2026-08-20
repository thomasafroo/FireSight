import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from firesight.evaluation.shap_analysis import (
    explain,
    mean_abs_shap_importance,
    top_contributions,
)


def _tiny_fitted_model_and_data() -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, size=200),
            "b": rng.uniform(0, 1, size=200),
            "noise": rng.uniform(0, 1, size=200),
        }
    )
    y = (X["a"] + rng.normal(0, 0.1, size=200) > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
    background = X.iloc[:50]
    sample = X.iloc[50:70].reset_index(drop=True)
    return model, background, sample


def test_explain_reconstructs_predict_proba_exactly():
    model, background, sample = _tiny_fitted_model_and_data()
    explanation = explain(model, sample, background)

    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    actual = model.predict_proba(sample)[:, 1]
    assert np.allclose(reconstructed, actual, atol=1e-4)


def test_mean_abs_shap_importance_ranks_the_real_signal_feature_first():
    # "a" is what y was actually generated from; "noise" carries no signal at all.
    model, background, sample = _tiny_fitted_model_and_data()
    explanation = explain(model, sample, background)

    ranking = mean_abs_shap_importance(explanation)

    assert next(iter(ranking["feature"])) == "a"
    assert ranking.set_index("feature").loc["a", "mean_abs_shap"] > ranking.set_index("feature").loc["noise", "mean_abs_shap"]


def test_top_contributions_are_sorted_by_absolute_magnitude_and_match_explanation_row():
    model, background, sample = _tiny_fitted_model_and_data()
    explanation = explain(model, sample, background)

    contributions = top_contributions(explanation, row_index=0, top_n=2)

    assert len(contributions) == 2
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # every reported contribution must actually come from that row's real SHAP values, not some
    # other row's -- check against the raw explanation directly.
    for c in contributions:
        feature_idx = explanation.feature_names.index(c["feature"])
        assert np.isclose(c["contribution"], explanation.values[0, feature_idx])


def test_top_contributions_respects_top_n():
    model, background, sample = _tiny_fitted_model_and_data()
    explanation = explain(model, sample, background)

    assert len(top_contributions(explanation, row_index=0, top_n=1)) == 1
    assert len(top_contributions(explanation, row_index=0, top_n=3)) == 3
