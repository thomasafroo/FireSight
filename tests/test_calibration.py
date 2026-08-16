import numpy as np

from firesight.evaluation.calibration import reliability_table
from firesight.evaluation.metrics import brier_score


def test_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_score) == 0.0


def test_brier_score_penalizes_confident_wrong_predictions_more_than_unsure_ones():
    y_true = np.array([0, 1])
    confident_wrong = brier_score(y_true, np.array([1.0, 0.0]))
    unsure = brier_score(y_true, np.array([0.5, 0.5]))
    assert confident_wrong > unsure


def test_reliability_table_shows_a_perfectly_calibrated_model_as_matching_columns():
    rng = np.random.default_rng(0)
    y_score = rng.uniform(0, 1, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < y_score).astype(int)

    table = reliability_table(y_true, y_score, n_bins=5)

    assert len(table) == 5
    assert table["count"].sum() == 2000
    for mean_predicted, observed_rate in zip(table["mean_predicted"], table["observed_rate"]):
        assert abs(mean_predicted - observed_rate) < 0.1


def test_reliability_table_flags_a_systematically_overconfident_model():
    rng = np.random.default_rng(0)
    true_rate = rng.uniform(0, 0.05, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < true_rate).astype(int)
    y_score = true_rate * 10  # overconfident by 10x, but same rank order

    table = reliability_table(y_true, y_score, n_bins=5)

    top_bin = table.iloc[-1]
    assert top_bin["mean_predicted"] > top_bin["observed_rate"] * 3
