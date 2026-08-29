import numpy as np

from firesight.evaluation.calibration import (
    apply_venn_abers_calibrator,
    fit_isotonic_calibrator,
    fit_venn_abers_calibrator,
    leave_one_year_out_calibration_check,
    reliability_table,
)
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


def test_fit_isotonic_calibrator_recovers_a_known_overconfident_scaling():
    rng = np.random.default_rng(0)
    n = 5000
    y_score = rng.uniform(0, 1, size=n)
    true_rate = y_score / 10  # same overconfident-by-10x pattern as the reliability_table test above
    y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)

    calibrator = fit_isotonic_calibrator(y_score, y_true)
    calibrated = calibrator.predict(y_score)

    raw_error = np.abs(y_score - true_rate).mean()
    calibrated_error = np.abs(calibrated - true_rate).mean()
    assert calibrated_error < raw_error / 3


def test_leave_one_year_out_calibration_check_improves_brier_when_miscalibration_is_consistent_across_years():
    rng = np.random.default_rng(1)
    fold_scores = {}
    for year in range(2017, 2022):
        n = 3000
        y_score = rng.uniform(0, 1, size=n)
        true_rate = y_score / 10  # same overconfidence factor every year
        y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)
        fold_scores[year] = (y_true, y_score)

    result = leave_one_year_out_calibration_check(fold_scores)

    assert set(result["year"]) == set(fold_scores.keys())
    assert (result["isotonic_brier"] < result["raw_brier"]).all()


def test_fit_venn_abers_calibrator_recovers_a_known_overconfident_scaling():
    rng = np.random.default_rng(0)
    n = 5000
    y_score = rng.uniform(0, 1, size=n)
    true_rate = y_score / 10  # same overconfident-by-10x pattern as the isotonic test above
    y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)

    va = fit_venn_abers_calibrator(y_score, y_true)
    p_prime, _interval_width = apply_venn_abers_calibrator(va, y_score)

    raw_error = np.abs(y_score - true_rate).mean()
    calibrated_error = np.abs(p_prime - true_rate).mean()
    assert calibrated_error < raw_error / 3


def test_apply_venn_abers_calibrator_widens_the_interval_with_a_smaller_calibration_set():
    """The multiprobability interval [p0, p1] is Venn-Abers' own signal for thin calibration-set
    support behind a prediction, docs/06's Venn-Abers proposal (#3) hinges on this actually being
    true, checked here directly rather than assumed from the paper."""
    rng = np.random.default_rng(3)
    y_score_large_cal = rng.uniform(0, 1, size=4000)
    y_true_large_cal = (rng.uniform(0, 1, size=4000) < y_score_large_cal).astype(int)
    y_score_small_cal = rng.uniform(0, 1, size=40)
    y_true_small_cal = (rng.uniform(0, 1, size=40) < y_score_small_cal).astype(int)

    y_test = rng.uniform(0, 1, size=500)

    va_large = fit_venn_abers_calibrator(y_score_large_cal, y_true_large_cal)
    va_small = fit_venn_abers_calibrator(y_score_small_cal, y_true_small_cal)
    _p_prime_large, width_large = apply_venn_abers_calibrator(va_large, y_test)
    _p_prime_small, width_small = apply_venn_abers_calibrator(va_small, y_test)

    assert width_small.mean() > width_large.mean()


def test_leave_one_year_out_calibration_check_includes_venn_abers_columns():
    rng = np.random.default_rng(1)
    fold_scores = {}
    for year in range(2017, 2022):
        n = 1000
        y_score = rng.uniform(0, 1, size=n)
        true_rate = y_score / 10
        y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)
        fold_scores[year] = (y_true, y_score)

    result = leave_one_year_out_calibration_check(fold_scores)

    assert "venn_abers_brier" in result.columns
    assert "venn_abers_top_bin_ratio" in result.columns
    assert "venn_abers_mean_interval_width" in result.columns
    assert (result["venn_abers_mean_interval_width"] > 0).all()


def test_leave_one_year_out_calibration_check_can_make_an_already_calibrated_year_worse():
    rng = np.random.default_rng(2)
    n = 4000
    # 2017-2019 are all overconfident by the same 50x factor; 2020's raw score is already the
    # true rate. Pooling "the other years" to calibrate 2020 pools in three years whose own
    # correction doesn't apply to it -- calibration isn't a free lunch just because it helped
    # everywhere else.
    factors = {2017: 50, 2018: 50, 2019: 50, 2020: 1}
    fold_scores = {}
    for year, factor in factors.items():
        y_score = rng.uniform(0, 1, size=n)
        true_rate = np.clip(y_score / factor, 0, 1)
        y_true = (rng.uniform(0, 1, size=n) < true_rate).astype(int)
        fold_scores[year] = (y_true, y_score)

    result = leave_one_year_out_calibration_check(fold_scores).set_index("year")

    assert result.loc[2020, "isotonic_brier"] > result.loc[2020, "raw_brier"]
