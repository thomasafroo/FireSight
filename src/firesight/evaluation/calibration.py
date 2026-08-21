"""Reliability check: does a predicted probability actually mean what it claims?

pr_auc/roc_auc/top_k_capture (metrics.py) are all *rank-only* — they only ask
"are positives scored higher than negatives," and are invariant to any
monotonic rescaling of the scores. A model can top those metrics while its
raw probabilities are wildly off in absolute terms, which matters here
specifically because `training/baseline.py::fit_random_forest` and
`fit_logistic_regression` both use `class_weight="balanced"`: that reweights
samples during fitting to counter the ~0.2% positive rate, which is exactly
the kind of transformation that can shift `predict_proba`'s output away from
the true empirical frequency while leaving rank order (and therefore every
metric in metrics.py) untouched. `api/main.py`'s `/predict` and
`/predict/live` both hand callers a raw `ignition_probability` float as if
it were a real probability, so this is worth checking, not assuming — see
docs/06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability
for what was actually found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def reliability_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions by predicted-probability *quantile*, not equal width.

    Equal-width bins (e.g. [0, 0.1), [0.1, 0.2), ...) would dump nearly every
    row into the lowest bin given how rare positives are here — most raw
    scores cluster near 0, so there's nothing to compare in the higher
    buckets. Quantile bins instead give each bucket a similar row count, so
    the highest-score bucket (the one `/risk-map`'s and `top_k_capture`'s
    "top 10%" actually corresponds to) still has enough rows to read a
    meaningful observed rate from.

    A well-calibrated model has `observed_rate` ~= `mean_predicted` in every
    row of the returned table. A model whose probabilities are well-*ranked*
    but not well-*calibrated* can still show `mean_predicted` badly off from
    `observed_rate` while every bucket is still correctly ordered.
    """
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df["bin"] = pd.qcut(df["y_score"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(count=("y_true", "size"), mean_predicted=("y_score", "mean"), observed_rate=("y_true", "mean"))
        .reset_index()
    )


def fit_isotonic_calibrator(y_score: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit a monotonic score -> true-frequency mapping directly on (score, outcome) pairs.

    Deliberately fit on raw pooled rows, not `CalibratedClassifierCV` — that class exists to
    wrap an estimator and manage its own internal cross-validation split, but
    `backtest.py::run_rolling_origin_backtest` already gives independently-trained, honest
    out-of-sample scores per year, so there's no CV split left for it to add. `y_min`/`y_max`
    clamp the fitted curve to a valid probability range; `out_of_bounds="clip"` extends the
    curve's endpoints rather than returning NaN for a score outside the fitted range (this
    module's raw scores can differ slightly year to year since a fresh model is refit per fold).
    """
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(y_score, y_true)


def fit_sigmoid_calibrator(y_score: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """Platt scaling: fit a 1-D logistic curve from raw score -> true frequency.

    Needs far fewer samples than `fit_isotonic_calibrator` to avoid overfitting (a straight
    sigmoid has 2 parameters vs. isotonic's one step per unique score), at the cost of assuming
    the miscalibration really is S-shaped rather than fitting whatever shape the data shows.
    """
    return LogisticRegression().fit(y_score.reshape(-1, 1), y_true)


def apply_sigmoid_calibrator(calibrator: LogisticRegression, y_score: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(y_score.reshape(-1, 1))[:, 1]


def fit_venn_abers_calibrator(y_score: np.ndarray, y_true: np.ndarray):
    """Fit the low-level `venn_abers.VennAbers` classifier-calibration class on pooled scores.

    Deliberately the *low-level* `VennAbers` class, not the package's `VennAbersCalibrator`
    wrapper: checked against the library source (not just its README) and confirmed the wrapper
    does its own random `cal_size` split internally, which would quietly reintroduce the
    random-split leakage this project's temporal-split discipline (`docs/06-modeling-and-
    evaluation.md#splitting-by-time-never-randomly`) exists to prevent. The low-level class takes
    pre-computed calibration-set probabilities/labels directly, so the caller (here, the existing
    leave-one-year-out rig) controls the split.

    `VennAbers.fit`/`.predict_proba` both want two-column `[P(class=0), P(class=1)]` arrays, not
    the 1-D positive-class score this project's other calibrators use — reshaped here so callers
    keep using the same 1-D-score convention as `fit_isotonic_calibrator`/`fit_sigmoid_calibrator`.
    """
    from venn_abers import VennAbers

    p_cal = np.column_stack([1 - y_score, y_score])
    va = VennAbers()
    va.fit(p_cal, y_true)
    return va


def apply_venn_abers_calibrator(va, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated point probability plus multiprobability interval width for `y_score`.

    Returns `(p_prime, interval_width)`: `p_prime` is the calibrated positive-class probability
    (comparable to isotonic/sigmoid's output), `interval_width` is `p1 - p0` — the size of the
    `[p0, p1]` interval Venn-Abers brackets that estimate with (arxiv.org/pdf/1511.00213.pdf
    section 4). A wide interval on a given row is Venn-Abers' own signal that the calibration-set
    support behind that prediction is thin; this is the number `docs/06`'s Venn-Abers proposal
    (#3) predicted should track the already-known sparse-year unreliability.
    """
    p_test = np.column_stack([1 - y_score, y_score])
    p_prime, p0_p1 = va.predict_proba(p_test)
    interval_width = p0_p1[:, 1] - p0_p1[:, 0]
    return p_prime[:, 1], interval_width


def leave_one_year_out_calibration_check(fold_scores: dict[int, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """For each year, calibrate on every *other* year pooled together, then score the held-out
    year — the honest test of "does a pooled calibrator generalize to a year it never saw,"
    not "does it fit the years it was trained on." `fold_scores` is `{year: (y_true, y_score)}`,
    e.g. built from `backtest.py::run_rolling_origin_backtest`'s per-fold results.

    Reports Brier score (whole-year fit) and the top-decile reliability ratio (mean_predicted /
    observed_rate — 1.0 is perfect, matching docs/06's existing per-year calibration table) both
    before and after each calibration method, so a pooled calibrator that merely shrinks the
    aggregate Brier number without fixing the top-decile — the bucket `/risk-map` and
    `top_10pct_capture` actually rely on — doesn't get mistaken for a real fix.
    """
    from firesight.evaluation.metrics import brier_score

    def top_bin_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
        table = reliability_table(y_true, y_score, n_bins=10)
        top_bin = table.iloc[-1]
        observed = float(top_bin["observed_rate"])
        return float(top_bin["mean_predicted"] / observed) if observed > 0 else float("nan")

    rows = []
    for year, (y_true, y_score) in fold_scores.items():
        other_true = np.concatenate([yt for y, (yt, _ys) in fold_scores.items() if y != year])
        other_score = np.concatenate([ys for y, (_yt, ys) in fold_scores.items() if y != year])

        iso = fit_isotonic_calibrator(other_score, other_true)
        sig = fit_sigmoid_calibrator(other_score, other_true)
        va = fit_venn_abers_calibrator(other_score, other_true)
        iso_score = iso.predict(y_score)
        sig_score = apply_sigmoid_calibrator(sig, y_score)
        va_score, va_interval_width = apply_venn_abers_calibrator(va, y_score)

        rows.append(
            {
                "year": year,
                "positives": int(y_true.sum()),
                "pooled_train_positives": int(other_true.sum()),
                "raw_brier": brier_score(y_true, y_score),
                "isotonic_brier": brier_score(y_true, iso_score),
                "sigmoid_brier": brier_score(y_true, sig_score),
                "venn_abers_brier": brier_score(y_true, va_score),
                "raw_top_bin_ratio": top_bin_ratio(y_true, y_score),
                "isotonic_top_bin_ratio": top_bin_ratio(y_true, iso_score),
                "sigmoid_top_bin_ratio": top_bin_ratio(y_true, sig_score),
                "venn_abers_top_bin_ratio": top_bin_ratio(y_true, va_score),
                "venn_abers_mean_interval_width": float(va_interval_width.mean()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from firesight.evaluation.metrics import brier_score
    from firesight.training.baseline import (
        DATASET_PATH,
        LABEL_COLUMN,
        TRAIN_END,
        VAL_END,
        filter_fire_season,
        temporal_split,
    )
    from firesight.training.export_model import MODEL_PATH
    from firesight.training.persist import load_model_bundle

    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    _, val, test = temporal_split(df, TRAIN_END, VAL_END)

    for name, split in [("val (2023)", val), ("test (2024)", test)]:
        y_true = split[LABEL_COLUMN].to_numpy()
        y_score = bundle.model.predict_proba(split[bundle.feature_columns])[:, 1]
        print(f"\n--- {name} ---", flush=True)
        print(f"brier score: {brier_score(y_true, y_score):.6f}  (base-rate-only floor: {y_true.mean() * (1 - y_true.mean()):.6f})", flush=True)
        print(f"observed positive rate: {y_true.mean():.6f}", flush=True)
        print(reliability_table(y_true, y_score).to_string(index=False), flush=True)

    # Pooled, leave-one-year-out calibration check: refits BEST_RANDOM_FOREST_PARAMS across the
    # same 8 rolling-origin folds backtest.py uses (not the single served bundle above, which is
    # scored on val/test only), so a calibrator can be fit and validated across many years rather
    # than one — see docs/06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability
    # for why a single-year fit was rejected.
    from firesight.evaluation.backtest import HOLDOUT_YEARS, run_rolling_origin_backtest

    print("\n=== pooled, leave-one-year-out calibration check ===", flush=True)
    folds = run_rolling_origin_backtest(df, HOLDOUT_YEARS)
    fold_scores = {fold.year: (fold.y_true, fold.y_score) for fold in folds}
    loyo = leave_one_year_out_calibration_check(fold_scores)
    print(loyo.to_string(index=False), flush=True)

    # The real test from docs/06's Venn-Abers proposal (#3): do intervals actually widen honestly
    # in the years already known least reliable (2019/2020/2022, from the isotonic/sigmoid columns
    # above), or is the interval claiming false confidence there too? Correlating against 1/positives
    # (not year identity) lets this hold for whichever years turn out sparse on this feature set,
    # rather than hardcoding the old weather-only model's specific sparse years.
    print("\n=== does venn_abers_mean_interval_width track known-sparse-year unreliability? ===", flush=True)
    sparse_years = {2019, 2020, 2022}
    loyo["known_sparse_year"] = loyo["year"].isin(sparse_years)
    print(
        loyo[["year", "positives", "known_sparse_year", "venn_abers_mean_interval_width", "venn_abers_top_bin_ratio"]]
        .to_string(index=False),
        flush=True,
    )
    inverse_positives_corr = loyo["venn_abers_mean_interval_width"].corr(1 / loyo["positives"])
    sparse_mean_width = loyo.loc[loyo["known_sparse_year"], "venn_abers_mean_interval_width"].mean()
    dense_mean_width = loyo.loc[~loyo["known_sparse_year"], "venn_abers_mean_interval_width"].mean()
    print(f"corr(mean_interval_width, 1/positives) across {len(loyo)} folds: {inverse_positives_corr:.3f}", flush=True)
    print(f"mean interval width, known-sparse years: {sparse_mean_width:.6f}", flush=True)
    print(f"mean interval width, other years:        {dense_mean_width:.6f}", flush=True)
    print(
        "-> "
        + (
            "widens honestly on sparse years: real signal, worth shipping (see docs/06 #3)."
            if sparse_mean_width > dense_mean_width
            else "does NOT widen on sparse years: negative result per docs/06 #3's own criterion, don't ship."
        ),
        flush=True,
    )

    print("\n=== reference: one calibrator fit on all 8 years pooled together ===", flush=True)
    print("(printed only, not persisted or wired into the served model)", flush=True)
    all_true = np.concatenate([yt for yt, _ys in fold_scores.values()])
    all_score = np.concatenate([ys for _yt, ys in fold_scores.values()])
    pooled_iso = fit_isotonic_calibrator(all_score, all_true)
    print(reliability_table(all_true, pooled_iso.predict(all_score)).to_string(index=False), flush=True)
