"""Rolling-origin backtest: how stable is the served model's reported performance across years?

`export_model.py` picks one arbitrary split (train <=2022, val 2023, test 2024) and reports a single
number per metric. [Investigating the val/test
gap](../../docs/06-modeling-and-evaluation.md#investigating-the-valtest-gap-a-monthly-breakdown)
already showed that single number isn't trustworthy on its own: `top_10pct_capture` swung from 41.9%
(val 2023) to 71.9% (test 2024) on the *same* model, purely because of which year it happened to be
evaluated against (2023's outlier September fire cluster vs. 2024's concentrated July/August peak).
This script checks whether that instability is the exception or the rule, by refitting the *same*
already-tuned hyperparameters (`BEST_RANDOM_FOREST_PARAMS`) on an expanding training window and
scoring against each subsequent year in turn, instead of trusting the one arbitrary 2023/2024 split.

Hyperparameters are deliberately NOT re-tuned per fold, that isolates "how much does the reported
number vary just by which year you happen to test on" from "how much would retuning per-fold change
things," which is a different (and much more expensive) question this script isn't trying to answer.
Re-running `RandomizedSearchCV` per fold would also multiply this script's cost by
`advanced_models.py::N_ITER` (15 candidate fits per fold instead of 1) for no benefit here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from firesight.training.baseline import DATE_COLUMN, LABEL_COLUMN

# 2012-2016 as a floor for the first fold's training window (5 years, avoids evaluating off a
# single sparse year of history), see docs/06 for why per-year fire counts vary so wildly (2017 and
# 2021 were BC's worst fire seasons on record; some years have as few as 30 fire-season positives).
HOLDOUT_YEARS = list(range(2017, 2025))


def rolling_origin_folds(df: pd.DataFrame, holdout_years: list[int]) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (year, train, holdout) for each holdout year, with an expanding training window.

    Unlike `temporal_split`'s single fixed boundary, this simulates "how would this model's reported
    performance have looked if evaluated on each subsequent year in turn", every fold's train set is
    still strictly earlier than its holdout year (no leakage), it just isn't the one arbitrary
    train/val/test cut `export_model.py` happens to use.
    """
    for year in holdout_years:
        train = df[df[DATE_COLUMN].dt.year < year]
        holdout = df[df[DATE_COLUMN].dt.year == year]
        yield year, train, holdout


def monthly_capture_breakdown(dates: pd.Series, y_true: np.ndarray, y_score: np.ndarray, k_fraction: float = 0.1) -> pd.DataFrame:
    """Break down top-k% capture by calendar month, instead of collapsing it into one number.

    Uses the exact same top-k cutoff `evaluation/metrics.py::top_k_capture` does (rank every row in
    `dates`/`y_true`/`y_score` globally, take the top `k_fraction`), this doesn't change what
    "captured" means, it just reports it per month so a single aggregate number can't hide which
    months are actually driving it. See docs/06-modeling-and-evaluation.md's "Investigating the
    val/test gap" section, which did this by hand for two years before this function generalized it
    to every backtest fold.

    Only months with at least one real fire are included (a month with zero fires has an undefined
    capture rate, not a 0% one).
    """
    n = len(y_score)
    k = max(1, int(n * k_fraction))
    top_k_idx = np.argsort(y_score)[-k:]
    caught = np.zeros(n, dtype=bool)
    caught[top_k_idx] = True

    df = pd.DataFrame(
        {
            "month": pd.DatetimeIndex(dates).month,
            "ignited": np.asarray(y_true).astype(bool),
            "caught": caught,
        }
    )
    fires = df[df["ignited"]]
    table = fires.groupby("month").agg(total=("ignited", "size"), caught=("caught", "sum")).reset_index()
    table["capture_rate"] = table["caught"] / table["total"]
    return table


@dataclass
class BacktestFold:
    """One rolling-origin fold's raw predictions, kept around for reuse beyond this script's own
    printing, e.g. `evaluation/calibration.py` pools `y_true`/`y_score` across every fold's
    `year` to fit and validate a calibrator, without re-running the (expensive) per-fold refit.
    """

    year: int
    train_rows: int
    y_true: np.ndarray
    y_score: np.ndarray
    dates: pd.Series
    scores: dict[str, float]


def run_rolling_origin_backtest(
    df: pd.DataFrame, holdout_years: list[int] = HOLDOUT_YEARS
) -> list[BacktestFold]:
    """Refit `BEST_RANDOM_FOREST_PARAMS` per fold and score each holdout year, returning the raw
    predictions rather than printing them, the reusable core this module's own `__main__` and
    `calibration.py`'s pooled-calibration check both build on.
    """
    from firesight.training.advanced_models import fit_random_forest
    from firesight.training.baseline import FEATURE_COLUMNS, score_model
    from firesight.training.export_model import BEST_RANDOM_FOREST_PARAMS

    folds = []
    for year, train, holdout in rolling_origin_folds(df, holdout_years):
        model = fit_random_forest(train, **BEST_RANDOM_FOREST_PARAMS)
        scores = score_model(model, holdout)
        y_true = holdout[LABEL_COLUMN].to_numpy()
        y_score = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
        folds.append(
            BacktestFold(
                year=year,
                train_rows=len(train),
                y_true=y_true,
                y_score=y_score,
                dates=holdout[DATE_COLUMN],
                scores=scores,
            )
        )
    return folds


if __name__ == "__main__":
    from firesight.evaluation.calibration import reliability_table
    from firesight.evaluation.metrics import brier_score
    from firesight.training.baseline import DATASET_PATH, filter_fire_season

    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)

    results = []
    monthly_tables = []
    for fold in run_rolling_origin_backtest(df, HOLDOUT_YEARS):
        year, y_true, y_score = fold.year, fold.y_true, fold.y_score
        positives = int(y_true.sum())
        print(f"\n--- holdout {year}: train={fold.train_rows:,} rows, holdout positives={positives} ---", flush=True)

        base_rate = float(y_true.mean())
        bs = brier_score(y_true, y_score)
        floor = base_rate * (1 - base_rate)
        table = reliability_table(y_true, y_score, n_bins=10)
        top_bin = table.iloc[-1]
        top_bin_observed = float(top_bin["observed_rate"])
        top_bin_ratio = float(top_bin["mean_predicted"] / top_bin_observed) if top_bin_observed > 0 else float("nan")

        print(f"holdout {year}: {fold.scores}", flush=True)
        print(f"holdout {year}: brier={bs:.6f} floor={floor:.6f} ratio={bs / floor:.1f}x", flush=True)
        print(table.to_string(index=False), flush=True)

        monthly = monthly_capture_breakdown(fold.dates, y_true, y_score)
        monthly.insert(0, "year", year)
        print(monthly.to_string(index=False), flush=True)
        jul_aug = monthly[monthly["month"].isin([7, 8])]
        jul_aug_share = float(jul_aug["total"].sum() / positives) if positives else float("nan")

        results.append(
            {
                "year": year,
                "train_rows": fold.train_rows,
                "positives": positives,
                **fold.scores,
                "brier": bs,
                "brier_floor": floor,
                "brier_ratio": bs / floor,
                "top_bin_mean_predicted": float(top_bin["mean_predicted"]),
                "top_bin_observed_rate": top_bin_observed,
                "top_bin_ratio": top_bin_ratio,
                "jul_aug_fire_share": jul_aug_share,
            }
        )
        monthly_tables.append(monthly)

    print("\n=== rolling-origin backtest summary ===", flush=True)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False), flush=True)

    print("\n=== monthly breakdown, every fold ===", flush=True)
    all_monthly = pd.concat(monthly_tables, ignore_index=True)
    print(all_monthly.to_string(index=False), flush=True)

    print("\n=== aggregate capture rate by month, across all 8 folds ===", flush=True)
    aggregate_monthly = all_monthly.groupby("month").agg(total=("total", "sum"), caught=("caught", "sum")).reset_index()
    aggregate_monthly["capture_rate"] = aggregate_monthly["caught"] / aggregate_monthly["total"]
    print(aggregate_monthly.to_string(index=False), flush=True)

    correlation = summary["jul_aug_fire_share"].corr(summary["top_10pct_capture"])
    print(f"\ncorrelation(jul_aug_fire_share, top_10pct_capture) across {len(summary)} folds: {correlation:.3f}", flush=True)
