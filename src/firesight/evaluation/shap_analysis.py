"""Local, per-prediction explanations, complementing the global MDI/permutation-importance work.

`training/advanced_models.py`'s feature importance (see
docs/06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on)
is entirely *global*, one ranking across the whole val/test set. It can say
"neighbor_fire_count matters most on average," not "why did this specific cell get flagged
high-risk on this specific day." SHAP gives *local*, additive, per-prediction attributions (each
feature's contribution sums exactly to that one prediction's score) instead.

Two gotchas, both handled explicitly rather than assumed, per the research note in
docs/06-modeling-and-evaluation.md#2-shap-explainability:

1. **Output shape is version-dependent.** Verified empirically against the installed `shap==0.52.0`
   (not assumed from docs/README): `TreeExplainer.shap_values` / calling the explainer both return a
   single `(n_samples, n_features, n_classes)` array, not the older "list of two arrays" shape some
   shap versions/configs use. `_positive_class_explanation` asserts this shape and fails loudly
   rather than silently misreading which slice is the positive class if a future shap upgrade changes
   it again.
2. **Raw margin vs. calibrated probability.** `TreeExplainer(..., feature_perturbation="interventional",
   model_output="probability")` gives SHAP values that sum exactly to `model.predict_proba`'s raw
   output (`ignition_probability`, the same rank-only score everything else in this project's
   evaluation is built on), verified below by reconstruction, not assumed. There is no SHAP
   decomposition of `calibrated_probability`: the attached isotonic calibrator
   (evaluation/calibration.py) is a separate post-hoc regression bolted on *after* the tree ensemble,
   not part of its structure, so nothing here can attribute a calibrated number to individual
   features. Every contribution reported by this module is in raw-`ignition_probability` units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _positive_class_explanation(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame,
) -> shap.Explanation:
    """Run TreeExplainer and slice down to the positive (ignited=1) class only.

    `background` should be a modest sample of *training* rows (a few hundred is plenty for
    `feature_perturbation="interventional"`), it defines the reference distribution SHAP values are
    computed relative to, not the rows being explained.
    """
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="probability"
    )
    explanation = explainer(X)

    if explanation.values.ndim != 3 or explanation.values.shape[-1] != 2:
        raise ValueError(
            f"Expected TreeExplainer output shaped (n_samples, n_features, 2) for this binary "
            f"classifier, got {explanation.values.shape}. shap's output shape is version-dependent "
            f"(see this module's docstring); re-verify against the installed shap version before "
            f"trusting the positive-class slice below."
        )

    return shap.Explanation(
        values=explanation.values[..., 1],
        base_values=explanation.base_values[..., 1],
        data=explanation.data,
        feature_names=list(X.columns),
    )


def explain(model, X: pd.DataFrame, background: pd.DataFrame) -> shap.Explanation:
    """Positive-class SHAP explanation for `X`, verified to reconstruct `predict_proba` exactly."""
    explanation = _positive_class_explanation(model, X, background)

    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    actual = model.predict_proba(X)[:, 1]
    if not np.allclose(reconstructed, actual, atol=1e-4):
        raise ValueError(
            "SHAP values + base value did not reconstruct predict_proba's raw output, the "
            "additivity guarantee model_output='probability' is supposed to give has broken, "
            "don't trust these attributions."
        )
    return explanation


def mean_abs_shap_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Global ranking: mean |SHAP value| per feature, the SHAP analogue of permutation importance.

    Pure aggregation over an already-computed `Explanation`, the natural cross-check against
    `training/advanced_models.py`'s MDI/permutation rankings (a third, independent method agreeing
    with the first two is stronger evidence the model's signal is real, not an artifact, see
    docs/06-modeling-and-evaluation.md#feature-importance-what-the-model-is-actually-leaning-on).
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_contributions(
    explanation: shap.Explanation,
    row_index: int,
    top_n: int = 5,
) -> list[dict]:
    """Sorted, human-readable breakdown of one row's prediction, a waterfall plot as data.

    Returns the `top_n` features with the largest |contribution| for this one row, each as
    `{"feature", "value", "contribution"}`, sorted by |contribution| descending. `contribution` is in
    raw-`ignition_probability` units (see module docstring) and signed: positive pushes the
    prediction up, negative pushes it down.
    """
    values = explanation.values[row_index]
    data = explanation.data[row_index]
    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {
            "feature": explanation.feature_names[i],
            "value": float(data[i]),
            "contribution": float(values[i]),
        }
        for i in order
    ]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    OUT_DIR = DATASET_PATH.parent  # data/processed/, gitignored build output, matching model.joblib

    bundle = load_model_bundle(MODEL_PATH)
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    train, _val, test = temporal_split(df, TRAIN_END, VAL_END)

    rng = np.random.default_rng(0)
    background = train[bundle.feature_columns].sample(n=300, random_state=0)
    # A moderate, *unbiased* random test-set sample, not the full ~242k rows: SHAP's interventional
    # mode re-evaluates the model against the background sample for every (row, feature) pair, so
    # cost scales with sample_size x len(background) x n_features -- a few thousand rows is plenty
    # to rank global importance at a small fraction of the full-test-set cost. Deliberately *not*
    # used to pick example fires below -- at this test set's ~0.1% positive rate a 3000-row random
    # draw has under 3 expected positives, too few to reliably land a real caught/missed fire.
    sample = test.sample(n=3000, random_state=0)
    explanation = explain(bundle.model, sample[bundle.feature_columns], background)

    print("=== mean |SHAP| global ranking (unbiased test sample, n=3000) ===", flush=True)
    ranking = mean_abs_shap_importance(explanation)
    print(ranking.to_string(index=False), flush=True)

    # This ranking will *not* match MDI's (neighbor_fire_count_7d dominant at 0.57) -- checked
    # directly rather than left as an unexplained discrepancy, since two "which feature matters"
    # methods disagreeing sharply is exactly the kind of thing docs/06's MDI-vs-permutation
    # cross-check exists to catch. The reason isn't a bug: neighbor_fire_count_7d is 0 for ~98.6%
    # of test rows (239,096 / 242,424), so a *population-average* |SHAP| is dominated by that
    # near-zero-contribution majority, however large its effect is on the rows where it's actually
    # nonzero. Split by that condition to show both halves of the story:
    values_by_feature = dict(zip(explanation.feature_names, explanation.values.T))
    neighbor_7d_shap = values_by_feature["neighbor_fire_count_7d"]
    neighbor_7d_nonzero = sample["neighbor_fire_count_7d"].to_numpy() > 0
    print(
        f"\nneighbor_fire_count_7d mean |SHAP| when 0 (n={(~neighbor_7d_nonzero).sum()}): "
        f"{np.abs(neighbor_7d_shap[~neighbor_7d_nonzero]).mean():.6f}",
        flush=True,
    )
    print(
        f"neighbor_fire_count_7d mean |SHAP| when >0 (n={neighbor_7d_nonzero.sum()}): "
        f"{np.abs(neighbor_7d_shap[neighbor_7d_nonzero]).mean():.6f}",
        flush=True,
    )

    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    fig.savefig(OUT_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT_DIR / 'shap_beeswarm.png'}", flush=True)

    # Real spot-check examples, from a sample built to actually contain them -- every real test-set
    # fire (only 242 rows, cheap to include in full) plus a random slice of non-fires, rather than
    # hoping an unbiased draw happens to land one (it didn't, in an earlier run of this script).
    fires = test[test[LABEL_COLUMN] == 1]
    non_fires = test[test[LABEL_COLUMN] == 0].sample(n=2000, random_state=0)
    example_pool = pd.concat([fires, non_fires])
    example_explanation = explain(bundle.model, example_pool[bundle.feature_columns], background)

    scores = bundle.model.predict_proba(example_pool[bundle.feature_columns])[:, 1]
    example_pool = example_pool.assign(_score=scores)
    # top-10% cutoff from the full test set's own score distribution (matching top_10pct_capture's
    # actual cutoff), not from this smaller pool, which is deliberately fire-heavy and would give a
    # too-easy-to-clear cutoff.
    top_10pct_cutoff = np.quantile(bundle.model.predict_proba(test[bundle.feature_columns])[:, 1], 0.9)

    examples = {
        "highest_scored_actual_fire": example_pool[example_pool[LABEL_COLUMN] == 1]["_score"].idxmax(),
        "highest_scored_non_fire": example_pool[example_pool[LABEL_COLUMN] == 0]["_score"].idxmax(),
    }
    missed = example_pool[(example_pool[LABEL_COLUMN] == 1) & (example_pool["_score"] < top_10pct_cutoff)]
    examples["highest_scored_missed_fire"] = missed["_score"].idxmax() if len(missed) else None

    pool_positions = {idx: pos for pos, idx in enumerate(example_pool.index)}
    for name, idx in examples.items():
        if idx is None:
            print(f"\n{name}: none in this pool", flush=True)
            continue
        pos = pool_positions[idx]
        row = example_pool.loc[idx]
        print(
            f"\n=== {name}: cell {row['cell_id']} on {row['date'].date()} "
            f"(ignited={row[LABEL_COLUMN]}, ignition_probability={row['_score']:.4f}) ===",
            flush=True,
        )
        for contrib in top_contributions(example_explanation, pos, top_n=6):
            print(f"  {contrib['feature']:25s} value={contrib['value']:>10.3f}  contribution={contrib['contribution']:+.4f}", flush=True)

        fig = plt.figure()
        shap.plots.waterfall(example_explanation[pos], show=False)
        out_path = OUT_DIR / f"shap_waterfall_{name}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}", flush=True)
