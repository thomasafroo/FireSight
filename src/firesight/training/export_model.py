"""Fit the currently-best model on the real dataset and persist it for serving.

Deliberately a separate, explicit step from `baseline.py`/`advanced_models.py`
(which explore/compare) rather than auto-saving from either — promoting a
model to "the one the API serves" should be a decision made after looking
at the comparison, not a side effect of running a training script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from firesight.training.advanced_models import fit_xgboost
from firesight.training.baseline import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TRAIN_END,
    VAL_END,
    score_model,
    temporal_split,
)
from firesight.training.persist import ModelBundle, save_model_bundle

MODEL_PATH = Path("data/processed/model.joblib")

# Winner of advanced_models.py's grid search by val (2023) PR-AUC — see
# docs/06-modeling-and-evaluation.md#randomforest-and-xgboost-trainingadvanced_modelspy
# for the full comparison, including the (close, worth revisiting) call
# against RandomForest, which had better test-set ROC-AUC/top-10% capture
# despite XGBoost's better PR-AUC on both val and test.
BEST_XGBOOST_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}


def export_current_best(dataset_path: Path = DATASET_PATH, out_path: Path = MODEL_PATH) -> ModelBundle:
    """Fit the tuned XGBoost model on train and save it.

    This is today's best *validated* model (see
    docs/06-modeling-and-evaluation.md) — beats both the Dummy floor and
    the LogisticRegression baseline by a wide margin on every val metric.
    Nothing else in the API needs to change if this is swapped for a
    different winner later, since it only depends on the ModelBundle
    contract (predict_proba(dict) -> float), not on the model class.
    """
    df = pd.read_parquet(dataset_path)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    model = fit_xgboost(train, **BEST_XGBOOST_PARAMS)
    val_scores = score_model(model, val)
    test_scores = score_model(model, test)

    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        metadata={
            "model_type": "XGBoost",
            "params": BEST_XGBOOST_PARAMS,
            "trained_through": TRAIN_END,
            "validated_on": f"{TRAIN_END} to {VAL_END}",
            "val_scores": val_scores,
            "test_scores": test_scores,
        },
    )
    save_model_bundle(bundle, out_path)
    return bundle


if __name__ == "__main__":
    bundle = export_current_best()
    print(f"Saved model bundle -> {MODEL_PATH}", flush=True)
    print(f"metadata: {bundle.metadata}", flush=True)
