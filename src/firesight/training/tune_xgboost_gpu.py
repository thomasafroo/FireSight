"""GPU-accelerated XGBoost tuning, via tree_method="hist"/device="cuda".

Same search (RANDOM_STATE, XGBOOST_DISTRIBUTIONS, N_ITER) as
advanced_models.py's "XGBoost (randomized search)" run, just on the local
NVIDIA GPU instead of CPU — kept as its own entry point rather than folded
into advanced_models.py's __main__ since it needs different hardware (fails
loudly if no CUDA-capable GPU is visible to XGBoost) and a different
RandomizedSearchCV n_jobs (1, not -1 — see tune_random_search's docstring
for why GPU candidates must run sequentially, not in parallel subprocesses).

Requires the NVIDIA driver installed to support the CUDA version this
xgboost build was compiled against (check via
`python -c "import xgboost; print(xgboost.build_info())"` -> CUDA_VERSION).
If the driver is too old, XGBoost does not error — it silently falls back to
CPU with a "No visible GPU is found" warning, so a successful run here isn't
by itself proof the GPU was actually used; check that warning is absent.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from firesight.training.advanced_models import (
    N_ITER,
    RANDOM_STATE,
    XGBOOST_DISTRIBUTIONS,
    _run_random_search_and_report,
    _scale_pos_weight,
)
from firesight.training.baseline import (
    DATASET_PATH,
    TRAIN_END,
    VAL_END,
    filter_fire_season,
    temporal_split,
)

if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    df = filter_fire_season(df)
    train, val, test = temporal_split(df, TRAIN_END, VAL_END)

    xgb_gpu_estimator = XGBClassifier(
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        scale_pos_weight=_scale_pos_weight(train),
        eval_metric="aucpr",
    )
    _run_random_search_and_report(
        "XGBoost (GPU, randomized search)",
        xgb_gpu_estimator,
        XGBOOST_DISTRIBUTIONS,
        N_ITER,
        train,
        val,
        test,
        n_jobs=1,
    )
