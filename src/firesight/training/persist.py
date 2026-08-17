"""Save/load a trained model together with what's needed to use it correctly.

A bare pickled model isn't enough to serve safely: predicting requires
knowing the exact feature columns, in the exact order, the model was
fit on — get that wrong and predict_proba fails loudly (wrong shape) or,
worse, silently returns nonsense (right shape, wrong column meaning).
Bundling the feature list and basic provenance alongside the model
avoids the API layer ever having to duplicate or guess that contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelBundle:
    model: Any
    feature_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Any object with a `.predict(np.ndarray) -> np.ndarray` interface (an `IsotonicRegression`
    # fit by `evaluation/calibration.py::fit_isotonic_calibrator`, in practice) that maps the raw
    # model's `predict_proba` output to a calibrated probability. `None` means this bundle was
    # exported without one — callers should treat the calibrated value as unavailable, not as 0.
    calibrator: Any | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        """Predict ignition probability for one row of features, given as a dict."""
        import pandas as pd

        row = pd.DataFrame([features])[self.feature_columns]
        return float(self.model.predict_proba(row)[0, 1])

    def predict_calibrated_proba(self, features: dict[str, float]) -> float | None:
        """Calibrated counterpart to `predict_proba` — `None` if no calibrator is attached."""
        if self.calibrator is None:
            return None
        raw = self.predict_proba(features)
        return float(self.calibrator.predict([raw])[0])


def save_model_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)
