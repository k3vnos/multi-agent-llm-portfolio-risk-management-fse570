"""Gradient boosting comparison model. Early stopping is OFF: sklearn's early stopping holds out a *random*
subset, which would break the chronological protocol."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class GBM:
    name = "gbm"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBM:
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def summary(self, names: list[str]) -> dict:
        return {}
