"""MAP logistic regression: the posterior mode of the Bayesian logit's Normal(0,1)-prior model (L2, C=1 on
standardised features). Used only as a fast stand-in for the posterior in the M4 validation grid."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MapLogit:
    name = "map_logit"

    def __init__(self, seed: int = 0):
        self.model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MapLogit:
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def summary(self, names: list[str]) -> dict:
        return {"coef": dict(zip(names, map(float, self.model.coef_[0])))}
