"""Simple baselines (M2): VIX level, Chicago Fed NFCI (point-in-time), rolling-correlation VaR.

Each returns a daily risk score known at the close of t. Thresholds are learned on validation data only, with
the same rule as the models. DCC-GARCH is added in M4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def vix_score(controls_pit: pd.DataFrame) -> pd.Series:
    return controls_pit["VIXCLS"].rename("vix")


def nfci_score(controls_pit: pd.DataFrame) -> pd.Series:
    return controls_pit["NFCI"].rename("nfci")


def corr_var_score(returns: pd.DataFrame, window: int = 63, alpha: float = 0.99) -> pd.Series:
    """1-day parametric VaR of an equal-weight portfolio from the trailing `window`-day covariance matrix."""
    w = np.full(returns.shape[1], 1 / returns.shape[1])
    arr = returns.to_numpy()
    z = norm.ppf(alpha)
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        X = arr[i - window + 1:i + 1]
        if np.isnan(X).any():
            continue
        out[i] = z * np.sqrt(w @ np.cov(X, rowvar=False) @ w)
    return pd.Series(out, index=returns.index, name="corr_var")
