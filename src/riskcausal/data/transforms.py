"""Stationary transforms. Only trailing operations — no centring, no back-fill, no full-sample statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def realized_vol(ret: pd.Series, window: int) -> pd.Series:
    """Annualised trailing realised volatility from daily log returns (uses returns <= t)."""
    return np.sqrt(252 * (ret ** 2).rolling(window, min_periods=window).mean())


def rolling_avg_corr(ret: pd.DataFrame, window: int) -> pd.Series:
    """Mean off-diagonal correlation of trailing `window` returns ending at t."""
    vals = []
    arr = ret.to_numpy()
    k = arr.shape[1]
    for i in range(len(arr)):
        if i + 1 < window:
            vals.append(np.nan)
            continue
        w = arr[i + 1 - window:i + 1]
        if np.isnan(w).any():
            vals.append(np.nan)
            continue
        c = np.corrcoef(w, rowvar=False)
        vals.append((c.sum() - k) / (k * (k - 1)))
    return pd.Series(vals, index=ret.index)
