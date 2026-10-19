"""Rolling VarLiNGAM (Hyvärinen et al., 2010): estimated *directed dependence*, not proven causality.

For each window ending at row i (rows i-W+1..i, i.e. data <= window end only) the data are standardised
within the window and VARLiNGAM is fit. Output arrays are padded to max_lag+1 so windows are comparable:
B[w, 0] is the contemporaneous matrix B0, B[w, l] the lag-l matrix; entry [i, j] = effect of node j on node i.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import lingam
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RollingGraphs:
    ends: pd.DatetimeIndex          # last observation date in each window
    nodes: list[str]
    B: np.ndarray                   # (n_windows, max_lag+1, k, k)
    lags: np.ndarray                # chosen lag order per window
    order: np.ndarray               # causal order per window (n_windows, k)

    def save(self, path) -> None:
        np.savez_compressed(path, ends=self.ends.values.astype("datetime64[ns]"), nodes=np.array(self.nodes),
                            B=self.B, lags=self.lags, order=self.order)

    @classmethod
    def load(cls, path) -> RollingGraphs:
        z = np.load(path, allow_pickle=False)
        return cls(pd.DatetimeIndex(z["ends"]), list(z["nodes"]), z["B"], z["lags"], z["order"])


def window_ends(n: int, window: int, step: int) -> list[int]:
    return list(range(window - 1, n, step))


def fit_window(X: np.ndarray, max_lag: int, criterion: str, prune: bool, seed: int) -> tuple[np.ndarray, int, np.ndarray]:
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    m = lingam.VARLiNGAM(lags=max_lag, criterion=None if criterion == "fixed" else criterion, prune=prune,
                         random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Z)
    A = m.adjacency_matrices_
    k = X.shape[1]
    out = np.zeros((max_lag + 1, k, k))
    out[: A.shape[0]] = A
    return out, A.shape[0] - 1, np.asarray(m.causal_order_)


def rolling_varlingam(panel: pd.DataFrame, window: int, step: int, max_lag: int, criterion: str = "bic",
                      prune: bool = True, seed: int = 0) -> RollingGraphs:
    data = panel.to_numpy(dtype=float)
    k = data.shape[1]
    ends, Bs, lags, orders = [], [], [], []
    skipped = 0
    for i in window_ends(len(data), window, step):
        X = data[i - window + 1: i + 1]
        if np.isnan(X).any():
            skipped += 1
            continue
        B, p, order = fit_window(X, max_lag, criterion, prune, seed)
        ends.append(panel.index[i])
        Bs.append(B)
        lags.append(p)
        orders.append(order)
    log.info("VarLiNGAM: %d windows fit, %d skipped (incomplete data)", len(ends), skipped)
    return RollingGraphs(pd.DatetimeIndex(ends), list(panel.columns), np.stack(Bs), np.array(lags),
                         np.stack(orders) if orders else np.zeros((0, k), int))
