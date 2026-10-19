"""Rolling pairwise Granger tests from a VAR fit per window (comparison method, E8).

For each window a VAR(p) is fit by OLS on standardised data (p = lag order chosen by VarLiNGAM's BIC for that
window, so both methods see the same lag structure). For each ordered pair (j -> i) a Wald F-test checks
whether all p lags of j are zero in the equation for i. p-values are BH-FDR corrected within each window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class RollingGranger:
    ends: pd.DatetimeIndex
    nodes: list[str]
    pvals: np.ndarray   # (n_windows, k, k), [i, j] = p-value for j Granger-causing i; diagonal NaN
    sig: np.ndarray     # FDR-significant (bool)

    def save(self, path) -> None:
        np.savez_compressed(path, ends=self.ends.values.astype("datetime64[ns]"), nodes=np.array(self.nodes),
                            pvals=self.pvals, sig=self.sig)

    @classmethod
    def load(cls, path) -> RollingGranger:
        z = np.load(path, allow_pickle=False)
        return cls(pd.DatetimeIndex(z["ends"]), list(z["nodes"]), z["pvals"], z["sig"])


def granger_pvalues(X: np.ndarray, p: int) -> np.ndarray:
    """Wald-F p-values for every ordered pair from one VAR(p) OLS fit."""
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    n, k = Z.shape
    Y = Z[p:]
    lagged = np.hstack([Z[p - l: n - l] for l in range(1, p + 1)])     # columns: lag1 vars, lag2 vars, ...
    R = np.hstack([np.ones((n - p, 1)), lagged])
    XtX_inv = np.linalg.pinv(R.T @ R)
    beta = XtX_inv @ R.T @ Y
    resid = Y - R @ beta
    dof = (n - p) - R.shape[1]
    sigma2 = (resid ** 2).sum(0) / dof
    P = np.full((k, k), np.nan)
    for j in range(k):
        idx = [1 + (l * k) + j for l in range(p)]
        V = XtX_inv[np.ix_(idx, idx)]
        Vinv = np.linalg.pinv(V)
        for i in range(k):
            if i == j:
                continue
            b = beta[idx, i]
            F = float(b @ Vinv @ b) / (p * sigma2[i])
            P[i, j] = stats.f.sf(F, p, dof)
    return P


def rolling_granger(panel: pd.DataFrame, ends: pd.DatetimeIndex, lags: np.ndarray, window: int,
                    alpha: float) -> RollingGranger:
    data = panel.to_numpy(dtype=float)
    pos = panel.index.get_indexer(ends)
    k = data.shape[1]
    pv, sg = [], []
    off = ~np.eye(k, dtype=bool)
    for i, p in zip(pos, lags):
        P = granger_pvalues(data[i - window + 1: i + 1], max(int(p), 1))
        rej = np.zeros((k, k), bool)
        rej[off] = multipletests(P[off], alpha=alpha, method="fdr_bh")[0]
        pv.append(P)
        sg.append(rej)
    return RollingGranger(ends, list(panel.columns), np.stack(pv), np.stack(sg))
