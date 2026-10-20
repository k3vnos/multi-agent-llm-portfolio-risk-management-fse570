"""DCC-GARCH baseline (Engle, 2002): GARCH(1,1) per asset + DCC(1,1) correlations, fit on training data only.

Parameters are estimated on data <= fit_end and then filtered forward with fixed parameters. The score at session
t is the one-step-ahead forecast for t+1 made at the close of t (uses returns <= t only):
- dcc_var : 99% 1-day parametric VaR of the equal-weight portfolio from H_{t+1|t} = D R D
- dcc_corr: mean off-diagonal forecast correlation R_{t+1|t}
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

log = logging.getLogger(__name__)


def _garch_params(r: np.ndarray) -> tuple[float, float, float, float]:
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
            disp="off", options={"maxiter": 500})
    p = res.params
    return float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])


def _garch_filter(r: np.ndarray, mu: float, omega: float, alpha: float, beta: float, s2_0: float):
    """sigma2[t] = variance of r_t given r_<t; sigma2_next[t] = variance of r_{t+1} given r_<=t."""
    n = len(r)
    s2 = np.empty(n)
    s2[0] = s2_0
    e = r - mu
    for t in range(1, n):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
    s2_next = omega + alpha * e ** 2 + beta * s2
    return s2, s2_next


def _dcc_Q(a: float, b: float, z: np.ndarray, Qbar: np.ndarray) -> np.ndarray:
    """Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_0 = Qbar — as one linear filter over all t."""
    from scipy.signal import lfilter

    n, k = z.shape
    x = np.empty((n, k, k))
    x[0] = Qbar
    x[1:] = (1 - a - b) * Qbar + a * np.einsum("ti,tj->tij", z[:-1], z[:-1])
    return lfilter([1.0], [1.0, -b], x, axis=0)


def _dcc_nll(theta: np.ndarray, z: np.ndarray, Qbar: np.ndarray) -> float:
    a, b = theta
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    Q = _dcc_Q(a, b, z, Qbar)
    d = 1 / np.sqrt(np.einsum("tii->ti", Q))
    R = Q * d[:, :, None] * d[:, None, :]
    sign, logdet = np.linalg.slogdet(R)
    if (sign <= 0).any():
        return 1e10
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(R, z[:, :, None])[:, :, 0])
    return float(0.5 * (logdet + quad - (z * z).sum(1)).sum())


def fit_dcc(returns: pd.DataFrame, fit_end: pd.Timestamp) -> dict:
    """QML estimation on rows <= fit_end with complete data. Returns are in decimal log-return units."""
    R = returns.loc[:fit_end].dropna() * 100.0
    garch = [_garch_params(R[c].to_numpy()) for c in R.columns]
    Z = np.column_stack([
        (R[c].to_numpy() - g[0]) / np.sqrt(_garch_filter(R[c].to_numpy(), *g, s2_0=R[c].var())[0])
        for c, g in zip(R.columns, garch)])
    Qbar = np.corrcoef(Z, rowvar=False)
    best = None
    for start in ([0.02, 0.95], [0.05, 0.90], [0.01, 0.97]):
        res = minimize(_dcc_nll, np.array(start), args=(Z, Qbar), method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    a, b = (float(x) for x in best.x)
    return {"fit_end": fit_end, "columns": list(R.columns), "garch": garch, "s2_0": [float(R[c].var()) for c in R.columns],
            "Qbar": Qbar, "a": a, "b": b, "start": R.index[0]}


def dcc_scores(returns: pd.DataFrame, params: dict, alpha: float = 0.99) -> pd.DataFrame:
    """Filter forward from the start of the fit sample with fixed parameters; scores for every session."""
    R = returns[params["columns"]].loc[params["start"]:] * 100.0
    ok = R.notna().all(axis=1).to_numpy()
    X = R.to_numpy()
    n, k = X.shape
    s2 = np.array(params["s2_0"], float)
    Q = params["Qbar"].copy()
    a, b, Qbar = params["a"], params["b"], params["Qbar"]
    w = np.full(k, 1 / k)
    var_out, corr_out = np.full(n, np.nan), np.full(n, np.nan)
    z_prev = None
    off = ~np.eye(k, dtype=bool)
    for t in range(n):
        if not ok[t]:          # incomplete row: carry state, no score (never happens after the fit start)
            continue
        e = np.array([X[t, i] - params["garch"][i][0] for i in range(k)])
        z = e / np.sqrt(s2)
        if z_prev is not None:
            Q = (1 - a - b) * Qbar + a * np.outer(z_prev, z_prev) + b * Q
        # forecast for t+1 made at close of t
        s2_next = np.array([g[1] + g[2] * e[i] ** 2 + g[3] * s2[i] for i, g in enumerate(params["garch"])])
        Q_next = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
        d = 1 / np.sqrt(np.diag(Q_next))
        R_next = Q_next * np.outer(d, d)
        sd = np.sqrt(s2_next)
        H = R_next * np.outer(sd, sd)
        var_out[t] = norm.ppf(alpha) * np.sqrt(w @ H @ w) / 100.0
        corr_out[t] = R_next[off].mean()
        s2, z_prev = s2_next, z
    idx = R.index
    out = pd.DataFrame({"dcc_var": var_out, "dcc_corr": corr_out}, index=idx)
    return out.reindex(returns.index)
