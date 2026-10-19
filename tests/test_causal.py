"""Directed-dependence estimators on synthetic data with a known structure."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from riskcausal.causal.granger_rolling import granger_pvalues
from riskcausal.causal.varlingam_rolling import fit_window, rolling_varlingam, window_ends


def _lingam_data(n=2000, seed=0):
    """x0 -> x1 -> x2 contemporaneously, x0 has a lag-1 effect on x2; non-Gaussian (uniform) shocks."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(-1, 1, size=(n, 3))
    X = np.zeros((n, 3))
    for t in range(1, n):
        x0 = e[t, 0]
        x1 = 0.8 * x0 + e[t, 1]
        x2 = 0.7 * x1 + 0.5 * X[t - 1, 0] + e[t, 2]
        X[t] = [x0, x1, x2]
    return X[1:]


def test_varlingam_recovers_known_contemporaneous_dag():
    B, _p, order = fit_window(_lingam_data(), max_lag=2, criterion="bic", prune=True, seed=0)
    B0 = B[0]
    assert abs(B0[1, 0]) > 0.3 and abs(B0[2, 1]) > 0.3       # true edges found
    assert abs(B0[0, 1]) < 0.05 and abs(B0[1, 2]) < 0.05     # no reversed edges
    assert list(order) == [0, 1, 2]
    assert abs(B[1][2, 0]) > 0.1                              # lagged effect x0(t-1) -> x2


def test_rolling_windows_only_use_past_rows():
    X = pd.DataFrame(_lingam_data(600), index=pd.bdate_range("2020-01-01", periods=599))
    g1 = rolling_varlingam(X, window=200, step=50, max_lag=1, seed=0)
    X2 = X.copy()
    X2.iloc[400:] = X2.iloc[400:] * 5 + 3          # perturb the future
    g2 = rolling_varlingam(X2, window=200, step=50, max_lag=1, seed=0)
    keep = g1.ends < X.index[400]
    assert keep.sum() > 0
    np.testing.assert_allclose(g1.B[keep], g2.B[keep])


def test_window_ends_first_window_complete():
    assert window_ends(10, 5, 2) == [4, 6, 8]


def test_granger_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 400
    X = np.zeros((n, 3))
    for t in range(1, n):
        X[t, 0] = rng.normal()
        X[t, 1] = 0.6 * X[t - 1, 0] + rng.normal()
        X[t, 2] = rng.normal()
    P = granger_pvalues(X, p=1)
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    # single-equation OLS of x1 on all lagged variables; F-test that the lag of x0 is zero
    ols = sm.OLS(Z[1:, 1], sm.add_constant(Z[:-1])).fit()
    sm_p = float(ols.f_test("x1 = 0").pvalue)
    np.testing.assert_allclose(P[1, 0], sm_p, rtol=1e-6)
    assert P[1, 0] < 1e-6 and P[2, 0] > 0.01
