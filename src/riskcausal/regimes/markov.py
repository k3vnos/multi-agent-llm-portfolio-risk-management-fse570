"""Markov regime switching (Hamilton, 1989) on log realised volatility → calm / stressed regimes.

Two kinds of output, kept apart (IMPLEMENTATION_PLAN §F3):
- `fit_ms` + `smoothed_stress_prob` on a sample ending at some date: smoothed probabilities use every
  observation *in that sample*. Fit on a training sample → training labels. Fit on the full sample →
  retrospective evaluation labels (`label_eval`), used only for scoring.
- `filtered_stress_prob` with parameters fit on training data: uses observations <= t only (sensitivity).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

log = logging.getLogger(__name__)


@dataclass
class MSFit:
    params: np.ndarray
    stress_regime: int
    sample_end: pd.Timestamp
    regime_means: list[float]
    smoothed: pd.Series            # P(stress) smoothed within the fitted sample
    llf: float = float("nan")


def _model(y: pd.Series, k: int) -> MarkovRegression:
    return MarkovRegression(y.to_numpy(), k_regimes=k, trend="c", switching_variance=True)


def _start_points(y: np.ndarray, names: list[str], k: int, n: int, seed: int) -> list[np.ndarray]:
    """Deterministic starting values: a quantile-based start plus `n` seeded random perturbations."""
    rng = np.random.default_rng(seed)
    qs = np.quantile(y, np.linspace(0.2, 0.8, k))
    var = float(np.var(y))
    starts = []
    for r in range(n + 1):
        means = qs if r == 0 else np.sort(rng.uniform(y.min(), y.max(), k))
        sig = np.full(k, var / k) if r == 0 else var * rng.uniform(0.05, 1.0, k)
        stay = 0.95 if r == 0 else rng.uniform(0.8, 0.99)
        sp = []
        for nm in names:
            if nm.startswith("p["):
                i, j = (int(x) for x in nm[2:-1].split("->"))
                sp.append(stay if i == j else (1 - stay) / (k - 1))
            elif nm.startswith("const["):
                sp.append(float(means[int(nm[6:-1])]))
            elif nm.startswith("sigma2["):
                sp.append(float(sig[int(nm[7:-1])]))
            else:
                raise ValueError(f"unexpected parameter {nm}")
        starts.append(np.array(sp))
    return starts


def fit_ms(y: pd.Series, k: int = 2, seed: int = 0, n_starts: int = 20) -> MSFit:
    """Deterministic multi-start ML fit: best log-likelihood over seeded starting points.

    statsmodels' built-in `search_reps` was found non-reproducible across calls with the same seed and
    occasionally converged to much worse optima, so starting points are generated here explicitly.
    """
    y = y.dropna()
    mod = _model(y, k)
    names = list(mod.param_names)
    best = None
    for sp in _start_points(y.to_numpy(), names, k, n_starts, seed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = mod.fit(start_params=sp, search_reps=0, disp=False)
            except (np.linalg.LinAlgError, ValueError):
                continue
        if np.isfinite(res.llf) and (best is None or res.llf > best.llf + 1e-9):
            best = res
    if best is None:
        raise RuntimeError("Markov-switching fit failed from every starting point")
    means = [float(best.params[names.index(f"const[{i}]")]) for i in range(k)]
    stress = int(np.argmax(means))
    sm = pd.Series(best.smoothed_marginal_probabilities[:, stress], index=y.index, name="p_stress")
    return MSFit(np.asarray(best.params), stress, y.index.max(), means, sm, float(best.llf))


def filtered_stress_prob(y: pd.Series, fit: MSFit, k: int = 2) -> pd.Series:
    """P(stress at t | y_<=t) using fixed parameters from `fit` (no re-estimation on new data)."""
    y = y.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(y, k).filter(fit.params)
    return pd.Series(res.filtered_marginal_probabilities[:, fit.stress_regime], index=y.index, name="p_stress")


def cached_fit_ms(y: pd.Series, k: int, seed: int, cache_dir, n_starts: int = 20) -> MSFit:
    """`fit_ms` with an on-disk cache keyed by the exact input series, k, seed and n_starts.

    The same fold-end fits recur across runs and grid configurations (labels do not depend on graph settings), so
    caching removes repeated multi-start optimisation without changing any result.
    """
    import hashlib
    from pathlib import Path

    y = y.dropna()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(y.to_numpy(float)).tobytes())
    h.update(np.ascontiguousarray(y.index.asi8).tobytes())
    h.update(f"k={k}|seed={seed}|n_starts={n_starts}|v1".encode())
    path = Path(cache_dir) / f"ms_{h.hexdigest()[:20]}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        sm = pd.Series(z["smoothed"], index=y.index, name="p_stress")
        return MSFit(z["params"], int(z["stress"]), y.index.max(), list(z["means"]), sm, float(z["llf"]))
    fit = fit_ms(y, k, seed=seed, n_starts=n_starts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, params=fit.params, stress=fit.stress_regime, means=np.array(fit.regime_means),
             smoothed=fit.smoothed.to_numpy(), llf=fit.llf)
    return fit


def predictive_loglik(y: pd.Series, fit: MSFit, k: int, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Per-observation one-step-ahead predictive log-likelihood on (start..end] rows, parameters fixed from `fit`.

    The filter runs over all data <= end; each contribution is log p(y_t | y_<t), so no row uses later data.
    """
    y = y.dropna().loc[:end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(y, k).filter(fit.params)
    ll = pd.Series(res.llf_obs, index=y.index)
    return float(ll.loc[start:end].mean())
