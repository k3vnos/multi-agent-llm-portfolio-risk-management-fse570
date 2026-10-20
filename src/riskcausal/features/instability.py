"""Graph-instability features: how much the estimated directed-dependence structure changes between
consecutive rolling windows (proposal §3 "Feature").

Each component is kept separately (E7). A window's value compares it with the previous window only, so it
depends on data up to that window's end. `window_availability` gives the first session at which every row of a
window had been published, and `to_daily` makes each window's value usable only from then (LEAKAGE_AUDIT row 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskcausal.causal.granger_rolling import RollingGranger
from riskcausal.causal.varlingam_rolling import RollingGraphs


def varlingam_instability(g: RollingGraphs, edge_threshold: float) -> pd.DataFrame:
    B = g.B
    A = np.abs(B) > edge_threshold                       # edge exists
    dB = np.diff(B, axis=0)
    dA = A[1:] != A[:-1]
    df = pd.DataFrame({
        "vl_shd_b0": dA[:, 0].sum(axis=(1, 2)),            # structural Hamming distance, contemporaneous
        "vl_shd_lag": dA[:, 1:].sum(axis=(1, 2, 3)),       # ... lagged matrices
        "vl_frob_b0": np.sqrt((dB[:, 0] ** 2).sum(axis=(1, 2))),
        "vl_frob_lag": np.sqrt((dB[:, 1:] ** 2).sum(axis=(1, 2, 3))),
        "vl_n_edges_b0": A[1:, 0].sum(axis=(1, 2)),        # level of connectedness (context feature)
        "vl_order_change": (g.order[1:] != g.order[:-1]).sum(axis=1),
    }, index=g.ends[1:]).astype(float)
    return df


def granger_instability(gr: RollingGranger) -> pd.DataFrame:
    k = gr.pvals.shape[1]
    off = ~np.eye(k, dtype=bool)
    logp = -np.log10(np.clip(np.nan_to_num(gr.pvals, nan=1.0), 1e-12, 1.0))
    d_logp = np.abs(np.diff(logp, axis=0))[:, off]
    dS = (gr.sig[1:] != gr.sig[:-1])[:, off]
    return pd.DataFrame({
        "gc_hamming": dS.sum(axis=1),
        "gc_mean_abs_dlogp": d_logp.mean(axis=1),
        "gc_n_edges": gr.sig[1:][:, off].sum(axis=1),
    }, index=gr.ends[1:]).astype(float)


def add_smoothed(df: pd.DataFrame, n_windows: int = 4) -> pd.DataFrame:
    """Trailing mean over the last n windows (≈ one month at step 5) — trailing only."""
    sm = df.rolling(n_windows, min_periods=n_windows).mean().add_suffix(f"_ma{n_windows}")
    return pd.concat([df, sm], axis=1)


def window_availability(ends: pd.DatetimeIndex, row_avail: pd.Series, window: int) -> pd.Series:
    """First session at which window ending at each `ends` date (and every earlier window) was fully published.

    A window uses rows end-window..end (the extra row feeds the first difference). Features compare a window with
    earlier windows, so availability is carried forward as a running maximum. NaT = never available.
    """
    pos = row_avail.index.get_indexer(ends)
    if (pos < 0).any():
        raise ValueError("window end dates must be rows of the graph panel")
    a = row_avail.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    a[np.isnat(row_avail.to_numpy(dtype="datetime64[ns]"))] = np.inf
    w = np.array([a[max(p - window, 0): p + 1].max() for p in pos])
    w = np.maximum.accumulate(w)
    out = pd.Series(pd.NaT, index=ends, dtype="datetime64[ns]")
    fin = np.isfinite(w)
    out[fin] = pd.to_datetime(w[fin].astype("int64"))
    return out


def source_window(ends: pd.DatetimeIndex, sessions: pd.DatetimeIndex, avail: pd.Series) -> np.ndarray:
    """For each session, position in `ends` of the latest-ending window available by then (-1 = none yet)."""
    av = pd.DatetimeIndex(avail.reindex(ends))
    marker = np.full(len(sessions), -1)
    ok = ~av.isna()
    for i, p in zip(np.flatnonzero(ok), sessions.searchsorted(av[ok])):
        if p < len(sessions):
            marker[p] = max(marker[p], i)
    return np.maximum.accumulate(marker)


def to_daily(per_window: pd.DataFrame, sessions: pd.DatetimeIndex, avail: pd.Series) -> pd.DataFrame:
    """Value at session t = the latest-ending window whose availability session is <= t."""
    marker = source_window(pd.DatetimeIndex(per_window.index), sessions, avail)
    vals = per_window.to_numpy(dtype=float)
    out = np.full((len(sessions), vals.shape[1]), np.nan)
    have = marker >= 0
    out[have] = vals[marker[have]]
    return pd.DataFrame(out, index=sessions, columns=per_window.columns)
