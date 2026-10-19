"""Graph-instability features: how much the estimated directed-dependence structure changes between
consecutive rolling windows (proposal §3 "Feature").

Each component is kept separately (E7). A window's value compares it with the previous window only, so it
depends on data up to that window's end. `to_daily` then makes each value usable `lag` sessions after the
window end (graph inputs include macro values that FRED publishes the next day).
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


def to_daily(per_window: pd.DataFrame, sessions: pd.DatetimeIndex, lag: int) -> pd.DataFrame:
    """Value at session t = latest window whose end is at least `lag` sessions before t."""
    pos = sessions.get_indexer(per_window.index)
    if (pos < 0).any():
        raise ValueError("window end dates must be NYSE sessions")
    out = pd.DataFrame(np.nan, index=sessions, columns=per_window.columns)
    avail_pos = pos + lag
    keep = avail_pos < len(sessions)
    out.iloc[avail_pos[keep]] = per_window.to_numpy()[keep]
    return out.ffill()
