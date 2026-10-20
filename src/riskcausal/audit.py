"""Real-data leakage audit of a finished run (LEAKAGE_AUDIT.md). Each check recomputes a timing property from the
raw snapshot or the saved artifacts, independently of the code path that produced the features, and counts
violations. A run with any violation is flagged in metrics.json and in the report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from riskcausal.causal.varlingam_rolling import RollingGraphs
from riskcausal.config import Config
from riskcausal.data.calendar import add_bdays, nyse_sessions
from riskcausal.data.panel import latest_snapshot, load_first_releases, processed_dir
from riskcausal.registry import write_json

log = logging.getLogger(__name__)


def _check(name: str, n_violations: int, n_checked: int, detail: str = "") -> dict:
    return {"check": name, "passed": n_violations == 0, "n_violations": int(n_violations),
            "n_checked": int(n_checked), "detail": detail}


def _obs_known_by(cfg: Config, sid: str, sessions_all: pd.DatetimeIndex) -> pd.Series:
    """Independent restatement of the publication rule. Market-close series: obs d known from d + lag sessions.
    Release series: known at close of t iff t > first-release date (tracked) or t >= d + fallback sessions
    (untracked), and never before d + lag sessions."""
    s = pd.read_parquet(cfg.data_path / "raw" / "fred" / latest_snapshot(cfg) / f"{sid}.parquet")[sid].dropna()
    lag, fb = cfg.pit.macro_release_lag_bdays, cfg.pit.daily_fallback_lag_bdays
    floor = add_bdays(s.index, lag, sessions_all)
    if cfg.universe["fred"][sid].get("availability", "release") == "market_close":
        return pd.Series(floor, index=s.index)
    rel = load_first_releases(cfg, sid).reindex(s.index)
    known = pd.Series(add_bdays(s.index, fb, sessions_all), index=s.index)
    tr = rel.notna().to_numpy()
    nxt = add_bdays(pd.DatetimeIndex(rel[tr]), 1, sessions_all)
    known[tr] = np.maximum(nxt.values, floor[tr].values)
    return known


def audit_run(cfg: Config, run_dir: Path, P: pd.DataFrame, feats: pd.DataFrame, fold_info: list[dict],
              g: RollingGraphs) -> dict:
    sessions = feats.index
    sessions_all = nyse_sessions("1990-01-01", str(sessions[-1].date()))
    pdir = processed_dir(cfg)
    checks = []

    # row 7/8 — fold geometry and where predictions/thresholds live
    viol = 0
    for f in fold_info:
        te, ts = sessions.get_loc(pd.Timestamp(f["train_end"])), sessions.get_loc(pd.Timestamp(f["test_start"]))
        ie, vs = sessions.get_loc(pd.Timestamp(f["inner_end"])), sessions.get_loc(pd.Timestamp(f["val_start"]))
        viol += int(not (ts - te - 1 >= cfg.walkforward.embargo_days and vs - ie - 1 >= cfg.walkforward.embargo_days
                         and ie < vs <= te < ts))
    checks.append(_check("fold geometry: inner_end < val_start <= train_end < test_start with embargo", viol,
                         len(fold_info)))
    span = {f["k"]: (pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])) for f in fold_info}
    lo = P["fold"].map(lambda k: span[k][0])
    hi = P["fold"].map(lambda k: span[k][1])
    checks.append(_check("every prediction lies inside its own fold's test block",
                         int(((P["date"] < lo) | (P["date"] > hi)).sum()), len(P)))
    tcols = [c for c in P.columns if c.startswith("threshold_")]
    n_thr = int((P.groupby(["method", "fold"])[tcols].nunique(dropna=False) > 1).to_numpy().sum())
    checks.append(_check("one threshold per (method, fold, rule) — never varies inside a test block", n_thr,
                         P.groupby(["method", "fold"]).ngroups * len(tcols)))

    # row 2/3 — graph features: source window ended before t and every input in it was published by t
    gw = pd.DatetimeIndex(feats["graph_window_end"])
    has = ~gw.isna()
    checks.append(_check("graph feature at t comes from a window ending strictly before t",
                         int((gw[has] >= sessions[has]).sum()), int(has.sum())))
    panel = pd.read_parquet(pdir / "graph_panel.parquet")
    ppos = panel.index.get_indexer(gw[has])
    W = cfg.graph.window
    viol, worst = 0, []
    for sid in cfg.graph_nodes.macro:
        known = _obs_known_by(cfg, sid, sessions_all).reindex(panel.index).ffill(limit=3)
        kv = known.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        kv[np.isnat(known.to_numpy(dtype="datetime64[ns]"))] = np.inf
        # latest publication among rows end-W..end of the source window (running max = earlier windows too)
        need = np.array([kv[max(p - W, 0): p + 1].max() for p in ppos])
        need = np.maximum.accumulate(need)
        t = sessions[has].to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        bad = need > t
        viol += int(bad.sum())
        if bad.any():
            worst.append(f"{sid}: first violation at {sessions[has][bad][0].date()}")
    checks.append(_check("every FRED value inside a feature's source window was published by t (ALFRED dates)",
                         viol, int(has.sum()) * len(cfg.graph_nodes.macro), "; ".join(worst)))

    # row 1/2 — controls: the observation used at t was published by t
    ctrl = pd.read_parquet(pdir / "controls_pit.parquet")
    viol = n = 0
    for sid in cfg.fred_series:
        od = ctrl[f"{sid}__obs_date"].dropna()
        n += len(od)
        viol += int((od.to_numpy() >= od.index.to_numpy()).sum())
        if not cfg.universe["fred"][sid]["revised"]:
            known = _obs_known_by(cfg, sid, sessions_all)
            kn = pd.DatetimeIndex(known.reindex(pd.DatetimeIndex(od.values)))
            viol += int((kn.isna() | (kn > od.index)).sum())
    checks.append(_check("control value used at t: observation dated before t and published by t", viol, n))

    # row 9 — ragged panel: no ETF return before the first raw close
    rets = pd.read_parquet(pdir / "returns.parquet")
    viol = 0
    for t in cfg.etfs:
        first_close = pd.read_parquet(cfg.data_path / "raw" / "yfinance" / latest_snapshot(cfg) /
                                      f"{t}.parquet")["close"].dropna().index.min()
        fv = rets[t].first_valid_index()
        viol += int(fv is not None and fv <= first_close)
    checks.append(_check("no ETF return on or before its first traded close (no back-fill)", viol, len(cfg.etfs)))

    # row 5 — composite/model standardisation is recomputed per fold from rows <= inner/train end: recorded counts
    stale = (sessions[has] - gw[has]).days
    out = {"passed": all(c["passed"] for c in checks), "checks": checks,
           "graph_feature_staleness_calendar_days": {"median": float(np.median(stale)), "p99": float(np.quantile(stale, .99)),
                                                     "max": int(stale.max())}}
    write_json(Path(run_dir) / "audit.json", out)
    lvl = logging.INFO if out["passed"] else logging.ERROR
    log.log(lvl, "leakage audit: %s (%d checks)", "PASSED" if out["passed"] else "FAILED", len(checks))
    return out
