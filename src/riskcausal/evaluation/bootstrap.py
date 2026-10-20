"""Confidence intervals from a finished run's saved predictions (pre-registered in robustness.yaml).

- Day level: paired stationary block bootstrap (Politis & Romano, 1994) over the common-span test days. The same
  resampled day indices are used for every method, so differences between methods are paired. Day-level metrics
  are computed, as in the run, on calm days with a defined target.
- Event level: Jeffreys Beta(0.5, 0.5) intervals for recall (detected / onsets) and precision
  (true / calm alert episodes) under the primary alert rule.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

from riskcausal.registry import read_json, write_json

log = logging.getLogger(__name__)


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample of 0..n-1: blocks start uniformly, lengths ~ Geometric(1/mean_block),
    wrapping around the end."""
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = rng.geometric(1 / mean_block)
        take = min(length, n - i)
        idx[i:i + take] = (start + np.arange(take)) % n
        i += take
    return idx


def jeffreys(k: int, n: int, level: float = 0.95) -> list[float] | None:
    if n == 0:
        return None
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 0.5, n - k + 0.5))
    return [lo, hi]


def _metric(y: np.ndarray, s: np.ndarray, which: str) -> float:
    m = ~np.isnan(s) & ~np.isnan(y)
    yy, ss = y[m].astype(int), s[m]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return np.nan
    return float(roc_auc_score(yy, ss) if which == "roc" else average_precision_score(yy, ss))


def run_cis(run_dir: Path, n_rep: int = 2000, mean_block: float = 63, seed: int = 570) -> dict:
    run_dir = Path(run_dir)
    M = read_json(run_dir / "metrics.json")
    P = pd.read_parquet(run_dir / "predictions.parquet")
    truth = pd.read_parquet(run_dir / "labels_eval.parquet")
    folds = M["common_span_folds"]
    dates = pd.DatetimeIndex(sorted(P.loc[P["fold"].isin(folds), "date"].unique()))
    tr = truth.reindex(dates)
    y = np.where(tr["stress"].fillna(0).to_numpy() == 0, tr["target"].to_numpy(float), np.nan)
    S = {m: g.set_index("date")["score"].reindex(dates).to_numpy(float) for m, g in P.groupby("method")}
    methods = sorted(S)
    pairs = [(m, "vix") for m in methods if m != "vix"]
    for fam in ("bayes_logit", "gbm"):
        if f"{fam}:combined" in S and f"{fam}:controls" in S:
            pairs.append((f"{fam}:combined", f"{fam}:controls"))

    point = {w: {m: _metric(y, S[m], w) for m in methods} for w in ("roc", "pr")}
    rng = np.random.default_rng(seed)
    boot = {w: {m: np.empty(n_rep) for m in methods} for w in ("roc", "pr")}
    for r in range(n_rep):
        ix = stationary_bootstrap_indices(len(dates), mean_block, rng)
        yb = y[ix]
        for m in methods:
            sb = S[m][ix]
            for w in ("roc", "pr"):
                boot[w][m][r] = _metric(yb, sb, w)

    def ci(a):
        a = a[~np.isnan(a)]
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) else None

    out = {"n_rep": n_rep, "mean_block": mean_block, "seed": seed, "span": [str(dates[0].date()), str(dates[-1].date())],
           "n_days_scored": int((~np.isnan(y)).sum()), "methods": {}, "differences": {}}
    for m in methods:
        out["methods"][m] = {w: {"point": point[w][m], "ci95": ci(boot[w][m])} for w in ("roc", "pr")}
    for a, b in pairs:
        d = {}
        for w in ("roc", "pr"):
            diff = boot[w][a] - boot[w][b]
            c = ci(diff)
            d[w] = {"point": point[w][a] - point[w][b], "ci95": c,
                    "supported": None if c is None else bool(c[0] > 0 or c[1] < 0)}
        out["differences"][f"{a} - {b}"] = d

    ev = M["by_rule"][M["primary_rule"]]["common_span_metrics"]
    out["event_level_primary_rule"] = {
        m: {"recall": r["event"]["recall"], "recall_ci95": jeffreys(r["event"]["n_detected"], r["event"]["n_onsets"]),
            "precision": r["event"]["precision"],
            "precision_ci95": jeffreys(r["event"]["n_true_episodes"],
                                       r["event"]["n_true_episodes"] + r["event"]["n_false_alarms"]),
            "n_onsets": r["event"]["n_onsets"]}
        for m, r in ev.items()}
    write_json(run_dir / "ci.json", out)
    _write_md(run_dir, out)
    return out


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _write_md(run_dir: Path, C: dict) -> None:
    L = [f"# Confidence intervals — `{run_dir.name}`", "",
         (f"_Generated from `ci.json`: paired stationary block bootstrap, {C['n_rep']} replicates, mean block "
          f"{C['mean_block']:g} sessions, seed {C['seed']}; span {C['span'][0]} → {C['span'][1]}, "
          f"{C['n_days_scored']} calm days scored._"), "",
         "## Day level (95% percentile intervals)", "", "| method | ROC-AUC | 95% CI | PR-AUC | 95% CI |",
         "|---|---|---|---|---|"]
    for m, r in sorted(C["methods"].items(), key=lambda x: -(x[1]["roc"]["point"] or 0)):
        rc, pc = r["roc"]["ci95"] or [None, None], r["pr"]["ci95"] or [None, None]
        L.append(f"| {m} | {_fmt(r['roc']['point'])} | [{_fmt(rc[0])}, {_fmt(rc[1])}] | {_fmt(r['pr']['point'])} | "
                 f"[{_fmt(pc[0])}, {_fmt(pc[1])}] |")
    L += ["", "## Paired differences", "", "| difference | ΔROC-AUC | 95% CI | excludes 0 | ΔPR-AUC | 95% CI | excludes 0 |",
          "|---|---|---|---|---|---|---|"]
    for k, d in C["differences"].items():
        rc, pc = d["roc"]["ci95"] or [None, None], d["pr"]["ci95"] or [None, None]
        L.append(f"| {k} | {_fmt(d['roc']['point'])} | [{_fmt(rc[0])}, {_fmt(rc[1])}] | {d['roc']['supported']} | "
                 f"{_fmt(d['pr']['point'])} | [{_fmt(pc[0])}, {_fmt(pc[1])}] | {d['pr']['supported']} |")
    L += ["", "## Event level, primary alert rule (Jeffreys 95% intervals)", "",
          "| method | onsets | recall | 95% CI | precision | 95% CI |", "|---|---|---|---|---|---|"]
    for m, r in sorted(C["event_level_primary_rule"].items()):
        rc, pc = r["recall_ci95"] or [None, None], r["precision_ci95"] or [None, None]
        L.append(f"| {m} | {r['n_onsets']} | {_fmt(r['recall'])} | [{_fmt(rc[0])}, {_fmt(rc[1])}] | "
                 f"{_fmt(r['precision'])} | [{_fmt(pc[0])}, {_fmt(pc[1])}] |")
    (run_dir / "ci.md").write_text("\n".join(L))
