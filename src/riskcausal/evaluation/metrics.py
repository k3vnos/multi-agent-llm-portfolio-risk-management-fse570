"""Event-level and day-level evaluation.

Definitions (IMPLEMENTATION_PLAN E14):
- alert episode: run of alert days, merging gaps of at most `merge_gap` sessions.
- an episode that starts in a stressed day is "in-stress" (neither hit nor false alarm).
- an episode starting on a calm day is TRUE if an onset falls in (start, end + h], otherwise a FALSE ALARM.
- an onset o is DETECTED if any alert occurs in [o - h, o - 1]; lead time = o - first such alert (sessions).
- onsets closer than h sessions to the start of the evaluated span are excluded (alerts before it unseen).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def alert_episodes(alert: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(alert)
    if len(idx) == 0:
        return []
    eps = []
    s = e = idx[0]
    for i in idx[1:]:
        if i - e - 1 <= merge_gap:
            e = i
        else:
            eps.append((s, e))
            s = e = i
    eps.append((s, e))
    return eps


def event_metrics(alert: np.ndarray, stress: np.ndarray, onset: np.ndarray, h: int, merge_gap: int,
                  n_years: float) -> dict:
    alert = np.asarray(alert, bool)
    eps = alert_episodes(alert, merge_gap)
    on_idx = np.flatnonzero(onset)
    true_ep = false_ep = in_stress = 0
    for s, e in eps:
        if stress[s] == 1:
            in_stress += 1
        elif ((on_idx > s) & (on_idx <= e + h)).any():
            true_ep += 1
        else:
            false_ep += 1
    eligible = on_idx[on_idx >= h]
    leads = []
    for o in eligible:
        w = np.flatnonzero(alert[o - h:o])
        leads.append(int(h - w[0]) if len(w) else None)
    det = [x for x in leads if x is not None]
    n_calm_eps = true_ep + false_ep
    return {
        "n_onsets": len(eligible), "n_detected": len(det),
        "recall": len(det) / len(eligible) if len(eligible) else None,
        "n_alert_episodes": len(eps), "n_true_episodes": true_ep, "n_false_alarms": false_ep,
        "n_in_stress_episodes": in_stress,
        "precision": true_ep / n_calm_eps if n_calm_eps else None,
        "false_alarms_per_year": false_ep / n_years if n_years else None,
        "lead_times": leads, "median_lead": float(np.median(det)) if det else None,
        "alert_day_share": float(alert.mean()) if len(alert) else None,
    }


def day_metrics(score: np.ndarray, target: np.ndarray, probabilistic: bool) -> dict:
    m = ~np.isnan(score) & ~np.isnan(target)
    y, s = target[m].astype(int), score[m]
    out = {"n_days": int(m.sum()), "n_pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
        out["base_rate"] = float(y.mean())
        if probabilistic:
            out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    return out


def choose_threshold(score: pd.Series, stress: pd.Series, onset: pd.Series, h: int, merge_gap: int,
                     max_fa_per_year: float) -> tuple[float, dict]:
    """Lowest threshold whose false-alarm rate on the validation span is within budget (max recall)."""
    s = score.to_numpy(dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() < 50:
        return float("nan"), {"status": "insufficient validation scores", "n_valid": int(ok.sum())}
    n_years = ok.sum() / 252
    cands = np.unique(np.quantile(s[ok], np.linspace(0.50, 0.999, 200)))
    st, on = stress.to_numpy(), onset.to_numpy()
    for thr in cands:
        alert = np.where(ok, s >= thr, False)
        m = event_metrics(alert, st, on, h, merge_gap, n_years)
        if m["false_alarms_per_year"] <= max_fa_per_year:
            return float(thr), {"status": "ok", "val_recall": m["recall"], "val_fa_per_year": m["false_alarms_per_year"],
                                "val_onsets": m["n_onsets"]}
    return float(np.nanmax(s) + 1e-12), {"status": "budget never met; threshold above max (no alerts)"}
