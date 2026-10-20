"""End-to-end core pipeline: panel → rolling graphs → features → fold-local labels → walk-forward models and
baselines → evaluation against retrospective labels. Every output lands in outputs/runs/<run_id>/.

Label discipline (IMPLEMENTATION_PLAN §F3):
- inner-train targets  : Markov model fit on data <= inner_end
- validation truth     : Markov model fit on data <= train_end   (used only to choose thresholds)
- outer-train targets  : Markov model fit on data <= train_end
- test truth (primary) : `label_eval`, Markov model fit on the full sample (retrospective, scoring only)
- test truth (sens.)   : filtered probabilities with train_end parameters (uses data <= t only)

`run_walkforward` never receives `label_eval`; it is built and used only in `run_pipeline` for scoring.
Alert thresholds are computed under every rule in ALERT_RULES from the same validation scores, so changing the
rule never requires refitting (M3: H0 rule kept for comparison next to the approved post-hoc rule).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from riskcausal.baselines.simple import corr_var_score, nfci_score, vix_score
from riskcausal.causal.granger_rolling import RollingGranger, rolling_granger
from riskcausal.causal.varlingam_rolling import RollingGraphs, rolling_varlingam
from riskcausal.config import Config
from riskcausal.data.panel import build_panel, latest_snapshot, processed_dir
from riskcausal.evaluation.metrics import choose_threshold, day_metrics, event_metrics
from riskcausal.features.controls import control_features
from riskcausal.features.instability import (
    add_smoothed,
    granger_instability,
    source_window,
    to_daily,
    varlingam_instability,
    window_availability,
)
from riskcausal.models.bayes_logit import BayesLogit
from riskcausal.models.gbm import GBM
from riskcausal.regimes.markov import filtered_stress_prob, fit_ms
from riskcausal.regimes.targets import label_set
from riskcausal.registry import new_run_dir, write_json
from riskcausal.validation.walkforward import make_folds

log = logging.getLogger(__name__)

VL = ["vl_shd_b0", "vl_shd_lag", "vl_frob_b0", "vl_frob_lag", "vl_n_edges_b0", "vl_order_change"]
GC = ["gc_hamming", "gc_mean_abs_dlogp", "gc_n_edges"]
CTRL = ["vix_level", "vix_chg_5d", "rv21", "avg_corr_63"]
FEATURE_SETS = {
    "instability": VL + [f"{c}_ma4" for c in VL],
    "granger": GC + [f"{c}_ma4" for c in GC],
    "controls": CTRL,
    "combined": VL + [f"{c}_ma4" for c in VL] + CTRL,
}
COMPOSITE = ["vl_shd_b0_ma4", "vl_frob_b0_ma4", "vl_shd_lag_ma4", "vl_frob_lag_ma4"]   # pre-specified (H0)
MODELS = [(m, fs) for m in ("bayes_logit", "gbm") for fs in FEATURE_SETS]
SCORES = ["inst_composite", "vix", "nfci", "corr_var"]
MERGE_GAP = 5              # alert episodes separated by <= 5 sessions are merged (H0)
VAL_YEARS = 2


# ---------------------------------------------------------------- graphs & features (cached by config)
def _graph_cache_dir(cfg: Config) -> Path:
    key = json.dumps({"snap": latest_snapshot(cfg), "nodes": cfg.graph_nodes.model_dump(),
                      "graph": cfg.graph.model_dump(), "granger": cfg.granger.model_dump(), "seed": cfg.seed},
                     sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    return cfg.data_path / "interim" / f"{cfg.profile}_graphs_{h}"


def build_graphs(cfg: Config) -> tuple[RollingGraphs, RollingGranger, Path]:
    d = _graph_cache_dir(cfg)
    if (d / "varlingam.npz").exists() and (d / "granger.npz").exists():
        log.info("graphs: cache hit %s", d)
        return RollingGraphs.load(d / "varlingam.npz"), RollingGranger.load(d / "granger.npz"), d
    d.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(processed_dir(cfg) / "graph_panel.parquet")
    g_cfg = cfg.graph
    t0 = time.time()
    g = rolling_varlingam(panel, g_cfg.window, g_cfg.step, g_cfg.max_lag, g_cfg.lag_criterion, g_cfg.prune,
                          cfg.seed)
    t1 = time.time()
    gr = rolling_granger(panel, g.ends, g.lags, g_cfg.window, cfg.granger.alpha)
    t2 = time.time()
    g.save(d / "varlingam.npz")
    gr.save(d / "granger.npz")
    write_json(d / "meta.json", {"n_windows": len(g.ends), "first_end": str(g.ends[0].date()),
                                 "last_end": str(g.ends[-1].date()), "varlingam_secs": round(t1 - t0, 1),
                                 "granger_secs": round(t2 - t1, 1),
                                 "lag_counts": pd.Series(g.lags).value_counts().sort_index().to_dict()})
    log.info("graphs: %d windows (VarLiNGAM %.0fs, Granger %.0fs)", len(g.ends), t1 - t0, t2 - t1)
    return g, gr, d


def build_features(cfg: Config, g: RollingGraphs, gr: RollingGranger) -> pd.DataFrame:
    pdir = processed_dir(cfg)
    rets = pd.read_parquet(pdir / "returns.parquet")
    ctrl_pit = pd.read_parquet(pdir / "controls_pit.parquet")
    sessions = rets.index
    row_avail = pd.read_parquet(pdir / "graph_avail.parquet")["available"]
    avail = window_availability(g.ends, row_avail, cfg.graph.window)
    vl_w = add_smoothed(varlingam_instability(g, cfg.graph.edge_threshold))
    vl = to_daily(vl_w, sessions, avail)
    gc = to_daily(add_smoothed(granger_instability(gr)), sessions, avail.reindex(gr.ends))
    src = source_window(pd.DatetimeIndex(vl_w.index), sessions, avail)
    vl["graph_window_end"] = pd.Series(np.where(src >= 0, vl_w.index.values[np.maximum(src, 0)],
                                                np.datetime64("NaT")), index=sessions)
    ctrl = control_features(rets, ctrl_pit, cfg.regime_series, cfg.regime.vol_window)
    base = pd.DataFrame({"vix": vix_score(ctrl_pit), "nfci": nfci_score(ctrl_pit),
                         "corr_var": corr_var_score(rets[cfg.etfs])})
    return pd.concat([vl, gc, ctrl, base], axis=1)


def regime_series(cfg: Config) -> pd.Series:
    from riskcausal.data.transforms import realized_vol

    rets = pd.read_parquet(processed_dir(cfg) / "returns.parquet")
    return np.log(realized_vol(rets[cfg.regime_series], cfg.regime.vol_window)).dropna()


# ---------------------------------------------------------------- walk-forward
def _labels(y: pd.Series, end: pd.Timestamp, cfg: Config, sessions: pd.DatetimeIndex):
    r = cfg.regime
    fit = fit_ms(y.loc[:end], r.n_regimes, seed=cfg.seed)
    L = label_set(fit.smoothed, r.min_stress_days, r.min_calm_days, cfg.target.horizon).reindex(sessions)
    return fit, L


def _make_model(name: str, seed: int):
    return BayesLogit(seed=seed) if name == "bayes_logit" else GBM(seed=seed)


def _fit_predict(name, X_tr, y_tr, X_pr, seed):
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    m = _make_model(name, seed).fit((X_tr - mu) / sd, y_tr)
    Xs = (X_pr - mu) / sd
    draws = m.predict_draws(Xs) if name == "bayes_logit" else None
    return m, m.predict_proba(Xs), draws


def alert_rules(cfg: Config) -> dict[str, float | None]:
    """Rule name -> alert-day budget. 'h0' = original M2 rule; 'm3' = approved post-hoc rule (primary)."""
    rules = {"h0": None}
    if cfg.alerts.max_alert_day_share is not None:
        rules["m3"] = cfg.alerts.max_alert_day_share
    return rules


def primary_rule(cfg: Config) -> str:
    return "m3" if cfg.alerts.max_alert_day_share is not None else "h0"


def _thresholds(cfg: Config, score: pd.Series, val_truth, h: int) -> tuple[dict, dict]:
    thr, info = {}, {}
    for rule, budget in alert_rules(cfg).items():
        thr[rule], info[rule] = choose_threshold(score, *val_truth, h, MERGE_GAP, cfg.alerts.max_false_alarms_per_year,
                                                 budget)
    return thr, info


def _record(cfg: Config, sessions: pd.DatetimeIndex, ctx: dict, method: str, score_val: pd.Series,
            score_test: pd.Series, extra: dict) -> dict:
    """Thresholds from validation scores under every rule; store test and validation scores for one method."""
    val, test, L_out = ctx["val"], ctx["test"], ctx["L_out"]
    thr, tinfo = _thresholds(cfg, score_val, ctx["val_truth"], cfg.target.horizon)
    ctx["preds"].append(pd.DataFrame({"date": sessions[test], "fold": ctx["k"], "method": method,
                                      "score": score_test.to_numpy(float),
                                      **{f"threshold_{r}": v for r, v in thr.items()}}))
    ctx["val_preds"].append(pd.DataFrame({"date": sessions[val], "fold": ctx["k"], "method": method,
                                          "score": score_val.to_numpy(float),
                                          "target": L_out["target"][val].to_numpy(float),
                                          "stress": L_out["stress"][val].to_numpy(float)}))
    ctx["info"]["methods"][method] = {"thresholds": thr, "threshold_info": tinfo, **extra}
    return thr


def run_walkforward(cfg: Config, feats: pd.DataFrame, y_rv: pd.Series, models=MODELS, scores=SCORES):
    """Walk-forward over annual test folds. Everything fit for fold k uses data <= train_end(k) only."""
    sessions = feats.index
    h = cfg.target.horizon
    first_feat_year = feats[FEATURE_SETS["combined"]].dropna().index[0].year
    folds = make_folds(sessions, first_feat_year + cfg.walkforward.min_train_years, sessions[-1].year,
                       cfg.walkforward.embargo_days, VAL_YEARS)
    preds, val_preds, fold_info, draws_store, filt_parts = [], [], [], {}, []
    for f in folds:
        t0 = time.time()
        ms_out, L_out = _labels(y_rv, f.train_end, cfg, sessions)
        _, L_in = _labels(y_rv, f.inner_end, cfg, sessions)
        val = (sessions >= f.val_start) & (sessions <= f.train_end)
        test = (sessions >= f.test_start) & (sessions <= f.test_end)
        val_truth = (L_out["stress"][val].fillna(0), L_out["onset"][val].fillna(0))
        info = {**f.as_dict(), "ms_regime_vol": [float(np.exp(m)) for m in ms_out.regime_means],
                "ms_llf": ms_out.llf, "n_onsets_train": int(L_out["onset"].loc[:f.train_end].sum()), "methods": {}}

        # sensitivity truth: filtered probabilities with training parameters, data <= t only
        fp = filtered_stress_prob(y_rv.loc[:f.test_end], ms_out, cfg.regime.n_regimes)
        Lf = label_set(fp, cfg.regime.min_stress_days, cfg.regime.min_calm_days, h).reindex(sessions)
        filt_parts.append(Lf[test])

        ctx = {"k": f.k, "val": val, "test": test, "val_truth": val_truth, "L_out": L_out, "info": info,
               "preds": preds, "val_preds": val_preds}
        record = functools.partial(_record, cfg, sessions, ctx)

        for name, fs in models:
            cols = FEATURE_SETS[fs]
            X = feats[cols]
            complete = X.notna().all(axis=1).to_numpy()
            method = f"{name}:{fs}"
            tr_in = (sessions <= f.inner_end) & complete & L_in["target"].notna().to_numpy()
            tr_out = (sessions <= f.train_end) & complete & L_out["target"].notna().to_numpy()
            vmask, tmask = val & complete, test & complete
            # inner fit (<= inner_end) -> validation scores -> thresholds
            _, pv, _ = _fit_predict(name, X[tr_in].to_numpy(), L_in["target"][tr_in].to_numpy().astype(int),
                                    X[vmask].to_numpy(), cfg.seed)
            sv = pd.Series(np.nan, index=sessions[val])
            sv.loc[sessions[vmask]] = pv
            # outer fit (<= train_end) -> test scores
            m, pt, dr = _fit_predict(name, X[tr_out].to_numpy(), L_out["target"][tr_out].to_numpy().astype(int),
                                     X[tmask].to_numpy(), cfg.seed)
            st = pd.Series(np.nan, index=sessions[test])
            st.loc[sessions[tmask]] = pt
            thr = record(method, sv, st, {"n_train": int(tr_out.sum()),
                                          "n_train_pos": int(L_out["target"][tr_out].sum()),
                                          "n_inner_train": int(tr_in.sum()), **m.summary(cols)})
            if dr is not None:
                full = np.full((dr.shape[0], test.sum()), np.nan)
                full[:, tmask[test]] = dr
                draws_store[(method, f.k)] = (full, thr)

        # pre-specified score methods (no fitting beyond standardisation on rows <= inner_end)
        zref = feats.loc[:f.inner_end, COMPOSITE]
        comp = ((feats[COMPOSITE] - zref.mean()) / zref.std()).mean(axis=1, skipna=False)
        for method in scores:
            s = comp if method == "inst_composite" else feats[method]
            record(method, s[val], s[test], {})
        fold_info.append(info)
        log.info("fold %d (test %d) done in %.0fs", f.k, f.test_year, time.time() - t0)

    P = pd.concat(preds, ignore_index=True)
    for rule in alert_rules(cfg):
        thr = P[f"threshold_{rule}"]
        P[f"alert_{rule}"] = (P["score"] >= thr) & P["score"].notna() & np.isfinite(thr)
    pr = primary_rule(cfg)
    P["threshold"], P["alert"] = P[f"threshold_{pr}"], P[f"alert_{pr}"]
    return P, pd.concat(val_preds, ignore_index=True), fold_info, draws_store, pd.concat(filt_parts)


# ---------------------------------------------------------------- evaluation
def _evaluate_span(P: pd.DataFrame, truth: pd.DataFrame, h: int, dates: pd.DatetimeIndex, rule: str) -> dict:
    out = {}
    tr = truth.reindex(dates)
    st, on, tg = tr["stress"].fillna(0).to_numpy(), tr["onset"].fillna(0).to_numpy(), tr["target"].to_numpy()
    n_years = len(dates) / 252
    for method, g in P.groupby("method"):
        g = g.set_index("date").reindex(dates)
        alert = g[f"alert_{rule}"].fillna(False).to_numpy(bool)
        ev = event_metrics(alert, st, on, h, MERGE_GAP, n_years)
        ev["calm_alert_share"] = float(alert[st == 0].mean()) if (st == 0).any() else None
        dm = day_metrics(g["score"].to_numpy(float), np.where(st == 0, tg, np.nan),
                         probabilistic=method.startswith(("bayes_logit", "gbm")))
        out[method] = {"event": ev, "day": dm, "score_coverage": float(g["score"].notna().mean())}
    return out


def validation_metrics(V: pd.DataFrame) -> dict:
    """Day-level metrics on each fold's inner validation span (truth = labels fit <= train_end), summarised as the
    median across folds — the pre-registered selection statistic for M4 (§F4). Validation spans of consecutive
    folds overlap, so folds are summarised, not pooled."""
    out = {}
    for method, g in V.groupby("method"):
        per = []
        for _, gf in g.groupby("fold"):
            tg = np.where(gf["stress"].to_numpy() == 0, gf["target"].to_numpy(), np.nan)
            d = day_metrics(gf["score"].to_numpy(float), tg, probabilistic=False)
            if "roc_auc" in d:
                per.append(d)
        out[method] = {"n_folds": len(per),
                       "median_roc_auc": float(np.median([d["roc_auc"] for d in per])) if per else None,
                       "median_pr_auc": float(np.median([d["pr_auc"] for d in per])) if per else None,
                       "median_base_rate": float(np.median([d["base_rate"] for d in per])) if per else None}
    return out


def bayes_lead_intervals(draws_store: dict, truth: pd.DataFrame, P: pd.DataFrame, h: int, rule: str) -> dict:
    """Per-onset detection probability and lead-time interval across posterior draws."""
    res = {}
    for method in sorted({m for m, _ in draws_store}):
        parts, idx = [], []
        for (m, k), (dr, thrs) in sorted(draws_store.items(), key=lambda x: x[0][1]):
            if m != method:
                continue
            thr = thrs[rule]
            parts.append(np.where(np.isfinite(thr), dr >= thr, False) & ~np.isnan(dr))
            idx.append(P[(P["method"] == m) & (P["fold"] == k)]["date"].to_numpy())
        A = np.concatenate(parts, axis=1)
        dates = pd.DatetimeIndex(np.concatenate(idx))
        tr = truth.reindex(dates)
        st, on = tr["stress"].fillna(0).to_numpy(), tr["onset"].fillna(0).to_numpy()
        per = [event_metrics(A[d], st, on, h, MERGE_GAP, len(dates) / 252) for d in range(A.shape[0])]
        onset_dates = dates[np.flatnonzero(on)][[o >= h for o in np.flatnonzero(on)]]
        per_onset = []
        for j, od in enumerate(onset_dates):
            leads = [p["lead_times"][j] for p in per if p["lead_times"][j] is not None]
            per_onset.append({"onset": str(od.date()), "p_detected": len(leads) / len(per),
                              "lead_median": float(np.median(leads)) if leads else None,
                              "lead_q05": float(np.quantile(leads, 0.05)) if leads else None,
                              "lead_q95": float(np.quantile(leads, 0.95)) if leads else None})
        rec = np.array([p["recall"] for p in per], float)
        fa = np.array([p["false_alarms_per_year"] for p in per], float)
        res[method] = {"n_draws": len(per), "recall_q05_q50_q95": np.nanquantile(rec, [.05, .5, .95]).tolist(),
                       "fa_per_year_q05_q50_q95": np.nanquantile(fa, [.05, .5, .95]).tolist(),
                       "per_onset": per_onset}
    return res


def run_pipeline(cfg: Config, rebuild_panel: bool = True) -> Path:
    t_start = time.time()
    if rebuild_panel or not (processed_dir(cfg) / "graph_panel.parquet").exists():
        build_panel(cfg)
    g, gr, gdir = build_graphs(cfg)
    feats = build_features(cfg, g, gr)
    y_rv = regime_series(cfg)
    sessions = feats.index

    P, V, fold_info, draws_store, label_filt = run_walkforward(cfg, feats, y_rv)

    # retrospective truth: built only after every prediction and threshold exists (scoring only)
    r = cfg.regime
    ms_full = fit_ms(y_rv, r.n_regimes, seed=cfg.seed)
    label_eval = label_set(ms_full.smoothed, r.min_stress_days, r.min_calm_days, cfg.target.horizon
                           ).reindex(sessions)
    run_dir = new_run_dir(cfg, tag=cfg.experiment or "")
    test_dates = pd.DatetimeIndex(sorted(P["date"].unique()))
    rules = list(alert_rules(cfg))
    valid_folds = [fi["k"] for fi in fold_info
                   if all(np.isfinite(v["thresholds"][rl]) for v in fi["methods"].values() for rl in rules)]
    common_dates = pd.DatetimeIndex(sorted(P.loc[P["fold"].isin(valid_folds), "date"].unique()))
    h = cfg.target.horizon
    by_rule = {}
    for rl in rules:
        by_rule[rl] = {
            "full_span": _evaluate_span(P, label_eval, h, test_dates, rl),
            "common_span_metrics": _evaluate_span(P, label_eval, h, common_dates, rl) if len(common_dates) else {},
            "sensitivity_filtered_labels": _evaluate_span(P, label_filt, h, test_dates, rl),
            "bayes_lead_intervals": bayes_lead_intervals(draws_store, label_eval, P, h, rl),
        }
    metrics = {
        "truth": "label_eval (retrospective full-sample smoothed Markov labels; scoring only)",
        "experiment": cfg.experiment, "n_regimes": r.n_regimes,
        "alert_rules": {rl: {"max_fa_per_year": cfg.alerts.max_false_alarms_per_year, "max_calm_alert_share": b}
                        for rl, b in alert_rules(cfg).items()},
        "primary_rule": primary_rule(cfg),
        "test_span": [str(test_dates[0].date()), str(test_dates[-1].date())],
        "common_span_folds": valid_folds,
        "common_span": [str(common_dates[0].date()), str(common_dates[-1].date())] if len(common_dates) else None,
        "by_rule": by_rule,
        "validation_median_over_folds": validation_metrics(V),
        "label_eval_summary": {
            "regime_annualised_vol": [float(np.exp(m)) for m in ms_full.regime_means],
            "stress_share": float(label_eval["stress"].mean()),
            "onsets": [str(d.date()) for d in label_eval.index[label_eval["onset"] == 1]],
        },
    }
    P.to_parquet(run_dir / "predictions.parquet")
    V.to_parquet(run_dir / "predictions_val.parquet")
    feats.to_parquet(run_dir / "features.parquet")
    label_eval.to_parquet(run_dir / "labels_eval.parquet")
    label_filt.to_parquet(run_dir / "labels_filtered_test.parquet")
    write_json(run_dir / "folds.json", fold_info)
    write_json(run_dir / "graph_cache.json", {"dir": str(gdir)})
    from riskcausal.audit import audit_run

    metrics["audit"] = audit_run(cfg, run_dir, P, feats, fold_info, g)
    metrics["runtime_secs"] = round(time.time() - t_start, 1)
    write_json(run_dir / "metrics.json", metrics)
    from riskcausal.evaluation.plots import make_figures
    from riskcausal.evaluation.report import write_report

    make_figures(run_dir)
    write_report(run_dir)
    log.info("run complete: %s (%.0fs)", run_dir, time.time() - t_start)
    return run_dir
