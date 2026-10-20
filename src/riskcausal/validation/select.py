"""M4 validation-only selection (configs/experiments/robustness.yaml, pre-registered).

Nothing here reads a test block: for every fold, models are fit on rows <= inner_end, scored on the validation
span (val_start..train_end), and judged against labels fit on data <= train_end. Results are written to a run
directory tagged `m4select`; the selected settings are written as an experiment overlay for the locked run.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import yaml

from riskcausal.config import CONFIG_DIR, Config, load_config
from riskcausal.evaluation.metrics import day_metrics
from riskcausal.regimes.markov import predictive_loglik
from riskcausal.regimes.targets import label_set, states_from_prob
from riskcausal.registry import new_run_dir, write_json
from riskcausal.validation.walkforward import Fold, make_folds

log = logging.getLogger(__name__)

GRID_MODELS = [("gbm", "instability"), ("map_logit", "instability")]
DESCRIPTIVE_MODELS = [("gbm", "combined"), ("map_logit", "combined"), ("gbm", "controls"), ("map_logit", "controls")]
TIE_MARGIN = 0.005


def _folds(cfg: Config, feats: pd.DataFrame) -> list[Fold]:
    from riskcausal.pipeline import FEATURE_SETS, VAL_YEARS

    s = feats.index
    first = feats[FEATURE_SETS["combined"]].dropna().index[0].year
    return make_folds(s, first + cfg.walkforward.min_train_years, s[-1].year, cfg.walkforward.embargo_days, VAL_YEARS)


def fold_fits(cfg: Config, y: pd.Series, folds: list[Fold]) -> dict:
    """Markov fits per fold on data <= inner_end and <= train_end (cached on disk)."""
    from riskcausal.pipeline import _ms_fit

    return {f.k: (_ms_fit(y.loc[:f.inner_end], cfg), _ms_fit(y.loc[:f.train_end], cfg)) for f in folds}


def label_selection(y: pd.Series, folds: list[Fold], base: Config) -> dict:
    out = {}
    for k in (2, 3):
        cfg = base.model_copy(update={"regime": base.regime.model_copy(update={"n_regimes": k})})
        t0 = time.time()
        fits = fold_fits(cfg, y, folds)
        rows = []
        for f in folds:
            fin, fout = fits[f.k]
            pll = predictive_loglik(y, fin, k, f.val_start, f.train_end)
            a = states_from_prob(fin.smoothed, cfg.regime.min_stress_days)
            b = states_from_prob(fout.smoothed.loc[:f.inner_end], cfg.regime.min_stress_days)
            rows.append({"fold": f.k, "val_pred_loglik": pll, "label_agreement": float((a == b.reindex(a.index)).mean()),
                         "stress_share_inner": float(a.mean())})
        df = pd.DataFrame(rows)
        out[k] = {"per_fold": rows, "median_val_pred_loglik": float(df["val_pred_loglik"].median()),
                  "median_label_agreement": float(df["label_agreement"].median()), "secs": round(time.time() - t0, 1)}
        log.info("label selection k=%d: median predictive loglik %.4f (%.0fs)", k,
                 out[k]["median_val_pred_loglik"], out[k]["secs"])
    wins3 = sum(r3["val_pred_loglik"] > r2["val_pred_loglik"] for r2, r3 in zip(out[2]["per_fold"], out[3]["per_fold"]))
    sel = 3 if out[3]["median_val_pred_loglik"] > out[2]["median_val_pred_loglik"] else 2
    return {"by_k": out, "folds_where_3_beats_2": wins3, "n_folds": len(folds), "validation_selected": sel,
            "primary": 2}


def validate_config(cfg: Config, feats: pd.DataFrame, folds: list[Fold], fits: dict, models,
                    h: int | None = None, min_stress: int | None = None) -> pd.DataFrame:
    """Validation day metrics per (fold, method). Uses rows <= train_end only."""
    from riskcausal.pipeline import COMPOSITE, FEATURE_SETS, _fit_predict

    h = h or cfg.target.horizon
    ms = min_stress or cfg.regime.min_stress_days
    s = feats.index
    rows = []
    for f in folds:
        fin, fout = fits[f.k]
        L_in = label_set(fin.smoothed, ms, cfg.regime.min_calm_days, h).reindex(s)
        L_out = label_set(fout.smoothed, ms, cfg.regime.min_calm_days, h).reindex(s)
        val = (s >= f.val_start) & (s <= f.train_end)
        truth = np.where(L_out["stress"][val].to_numpy() == 0, L_out["target"][val].to_numpy(float), np.nan)
        scores = {}
        for name, fs in models:
            X = feats[FEATURE_SETS[fs]]
            complete = X.notna().all(axis=1).to_numpy()
            tr = (s <= f.inner_end) & complete & L_in["target"].notna().to_numpy()
            vm = val & complete
            yt = L_in["target"][tr].to_numpy().astype(int)
            if yt.min() == yt.max():
                continue
            _, pv, _ = _fit_predict(name, X[tr].to_numpy(), yt, X[vm].to_numpy(), cfg.seed)
            sc = pd.Series(np.nan, index=s[val])
            sc.loc[s[vm]] = pv
            scores[f"{name}:{fs}"] = sc.to_numpy()
        zref = feats.loc[:f.inner_end, COMPOSITE]
        comp = ((feats[COMPOSITE] - zref.mean()) / zref.std()).mean(axis=1, skipna=False)
        scores["inst_composite"] = comp[val].to_numpy()
        scores["vix"] = feats["vix"][val].to_numpy()
        for m, sc in scores.items():
            d = day_metrics(sc.astype(float), truth, probabilistic=False)
            if "roc_auc" in d:
                rows.append({"fold": f.k, "method": m, **d})
    return pd.DataFrame(rows)


def _summ(df: pd.DataFrame) -> dict:
    g = df.groupby("method")
    return {m: {"n_folds": len(x), "median_pr_auc": float(x["pr_auc"].median()),
                "median_roc_auc": float(x["roc_auc"].median()), "median_base_rate": float(x["base_rate"].median())}
            for m, x in g}


def run_selection(profile: str = "dev") -> dict:
    from riskcausal.pipeline import build_features, build_graphs, regime_series

    base = load_config(profile)
    run_dir = new_run_dir(base, tag="m4select")
    pre = yaml.safe_load((CONFIG_DIR / "experiments" / "robustness.yaml").read_text())
    y = regime_series(base)

    # graph grid features (graph caches are keyed by window; edge threshold only changes features)
    feats_by = {}
    for W in pre["graph_grid"]["window"]:
        for thr in pre["graph_grid"]["edge_threshold"]:
            c = load_config(profile, overrides={"graph": {"window": W, "edge_threshold": thr}})
            g, gr, _ = build_graphs(c)
            feats_by[(W, thr)] = build_features(c, g, gr)
    folds = _folds(base, feats_by[(base.graph.window, base.graph.edge_threshold)])

    labels = label_selection(y, folds, base)
    grid_labels = [2] + ([3] if labels["validation_selected"] == 3 else [])

    grid = {}
    for k in grid_labels:
        ck = base.model_copy(update={"regime": base.regime.model_copy(update={"n_regimes": k})})
        fits = fold_fits(ck, y, folds)
        res = {}
        for (W, thr), feats in feats_by.items():
            t0 = time.time()
            df = validate_config(ck, feats, folds, fits, GRID_MODELS + DESCRIPTIVE_MODELS)
            sm = _summ(df)
            stat = float(np.mean([sm[f"{n}:{fs}"]["median_pr_auc"] for n, fs in GRID_MODELS]))
            res[f"W{W}_e{thr}"] = {"window": W, "edge_threshold": thr, "statistic": stat, "summary": sm}
            df.assign(config=f"W{W}_e{thr}", n_regimes=k).to_csv(run_dir / f"grid_k{k}_W{W}_e{thr}.csv", index=False)
            log.info("grid k=%d W=%d edge=%.2f: statistic %.4f (%.0fs)", k, W, thr, stat, time.time() - t0)
        h0 = f"W{base.graph.window}_e{base.graph.edge_threshold}"
        best = max(res, key=lambda c: res[c]["statistic"])
        chosen = h0 if res[best]["statistic"] - res[h0]["statistic"] <= TIE_MARGIN else best
        grid[k] = {"configs": res, "best": best, "h0": h0, "selected": chosen}

    # target sensitivity: validation only, selected config, primary labels; no selection
    sel = grid[2]["configs"][grid[2]["selected"]]
    feats = feats_by[(sel["window"], sel["edge_threshold"])]
    fits2 = fold_fits(base, y, folds)
    sens = {}
    for key, vals in pre["target_sensitivity"]["vary_one_at_a_time"].items():
        for v in vals:
            kw = {"h": v} if key == "horizon" else {"min_stress": v}
            df = validate_config(base, feats, folds, fits2, GRID_MODELS + DESCRIPTIVE_MODELS, **kw)
            sens[f"{key}={v}"] = _summ(df)
            log.info("target sensitivity %s=%s done", key, v)

    overlay = {"description": f"M4 validation-selected graph settings (generated by run {run_dir.name}; "
                              "pre-registered rule in robustness.yaml)",
               "graph": {"window": int(sel["window"]), "edge_threshold": float(sel["edge_threshold"])}}
    (CONFIG_DIR / "experiments" / "m4_selected.yaml").write_text(yaml.safe_dump(overlay, sort_keys=False))
    out = {"label_selection": labels, "graph_grid": grid, "target_sensitivity": sens,
           "selected_overlay": overlay, "folds": [f.as_dict() for f in folds]}
    write_json(run_dir / "selection.json", out)
    _write_md(run_dir, out)
    log.info("selection written to %s", run_dir)
    return out


def _write_md(run_dir, S: dict) -> None:
    L = [f"# M4 validation-only selection — `{run_dir.name}`", "",
         "_Generated from `selection.json`. Inner-validation spans only; no test block is read._", "",
         "## Label model (2 vs 3 regimes)", "",
         "| regimes | median validation predictive log-lik / obs | median label agreement (inner vs outer fit) |",
         "|---|---|---|"]
    for k, r in S["label_selection"]["by_k"].items():
        L.append(f"| {k} | {r['median_val_pred_loglik']:.4f} | {r['median_label_agreement']:.3f} |")
    ls = S["label_selection"]
    L += ["", (f"3 regimes beat 2 in {ls['folds_where_3_beats_2']} of {ls['n_folds']} folds. "
          f"Validation-selected: **{ls['validation_selected']} regimes**; primary (team decision): 2 regimes."), ""]
    for k, g in S["graph_grid"].items():
        L += [f"## Graph grid — {k}-regime labels", "",
              ("| config | statistic (mean of median val PR-AUC) | gbm:instability | map_logit:instability | "
              "map_logit:combined | map_logit:controls | composite | VIX |"), "|---|---|---|---|---|---|---|---|"]
        for c, r in g["configs"].items():
            sm = r["summary"]
            def pr(m, sm=sm):
                return f"{sm[m]['median_pr_auc']:.3f}" if m in sm else "—"
            L.append(f"| {c} | {r['statistic']:.4f} | {pr('gbm:instability')} | {pr('map_logit:instability')} | "
                     f"{pr('map_logit:combined')} | {pr('map_logit:controls')} | {pr('inst_composite')} | {pr('vix')} |")
        L += ["", (f"Best: `{g['best']}`; H0: `{g['h0']}`; **selected: `{g['selected']}`** "
              f"(H0 kept if within {TIE_MARGIN})."), ""]
    L += ["## Target-definition sensitivity (validation, selected config, 2 regimes; descriptive only)", "",
          ("| variant | map_logit:instability | map_logit:combined | map_logit:controls | gbm:instability | VIX | "
          "median base rate |"), "|---|---|---|---|---|---|---|"]
    for v, sm in S["target_sensitivity"].items():
        def pr(m, sm=sm):
            return f"{sm[m]['median_pr_auc']:.3f}" if m in sm else "—"
        L.append(f"| {v} | {pr('map_logit:instability')} | {pr('map_logit:combined')} | {pr('map_logit:controls')} "
                 f"| {pr('gbm:instability')} | {pr('vix')} | {sm['vix']['median_base_rate']:.3f} |")
    L += ["", "Values are median-over-folds validation PR-AUC (calm days)."]
    (run_dir / "selection.md").write_text("\n".join(L))
