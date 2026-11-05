"""Read-only local API over saved runs (M7). Binds 127.0.0.1 (api.host is validated as local). All routes are GET,
except the two M10 routes that start / cancel a live local-LLM agent job (`agents_live.py`); those write nothing to
disk and never change a score, alert or metric.

Every response is `{"meta": {run_id, role, git_sha, synthetic, audit_passed, kind}, "data": ...}` where `kind` is
`research_result` (from metrics.json / ci.json / predictions), `evidence` (point-in-time inputs, deterministic
descriptions), `illustrative` (graph snapshots) or `unverified_draft` (LLM explanations). Synthetic runs are
refused as results; runs whose leakage audit failed are flagged in every response.

Run locally: `.venv/bin/uvicorn riskcausal_api.main:app --host 127.0.0.1 --port 8000`
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from riskcausal_api import describe
from riskcausal_api.store import MERGE_GAP, Run, dashboard_config, reproducibility, runs

app = FastAPI(title="riskcausal (read-only, local)", version="m7",
              description="Saved research results of the FSE 570 capstone. Read-only; no model is run.")


def _run(run_id: str, allow_synthetic: bool = False) -> Run:
    r = runs().get(run_id)
    if r is None:
        raise HTTPException(404, f"unknown run {run_id}")
    if r.synthetic and not allow_synthetic:
        raise HTTPException(409, "synthetic run: never presented as a research result")
    return r


def _out(r: Run, kind: str, data) -> dict:
    return {"meta": r.header(kind), "data": data}


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "read_only": True}


@app.get("/runs")
def list_runs() -> dict:
    cfg = dashboard_config()
    out = []
    for r in runs().values():
        m = r.metrics
        out.append({**r.header("metadata"), "profile": r.config["profile"], "experiment": m.get("experiment"),
                    "test_span": m["test_span"], "target_series": m.get("target_series", "SPY"),
                    "n_regimes": m.get("n_regimes"), "primary_rule": m["primary_rule"],
                    "has_ci": r.ci is not None})
    return {"runs": out, "reproducibility_of_primary": reproducibility(),
            "explained_method": cfg["explained_method"], "headline_methods": cfg["headline_methods"]}


@app.get("/runs/{run_id}/metrics")
def metrics(run_id: str) -> dict:
    r = _run(run_id)
    m, ci = r.metrics, r.ci
    rule = m["primary_rule"]
    rows = []
    for meth, v in m["by_rule"][rule]["common_span_metrics"].items():
        e, d = v["event"], v["day"]
        c = (ci or {}).get("methods", {}).get(meth, {})
        ev = (ci or {}).get("event_level_primary_rule", {}).get(meth, {})
        rows.append({"method": meth, "roc_auc": d.get("roc_auc"), "roc_ci95": c.get("roc", {}).get("ci95"),
                     "pr_auc": d.get("pr_auc"), "pr_ci95": c.get("pr", {}).get("ci95"),
                     "base_rate": d.get("base_rate"), "n_onsets": e["n_onsets"], "n_detected": e["n_detected"],
                     "recall": e["recall"], "recall_ci95": ev.get("recall_ci95"), "precision": e["precision"],
                     "precision_ci95": ev.get("precision_ci95"), "false_alarms_per_year": e["false_alarms_per_year"],
                     "calm_alert_share": e.get("calm_alert_share"), "median_lead": e["median_lead"]})
    rows.sort(key=lambda x: -(x["roc_auc"] or 0))
    diffs = [{"comparison": k, "delta_roc": v["roc"]["point"], "roc_ci95": v["roc"]["ci95"],
              "excludes_zero": v["roc"]["supported"], "delta_pr": v["pr"]["point"], "pr_ci95": v["pr"]["ci95"]}
             for k, v in (ci or {}).get("differences", {}).items()]
    return _out(r, "research_result", _clean({
        "span": m["common_span"], "alert_rule": rule, "alert_rule_spec": m["alert_rules"][rule],
        "truth": m["truth"], "ci_method": None if not ci else
        f"paired stationary block bootstrap, {ci['n_rep']} reps, mean block {ci['mean_block']:g}; Jeffreys for events",
        "methods": rows, "differences": diffs, "audit": m.get("audit", {}).get("checks", []),
        "label_summary": m["label_eval_summary"]}))


@app.get("/runs/{run_id}/timeline")
def timeline(run_id: str, method: str) -> dict:
    r = _run(run_id)
    g = r.predictions[r.predictions["method"] == method].sort_values("date")
    if g.empty:
        raise HTTPException(404, f"no predictions for {method}")
    L = r.labels.reindex(pd.DatetimeIndex(g["date"]))
    data = {"method": method, "dates": [str(d.date()) for d in g["date"]], "score": g["score"].tolist(),
            "threshold": g["threshold"].tolist(), "alert": g["alert"].astype(bool).tolist(),
            "retrospective_stress": L["stress"].tolist(), "retrospective_onset": L["onset"].tolist(),
            "target_rv": L["target_rv"].tolist() if "target_rv" in L else None,
            "target_series": r.metrics.get("target_series", "SPY"),
            "note": "retrospective_* fields are full-sample labels known only after the fact; used for scoring"}
    return _out(r, "research_result", _clean(data))


@app.get("/runs/{run_id}/alerts")
def alerts(run_id: str, method: str) -> dict:
    from riskcausal.evaluation.metrics import alert_episodes

    r = _run(run_id)
    g = r.predictions[r.predictions["method"] == method].sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(g["date"])
    L = r.labels.reindex(dates)
    st, on = L["stress"].fillna(0).to_numpy(), np.flatnonzero(L["onset"].fillna(0).to_numpy())
    h = int(r.config["target"]["horizon"])
    eps = []
    for s, e in alert_episodes(g["alert"].to_numpy(bool), MERGE_GAP):
        outcome = "in_stress" if st[s] == 1 else ("true_alert" if ((on > s) & (on <= e + h)).any() else "false_alarm")
        eps.append({"start": str(dates[s].date()), "end": str(dates[e].date()), "outcome_retrospective": outcome,
                    "peak_score": float(g["score"].iloc[s:e + 1].max())})
    return _out(r, "research_result", {"method": method, "episodes": eps, "merge_gap": MERGE_GAP})


def _feature_pct(r: Run, col: str, t: pd.Timestamp) -> float | None:
    s = r.features[col]
    hist = s.loc[t - pd.DateOffset(years=5):t].dropna()
    return None if len(hist) < 100 or pd.isna(s.get(t)) else float((hist <= s.loc[t]).mean() * 100)


@app.get("/runs/{run_id}/graph/{date}")
def graph(run_id: str, date: str) -> dict:
    r = _run(run_id)
    t = pd.Timestamp(date)
    if t not in r.features.index:
        raise HTTPException(404, f"{date} is not a session in this run")
    we = r.features["graph_window_end"].get(t)
    if we is None or pd.isna(we):
        raise HTTPException(404, f"no graph window available at {date}")
    G = r.graphs
    w = G["ends"].get_loc(pd.Timestamp(we))
    nodes, B = G["nodes"], G["B"][w, 0]
    thr = float(r.config["graph"]["edge_threshold"])
    edges = [{"source": nodes[j], "target": nodes[i], "coef": float(B[i, j])}
             for i in range(len(nodes)) for j in range(len(nodes)) if i != j and abs(B[i, j]) > thr]
    changes = []
    if w > 0:
        P = G["B"][w - 1, 0]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    c = describe.edge_change(nodes[j], nodes[i], float(P[i, j]), float(B[i, j]), tol=thr)
                    if c["change"] != "absent" and abs(B[i, j] - P[i, j]) >= 0.01:
                        changes.append(c)
        changes.sort(key=lambda c: -abs(c["after"] - c["before"]))
    feats = {c: {"value": float(r.features[c].loc[t]), "pct": _feature_pct(r, c, t)}
             for c in ["vl_shd_b0", "vl_frob_b0", "vl_n_edges_b0", "vl_order_change", "vl_frob_b0_ma4"]
             if c in r.features and not pd.isna(r.features[c].loc[t])}
    return _out(r, "illustrative", _clean({
        "date": date, "window_end": str(pd.Timestamp(we).date()),
        "previous_window_end": str(G["ends"][w - 1].date()) if w > 0 else None,
        "window_length": int(r.config["graph"]["window"]), "edge_threshold": thr, "nodes": nodes,
        "B0": B.round(4).tolist(), "edges": edges, "changes": changes, "instability_features": feats,
        "caveat": ("Estimated directed dependence (VarLiNGAM, contemporaneous B0), not causality. Bootstrap refits of "
                   "one window disagree on ~21 of 72 edges (dev profile noise floor), so single changes are noisy."),
    }))


@lru_cache(maxsize=4)
def _run_evidence(run_id: str):
    from riskcausal.config import PROJECT_ROOT
    from riskcausal_agents.evidence import RunEvidence

    nd = PROJECT_ROOT / "data" / "raw" / "fnspid" / run_id
    news = end = None
    if (nd / "news_subset.parquet").exists():
        news = pd.read_parquet(nd / "news_subset.parquet")
        end = pd.Timestamp(json.loads((nd / "manifest.json").read_text())["max_utc"])
    return RunEvidence(runs()[run_id].dir, news, end)


@app.get("/runs/{run_id}/evidence/{method}/{date}")
def evidence(run_id: str, method: str, date: str) -> dict:
    """Point-in-time evidence for one decision date with deterministic descriptions (the primary explanation)."""
    r = _run(run_id)
    try:
        b = _run_evidence(run_id).bundle(method, date)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    items = []
    for e in b.items.values():
        desc = e.text
        if e.kind == "alert":
            desc = describe.alert(method, e.values["score"], e.values["threshold"])
        elif e.kind == "macro":
            series = e.id.split(":")[1].split("@")[0]
            desc = describe.indicator(series, e.values["value"], e.values.get("pct"))
        elif e.id.startswith("edge:"):
            src, dst = e.id.split(":", 1)[1].split("@")[0].split("->")
            desc = describe.edge_change(src, dst, e.values["before"], e.values["after"],
                                        tol=float(r.config["graph"]["edge_threshold"]))["text"]
        items.append({"id": e.id, "kind": e.kind, "available": e.available, "description": desc,
                      "source_text": e.text, "values": e.values})
    return _out(r, "evidence", _clean({"method": method, "date": b.date, "cutoff_utc": b.cutoff_utc, "items": items,
                                       "coverage_notes": b.notes, "method_reliability_retrospective": b.reliability}))


def _drafts(r: Run) -> dict[str, dict]:
    """Hand-check rows for the evaluated drafts, keyed by alert id (tuning set + held-out set)."""
    out = {}
    for name in ("eval", "heldout"):
        p = r.dir / "explanations" / name / "eval_set.csv"
        if p.exists():
            for _, row in pd.read_csv(p).iterrows():
                out[row["alert_id"]] = {**{k: (None if pd.isna(v) else v) for k, v in row.items()}, "set": name}
    return out


def _status(rec: dict, hc: dict | None) -> tuple[str, list[str]]:
    flags = []
    ex = rec["explanation"]
    if ex["abstain"] or not ex["claims"]:
        return "suppressed: no claim passed validation", [ex.get("abstain_reason") or "abstained"]
    if hc:
        k, n = (int(x) for x in str(hc.get("R2_faithful_claims") or "0/0").split("/"))
        if n and k < n:
            flags.append(f"hand check: {n - k} of {n} claims unfaithful to their evidence")
        for col, label in [("R3_hindsight_free", "hindsight"), ("R5_uncertainty_adequate", "uncertainty statement"),
                           ("R6_abstention_appropriate", "abstention")]:
            if str(hc.get(col)) == "0":
                flags.append(f"hand check: {label} inadequate")
        if hc.get("checker_notes"):
            flags.append(f"checker notes: {hc['checker_notes']}")
    return ("unverified draft — flagged" if flags else "unverified draft"), flags


@app.get("/runs/{run_id}/explanations")
def explanations(run_id: str) -> dict:
    r = _run(run_id)
    hcs = _drafts(r)
    out = []
    for aid, hc in sorted(hcs.items()):
        rec = json.loads((r.dir / "explanations" / hc["file"]).read_text())
        status, flags = _status(rec, hc)
        out.append({"alert_id": aid, "date": rec["decision_date"], "set": hc["set"], "stratum": hc["stratum"],
                    "status": status, "n_flags": len(flags), "prompt_version": rec["prompt_version"]})
    return _out(r, "unverified_draft", {"drafts": out, "label": "LLM drafts are unverified; see status and flags"})


@app.get("/runs/{run_id}/explanations/{alert_id}")
def explanation(run_id: str, alert_id: str) -> dict:
    r = _run(run_id)
    hc = _drafts(r).get(alert_id)
    if hc is None:
        raise HTTPException(404, f"no evaluated draft for {alert_id}")
    rec = json.loads((r.dir / "explanations" / hc["file"]).read_text())
    status, flags = _status(rec, hc)
    shown = [] if status.startswith("suppressed") else rec["explanation"]["claims"]
    n_rej = sum(len(s.get("rejected", [])) for s in rec["specialists"]) + len(rec["synthesis"].get("rejected", []))
    return _out(r, "unverified_draft", _clean({
        "alert_id": alert_id, "status": status, "flags": flags, "claims": shown,
        "uncertainty": rec["explanation"]["uncertainty"], "n_claims_rejected_by_validator": n_rej,
        "coverage_notes": rec["coverage_notes"], "model": f"{rec['backend']} {rec['model_digest']}",
        "prompt_version": rec["prompt_version"], "set": hc["set"],
        "hand_check": {k: hc.get(k) for k in ("R2_faithful_claims", "R3_hindsight_free", "R5_uncertainty_adequate",
                                               "R6_abstention_appropriate", "checked_by")},
        "label": "UNVERIFIED LLM DRAFT — not a verified conclusion. The deterministic evidence is authoritative."}))


# ---------------------------------------------------------------- product UI (M8)
from riskcausal_api import agents_live, portfolio, replay

app.include_router(replay.router)
app.include_router(portfolio.router)        # exploratory Portfolio Lens (GET only)
app.include_router(agents_live.router)      # live local-LLM agent runs: the only POST routes (start / cancel a job)

_DIST = __import__("riskcausal.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT / "frontend" / "dist"
if _DIST.is_dir():                                    # built React app, served by the same local read-only server
    from fastapi.staticfiles import StaticFiles

    class _UI(StaticFiles):
        """index.html is never cached (it names the current build's files); hashed assets may be."""

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            if not path.startswith("assets/"):
                resp.headers["Cache-Control"] = "no-cache"
            return resp

    app.mount("/", _UI(directory=_DIST, html=True), name="ui")
