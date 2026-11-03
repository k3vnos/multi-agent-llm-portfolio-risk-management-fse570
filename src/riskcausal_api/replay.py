"""Historical-replay endpoints for the product UI (M8). GET only, read-only, saved artifacts only.

Point-in-time rules for everything returned "as of" a replay date D:
- indicator values are the saved point-in-time features at D (release lags already applied in the pipeline);
- percentiles compare a value with the same series over the previous 5 years up to D only;
- the regime shown is the *filtered* Markov stress probability at D (training-period parameters, data <= D);
  retrospective (full-sample, hindsight) labels are returned only in explicitly named `retrospective_*` fields;
- the graph is the latest estimation window that was fully published by D (saved `graph_window_end`);
- alerts listed are those issued on or before D; their retrospective outcome is not included here.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from riskcausal_api import describe
from riskcausal_api.store import MERGE_GAP, Run, dashboard_config, reproducibility, runs

router = APIRouter(prefix="/replay", tags=["replay"])


def json_safe(fn):
    """Route decorator: convert numpy scalars / NaN in the response to plain JSON values."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from riskcausal_api.main import _clean

        return _clean(fn(*args, **kwargs))

    return wrapper

HIGH, LOW, YEARS, MIN_HIST = 80.0, 20.0, 5, 100
NODE_INFO = {
    "SPY": ("US large-cap stocks", "equity"), "IWM": ("US small-cap stocks", "equity"),
    "EFA": ("Developed-market stocks ex-US", "equity"), "EEM": ("Emerging-market stocks", "equity"),
    "XLF": ("US financial sector", "equity"), "TLT": ("Long-term US Treasuries", "rates"),
    "LQD": ("Investment-grade corporate bonds", "credit"), "HYG": ("High-yield corporate bonds", "credit"),
    "GLD": ("Gold", "commodity"), "UUP": ("US dollar index", "fx"),
    "d_VIXCLS": ("Daily change in VIX (volatility index)", "macro"),
    "d_BAA10Y": ("Daily change in Baa credit spread", "macro"),
    "d_T10Y2Y": ("Daily change in 10y–2y yield curve", "macro"),
}
NAMED_EVENTS = {
    "2020-02-21": ("COVID-19 market stress", "Feb–Mar 2020"),
}


def _primary() -> Run:
    return runs()[dashboard_config()["runs"]["primary"]]


def _explained() -> str:
    return dashboard_config()["explained_method"]


def pit_percentile(s: pd.Series, years: int = YEARS, min_hist: int = MIN_HIST) -> pd.Series:
    """Percentile of each value within the same series over the previous `years` years up to that date (inclusive).
    Uses only values dated <= t. NaN when fewer than `min_hist` observations exist."""
    s = s.dropna()
    v, idx = s.to_numpy(float), s.index
    lo = idx.searchsorted(idx - pd.DateOffset(years=years))
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[lo[i]:i + 1]
        if len(w) >= min_hist:
            out[i] = (w <= v[i]).mean() * 100
    return pd.Series(out, index=idx)


@lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    """Daily replay frame over the primary run's test span (all columns point-in-time except retrospective_*)."""
    r = _primary()
    P, F = r.predictions, r.features
    m = _explained()

    def method(name):
        return P[P["method"] == name].set_index("date").sort_index()

    model, comp = method(m), method("inst_composite")
    dates = model.index
    Lf = pd.read_parquet(r.dir / "labels_filtered_test.parquet")
    Le = r.labels
    df = pd.DataFrame(index=dates)
    df["sii"] = comp["score"]
    df["sii_pct"] = pit_percentile(comp["score"])
    for col, name in [("vix", "vix"), ("nfci", "nfci"), ("vix_chg_5d", "vix_chg_5d"), ("rv21", "rv21"),
                      ("vl_shd_b0", "edge_turnover"), ("vl_shd_b0_ma4", "edge_turnover_ma4")]:
        df[name] = F[col].reindex(dates)
        df[f"{name}_pct"] = pit_percentile(F[col]).reindex(dates)       # history from 2005, <= t only
    df["model_score"], df["model_threshold"], df["model_alert"] = model["score"], model["threshold"], model["alert"]
    df["vix_alert"] = method("vix")["alert"]
    df["regime_p_stress"] = Lf["p_stress"].reindex(dates)              # filtered probability, data <= t
    df["retrospective_stress"] = Le["stress"].reindex(dates)
    df["retrospective_onset"] = Le["onset"].reindex(dates)
    df["graph_window_end"] = F["graph_window_end"].reindex(dates)
    return df


def _date(d: str) -> pd.Timestamp:
    t = pd.Timestamp(d)
    df = _frame()
    if t < df.index[0] or t > df.index[-1]:
        raise HTTPException(404, f"replay covers {df.index[0].date()} to {df.index[-1].date()}")
    return df.index[df.index.searchsorted(t, side="right") - 1]        # last session on or before d


def _num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else float(x)


def status(row: pd.Series) -> dict:
    """Documented rule: ALERT if the monitored model alerted at D; else UNUSUAL if any monitored indicator is above
    its 80th percentile of the previous 5 years; else NORMAL."""
    checks = [("Structural instability (SII)", row["sii_pct"]), ("VIX level", row["vix_pct"]),
              ("5-day VIX change", row["vix_chg_5d_pct"]), ("NFCI financial conditions", row["nfci_pct"]),
              ("Graph edge turnover", row["edge_turnover_pct"])]
    high = [(n, p) for n, p in checks if p is not None and not pd.isna(p) and p > HIGH]
    if bool(row["model_alert"]):
        level, text = "alert", (f"The monitored model ({_explained()}) issued an alert at this close: its score "
                                f"{row['model_score']:.3f} was above its threshold {row['model_threshold']:.3f}.")
    elif high:
        level, text = "unusual", "No model alert, but some indicators are unusually high for the past 5 years."
    else:
        level, text = "normal", "No model alert and all monitored indicators are within their usual 5-year range."
    return {"level": level, "summary": text,
            "unusual": [{"indicator": n, "percentile": round(float(p))} for n, p in high],
            "rule": ("ALERT = monitored model alert on this date; UNUSUAL = no alert but at least one indicator above "
                     "its 80th percentile of the previous 5 years; NORMAL = otherwise. Not a forecast.")}


@router.get("/meta")
@json_safe
def meta() -> dict:
    r, df = _primary(), _frame()
    onsets = [d for d in df.index[df["retrospective_onset"] == 1]]
    events = []
    for d in onsets:
        name, period = NAMED_EVENTS.get(str(d.date()), (f"Stress onset · {d:%b %Y}", f"{d:%b %Y}"))
        start = df.index[max(df.index.get_loc(d) - 25, 0)]
        events.append({"id": str(d.date()), "label": name, "period": period, "replay_start": str(start.date()),
                       "focus": str(d.date()), "target": "primary", "available": True})
    kre = runs()[dashboard_config()["runs"]["exploratory"]]
    if any(o == "2023-03-09" for o in kre.metrics["label_eval_summary"]["onsets"]):
        events.append({"id": "2023-03-09-kre", "label": "Regional banking stress (exploratory)", "period": "Mar 2023",
                       "replay_start": str(df.index[df.index.searchsorted(pd.Timestamp("2023-02-01"))].date()),
                       "focus": "2023-03-09", "target": "exploratory", "available": True,
                       "note": ("A stress onset only under the exploratory KRE banking-stress target; under the primary "
                                "SPY-volatility target March 2023 is not a new onset (2022's high-volatility regime "
                                "had not ended).")})
    events.append({"id": "2008", "label": "Global financial crisis", "period": "2008", "available": False,
                   "note": "Not available: the locked evaluation's test span starts in 2014."})
    events.sort(key=lambda e: e.get("focus", "0000"))
    G = r.graphs
    return {"mode": "historical_replay", "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "monitored_model": _explained(), "events": events,
            "nodes": [{"id": n, "name": NODE_INFO.get(n, (n, "other"))[0], "group": NODE_INFO.get(n, (n, "other"))[1]}
                      for n in G["nodes"]],
            "universe_etfs": list(r.config["etfs"]), "graph_window": int(r.config["graph"]["window"]),
            "graph_step": int(r.config["graph"]["step"]), "edge_threshold": float(r.config["graph"]["edge_threshold"]),
            "news_coverage_end": "2020-06-11", "status_rule": status(df.iloc[-1])["rule"]}


@router.get("/series")
@json_safe
def series(start: str | None = None, end: str | None = None) -> dict:
    df = _frame()
    lo = pd.Timestamp(start) if start else df.index[0]
    hi = _date(end) if end else df.index[-1]
    d = df.loc[lo:hi]
    cols = ["sii", "sii_pct", "vix", "vix_pct", "nfci", "nfci_pct", "model_score", "model_threshold", "regime_p_stress"]
    out = {"dates": [str(x.date()) for x in d.index], **{c: [_num(v) for v in d[c]] for c in cols},
           "model_alert": [bool(x) for x in d["model_alert"]], "vix_alert": [bool(x) for x in d["vix_alert"]],
           "retrospective_stress": [None if pd.isna(x) else int(x) for x in d["retrospective_stress"]],
           "note": "All fields are point-in-time except retrospective_stress (full-sample hindsight labels)."}
    return out


def _recent_alerts(t: pd.Timestamp, n: int = 8) -> list[dict]:
    from riskcausal.evaluation.metrics import alert_episodes

    P = _primary().predictions
    out = []
    for m in (_explained(), "vix", "inst_composite"):
        g = P[(P["method"] == m) & (P["date"] <= t)].sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(g["date"])
        for s, e in alert_episodes(g["alert"].to_numpy(bool), MERGE_GAP):
            out.append({"method": m, "start": str(dates[s].date()), "end": str(dates[e].date()),
                        "ongoing": bool(dates[e] == t), "peak_score": float(g["score"].iloc[s:e + 1].max()),
                        "threshold": float(g["threshold"].iloc[s])})
    return sorted(out, key=lambda a: a["start"], reverse=True)[:n]


@router.get("/snapshot/{date}")
@json_safe
def snapshot(date: str) -> dict:
    df = _frame()
    t = _date(date)
    row = df.loc[t]
    i = df.index.get_loc(t)
    spark = df.iloc[max(0, i - 62):i + 1]
    prev = df.iloc[max(0, i - 21)]
    p = _num(row["regime_p_stress"])
    return {
        "date": str(t.date()), "requested": date, "mode": "historical_replay",
        "status": status(row),
        "sii": {"value": _num(row["sii"]), "percentile": _num(row["sii_pct"]),
                "change_21d": _num(row["sii"] - prev["sii"]) if not pd.isna(prev["sii"]) else None,
                "spark_dates": [str(x.date()) for x in spark.index], "spark": [_num(v) for v in spark["sii"]]},
        "regime": {"p_stress": p, "state": None if p is None else ("stressed" if p > 0.5 else "calm"),
                   "basis": "filtered Markov-switching probability on SPY realised volatility, data up to this date"},
        "vix": {"value": _num(row["vix"]), "percentile": _num(row["vix_pct"]),
                "change_5d": _num(row["vix_chg_5d"]), "change_5d_percentile": _num(row["vix_chg_5d_pct"])},
        "nfci": {"value": _num(row["nfci"]), "percentile": _num(row["nfci_pct"])},
        "edge_turnover": {"value": _num(row["edge_turnover"]), "percentile": _num(row["edge_turnover_pct"]),
                          "avg_4_windows": _num(row["edge_turnover_ma4"])},
        "model": {"method": _explained(), "score": _num(row["model_score"]),
                  "threshold": _num(row["model_threshold"]), "alert": bool(row["model_alert"])},
        "graph_window_end": None if pd.isna(row["graph_window_end"]) else str(pd.Timestamp(row["graph_window_end"]).date()),
        "recent_alerts": _recent_alerts(t),
    }


@router.get("/network/{date}")
@json_safe
def network(date: str, lag: int = 1) -> dict:
    """Estimated directed-dependence graph used at D and its change vs the window `lag` steps earlier."""
    r, df = _primary(), _frame()
    t = _date(date)
    we = df.loc[t, "graph_window_end"]
    if pd.isna(we):
        raise HTTPException(404, "no graph window available")
    G = r.graphs
    w = G["ends"].get_loc(pd.Timestamp(we))
    lag = max(1, min(int(lag), 12))
    wp = max(w - lag, 0)
    nodes, B, Bp = G["nodes"], G["B"][w, 0], G["B"][wp, 0]
    thr = float(r.config["graph"]["edge_threshold"])
    edges = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j or (abs(B[i, j]) <= thr and abs(Bp[i, j]) <= thr):
                continue
            c = describe.edge_change(nodes[j], nodes[i], float(Bp[i, j]), float(B[i, j]), tol=thr)
            edges.append({"id": f"{nodes[j]}->{nodes[i]}", **c, "active": bool(abs(B[i, j]) > thr),
                          "sign": "positive" if B[i, j] > 0 else "negative" if B[i, j] < 0 else "zero"})
    edges.sort(key=lambda e: -abs(e["after"] - e["before"]))
    stats = []
    for n in nodes:
        out_e = [e for e in edges if e["source"] == n and e["active"]]
        in_e = [e for e in edges if e["target"] == n and e["active"]]
        stats.append({"id": n, "name": NODE_INFO.get(n, (n, ""))[0], "group": NODE_INFO.get(n, (n, "other"))[1],
                      "out_degree": len(out_e), "in_degree": len(in_e),
                      "changed": sum(1 for e in edges if n in (e["source"], e["target"]) and _material(e))})
    return {"date": str(t.date()), "window_end": str(G["ends"][w].date()), "previous_window_end": str(G["ends"][wp].date()),
            "lag_windows": lag, "edge_threshold": thr, "nodes": stats, "edges": edges,
            "B0": B.round(4).tolist(), "B0_previous": Bp.round(4).tolist(), "node_order": nodes,
            "n_active": sum(e["active"] for e in edges),
            "counts": {k: sum(e["change"] == k for e in edges) for k in
                       ("appeared", "disappeared", "sign flipped", "strengthened", "weakened")},
            "caveat": ("Arrows are estimated directed statistical dependence from VarLiNGAM on the previous "
                       f"{int(r.config['graph']['window'])} trading days — not proven causal effects. Single "
                       "changes are noisy: refitting the same window on resampled data changes many edges.")}


def _material(e: dict) -> bool:
    """A change worth highlighting: an edge appeared, disappeared or flipped sign, or moved by more than 0.1."""
    return e["change"] in ("appeared", "disappeared", "sign flipped") or abs(e["after"] - e["before"]) > 0.1


def _briefing(items: list[dict], notes: list[str], method: str, d: str) -> list[dict]:
    """Deterministic evidence briefing (no LLM): each sentence cites the evidence it is built from."""
    by = {k: [i for i in items if i["kind"] == k] for k in ("alert", "macro", "graph", "news")}
    out = []
    if by["alert"]:
        a = by["alert"][0]
        out.append({"text": a["description"], "evidence_ids": [a["id"]]})
    unusual = [i for i in by["macro"] if i["values"].get("pct") is not None and
               (i["values"]["pct"] > HIGH or i["values"]["pct"] < LOW)]
    if unusual:
        parts = [f"{i['label']} ({describe.pct_label(i['values']['pct'])}, {i['values']['pct']}th percentile)"
                 for i in unusual]
        out.append({"text": "Readings outside their usual 5-year range: " + "; ".join(parts) + ".",
                    "evidence_ids": [i["id"] for i in unusual]})
    elif by["macro"]:
        out.append({"text": "All monitored market and macro indicators were within their usual 5-year range.",
                    "evidence_ids": [i["id"] for i in by["macro"]]})
    edges = [i for i in by["graph"] if i["id"].startswith("edge:")]
    turn = next((i for i in by["graph"] if "vl_shd_b0@" in i["id"]), None)
    if turn:
        pct = turn["values"].get("pct")
        out.append({"text": f"The estimated relationship graph had {turn['values']['value']:.0f} edges added or removed "
                            f"versus the previous window" + (f" ({describe.pct_label(pct)} for the past 5 years)."
                                                             if pct is not None else "."),
                    "evidence_ids": [turn["id"]]})
    if edges:
        out.append({"text": "Largest change: " + edges[0]["description"], "evidence_ids": [edges[0]["id"]]})
    if by["news"]:
        out.append({"text": f"{len(by['news'])} news headlines published in the 7 days before the 4pm New York cutoff "
                            "were retrieved (see News agent).", "evidence_ids": [i["id"] for i in by["news"]]})
    else:
        out.append({"text": "No news evidence is available for this date — this is missing data, not neutral news.",
                    "evidence_ids": []})
    out.append({"text": f"This briefing describes what was observable at the close of {d}. It is not a forecast, and "
                        "estimated graph relationships are not causal effects.", "evidence_ids": []})
    return out


MACRO_LABEL = {"VIXCLS": "VIX level", "NFCI": "NFCI financial conditions", "BAA10Y": "Baa credit spread",
               "T10Y2Y": "10y–2y yield curve", "rv21": "SPY realised volatility", "vix_chg_5d": "5-day VIX change",
               "avg_corr_63": "Average cross-asset correlation"}


@router.get("/agents/{date}")
@json_safe
def agents(date: str, method: str | None = None) -> dict:
    """Four-agent evidence view for one replay date: deterministic evidence per agent + deterministic synthesis
    briefing (the reliable default) + the evaluated LLM draft for this alert, if one exists (unverified)."""
    from riskcausal_api.main import _drafts, _run_evidence, _status

    method = method or _explained()
    t = _date(date)
    r = _primary()
    try:
        b = _run_evidence(r.run_id).bundle(method, str(t.date()))
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    thr = float(r.config["graph"]["edge_threshold"])
    items = []
    for e in b.items.values():
        it = {"id": e.id, "kind": e.kind, "available": e.available, "values": e.values, "source_text": e.text}
        if e.kind == "alert":
            it["description"] = describe.alert(method, e.values["score"], e.values["threshold"])
            it["label"] = "Model alert"
        elif e.kind == "macro":
            s = e.id.split(":")[1].split("@")[0]
            it["description"], it["label"] = describe.indicator(s, e.values["value"], e.values.get("pct")), MACRO_LABEL.get(s, s)
        elif e.id.startswith("edge:"):
            src, dst = e.id.split(":", 1)[1].split("@")[0].split("->")
            it["description"] = describe.edge_change(src, dst, e.values["before"], e.values["after"], tol=thr)["text"]
            it["label"] = f"{src} → {dst}"
        else:
            it["description"], it["label"] = e.text, e.kind
        items.append(it)
    news_note = [n for n in b.notes if n.startswith("news")]
    draft = None
    aid = f"{method}@{t.date()}"
    hc = _drafts(r).get(aid)
    if hc:
        rec = json.loads((r.dir / "explanations" / hc["file"]).read_text())
        st, flags = _status(rec, hc)
        draft = {"status": st, "flags": flags, "claims": [] if st.startswith("suppressed") else rec["explanation"]["claims"],
                 "uncertainty": rec["explanation"]["uncertainty"], "model": "qwen2.5:7b (local, Ollama)",
                 "evaluation_set": hc["set"], "specialists": {
                     s["role"]: {"abstain": s["abstain"], "abstain_reason": s.get("abstain_reason", ""),
                                 "claims": s["claims"], "n_rejected": len(s.get("rejected", []))}
                     for s in rec["specialists"]},
                 "label": "UNVERIFIED LLM DRAFT — not a verified conclusion; the evidence above is authoritative."}
    return {
        "date": str(t.date()), "method": method, "cutoff_utc": b.cutoff_utc,
        "agents": {
            "graph": {"title": "Graph agent", "role": "Changes in estimated financial relationships",
                      "evidence": [i for i in items if i["kind"] == "graph"]},
            "macro": {"title": "Macro agent", "role": "Economic and market indicators as known that day",
                      "evidence": [i for i in items if i["kind"] in ("macro", "alert")]},
            "news": {"title": "News agent", "role": "Headlines published before the 4pm New York cutoff",
                     "evidence": [i for i in items if i["kind"] == "news"], "unavailable": news_note,
                     "abstained": not any(i["kind"] == "news" for i in items)},
        },
        "synthesis": {"title": "Synthesis agent", "kind": "deterministic_briefing",
                      "briefing": _briefing(items, b.notes, method, str(t.date()))},
        "llm_draft": draft,
        "llm_draft_available_for": sorted(k.split("@")[1] for k in _drafts(r)),
        "reliability_retrospective": b.reliability,
    }


@router.get("/research")
@json_safe
def research() -> dict:
    """Locked research results for the Research section (unchanged artifacts; read-only)."""
    from riskcausal_api.main import metrics as metrics_ep

    cfg = dashboard_config()["runs"]
    prim = metrics_ep(cfg["primary"])
    expl = metrics_ep(cfg["exploratory"])
    M = runs()[cfg["primary"]].metrics
    ev = M["by_rule"][M["primary_rule"]]["common_span_metrics"]
    leads = {m: [x for x in v["event"]["lead_times"] if x is not None] for m, v in ev.items()}
    return {"primary": prim, "exploratory": expl, "lead_times": leads,
            "reproducibility": reproducibility(),
            "conclusion": ("In the locked full-profile evaluation (2014–2026, 21 stress onsets) the graph-instability "
                           "measures did not give earlier or better warning than the VIX level, and adding them to "
                           "standard controls did not improve the model.")}
