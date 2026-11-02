"""Read-only local dashboard (M7). Run: `.venv/bin/streamlit run dashboard/app.py` (binds 127.0.0.1, no telemetry —
see .streamlit/config.toml). Everything shown comes from saved runs through the read-only API."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ApiClient

# Reference palette (dataviz skill, validated: 3 slots pass; aqua < 3:1 contrast -> every chart has a table view)
INK, INK2, GRID, SURFACE, STRESS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d9d8d3"
METHOD_COLOR = {"vix": "#2a78d6", "bayes_logit:combined": "#eb6834", "bayes_logit:controls": "#1baf7a",
                "dcc_var": "#eda100", "inst_composite": "#6250d6"}     # fixed per entity, never by rank
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
PANEL_W = 820          # stacked (vconcat) charts ignore container width; fixed width avoids clipping
XAXIS = alt.Axis(format="%b %Y", labelAngle=0, tickCount=8, title=None)
PAGES = ["Overview", "Stress timeline", "Model comparison", "Graph snapshot", "Alert explorer",
         "Exploratory: banking stress"]

st.set_page_config(page_title="riskcausal — capstone results (local)", layout="wide")


@st.cache_resource
def api() -> ApiClient:
    return ApiClient()


@st.cache_data
def get(path: str, **params) -> dict:
    return api().get(path, **params)


def fmt_ci(v) -> str:
    return "—" if not v else f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def ci_table(rows: list[dict]) -> pd.DataFrame:
    """Round point estimates to 3 decimals and render every *_ci95 pair as a readable interval."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.endswith("ci95"):
            df[c] = df[c].map(fmt_ci)
        elif df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    return df


def chart_style(c: alt.Chart) -> alt.Chart:
    return (c.configure_view(stroke=None, fill=SURFACE).configure_axis(gridColor=GRID, domainColor=GRID,
                                                                       labelColor=INK2, titleColor=INK2)
            .configure_title(color=INK, anchor="start", fontSize=13))


def banner(meta: dict) -> None:
    kind = meta["kind"]
    label = {"research_result": "Research result (from saved metrics/CIs of a locked run)",
             "evidence": "Point-in-time evidence with deterministic descriptions",
             "illustrative": "Illustrative (estimated directed dependence, not causality)",
             "unverified_draft": "UNVERIFIED LLM DRAFT — not a verified conclusion"}[kind]
    audit = "audit passed" if meta["audit_passed"] else "AUDIT FAILED"
    st.caption(f"{label} · run `{meta['run_id']}` ({meta['role']}) · git `{(meta['git_sha'] or '')[:10]}` · {audit}")


def runs() -> dict:
    return get("/runs")


def run_id(role: str) -> str:
    return next(r["run_id"] for r in runs()["runs"] if r["role"] == role)


# ------------------------------------------------------------------ pages
def overview() -> None:
    R = runs()
    st.title("Directed-dependence instability as an early-warning signal")
    st.markdown("**Research question.** Does instability in rolling VarLiNGAM estimates of *directed dependence* "
                "give earlier warning of stress than VIX, NFCI, correlation-based VaR and DCC-GARCH?")
    prim = next(r for r in R["runs"] if r["role"] == "primary")
    m = get(f"/runs/{prim['run_id']}/metrics")
    banner(m["meta"])
    rows = {x["method"]: x for x in m["data"]["methods"]}
    diffs = {d["comparison"]: d for d in m["data"]["differences"]}
    st.subheader("Answer on the locked full-profile evaluation: no")
    c1, c2, c3 = st.columns(3)
    vix, ins = rows["vix"], rows["bayes_logit:instability"]
    c1.metric("VIX level — ROC-AUC", f"{vix['roc_auc']:.3f}", help=f"95% CI {vix['roc_ci95']}")
    c2.metric("Graph instability only (Bayes) — ROC-AUC", f"{ins['roc_auc']:.3f}",
              delta=f"{diffs['bayes_logit:instability - vix']['delta_roc']:+.3f} vs VIX", delta_color="normal",
              help=f"Δ 95% CI {diffs['bayes_logit:instability - vix']['roc_ci95']}")
    d = diffs["bayes_logit:combined - bayes_logit:controls"]
    c3.metric("Adding instability to controls (Bayes) — ΔROC-AUC", f"{d['delta_roc']:+.3f}",
              help=f"95% CI {d['roc_ci95']}; excludes 0: {d['excludes_zero']}")
    st.markdown(f"- Test span {m['data']['span'][0]} → {m['data']['span'][1]}, "
                f"{vix['n_onsets']} retrospective stress onsets; alert rule `{m['data']['alert_rule']}`.\n"
                f"- CIs: {m['data']['ci_method']}.\n"
                "- Instability-only models are significantly worse than VIX; adding them to the controls gives no "
                "supported improvement. Graphs are *estimated directed dependence*, not causal structure.")
    rep = R["reproducibility_of_primary"]
    st.markdown(f"**Leakage audit:** {'passed' if prim['audit_passed'] else 'FAILED'} "
                f"({len(m['data']['audit'])} checks, "
                f"{sum(c['n_violations'] for c in m['data']['audit'])} violations) · "
                f"**Reproducibility:** {rep['status']} vs re-run `{rep.get('twin_run_id', '—')}`")
    st.dataframe(pd.DataFrame(R["runs"])[["role", "run_id", "profile", "test_span", "target_series", "audit_passed",
                                          "git_sha"]], hide_index=True)


def timeline() -> None:
    st.title("Stress timeline")
    R = runs()
    role = st.selectbox("Run", [r["role"] for r in R["runs"] if r["role"] != "exploratory"])
    rid = run_id(role)
    methods = st.multiselect("Methods (one panel each)", R["headline_methods"], default=["vix", "bayes_logit:combined"],
                             max_selections=3)
    span = st.slider("Years", 2014 if role == "primary" else 2012, 2026, (2019, 2021))
    first = get(f"/runs/{rid}/timeline", method=methods[0] if methods else "vix")
    banner(first["meta"])
    t = first["data"]
    df = pd.DataFrame({"date": pd.to_datetime(t["dates"]), "rv": t["target_rv"], "stress": t["retrospective_stress"]})
    df = df[(df.date.dt.year >= span[0]) & (df.date.dt.year <= span[1])]
    runs_ = (df["stress"].fillna(0) != df["stress"].fillna(0).shift()).cumsum()
    shade = df[df["stress"] == 1].groupby(runs_[df["stress"] == 1]).agg(start=("date", "min"), end=("date", "max"))
    sh = alt.Chart(shade).mark_rect(color=STRESS, opacity=0.7).encode(x="start:T", x2="end:T")
    vol = alt.Chart(df).mark_line(color=INK2, strokeWidth=2).encode(
        x=alt.X("date:T", axis=XAXIS), y=alt.Y("rv:Q", title=f"{t['target_series']} 21-day realised vol"),
        tooltip=["date:T", alt.Tooltip("rv:Q", format=".2f")])
    panels = [(sh + vol).properties(width=PANEL_W, height=140, title="Realised volatility; shaded = retrospective stress regime")]
    tables = []
    for m in methods:
        d = get(f"/runs/{rid}/timeline", method=m)["data"]
        x = pd.DataFrame({"date": pd.to_datetime(d["dates"]), "score": d["score"], "threshold": d["threshold"],
                          "alert": d["alert"]})
        x = x[(x.date.dt.year >= span[0]) & (x.date.dt.year <= span[1])]
        tables.append(x.assign(method=m))
        base = alt.Chart(x).encode(x=alt.X("date:T", axis=XAXIS))
        line = base.mark_line(color=METHOD_COLOR.get(m, INK2), strokeWidth=2).encode(
            y=alt.Y("score:Q", title="score"), tooltip=["date:T", alt.Tooltip("score:Q", format=".3f"),
                                                        alt.Tooltip("threshold:Q", format=".3f"), "alert:N"])
        thr = base.mark_line(color=INK, strokeDash=[4, 3], strokeWidth=1).encode(y="threshold:Q")
        al = alt.Chart(x[x["alert"]]).mark_tick(color=INK, thickness=2, size=10).encode(x="date:T", y=alt.value(0))
        panels.append((sh + line + thr + al).properties(width=PANEL_W, height=140, title=f"{m} — dashed: validation threshold; "
                                                                          "ticks: alert days"))
    st.altair_chart(chart_style(alt.vconcat(*panels).resolve_scale(x="shared")), use_container_width=False)
    st.caption("Retrospective stress shading uses full-sample labels known only after the fact (scoring truth).")
    with st.expander("Table view"):
        st.dataframe(pd.concat(tables) if tables else df, hide_index=True)


def comparison() -> None:
    st.title("Model comparison")
    role = st.selectbox("Run", [r["role"] for r in runs()["runs"]])
    m = get(f"/runs/{run_id(role)}/metrics")
    banner(m["meta"])
    df = pd.DataFrame(m["data"]["methods"])
    df[["lo", "hi"]] = pd.DataFrame(df["roc_ci95"].tolist(), index=df.index)
    vix = float(df.loc[df.method == "vix", "roc_auc"].iloc[0])
    order = df.sort_values("roc_auc", ascending=False)["method"].tolist()
    base = alt.Chart(df).encode(y=alt.Y("method:N", sort=order, title=None,
                                         axis=alt.Axis(labelLimit=260, labelOverlap=False)))
    ci = base.mark_rule(color=INK2, strokeWidth=2).encode(x=alt.X("lo:Q", title="ROC-AUC (95% CI)",
                                                                   scale=alt.Scale(domain=[0.25, 0.9])), x2="hi:Q")
    pt = base.mark_point(filled=True, size=90, color=METHOD_COLOR["vix"]).encode(
        x="roc_auc:Q", tooltip=["method", alt.Tooltip("roc_auc:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"),
                                alt.Tooltip("hi:Q", format=".3f"), "n_detected", "n_onsets"])
    ref = alt.Chart(pd.DataFrame({"x": [vix, 0.5]}, index=["VIX", "chance"])).mark_rule(
        strokeDash=[4, 3], color=INK).encode(x="x:Q")
    st.altair_chart(chart_style((ci + pt + ref).properties(
        height=28 * len(df), title=f"ROC-AUC with 95% block-bootstrap CI · dashed: VIX ({vix:.3f}) and chance (0.5)")),
        use_container_width=True)
    st.subheader("Paired differences (95% CI)")
    st.dataframe(ci_table(m["data"]["differences"]), hide_index=True)
    st.subheader("Event-level metrics (primary alert rule; Jeffreys intervals)")
    st.dataframe(ci_table(m["data"]["methods"])[["method", "n_onsets", "n_detected", "recall", "recall_ci95",
                                                   "precision", "precision_ci95", "false_alarms_per_year",
                                                   "calm_alert_share", "median_lead", "pr_auc", "base_rate"]],
                 hide_index=True)


def graph_page() -> None:
    st.title("Graph snapshot — estimated directed dependence")
    rid = run_id("primary")
    date = st.date_input("Decision date", value=pd.Timestamp("2020-02-20"), min_value=pd.Timestamp("2014-01-02"),
                         max_value=pd.Timestamp("2026-10-16"))
    try:
        g = get(f"/runs/{rid}/graph/{pd.Timestamp(date).date()}")
    except Exception:  # noqa: BLE001
        st.warning("Not a trading session with an available graph window; pick another date.")
        return
    banner(g["meta"])
    d = g["data"]
    st.markdown(f"Window used on {d['date']}: **ends {d['window_end']}** ({d['window_length']} sessions; previous "
                f"window ends {d['previous_window_end']}). Edge = |coefficient| > {d['edge_threshold']}.")
    st.info(d["caveat"])
    n = d["nodes"]
    hm = pd.DataFrame([{"effect of (source)": n[j], "on (target)": n[i], "coef": d["B0"][i][j]}
                       for i in range(len(n)) for j in range(len(n)) if i != j])
    lim = max(0.5, float(hm["coef"].abs().max()))
    chart = alt.Chart(hm).mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("effect of (source):N", sort=n, axis=alt.Axis(labelOverlap=False, labelAngle=-45)),
        y=alt.Y("on (target):N", sort=n, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("coef:Q", scale=alt.Scale(domain=[-lim, 0, lim], range=DIVERGING, interpolate="lab"),
                        title="B0 coefficient"),
        tooltip=["effect of (source)", "on (target)", alt.Tooltip("coef:Q", format=".3f")])
    st.altair_chart(chart_style(chart.properties(height=alt.Step(26), width=alt.Step(52), title="Contemporaneous B0: column = source, "
                                                 "row = target (blue negative, red positive, gray ≈ 0)")),
                    use_container_width=False)
    st.subheader("Changes vs the previous window (deterministic descriptions)")
    st.dataframe(pd.DataFrame(d["changes"])[["text", "change", "before", "after"]], hide_index=True)
    st.subheader("Instability features at this date")
    st.dataframe(pd.DataFrame(d["instability_features"]).T, use_container_width=True)


def alert_explorer() -> None:
    st.title("Alert explorer")
    R = runs()
    rid = run_id("primary")
    method = st.selectbox("Method", R["headline_methods"], index=R["headline_methods"].index(R["explained_method"]))
    eps = get(f"/runs/{rid}/alerts", method=method)
    banner(eps["meta"])
    E = pd.DataFrame(eps["data"]["episodes"])
    st.dataframe(E, hide_index=True, height=220)
    start = st.selectbox("Alert episode (start date)", E["start"].tolist()[::-1])
    ev = get(f"/runs/{rid}/evidence/{method}/{start}")
    st.subheader(f"Evidence available at the close of {start}")
    banner(ev["meta"])
    for note in ev["data"]["coverage_notes"]:
        st.warning(note)
    for kind, title in [("alert", "Alert"), ("macro", "Market & macro indicators (point-in-time)"),
                        ("graph", "Estimated directed dependence (VarLiNGAM)"), ("news", "News published before 16:00 New York")]:
        items = [i for i in ev["data"]["items"] if i["kind"] == kind]
        if items:
            st.markdown(f"**{title}**")
            for i in items:
                st.markdown(f"- {i['description']}  \n  <small>`{i['id']}` · available {i['available']}</small>",
                            unsafe_allow_html=True)
    with st.expander("Method reliability (retrospective — NOT known at alert time)"):
        for line in ev["data"]["method_reliability_retrospective"]:
            st.markdown(f"- {line}")
    drafts = {d["alert_id"]: d for d in get(f"/runs/{rid}/explanations")["data"]["drafts"]}
    aid = f"{method}@{start}"
    st.subheader("LLM explanation")
    if aid not in drafts:
        st.caption("No LLM draft for this alert (drafts exist only for the 24 evaluated alerts of "
                   f"{R['explained_method']}). The evidence above is the explanation.")
        return
    x = get(f"/runs/{rid}/explanations/{aid}")
    banner(x["meta"])
    d = x["data"]
    if d["flags"]:
        st.error(d["label"])
    else:
        st.warning(d["label"])
    st.markdown(f"**Status:** {d['status']} · validator rejected {d['n_claims_rejected_by_validator']} claim(s) · "
                f"{d['model']} · prompt {d['prompt_version']} · evaluation set: {d['set']}")
    for f in d["flags"]:
        st.markdown(f"- ⚠ {f}")
    with st.expander("Show unverified draft text", expanded=False):
        for c in d["claims"]:
            st.markdown(f"- {c['text']}  \n  <small>cites {', '.join(c['evidence_ids'])}</small>",
                        unsafe_allow_html=True)
        if d["uncertainty"]:
            st.markdown(f"*Uncertainty (draft):* {d['uncertainty']}")
        st.caption(f"Hand check: {d['hand_check']}")


def exploratory() -> None:
    st.title("Exploratory: banking-stress target (KRE volatility)")
    st.warning("EXPLORATORY — a separately defined target (Markov regimes on KRE realised volatility), kept apart "
               "from the primary SPY-volatility target and never used for model selection.")
    rid = run_id("exploratory")
    m = get(f"/runs/{rid}/metrics")
    banner(m["meta"])
    pd.DataFrame(m["data"]["methods"])
    st.dataframe(ci_table(m["data"]["methods"])[["method", "roc_auc", "roc_ci95", "pr_auc", "n_onsets", "n_detected",
                                                   "precision"]], hide_index=True)
    onsets = [o for o in m["data"]["label_summary"]["onsets"] if o.startswith("2023")]
    st.markdown(f"Onsets in 2023 under this target: {', '.join(onsets)} (2023-03-09 = SVB week).")
    t = get(f"/runs/{rid}/timeline", method="dcc_corr")["data"]
    x = pd.DataFrame({"date": pd.to_datetime(t["dates"]), "rv": t["target_rv"], "score": t["score"],
                      "threshold": t["threshold"]})
    x = x[(x.date >= "2022-12-01") & (x.date <= "2023-05-31")]
    a = alt.Chart(x).mark_line(color=INK2, strokeWidth=2).encode(x=alt.X("date:T", axis=XAXIS),
                                                                 y=alt.Y("rv:Q", title="KRE realised vol"))
    b = alt.Chart(x).mark_line(color=METHOD_COLOR["vix"], strokeWidth=2).encode(x="date:T", y=alt.Y(
        "score:Q", title="DCC correlation score"), tooltip=["date:T", alt.Tooltip("score:Q", format=".3f")])
    c = alt.Chart(x).mark_line(color=INK, strokeDash=[4, 3]).encode(x="date:T", y="threshold:Q")
    on = alt.Chart(pd.DataFrame({"d": pd.to_datetime(onsets)})).mark_rule(color=INK, strokeDash=[2, 2]).encode(x="d:T")
    st.altair_chart(chart_style(alt.vconcat((a + on).properties(width=PANEL_W, height=120, title="KRE 21-day realised volatility"),
                                            (b + c + on).properties(width=PANEL_W, height=120, title="DCC correlation score "
                                                                    "(the only method alerting before SVB) — one "
                                                                    "event, anecdotal"))), use_container_width=False)


page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(f"Data source: {api().mode}. Read-only; local only; no model is run.")
{"Overview": overview, "Stress timeline": timeline, "Model comparison": comparison, "Graph snapshot": graph_page,
 "Alert explorer": alert_explorer, "Exploratory: banking stress": exploratory}[page]()
