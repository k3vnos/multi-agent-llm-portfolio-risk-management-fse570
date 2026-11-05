"""Portfolio Lens (M10, EXPLORATORY — separate from the locked research evaluation).

A user-chosen portfolio of the project's 25 ETFs, evaluated as of a historical replay date D using only daily
returns up to and including D (known at the close of D). Nothing here feeds the warning model, its score, its alerts
or any locked metric.

Definitions (all from saved daily log returns of adjusted closes, converted to simple returns):
- portfolio daily return = Σ w_i r_i with weights reset to the targets every day (daily rebalancing);
- volatility = sample standard deviation of daily portfolio returns × √252, over the last 63 / 252 sessions;
- historical VaR95 / CVaR95 (1 day) = the 5th-percentile daily loss / mean loss beyond it, last 252 sessions;
- parametric VaR95 (1 day) = 1.645 × daily σ from the 63-session covariance (normal approximation, mean ignored);
- max drawdown = largest peak-to-trough fall of the portfolio value over the last 252 sessions;
- risk contribution (Euler) of holding i = w_i (Σw)_i / (wᵀΣw) on the 63-session covariance; contributions sum to
  100% and show which holdings account for the portfolio's variance;
- average pairwise co-movement = mean Pearson correlation between holdings over the last 63 sessions, compared with
  the 63 sessions that ended 63 sessions earlier;
- network links: holdings are mapped to the 10 asset nodes of the estimated directed-dependence graph (documented
  proxies for ETFs that are not graph nodes) and changed links touching those nodes are listed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from riskcausal_api.replay import _date, _material, json_safe, network

router = APIRouter(prefix="/portfolio", tags=["portfolio (exploratory)"])

ANN, Z95 = np.sqrt(252), 1.6448536269514722
W_SHORT, W_LONG, MIN_OBS = 63, 252, 60
NAMES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "US small caps", "EFA": "Developed ex-US stocks",
         "EEM": "Emerging-market stocks", "XLF": "Financials", "XLK": "Technology", "XLE": "Energy",
         "XLV": "Health care", "XLI": "Industrials", "XLY": "Consumer discretionary", "XLP": "Consumer staples",
         "XLU": "Utilities", "XLB": "Materials", "KRE": "Regional banks", "SHY": "1–3y Treasuries",
         "IEF": "7–10y Treasuries", "TLT": "20+y Treasuries", "TIP": "Inflation-protected Treasuries",
         "LQD": "Investment-grade corporate bonds", "HYG": "High-yield corporate bonds", "GLD": "Gold",
         "DBC": "Broad commodities", "USO": "Crude oil", "UUP": "US dollar"}
# graph node used to place a holding in the estimated network (None = not represented in the graph)
NODE_PROXY = {"SPY": "SPY", "QQQ": "SPY", "XLK": "SPY", "XLY": "SPY", "XLI": "SPY", "XLB": "SPY", "XLV": "SPY",
              "XLP": "SPY", "XLU": "SPY", "XLE": "SPY", "IWM": "IWM", "EFA": "EFA", "EEM": "EEM", "XLF": "XLF",
              "KRE": "XLF", "SHY": "TLT", "IEF": "TLT", "TLT": "TLT", "TIP": "TLT", "LQD": "LQD", "HYG": "HYG",
              "GLD": "GLD", "UUP": "UUP", "DBC": None, "USO": None}
EXAMPLES = [
    {"id": "60-40", "name": "Classic 60/40", "spec": "SPY:60,IEF:40",
     "about": "60% US stocks, 40% 7–10 year Treasuries — a common balanced benchmark."},
    {"id": "all-equity", "name": "All stocks, global", "spec": "SPY:50,EFA:25,EEM:15,IWM:10",
     "about": "Only equities, spread across US large/small caps, developed and emerging markets."},
    {"id": "defensive", "name": "Defensive mix", "spec": "SPY:30,TLT:30,GLD:20,XLP:20",
     "about": "Stocks balanced with long Treasuries, gold and consumer staples."},
    {"id": "financials", "name": "Bank-heavy", "spec": "XLF:40,KRE:30,HYG:30",
     "about": "Concentrated in financials, regional banks and high-yield credit (stress-sensitive)."},
]
EXPLORATORY = ("EXPLORATORY — Portfolio Lens is not part of the locked research evaluation. Metrics are "
               "historical, computed from daily returns up to the replay date only; they are not forecasts or "
               "investment advice.")


@lru_cache(maxsize=1)
def returns() -> pd.DataFrame:
    """Simple daily returns of the 25 ETFs (from the saved processed log returns of the primary run's profile)."""
    from riskcausal.data.panel import processed_dir
    from riskcausal_api.replay import _primary

    r = _primary()
    from riskcausal.config import load_config

    cfg = load_config(r.config["profile"], overrides=None, experiment=None)
    return np.expm1(pd.read_parquet(processed_dir(cfg) / "returns.parquet"))


def parse(spec: str) -> dict[str, float]:
    """"SPY:60,IEF:40" -> {"SPY": 0.6, "IEF": 0.4}. Weights in percent, must be positive and total 100."""
    w: dict[str, float] = {}
    for part in [p for p in (spec or "").split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(f"bad holding {part!r}; use TICKER:PERCENT")
        t, v = part.split(":", 1)
        t = t.strip().upper()
        if t not in NAMES:
            raise ValueError(f"{t} is not one of the project's ETFs")
        if t in w:
            raise ValueError(f"{t} listed twice")
        x = float(v)
        if not 0 < x <= 100:
            raise ValueError(f"weight for {t} must be between 0 and 100")
        w[t] = x
    if not w:
        raise ValueError("portfolio is empty")
    if abs(sum(w.values()) - 100) > 0.01:
        raise ValueError(f"weights total {sum(w.values()):g}%, they must total 100%")
    return {k: v / 100 for k, v in w.items()}


def _pct(s: pd.Series, years: int = 5) -> float | None:
    s = s.dropna()
    if len(s) < 100:
        return None
    hist = s.loc[s.index[-1] - pd.DateOffset(years=years):]
    return round(float((hist <= s.iloc[-1]).mean() * 100))


def portfolio_returns(R: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    sub = R[list(w)].dropna()
    return sub @ pd.Series(w)


def max_drawdown(r: pd.Series) -> float:
    v = (1 + r).cumprod()
    return float((v / v.cummax() - 1).min())


def euler(cov: pd.DataFrame, w: pd.Series) -> pd.Series:
    mrc = cov @ w
    return w * mrc / float(w @ mrc)


def analyze(spec: str, date: str, R: pd.DataFrame | None = None, net: dict | None = None) -> dict:
    w = parse(spec)
    t = _date(date) if R is None else pd.Timestamp(date)
    R = (returns() if R is None else R).loc[:t]                       # point-in-time: nothing after the close of D
    late = [k for k in w if R[k].loc[:t].dropna().shape[0] < MIN_OBS]
    if late:
        raise ValueError(f"not enough history before {t.date()} for {', '.join(late)} (fund too new on this date)")
    p = portfolio_returns(R, w)
    last = p.iloc[-W_LONG:]
    sub = R[list(w)].dropna()
    cov = sub.iloc[-W_SHORT:].cov()
    ws = pd.Series(w)
    rc = euler(cov, ws)
    vol63_series = p.rolling(W_SHORT).std() * ANN
    ret21 = (1 + p).rolling(21).apply(np.prod, raw=True) - 1
    q = float(np.quantile(last, 0.05))
    corr_now = sub.iloc[-W_SHORT:].corr().to_numpy()
    corr_then = sub.iloc[-2 * W_SHORT:-W_SHORT].corr().to_numpy()
    iu = np.triu_indices(len(w), 1)
    avg_c = (float(np.nanmean(corr_now[iu])), float(np.nanmean(corr_then[iu]))) if len(w) > 1 else (None, None)
    vol_each = sub.iloc[-W_SHORT:].std() * ANN
    holdings = []
    for k in w:
        holdings.append({"ticker": k, "name": NAMES[k], "weight": round(w[k] * 100, 2),
                         "risk_share": round(float(rc[k]) * 100, 2), "vol_63d": round(float(vol_each[k]) * 100, 2),
                         "return_21d": round(float((1 + sub[k].iloc[-21:]).prod() - 1) * 100, 2),
                         "node": NODE_PROXY[k], "node_is_proxy": NODE_PROXY[k] not in (None, k)})
    holdings.sort(key=lambda h: -h["risk_share"])
    # network: links of the estimated graph (window known at D) touching the portfolio's nodes
    net = net if net is not None else network(str(t.date()))
    nodes = {h["node"]: [] for h in holdings if h["node"]}
    for h in holdings:
        if h["node"]:
            nodes[h["node"]].append(h["ticker"])
    node_w = {n: sum(w[x] for x in xs) for n, xs in nodes.items()}
    touching = [e for e in net["edges"] if (e["source"] in nodes or e["target"] in nodes) and _material(e)]
    links = [{**{k: e[k] for k in ("id", "source", "target", "before", "after", "change", "text")},
              "holdings": sorted(set(nodes.get(e["source"], []) + nodes.get(e["target"], []))),
              "weight_touched": round(100 * (node_w.get(e["source"], 0) + node_w.get(e["target"], 0)
                                              if e["source"] != e["target"] else 0), 1)} for e in touching]
    links.sort(key=lambda x: (-x["weight_touched"], -abs(x["after"] - x["before"])))
    exposed = {n for e in touching for n in (e["source"], e["target"]) if n in nodes}
    return {
        "date": str(t.date()), "spec": ",".join(f"{k}:{round(v * 100, 2):g}" for k, v in w.items()),
        "label": EXPLORATORY, "data_through": str(R.index[-1].date()), "n_sessions": len(p),
        "metrics": {
            "return_21d": round(float(ret21.iloc[-1]) * 100, 2), "return_21d_pct": _pct(ret21),
            "return_252d": round(float((1 + last).prod() - 1) * 100, 2),
            "vol_63d": round(float(vol63_series.iloc[-1]) * 100, 2), "vol_63d_pct": _pct(vol63_series),
            "vol_252d": round(float(last.std() * ANN) * 100, 2),
            "var95_hist": round(-q * 100, 2), "cvar95_hist": round(-float(last[last <= q].mean()) * 100, 2),
            "var95_param": round(Z95 * float(np.sqrt(ws @ cov @ ws)) * 100, 2),
            "max_drawdown_252d": round(max_drawdown(last) * 100, 2),
            "avg_comovement_63d": None if avg_c[0] is None else round(avg_c[0], 3),
            "avg_comovement_prev_63d": None if avg_c[1] is None else round(avg_c[1], 3),
        },
        "holdings": holdings,
        "network": {"window_end": net["window_end"], "previous_window_end": net["previous_window_end"],
                    "nodes": {n: xs for n, xs in nodes.items()}, "not_in_graph": [k for k in w if NODE_PROXY[k] is None],
                    "links": links[:12], "n_links": len(links),
                    "weight_on_changed_nodes": round(100 * sum(node_w[n] for n in exposed), 1)},
        "series": {"dates": [str(d.date()) for d in last.index],
                   "value": [round(float(x), 5) for x in (1 + last).cumprod()],
                   "drawdown": [round(float(x), 5) for x in ((1 + last).cumprod() / (1 + last).cumprod().cummax() - 1)]},
        "definitions": __doc__.split("Definitions", 1)[1].strip(),
    }


def portfolio_facts(a: dict):
    """Portfolio facts for the agents (labels computed from the numbers, as for market facts)."""
    from riskcausal_agents.facts import (
        CHANGE_LABELS,
        EDGE_LABELS,
        LEVEL_LABELS,
        SHARE_LABELS,
        Fact,
        change_label,
        edge_prompt,
        level_label,
    )

    m, d = a["metrics"], a["date"]
    out = [Fact("", "macro", "level", level_label(m["vol_63d_pct"]), LEVEL_LABELS,
                f"your portfolio's 63-day volatility = {m['vol_63d']}% annualised"
                + (f", percentile within its own previous 5 years = {m['vol_63d_pct']}" if m["vol_63d_pct"] is not None else ", no 5-year history"),
                f"Your portfolio's 63-day volatility was {m['vol_63d']:.1f}% annualised"
                + (f", {m['vol_63d_pct']}th percentile of its previous 5 years ({level_label(m['vol_63d_pct'])})." if m["vol_63d_pct"] is not None else "."),
                "portfolio:vol_63d", d, {"value": m["vol_63d"], "pct": m["vol_63d_pct"]}),
           Fact("", "macro", "change", change_label(m["return_21d"], m["return_21d_pct"]), CHANGE_LABELS,
                f"your portfolio's return over the last 21 sessions = {m['return_21d']:+.2f}%"
                + (f", percentile within its own previous 5 years = {m['return_21d_pct']}" if m["return_21d_pct"] is not None else ""),
                f"Your portfolio's value changed {m['return_21d']:+.2f}% over the last 21 sessions"
                + (f" ({m['return_21d_pct']}th percentile of its 1-month returns over the previous 5 years)." if m["return_21d_pct"] is not None else "."),
                "portfolio:return_21d", d, {"value": m["return_21d"], "pct": m["return_21d_pct"]})]
    hs = a["holdings"]
    for i, h in enumerate(hs[:3] if len(hs) > 1 else []):
        diff = h["risk_share"] - h["weight"]
        lab = "largest contributor" if i == 0 else "above its weight" if diff > 5 else "below its weight" if diff < -5 else "in line with its weight"
        out.append(Fact("", "macro", "share", lab, SHARE_LABELS,
                        f"holding {h['ticker']}: portfolio weight {h['weight']:g}%, share of portfolio risk {h['risk_share']:g}%",
                        f"{h['ticker']} ({h['name']}) is {h['weight']:g}% of the portfolio and accounts for "
                        f"{h['risk_share']:.1f}% of its 63-day risk (Euler contribution"
                        + ("; a negative share means it offset the other holdings' swings)." if h["risk_share"] < 0 else ")."),
                        f"portfolio:risk_share:{h['ticker']}", d, {"weight": h["weight"], "risk_share": h["risk_share"]}))
    for e in a["network"]["links"][:3]:
        out.append(Fact("", "graph", "edge", e["change"], EDGE_LABELS,
                        edge_prompt(e["source"], e["target"], e["before"], e["after"], 0.05)
                        + f" (touches your holdings {', '.join(e['holdings'])})",
                        e["text"] + f" Touches your holdings: {', '.join(e['holdings'])}.",
                        f"portfolio:edge:{e['id']}@{a['network']['window_end']}", d, {"before": e["before"], "after": e["after"]}))
    return out


def context_line(a: dict) -> str:
    hs = ", ".join(f"{h['ticker']} {h['weight']:g}%" for h in a["holdings"])
    return f"The user's portfolio (exploratory analysis): {hs}."


@router.get("/universe")
@json_safe
def universe() -> dict:
    R = returns()
    return {"etfs": [{"ticker": k, "name": v, "node": NODE_PROXY[k], "first_date": str(R[k].first_valid_index().date())}
                     for k, v in NAMES.items()], "examples": EXAMPLES, "label": EXPLORATORY}


@router.get("/analyze")
@json_safe
def analyze_ep(spec: str, date: str) -> dict:
    try:
        return analyze(spec, date)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
