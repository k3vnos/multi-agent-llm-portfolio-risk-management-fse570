"""Point-in-time evidence bundles for one alert (decision date t, close of the NYSE session).

Every evidence item carries the time it became available. Nothing available after the decision cutoff
(16:00 America/New_York on t) can enter a bundle, retrospective regime labels are never included, and graph
items are described as *estimated directed dependence*, never causality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NY = "America/New_York"
LOOKBACK_DAYS = 7
MAX_NEWS = 8
STRESS_TERMS = ["sell-off", "selloff", "plunge", "tumble", "crash", "slump", "volatil", "fear", "recession",
                "fed ", "federal reserve", "rate", "credit", "default", "bank", "liquidity", "virus", "coronavirus",
                "outbreak", "tariff", "trade war", "downgrade", "spread", "yield", "treasury", "dow", "s&p",
                "nasdaq", "market", "stocks", "oil", "crisis", "panic", "loss", "drop", "fall"]


@dataclass
class Evidence:
    id: str
    kind: str                      # alert | macro | graph | news
    available: str                 # ISO time/date the item was known (<= cutoff)
    text: str                      # the fact, in words, with its numbers
    values: dict = field(default_factory=dict)


@dataclass
class Bundle:
    run_id: str
    method: str
    date: str                      # decision date t (NYSE session)
    cutoff_utc: str                # 16:00 New York on t, in UTC
    items: dict[str, Evidence]     # point-in-time evidence: the only thing agents see and may cite
    notes: list[str]               # coverage notes (e.g. no news coverage) — shown to agents and users
    reliability: list[str] = field(default_factory=list)   # retrospective evaluation facts: shown to users,
    #                                                          never to agents, never citable

    def by_kind(self, kind: str) -> list[Evidence]:
        return [e for e in self.items.values() if e.kind == kind]

    def to_json(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "items"},
                "items": {k: asdict(v) for k, v in self.items.items()}}


def decision_cutoff_utc(date: pd.Timestamp) -> pd.Timestamp:
    """16:00 New York time on the decision date (handles daylight saving), as UTC."""
    return pd.Timestamp(pd.Timestamp(date).date()).tz_localize(NY).replace(hour=16).tz_convert("UTC")


def _pct(series: pd.Series, t: pd.Timestamp, years: int = 5) -> float | None:
    """Empirical percentile of the value at t within the trailing `years` of values known up to t."""
    hist = series.loc[t - pd.DateOffset(years=years):t].dropna()
    if len(hist) < 100 or pd.isna(series.get(t)):
        return None
    return float((hist <= series.loc[t]).mean() * 100)


def retrieve_news(news: pd.DataFrame | None, t: pd.Timestamp, coverage_end: pd.Timestamp | None,
                  k: int = MAX_NEWS, lookback_days: int = LOOKBACK_DAYS) -> tuple[list[Evidence], list[str]]:
    """Articles published in (cutoff - lookback, cutoff], ranked by a transparent stress-term score.

    Returns explicit notes when coverage is missing: missing news is NOT treated as neutral.
    """
    cutoff = decision_cutoff_utc(t)
    notes = []
    if news is None or coverage_end is None:
        return [], ["news: no news dataset loaded — the news agent abstains; absence of news is not evidence of calm"]
    if cutoff - pd.Timedelta(days=lookback_days) > coverage_end:
        return [], [(f"news: FNSPID coverage ends {coverage_end.date()}; no articles exist for this date in the "
                    "dataset — the news agent abstains; this is missing data, not neutral sentiment")]
    win = news[(news["published_utc"] <= cutoff) & (news["published_utc"] > cutoff - pd.Timedelta(days=lookback_days))]
    if win.empty:
        return [], [f"news: no FNSPID articles in the {lookback_days} days before the cutoff"]
    text = (win["Article_title"].fillna("") + " " + win["Lsa_summary"].fillna("")).str.lower()
    score = sum(text.str.count(term) for term in STRESS_TERMS)
    win = win.assign(_score=score.to_numpy()).sort_values(["_score", "published_utc"], ascending=[False, False])
    win = win.drop_duplicates("Article_title").head(k)
    out = []
    for _, r in win.iterrows():
        nid = "news:" + hashlib.sha1(f"{r['Url']}|{r['published_utc']}".encode()).hexdigest()[:10]
        summ = (r["Lsa_summary"] or "")[:300]
        out.append(Evidence(nid, "news", str(r["published_utc"]),
                            f"{r['published_utc']:%Y-%m-%d %H:%M} UTC, {r['Publisher'] or 'unknown publisher'}, "
                            f"ticker {r['Stock_symbol']}: \"{r['Article_title']}\"" + (f" — {summ}" if summ else ""),
                            {}))
    return out, notes


class RunEvidence:
    """Loads one run's saved artifacts once; builds bundles for any (method, date)."""

    def __init__(self, run_dir: str | Path, news: pd.DataFrame | None = None,
                 news_coverage_end: pd.Timestamp | None = None):
        from riskcausal.config import load_config
        from riskcausal.data.panel import processed_dir

        self.run_dir = Path(run_dir)
        conf = yaml.safe_load((self.run_dir / "config.yaml").read_text())
        self.cfg = load_config(conf["profile"], overrides={k: conf[k] for k in ("graph", "regime", "alerts")},
                               experiment=None)
        self.run_id = self.run_dir.name
        self.P = pd.read_parquet(self.run_dir / "predictions.parquet")
        self.F = pd.read_parquet(self.run_dir / "features.parquet")
        self.C = pd.read_parquet(processed_dir(self.cfg) / "controls_pit.parquet")
        self.metrics = json.loads((self.run_dir / "metrics.json").read_text())
        ci = self.run_dir / "ci.json"
        self.ci = json.loads(ci.read_text()) if ci.exists() else None
        gdir = Path(json.loads((self.run_dir / "graph_cache.json").read_text())["dir"])
        z = np.load(gdir / "varlingam.npz", allow_pickle=False)
        self.g_ends, self.g_nodes, self.g_B = pd.DatetimeIndex(z["ends"]), list(z["nodes"]), z["B"]
        self.news, self.news_end = news, news_coverage_end

    # ------------------------------------------------------------------ evidence builders
    def _alert(self, method: str, t: pd.Timestamp) -> Evidence:
        row = self.P[(self.P["method"] == method) & (self.P["date"] == t)]
        if row.empty:
            raise KeyError(f"no saved prediction for {method} on {t.date()}")
        r = row.iloc[0]
        rule = self.metrics["primary_rule"]
        return Evidence(f"alert:{method}@{t.date()}", "alert", str(t.date()),
                        f"{method} score {r['score']:.3f} vs threshold {r['threshold']:.3f} (rule {rule}, fold "
                        f"{int(r['fold'])}) at the close of {t.date()}: alert={'yes' if r['alert'] else 'no'}",
                        {"score": float(r["score"]), "threshold": float(r["threshold"])})

    def _macro(self, t: pd.Timestamp) -> list[Evidence]:
        out = []
        specs = [("VIXCLS", "VIX index level", 2), ("NFCI", "Chicago Fed NFCI (as published then)", 3),
                 ("BAA10Y", "Moody's Baa minus 10y Treasury spread, %", 2), ("T10Y2Y", "10y minus 2y Treasury, %", 2)]
        for sid, label, nd in specs:
            if sid not in self.C or pd.isna(self.C[sid].get(t)):
                continue
            v, obs = float(self.C[sid].loc[t]), pd.Timestamp(self.C[f"{sid}__obs_date"].loc[t])
            p = _pct(self.C[sid], t)
            out.append(Evidence(f"macro:{sid}@{t.date()}", "macro", str(t.date()),
                                f"{label} = {v:.{nd}f} (observation dated {obs.date()}, usable by {t.date()})"
                                + (f"; {p:.0f}th percentile of the previous 5 years" if p is not None else ""),
                                {"value": round(v, nd), **({"pct": round(p)} if p is not None else {})}))
        for col, label, nd, scale in [("rv21", "SPY 21-day realised volatility, annualised %", 1, 100),
                                      ("vix_chg_5d", "5-day change in VIX, points", 2, 1),
                                      ("avg_corr_63", "average pairwise 63-day correlation of the profile ETFs", 3, 1)]:
            if col in self.F and not pd.isna(self.F[col].get(t)):
                v = float(self.F[col].loc[t]) * scale
                p = _pct(self.F[col], t)
                out.append(Evidence(f"macro:{col}@{t.date()}", "macro", str(t.date()),
                                    f"{label} = {v:.{nd}f}" + (f"; {p:.0f}th percentile of the previous 5 years"
                                                                if p is not None else ""),
                                    {"value": round(v, nd), **({"pct": round(p)} if p is not None else {})}))
        return out

    def _graph(self, t: pd.Timestamp) -> list[Evidence]:
        we = self.F["graph_window_end"].get(t)
        if we is None or pd.isna(we):
            return []
        we = pd.Timestamp(we)
        out = []
        for col, label in [("vl_shd_b0", "edges added/removed in the contemporaneous graph vs the previous window"),
                           ("vl_frob_b0_ma4", "size of contemporaneous coefficient changes (Frobenius, 4-window mean)"),
                           ("vl_n_edges_b0", "number of contemporaneous edges")]:
            if col in self.F and not pd.isna(self.F[col].get(t)):
                v, p = float(self.F[col].loc[t]), _pct(self.F[col], t)
                out.append(Evidence(f"graph:{col}@{we.date()}", "graph", str(t.date()),
                                    f"estimated directed dependence (VarLiNGAM, window ending {we.date()}): {label} = "
                                    f"{v:.2f}" + (f"; {p:.0f}th percentile of the previous 5 years" if p is not None else ""),
                                    {"value": round(v, 2), **({"pct": round(p)} if p is not None else {})}))
        w = self.g_ends.get_loc(we)
        if w > 0:
            d = self.g_B[w, 0] - self.g_B[w - 1, 0]
            for i, j in sorted(np.ndindex(d.shape), key=lambda ij: -abs(d[ij]))[:3]:
                if abs(d[i, j]) < 0.01:
                    continue
                src, dst = self.g_nodes[j], self.g_nodes[i]
                out.append(Evidence(f"edge:{src}->{dst}@{we.date()}", "graph", str(t.date()),
                                    f"estimated directed-dependence coefficient {src} -> {dst} changed from "
                                    f"{self.g_B[w - 1, 0][i, j]:.2f} to {self.g_B[w, 0][i, j]:.2f} between windows "
                                    f"ending {self.g_ends[w - 1].date()} and {we.date()} (an estimate, not causality)",
                                    {"before": round(float(self.g_B[w - 1, 0][i, j]), 2),
                                     "after": round(float(self.g_B[w, 0][i, j]), 2)}))
        return out

    def reliability(self, method: str) -> list[str]:
        """Retrospective evaluation facts about the method (NOT known at alert time): displayed next to an
        explanation as a fixed, clearly labelled caveat; never given to an agent and never citable."""
        if not self.ci:
            return ["retrospective reliability: confidence intervals not computed for this run"]
        span = f"{self.ci['span'][0]} to {self.ci['span'][1]}"
        out = []
        for m in dict.fromkeys([method, "vix"]):
            r = self.ci["methods"].get(m)
            if r and r["roc"]["ci95"]:
                lo, hi = r["roc"]["ci95"]
                out.append(f"[retrospective, {span}] {m}: ROC-AUC {r['roc']['point']:.3f} (95% CI {lo:.3f}–{hi:.3f})")
        fam = method.split(":")[0]
        k = f"{fam}:combined - {fam}:controls"
        if k in self.ci["differences"]:
            d = self.ci["differences"][k]["roc"]
            out.append(f"[retrospective, {span}] adding graph-instability features to the {fam} controls model: "
                       f"ΔROC-AUC {d['point']:+.3f} (95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) — "
                       f"{'not ' if not d['supported'] else ''}distinguishable from zero")
        out.append("[retrospective] graph-instability features did not give earlier or better warning than VIX in "
                   "this project's locked evaluation; an alert is not a prediction that stress will occur")
        return out

    def bundle(self, method: str, date) -> Bundle:
        t = pd.Timestamp(date)
        items = [self._alert(method, t), *self._macro(t), *self._graph(t)]
        news, notes = retrieve_news(self.news, t, self.news_end)
        items += news
        b = Bundle(self.run_id, method, str(t.date()), str(decision_cutoff_utc(t)), {e.id: e for e in items}, notes,
                   self.reliability(method))
        assert_point_in_time(b)
        return b


def assert_point_in_time(b: Bundle) -> None:
    """Hard guard: no citable item in a bundle may postdate the decision cutoff."""
    cutoff = pd.Timestamp(b.cutoff_utc)
    for e in b.items.values():
        a = pd.Timestamp(e.available)
        a = a.tz_localize(NY).replace(hour=16).tz_convert("UTC") if a.tzinfo is None else a
        if a > cutoff:
            raise ValueError(f"evidence {e.id} available {e.available} is after the cutoff {b.cutoff_utc}")
