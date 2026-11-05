"""Structured facts for the v4 agents (M10).

Every point-in-time evidence item becomes one *fact*: a short id (F1, F2, …), the agent that may use it, a kind, the
label that is TRUE according to the saved numbers, the labels the model may choose from, the raw numbers the model is
shown (without the label), and a sentence rendered by fixed rules. The LLM selects facts and asserts a label; the
validator compares that label with the true one. Shown sentences are always rendered from the fact itself, so a model
misreading cannot change what a fact says.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from riskcausal_agents.evidence import Bundle
from riskcausal_api import describe

LEVEL_LABELS = ["high", "typical", "low", "unknown"]
CHANGE_LABELS = ["rose sharply", "rose", "little changed", "fell", "fell sharply"]
EDGE_LABELS = ["appeared", "disappeared", "strengthened", "weakened", "sign flipped"]
ALERT_LABELS = ["above alert line", "below alert line"]
SHARE_LABELS = ["largest contributor", "above its weight", "in line with its weight", "below its weight"]
# news: the model assigns a theme; a theme is valid only if one of its keywords is in the headline or summary
THEMES = {
    "market decline or volatility": ["sell-off", "selloff", "plunge", "tumble", "crash", "slump", "volatil", "drop",
                                     "fall", "fell", "sink", "slide", "losses", "rout", "correction", "lower", "weigh",
                                     "fears", "worries", "slip", "sheds", "retreat", "selling", "down "],
    "virus or health": ["virus", "coronavirus", "covid", "outbreak", "pandemic", "epidemic"],
    "interest rates or central bank": ["fed ", "federal reserve", "rate cut", "rate hike", "rates", "powell",
                                       "central bank", "yield", "treasur", "ecb"],
    "credit or banking": ["credit", "bank", "default", "debt", "lending", "loan", "liquidity", "bond"],
    "trade or tariffs": ["tariff", "trade war", "trade deal", "trade talks", "china trade", "import", "export"],
    "economy or recession": ["recession", "gdp", "jobs", "unemployment", "economy", "economic", "inflation",
                             "manufacturing"],
    "company news": ["earnings", "revenue", "profit", "guidance", "quarter", "ceo", "shares", "stock", "dividend",
                     "acquisition", "merger", "downgrad", "upgrad", "analyst", "price target", "outperform",
                     "underweight", "overweight", "neutral"],
}
NEWS_LABELS = list(THEMES)
ALL_LABELS = sorted(set(LEVEL_LABELS + CHANGE_LABELS + EDGE_LABELS + ALERT_LABELS + SHARE_LABELS + NEWS_LABELS))

# direction implied by a label (used by the semantic validator on the model's free-text note)
POLARITY = {"high": "up", "low": "down", "rose sharply": "up", "rose": "up", "fell": "down", "fell sharply": "down",
            "appeared": "up", "strengthened": "up", "disappeared": "down", "weakened": "down",
            "above alert line": "up", "below alert line": "down", "above its weight": "up",
            "below its weight": "down", "largest contributor": "up"}
UNUSUAL = {"high", "low", "rose sharply", "fell sharply", "appeared", "disappeared", "sign flipped",
           "above alert line", "largest contributor", "above its weight"}

GRAPH_NAME = {"vl_shd_b0": "links added or removed vs the previous window",
              "vl_frob_b0_ma4": "size of relationship changes (4-window average)",
              "vl_n_edges_b0": "number of active estimated links"}
MACRO_NAME = {"VIXCLS": "VIX level", "NFCI": "NFCI financial conditions (as published then)",
              "BAA10Y": "Baa–10y Treasury credit spread (pp)", "T10Y2Y": "10y–2y Treasury spread (pp)",
              "rv21": "SPY 21-day realised volatility (% annualised)", "vix_chg_5d": "5-day change in VIX (points)",
              "avg_corr_63": "average 63-day pairwise co-movement of the profile ETFs"}


@dataclass
class Fact:
    id: str                       # short id shown to the model: F1, F2, …
    agent: str                    # graph | macro | news (portfolio facts are routed to graph/macro)
    kind: str                     # level | change | edge | alert | news | share
    label: str                    # true label from the saved numbers ("" for news: validated by keywords)
    allowed: list[str]
    prompt: str                   # what the model sees: names and numbers, never the label
    text: str                     # rendered by fixed rules — the only wording ever shown for this fact
    source: str                   # evidence id (point-in-time) or "portfolio:<...>"
    available: str
    values: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def level_label(p) -> str:
    return "unknown" if p is None else ("high" if p > 80 else "low" if p < 20 else "typical")


def change_label(chg: float, pct: float | None, eps: float = 0.5) -> str:
    if abs(chg) < eps:
        return "little changed"
    if chg > 0:
        return "rose sharply" if pct is not None and pct > 80 else "rose"
    return "fell sharply" if pct is not None and pct < 20 else "fell"


def themes_of(text: str) -> list[str]:
    """Themes whose keywords occur in the text (word-start match, so "Markets Down;" matches "down")."""
    t = text.lower()
    return [k for k, words in THEMES.items()
            if any(re.search(r"\b" + re.escape(w.strip()) + (r"\b" if w.endswith(" ") else ""), t) for w in words)]


def _pct(p) -> str:
    return f", percentile within the previous 5 years = {p}" if p is not None else ", no 5-year history"


def bundle_facts(b: Bundle, edge_threshold: float) -> list[Fact]:
    """Facts for the market-wide evidence of one decision date (ids assigned by `number`)."""
    out: list[Fact] = []
    for e in b.items.values():
        v = e.values
        if e.kind == "alert":
            above = v["score"] >= v["threshold"]
            out.append(Fact("", "macro", "alert", ALERT_LABELS[0] if above else ALERT_LABELS[1], ALERT_LABELS,
                            f"warning-model score {v['score']:.3f}; alert line {v['threshold']:.3f}",
                            describe.alert(b.method, v["score"], v["threshold"]), e.id, e.available, v))
        elif e.kind == "macro":
            s = e.id.split(":")[1].split("@")[0]
            p = v.get("pct")
            kind, lab, allowed = (("change", change_label(v["value"], p), CHANGE_LABELS) if s == "vix_chg_5d"
                                  else ("level", level_label(p), LEVEL_LABELS))
            text = describe.indicator(s, v["value"], p)
            if s == "avg_corr_63":   # wording kept neutral; the agents must not make correlation claims
                text = text.replace("correlation", "co-movement")
            out.append(Fact("", "macro", kind, lab, allowed, f"{MACRO_NAME.get(s, s)} = {v['value']:g}{_pct(p)}",
                            text, e.id, e.available, v))
        elif e.id.startswith("edge:"):
            src, dst = e.id.split(":", 1)[1].split("@")[0].split("->")
            c = describe.edge_change(src, dst, v["before"], v["after"], tol=edge_threshold)
            if c["change"] == "absent":
                continue
            out.append(Fact("", "graph", "edge", c["change"], EDGE_LABELS,
                            edge_prompt(src, dst, v["before"], v["after"], edge_threshold),
                            c["text"], e.id, e.available, v))
        elif e.kind == "graph":
            feat = e.id.split(":")[1].split("@")[0]
            p = v.get("pct")
            name = GRAPH_NAME.get(feat, feat)
            out.append(Fact("", "graph", "level", level_label(p), LEVEL_LABELS, f"{name} = {v['value']:g}{_pct(p)}",
                            f"Estimated directed dependence: {name} = {v['value']:g}"
                            + (f", {p}th percentile of the previous 5 years ({level_label(p)})" if p is not None else "")
                            + ". A statistical estimate, not causality.", e.id, e.available, v))
        elif e.kind == "news":
            th = themes_of(e.text)
            out.append(Fact("", "news", "news", "", NEWS_LABELS, re.sub(r"\s+", " ", e.text)[:260], e.text,
                            e.id, e.available, {"themes": th}))
    return out


def edge_prompt(src: str, dst: str, before: float, after: float, thr: float) -> str:
    on = lambda x: "link" if abs(x) > thr else "no link"
    return (f"estimated link {src} → {dst}: last window {before:+.2f} ({on(before)}), now {after:+.2f} ({on(after)}); "
            f"a link exists when |coefficient| > {thr}")


def number(facts: list[Fact]) -> list[Fact]:
    for i, f in enumerate(facts, 1):
        f.id = f"F{i}"
    return facts


def label_ok(f: Fact, label: str) -> bool:
    if f.kind == "news":
        return label in f.values.get("themes", [])
    return label == f.label


def overall_label(facts: list[Fact]) -> str:
    """Deterministic synthesis label from the true labels of the validated facts (news excluded)."""
    scored = [f for f in facts if f.kind != "news" and f.label != "unknown"]
    if not scored:
        return "not enough evidence"
    share = sum(f.label in UNUSUAL for f in scored) / len(scored)
    return "mostly unusual" if share >= 0.6 else "mostly typical" if share <= 0.3 else "mixed"


OVERALL_LABELS = ["mostly unusual", "mixed", "mostly typical", "not enough evidence"]
