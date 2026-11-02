"""Deterministic validation of agent output. The LLM is never trusted to police itself.

A claim is kept only if ALL of these hold:
1. schema: non-empty text (<= 60 words) and >= 1 evidence id;
2. citations: every cited id exists in the bundle given to that agent (so it is point-in-time by construction);
3. numbers: every number in the text matches a number in the cited evidence (dates/years exempt);
4. names: every capitalised term or acronym in the text appears in the cited evidence text (hindsight guard:
   the model cannot introduce events or entities its evidence does not mention);
5. language: no causal-proof, predictive-superiority or forecasting language;
6. dates: no date later than the decision date.
Rejected claims are kept with their reasons for auditing; they are never shown as explanation.
"""

from __future__ import annotations

import re

import pandas as pd

from riskcausal_agents.evidence import Bundle, Evidence

MAX_WORDS = 60
FORBIDDEN = [
    (r"\bcaus(e|es|ed|ing|al|ally|ality|ation)\b", "causal language (graphs are estimated directed dependence)"),
    (r"\b(prove|proves|proved|proven|proof)\b", "claims of proof"),
    (r"\b(outperform\w*|superior|beats?|beat|better than|more accurate than)\b", "predictive-superiority claim"),
    (r"\breliabl[ey]\b.{0,30}\b(predict|warn|forecast|signal)", "reliability claim"),
    (r"\b(will|is going to|are going to)\s+(crash|fall|drop|decline|rise|rally|collapse|enter)", "forecast language"),
    (r"\b(guarantee\w*|certain(ly)?|definitely)\b", "certainty language"),
    (r"\b(led to|resulted in|drove|driven by|because of|due to)\b", "causal attribution"),
]
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?")
DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
NAME = re.compile(r"\b([A-Z][A-Za-z&]*[A-Z]+[A-Za-z]*|[A-Z]{2,}|[A-Z][a-z]{2,})\b")
COMMON_CAPS = {"The", "This", "These", "That", "There", "In", "On", "At", "As", "A", "An", "And", "But", "Over",
               "During", "With", "While", "Its", "It", "No", "Not", "Both", "Several", "Evidence", "News",
               "Graph", "Macro", "Alert", "Score", "Threshold", "Estimated", "Directed", "Dependence", "Signals",
               "Market", "Markets", "Stress", "Volatility", "Index", "Level", "Percentile", "Window", "Edge",
               "Edges", "Coefficient", "Spread", "Treasury", "Yield", "Change", "Changes", "Recent", "Data",
               "UTC", "ROC", "AUC", "Insufficient", "Article", "Articles", "Headline", "Headlines", "Multiple", "Some", "One", "Two", "Three", "Four", "Five", "Price", "Prices", "Stocks", "Stock",
               "Shares", "Ticker", "Publisher", "Close", "Previous", "Year", "Years", "Days", "Week", "Weeks"}


def _numbers(text: str) -> list[float]:
    text = DATE.sub(" ", text)
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _evidence_numbers(evs: list[Evidence]) -> list[float]:
    nums = []
    for e in evs:
        nums += _numbers(e.text)
        nums += [float(v) for v in e.values.values() if isinstance(v, int | float)]
    return nums


def _num_ok(x: float, pool: list[float]) -> bool:
    if 1990 <= x <= 2035 and float(x).is_integer():          # years
        return True
    return any(abs(x - y) <= max(0.011, 0.02 * abs(y)) or abs(abs(x) - abs(y)) <= max(0.011, 0.02 * abs(y))
               for y in pool)


def normalise_id(raw: str, allowed: dict[str, Evidence]) -> str:
    """Strip formatting the model adds around ids ([...], quotes, spaces); restore a dropped "kind:" prefix only
    when exactly one allowed id ends with the given text. Anything else stays as written (and will be rejected)."""
    x = str(raw).strip().strip("[]()\"'` ").strip()
    if x in allowed:
        return x
    hits = [k for k in allowed if k.endswith(":" + x)]
    return hits[0] if len(hits) == 1 else x


def validate_claim(claim: dict, allowed: dict[str, Evidence], decision_date: str) -> list[str]:
    reasons = []
    text = str(claim.get("text", "")).strip()
    ids = claim.get("evidence_ids") or []
    if isinstance(ids, list):
        ids = [normalise_id(i, allowed) for i in ids]
        claim["evidence_ids"] = ids
    if not text or len(text.split()) > MAX_WORDS:
        reasons.append("schema: empty or too long")
    if not isinstance(ids, list) or not ids:
        reasons.append("schema: no evidence ids")
        ids = []
    unknown = [i for i in ids if i not in allowed]
    if unknown:
        reasons.append(f"citation: unknown or not-allowed evidence id(s) {unknown}")
    cited = [allowed[i] for i in ids if i in allowed]
    if cited:
        pool = _evidence_numbers(cited)
        bad = [x for x in _numbers(text) if not _num_ok(x, pool)]
        if bad:
            reasons.append(f"numbers: {bad} not found in cited evidence")
        ev_text = " ".join(e.text for e in cited).lower() + " " + " ".join(ids).lower()
        names = {n for n in NAME.findall(text) if n not in COMMON_CAPS}
        missing = sorted(n for n in names if n.lower() not in ev_text)
        if missing:
            reasons.append(f"names: {missing} not mentioned in cited evidence (possible hindsight)")
    for pat, why in FORBIDDEN:
        if re.search(pat, text, flags=re.IGNORECASE):
            reasons.append(f"language: {why}")
    for m in re.finditer(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text):
        if pd.Timestamp(m.group(0)) > pd.Timestamp(decision_date):
            reasons.append(f"dates: {m.group(0)} is after the decision date {decision_date}")
    return reasons


def validate_output(out: dict, allowed: dict[str, Evidence], decision_date: str) -> dict:
    claims = out.get("claims") if isinstance(out.get("claims"), list) else []
    kept, rejected = [], []
    for c in claims:
        c = c if isinstance(c, dict) else {"text": str(c)}
        r = validate_claim(c, allowed, decision_date)
        (rejected if r else kept).append({**c, **({"reasons": r} if r else {})})
    unc = str(out.get("uncertainty", ""))[:400]
    unc_bad = [why for pat, why in FORBIDDEN if re.search(pat, unc, flags=re.IGNORECASE)]
    ev_text = " ".join(e.text for e in allowed.values()).lower()
    unc_bad += [f"name {n!r} not in evidence" for n in NAME.findall(unc)
                if n not in COMMON_CAPS and n.lower() not in ev_text]
    return {"claims": kept, "rejected": rejected, "abstain": bool(out.get("abstain")) and not kept,
            "abstain_reason": out.get("abstain_reason", ""), "uncertainty": "" if unc_bad else unc,
            **({"uncertainty_rejected": {"text": unc, "reasons": unc_bad}} if unc_bad else {})}


def allowed_for(bundle: Bundle, kinds: set[str]) -> dict[str, Evidence]:
    return {k: v for k, v in bundle.items.items() if v.kind in kinds}
