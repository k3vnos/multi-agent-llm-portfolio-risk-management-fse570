"""Deterministic, evidence-based descriptions (no LLM). Every sentence is computed from numbers, so the meaning of
credit spreads, VIX levels vs changes, edge direction and sign changes cannot be misread.
"""

from __future__ import annotations


def pct_label(p: float | None) -> str:
    if p is None:
        return "no 5-year history"
    return "low" if p < 20 else "high" if p > 80 else "typical"


def _pct(p: float | None) -> str:
    return "" if p is None else f", {p:.0f}th percentile of the previous 5 years ({pct_label(p)})"


def indicator(series: str, value: float, pct: float | None) -> str:
    lab = pct_label(pct)
    if series == "BAA10Y":
        meaning = {"low": "investors demanded a smaller-than-usual yield premium for Baa corporate credit "
                          "(credit-risk pricing was benign, not elevated)",
                   "high": "investors demanded a larger-than-usual yield premium for Baa corporate credit "
                           "(credit-risk pricing was elevated)",
                   "typical": "the premium for Baa corporate credit was within its usual range"}
        return f"Baa–10y Treasury spread {value:.2f} pp{_pct(pct)}: " + meaning.get(lab, "no 5-year comparison")
    if series == "VIXCLS":
        meaning = {"low": "option-implied 30-day S&P 500 volatility was below its usual range",
                   "high": "option-implied 30-day S&P 500 volatility was above its usual range",
                   "typical": "option-implied 30-day S&P 500 volatility was within its usual range"}
        return f"VIX level {value:.2f}{_pct(pct)}: " + meaning.get(lab, "no 5-year comparison")
    if series == "vix_chg_5d":
        direction = "rose" if value > 0 else "fell" if value < 0 else "was unchanged"
        return (f"VIX {direction} by {abs(value):.2f} points over 5 sessions{_pct(pct)} — this describes the "
                "change, not the level of volatility")
    if series == "NFCI":
        side = "tighter" if value > 0 else "looser"
        return (f"NFCI {value:.3f} (as published then){_pct(pct)}: financial conditions {side} than the long-run "
                "average (0 = average; positive = tighter)")
    if series == "T10Y2Y":
        shape = "inverted (2-year yield above 10-year)" if value < 0 else "upward-sloping"
        return (f"10y–2y Treasury spread {value:.2f} pp{_pct(pct)}: the curve was {shape}; this is a level, it says "
                "nothing by itself about whether the curve was flattening or steepening")
    if series == "rv21":
        return f"SPY 21-day realised volatility {value:.1f}% annualised{_pct(pct)}"
    if series == "avg_corr_63":
        return f"average 63-day pairwise correlation of the profile ETFs {value:.3f}{_pct(pct)}"
    return f"{series} {value:.3f}{_pct(pct)}"


def edge_change(src: str, dst: str, before: float, after: float, tol: float = 0.05) -> dict:
    """Classify the change of the estimated effect of src on dst (B0[dst, src]) between two windows.

    `tol` is the run's edge threshold: an edge exists only if |coefficient| > tol (same rule as the features).
    """
    if abs(before) <= tol < abs(after):
        kind = "appeared"
    elif abs(after) <= tol < abs(before):
        kind = "disappeared"
    elif abs(before) <= tol and abs(after) <= tol:
        kind = "absent"
    elif before * after < 0:
        kind = "sign flipped"
    else:
        kind = "strengthened" if abs(after) > abs(before) else "weakened"
    sign = "positive" if after > 0 else "negative" if after < 0 else "zero"
    text = (f"Estimated effect of {src} on {dst} ({src} → {dst}) {kind}: {before:+.2f} → {after:+.2f}"
            + (f" (now {sign})" if kind in ("appeared", "sign flipped") else "")
            + ". Estimated directed dependence (VarLiNGAM), not a causal effect.")
    return {"source": src, "target": dst, "before": round(before, 4), "after": round(after, 4), "change": kind,
            "text": text}


def alert(method: str, score: float, threshold: float) -> str:
    rel = "above" if score >= threshold else "below"
    return (f"{method} score {score:.3f} was {rel} its validation-chosen threshold {threshold:.3f} "
            f"(margin {score - threshold:+.3f}); {'an alert was issued' if rel == 'above' else 'no alert'}. "
            "An alert is a model output, not a forecast that stress will occur.")
