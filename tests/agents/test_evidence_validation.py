"""M6 evidence validation (gate before M7). Synthetic evidence and a fake backend are used ONLY here, in tests."""

from pathlib import Path

import pandas as pd
import pytest

from riskcausal_agents.agents import explain, run_specialist, run_synthesis
from riskcausal_agents.backends import AgentsDisabledError, DisabledBackend, OllamaBackend
from riskcausal_agents.evidence import (
    Bundle,
    Evidence,
    assert_point_in_time,
    decision_cutoff_utc,
    retrieve_news,
)
from riskcausal_agents.validator import validate_claim, validate_output

RUN = Path(__file__).resolve().parents[2] / "outputs" / "runs" / "20261030T225627Z_full_c1cccc3aa2_m4_selected"


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _news(rows):
    return pd.DataFrame([{"Article_title": t, "Lsa_summary": "", "Url": f"u{i}", "Publisher": "P",
                          "Stock_symbol": "SPY", "published_utc": _utc(ts)}
                         for i, (t, ts) in enumerate(rows)])


# ------------------------------------------------------------------ point-in-time news retrieval
@pytest.mark.parametrize("day,offset_utc", [("2020-02-21", 5), ("2020-03-10", 4)])    # EST, then EDT (DST)
def test_news_cutoff_is_1600_new_york_with_dst(day, offset_utc):
    cutoff = decision_cutoff_utc(pd.Timestamp(day))
    assert cutoff == pd.Timestamp(f"{day} {16 + offset_utc}:00", tz="UTC")
    before, after = cutoff - pd.Timedelta(minutes=1), cutoff + pd.Timedelta(minutes=1)
    news = _news([("Stocks fall in market sell-off", before), ("Stocks fall again in market sell-off", after)])
    items, _notes = retrieve_news(news, pd.Timestamp(day), pd.Timestamp("2020-06-11", tz="UTC"))
    assert [e.text.split('"')[1] for e in items] == ["Stocks fall in market sell-off"]
    assert all(pd.Timestamp(e.available) <= cutoff for e in items)


def test_news_lookback_window_is_bounded():
    t = pd.Timestamp("2020-02-21")
    old = decision_cutoff_utc(t) - pd.Timedelta(days=7, minutes=1)
    items, _ = retrieve_news(_news([("Old market news", old)]), t, pd.Timestamp("2020-06-11", tz="UTC"))
    assert items == []


def test_missing_coverage_is_explicit_not_neutral():
    items, notes = retrieve_news(_news([("x", "2020-01-02")]), pd.Timestamp("2023-03-08"),
                                 pd.Timestamp("2020-06-11", tz="UTC"))
    assert items == [] and "not neutral" in notes[0] and "2020-06-11" in notes[0]
    items, notes = retrieve_news(None, pd.Timestamp("2023-03-08"), None)
    assert items == [] and "not evidence of calm" in notes[0]


# ------------------------------------------------------------------ claim validation
EV = {
    "macro:VIXCLS@2020-02-21": Evidence("macro:VIXCLS@2020-02-21", "macro", "2020-02-21",
                                        "VIX index level = 15.56; 40th percentile of the previous 5 years",
                                        {"value": 15.56, "pct": 40}),
    "news:abc": Evidence("news:abc", "news", "2020-02-20 21:00:00+00:00",
                         '2020-02-20 21:00 UTC, P, ticker SPY: "Coronavirus outbreak fears weigh on stocks"'),
}


def _claim(text, ids):
    return validate_claim({"text": text, "evidence_ids": ids}, EV, "2020-02-21")


def test_valid_claim_passes():
    assert _claim("VIX stood at 15.56, around the 40th percentile of the previous 5 years.",
                  ["macro:VIXCLS@2020-02-21"]) == []
    assert _claim("Headlines mentioned coronavirus outbreak fears.", ["news:abc"]) == []


@pytest.mark.parametrize("text,ids,why", [
    ("VIX was 15.56.", [], "schema"),
    ("VIX was 15.56.", ["macro:NOPE"], "citation"),
    ("VIX was 18.20.", ["macro:VIXCLS@2020-02-21"], "numbers"),
    ("Fears about COVID-19 lockdowns in Italy were reported.", ["news:abc"], "names"),
    ("Coronavirus fears caused the VIX level.", ["news:abc", "macro:VIXCLS@2020-02-21"], "language"),
    ("The graph signal outperforms VIX.", ["macro:VIXCLS@2020-02-21"], "language"),
    ("Stocks will fall further.", ["news:abc"], "language"),
    ("VIX rose because of coronavirus fears.", ["news:abc", "macro:VIXCLS@2020-02-21"], "language"),
    ("By 2020-03-16 VIX was 15.56.", ["macro:VIXCLS@2020-02-21"], "dates"),
])
def test_invalid_claims_are_rejected(text, ids, why):
    reasons = _claim(text, ids)
    assert any(r.startswith(why) for r in reasons), reasons


def test_validate_output_keeps_only_valid_claims_and_never_trusts_model_abstain_flag():
    out = {"claims": [{"text": "VIX stood at 15.56.", "evidence_ids": ["macro:VIXCLS@2020-02-21"]},
                      {"text": "Stocks will crash.", "evidence_ids": ["news:abc"]}], "abstain": True}
    v = validate_output(out, EV, "2020-02-21")
    assert len(v["claims"]) == 1 and len(v["rejected"]) == 1 and v["abstain"] is False


def test_bundle_rejects_future_evidence():
    b = Bundle("r", "m", "2020-02-21", str(decision_cutoff_utc(pd.Timestamp("2020-02-21"))),
               {"news:late": Evidence("news:late", "news", "2020-02-21 22:00:00+00:00", "late")}, [])
    with pytest.raises(ValueError, match="after the cutoff"):
        assert_point_in_time(b)


# ------------------------------------------------------------------ agents, abstention, backends
class FakeBackend:
    """Test-only stand-in for the LLM: returns canned JSON and records calls."""
    name = "fake"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def complete_json(self, system, prompt):
        self.calls += 1
        return self.outputs.pop(0), {}


def _bundle(with_news=True):
    items = {"alert:m@2020-02-21": Evidence("alert:m@2020-02-21", "alert", "2020-02-21", "m score 0.4 vs threshold 0.3"),
             **{k: v for k, v in EV.items() if with_news or v.kind != "news"}}
    notes = [] if with_news else ["news: FNSPID coverage ends 2020-06-11 — not neutral"]
    return Bundle("r", "m", "2020-02-21", str(decision_cutoff_utc(pd.Timestamp("2020-02-21"))), items, notes,
                  ["[retrospective] m: ROC-AUC 0.671"])


def test_specialist_without_evidence_abstains_without_calling_model():
    fb = FakeBackend([])
    r = run_specialist("news", _bundle(with_news=False), fb)
    assert r["abstain"] and fb.calls == 0 and "not neutral" in r["abstain_reason"]
    r = run_specialist("graph", _bundle(), fb)
    assert r["abstain"] and fb.calls == 0


def test_synthesis_abstains_on_insufficient_evidence_without_calling_model():
    fb = FakeBackend([])
    spec = [{"role": "macro", "claims": [{"text": "VIX stood at 15.56.", "evidence_ids": ["macro:VIXCLS@2020-02-21"]}],
             "abstain": False}, {"role": "news", "claims": [], "abstain": True, "abstain_reason": "none"}]
    r = run_synthesis(_bundle(), spec, fb)
    assert r["abstain"] and fb.calls == 0 and "insufficient" in r["abstain_reason"]


def test_end_to_end_only_validated_claims_reach_the_explanation():
    fb = FakeBackend([
        {"claims": [{"text": "Headlines mentioned coronavirus outbreak fears.", "evidence_ids": ["news:abc"]}]},
        {"claims": [{"text": "VIX stood at 15.56.", "evidence_ids": ["macro:VIXCLS@2020-02-21"]},
                    {"text": "VIX will rise.", "evidence_ids": ["macro:VIXCLS@2020-02-21"]}]},
        {"claims": [{"text": "Headlines mentioned coronavirus outbreak fears while VIX stood at 15.56.",
                     "evidence_ids": ["news:abc", "macro:VIXCLS@2020-02-21"]},
                    {"text": "The graph proves contagion.", "evidence_ids": ["news:abc"]}],
         "uncertainty": "The evidence does not show whether stress followed."},
    ])
    rec = explain(_bundle(), fb)
    assert fb.calls == 3                               # graph specialist had no evidence: no call
    texts = [c["text"] for c in rec["explanation"]["claims"]]
    assert texts == ["Headlines mentioned coronavirus outbreak fears while VIX stood at 15.56."]
    assert rec["synthesis"]["rejected"][0]["reasons"]
    assert rec["method_reliability_retrospective"] and "not causality" in rec["disclaimer"]


def test_backends_are_local_only_and_disabled_by_default():
    with pytest.raises(ValueError, match="must be local"):
        OllamaBackend(host="http://example.com:11434")
    with pytest.raises(ValueError, match="must be local"):
        OllamaBackend(host="10.0.0.5:11434")
    OllamaBackend(host="http://127.0.0.1:11434")
    with pytest.raises(AgentsDisabledError):
        DisabledBackend().complete_json("s", "p")


# ------------------------------------------------------------------ real saved run (skipped if absent)
@pytest.mark.skipif(not RUN.exists(), reason="M5 locked run not present")
def test_real_bundle_is_point_in_time_and_has_no_labels_or_results():
    from riskcausal_agents.evidence import RunEvidence

    re_ = RunEvidence(RUN)
    P = re_.P[(re_.P["method"] == "bayes_logit:combined") & re_.P["alert"]]
    for t in P["date"].iloc[[0, len(P) // 2, -1]]:
        b = re_.bundle("bayes_logit:combined", t)
        assert {e.kind for e in b.items.values()} <= {"alert", "macro", "graph", "news"}
        for e in b.items.values():
            assert "ROC" not in e.text and "onset" not in e.text.lower() and "stress regime" not in e.text.lower()
            if e.id.startswith("edge:") or e.id.startswith("graph:"):
                assert pd.Timestamp(e.id.split("@")[1]) < t
        assert b.reliability and all(r.startswith("[retrospective") for r in b.reliability)
