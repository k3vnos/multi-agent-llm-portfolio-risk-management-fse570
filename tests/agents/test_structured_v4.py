"""M10 v4 structured agents: facts, constrained schema, semantic validator, pipeline, cancellation.
Synthetic bundles and a fake backend are used ONLY here, in tests."""

import re

import pytest
from fake_llm import FakeBackend

from riskcausal_agents.evidence import Bundle, Evidence
from riskcausal_agents.facts import (
    Fact,
    bundle_facts,
    change_label,
    label_ok,
    level_label,
    number,
    overall_label,
    themes_of,
)
from riskcausal_agents.structured import (
    Control,
    Stop,
    check_note,
    explain,
    schema_for,
    summarise,
    validate_specialist,
)


def _bundle(news=True) -> Bundle:
    d = "2020-02-21"
    items = [
        Evidence(f"alert:m@{d}", "alert", d, "", {"score": 0.6, "threshold": 0.4}),
        Evidence(f"macro:VIXCLS@{d}", "macro", d, "", {"value": 17.08, "pct": 72}),
        Evidence(f"macro:BAA10Y@{d}", "macro", d, "", {"value": 1.9, "pct": 8}),
        Evidence(f"macro:vix_chg_5d@{d}", "macro", d, "", {"value": 2.9, "pct": 93}),
        Evidence(f"graph:vl_shd_b0@{d}", "graph", d, "", {"value": 14, "pct": 88}),
        Evidence(f"edge:SPY->TLT@{d}", "graph", d, "", {"before": 0.0, "after": -0.4}),
        Evidence(f"edge:GLD->UUP@{d}", "graph", d, "", {"before": 0.3, "after": -0.2}),
    ]
    if news:
        items.append(Evidence("news:abc", "news", "2020-02-21 14:00:00+00:00",
                              '2020-02-21 14:00 UTC, P, ticker SPY: "Stocks tumble as virus fears spread"', {}))
    return Bundle("run", "m", d, "2020-02-21 21:00:00+00:00", {e.id: e for e in items},
                  [] if news else ["news: FNSPID coverage ends; the news agent abstains"])


@pytest.fixture
def facts():
    return {f.source: f for f in number(bundle_facts(_bundle(), 0.05))}


def test_true_labels_follow_the_fixed_rules(facts):
    assert level_label(81) == "high" and level_label(80) == "typical" and level_label(19) == "low"
    assert level_label(None) == "unknown"
    assert change_label(0.3, 99) == "little changed" and change_label(2.9, 93) == "rose sharply"
    assert change_label(-1.0, 50) == "fell" and change_label(-1.0, 10) == "fell sharply"
    assert facts["macro:VIXCLS@2020-02-21"].label == "typical"
    assert facts["macro:BAA10Y@2020-02-21"].label == "low"
    assert facts["alert:m@2020-02-21"].label == "above alert line"
    assert facts["edge:SPY->TLT@2020-02-21"].label == "appeared"
    assert facts["edge:GLD->UUP@2020-02-21"].label == "sign flipped"
    assert facts["graph:vl_shd_b0@2020-02-21"].label == "high"


def test_model_prompt_never_contains_the_label(facts):
    for f in facts.values():
        if f.kind in ("level", "change"):
            assert not re.search(r"\b(high|low|typical|rose|fell)\b", f.prompt), f.prompt


def test_schema_pairs_each_fact_with_its_own_label_vocabulary(facts):
    fs = list(facts.values())
    s = schema_for(fs)
    alts = s["properties"]["claims"]["items"]["anyOf"]
    assert [a["properties"]["fact_id"]["enum"] for a in alts] == [[f.id] for f in fs]
    for a, f in zip(alts, fs, strict=True):
        assert a["properties"]["label"]["enum"] == f.allowed


def test_validator_rejects_label_mismatch_unknown_and_duplicate_ids(facts):
    vix, baa = facts["macro:VIXCLS@2020-02-21"], facts["macro:BAA10Y@2020-02-21"]
    out = {"claims": [{"fact_id": vix.id, "label": "high", "note": ""},          # wrong: percentile 72 is typical
                      {"fact_id": baa.id, "label": "low", "note": ""},           # right
                      {"fact_id": baa.id, "label": "low", "note": ""},           # duplicate
                      {"fact_id": "F99", "label": "low", "note": ""}]}           # not in the list
    v = validate_specialist(out, list(facts.values()), "2020-02-21")
    assert [c["fact_id"] for c in v["claims"]] == [baa.id]
    reasons = [r for c in v["rejected"] for r in c["reasons"]]
    assert any(r.startswith("label mismatch") for r in reasons)
    assert any(r.startswith("duplicate") for r in reasons)
    assert any(r.startswith("citation") for r in reasons)
    assert v["claims"][0]["text"] == baa.text                                    # shown text is the fact's own


@pytest.mark.parametrize("src,note,why", [
    ("macro:BAA10Y@2020-02-21", "Credit spreads were correlated with stocks.", "correlation"),
    ("macro:BAA10Y@2020-02-21", "Credit risk was high.", "risk-level"),
    ("macro:BAA10Y@2020-02-21", "The credit spread rose this week.", "change over time"),
    ("macro:BAA10Y@2020-02-21", "The credit spread was elevated.", "direction"),
    ("macro:vix_chg_5d@2020-02-21", "VIX fell over the week.", "direction"),
    ("macro:vix_chg_5d@2020-02-21", "VIX will keep rising.", "forecast"),
    ("macro:vix_chg_5d@2020-02-21", "VIX rose 7.5 points.", "numbers"),
    ("macro:vix_chg_5d@2020-02-21", "VIX rose after the Fed meeting.", "names"),
    ("edge:SPY->TLT@2020-02-21", "SPY now has an effect on TLT.", "causal or effect"),
    ("news:abc", "Stocks rallied on virus news.", "not in the headline"),
    ("macro:BAA10Y@2020-02-21", "Investors should sell credit.", "advice"),
])
def test_note_semantic_checks(facts, src, note, why):
    reasons = check_note(note, [facts[src]], "2020-02-21")
    assert any(why in r for r in reasons), reasons


@pytest.mark.parametrize("src,note", [
    ("macro:BAA10Y@2020-02-21", "The Baa credit spread was below its usual range."),
    ("macro:vix_chg_5d@2020-02-21", "VIX rose sharply over 5 sessions."),
    ("news:abc", "Stocks tumble on virus fears."),
    ("edge:SPY->TLT@2020-02-21", "A new estimated link from SPY to TLT."),
])
def test_faithful_notes_pass(facts, src, note):
    assert check_note(note, [facts[src]], "2020-02-21") == []


def test_rejected_note_keeps_the_claim_but_drops_the_note(facts):
    baa = facts["macro:BAA10Y@2020-02-21"]
    v = validate_specialist({"claims": [{"fact_id": baa.id, "label": "low", "note": "Credit risk was high."}]},
                            [baa], "2020-02-21")
    assert v["claims"][0]["note"] == "" and v["claims"][0]["note_rejected"]["reasons"]


def test_news_theme_needs_a_keyword_in_the_headline(facts):
    n = facts["news:abc"]
    assert set(themes_of(n.text)) >= {"market decline or volatility", "virus or health"}
    assert label_ok(n, "virus or health") and not label_ok(n, "trade or tariffs")


def test_overall_label_rule():
    mk = lambda lab, kind="level": Fact("F", "macro", kind, lab, [], "", "", "", "")
    assert overall_label([mk("high"), mk("low"), mk("typical")]) == "mostly unusual"
    assert overall_label([mk("high"), mk("typical"), mk("typical")]) == "mixed"
    assert overall_label([mk("typical"), mk("typical"), mk("typical"), mk("high")]) == "mostly typical"
    assert overall_label([mk("", "news")]) == "not enough evidence"


def test_pipeline_runs_four_agents_and_shown_text_is_rendered_from_facts():
    b = _bundle()
    rec = explain(b, FakeBackend(), 0.05)
    roles = [s["role"] for s in rec["specialists"]]
    assert roles == ["graph", "macro", "news"]
    by = {f["id"]: f for f in rec["facts"]}
    for s in rec["specialists"]:
        for c in s["claims"]:
            assert c["label"] == c["true_label"] or by[c["fact_id"]]["kind"] == "news"
            assert c["text"] == by[c["fact_id"]]["text"]
    syn = rec["synthesis"]
    assert syn["called_model"] and syn["validated"] and syn["overall"] == syn["true_overall"]
    s = summarise(rec)
    assert s["calls"] == 4 and s["label_correct"] <= s["asserted"]


def test_synthesis_with_wrong_overall_is_rejected_and_falls_back():
    rec = explain(_bundle(), FakeBackend(correct_synthesis=False), 0.05)
    syn = rec["synthesis"]
    if syn["true_overall"] != "mixed":
        assert not syn["validated"] and syn["text"] is None and syn["fallback_text"]
        assert "overall mismatch" in syn["rejected"][0]["reasons"][0]


def test_agent_without_evidence_abstains_without_calling_the_model():
    fb = FakeBackend()
    rec = explain(_bundle(news=False), fb, 0.05)
    news = next(s for s in rec["specialists"] if s["role"] == "news")
    assert news["abstain"] and not news["called_model"] and "coverage" in news["abstain_reason"]
    assert len(fb.calls) == 3                                                   # graph, macro, synthesis


def test_cancellation_stops_the_run():
    with pytest.raises(Stop):
        explain(_bundle(), FakeBackend(), 0.05, ctl=Control(cancelled=lambda: True))


def test_progress_events_are_emitted_in_order():
    ev = []
    explain(_bundle(), FakeBackend(), 0.05, ctl=Control(emit=ev.append))
    starts = [e["agent"] for e in ev if e["type"] == "agent_start"]
    assert starts == ["graph", "macro", "news", "synthesis"]
