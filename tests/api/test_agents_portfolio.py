"""M10: live agent job runner (fake backend), Portfolio Lens point-in-time maths, and the route policy."""

import time

import numpy as np
import pandas as pd
import pytest

from riskcausal.config import PROJECT_ROOT
from riskcausal_api.store import dashboard_config

RUN = PROJECT_ROOT / "outputs" / "runs" / dashboard_config()["runs"]["primary"]
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="M5 locked run not present")
APPROVED_POST = {"/agents/run", "/agents/jobs/{jid}/cancel"}


@pytest.fixture(scope="module")
def c():
    from fastapi.testclient import TestClient

    from riskcausal_api.main import app

    return TestClient(app)


@pytest.fixture
def fake(monkeypatch):
    from fake_llm import FakeBackend

    from riskcausal_api import agents_live

    holder = {"slow": 0.0}
    monkeypatch.setattr(agents_live, "_backend_factory", lambda: FakeBackend(slow=holder["slow"]))
    agents_live._jobs.clear()
    yield holder
    for j in agents_live._jobs.values():
        j["cancel"] = True


def _wait(c, jid, until=("done", "error", "cancelled", "timeout"), secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        j = c.get(f"/agents/jobs/{jid}").json()
        if j["status"] in until:
            return j
        time.sleep(0.1)
    raise AssertionError(f"job stuck: {j['status']}")


# ------------------------------------------------------------------ route policy
def test_only_the_two_approved_routes_accept_post(c):
    from riskcausal_api.main import app

    ops = {(path, m.upper()) for path, item in app.openapi()["paths"].items() for m in item}
    assert {o for o in ops if o[1] != "GET"} == {(p, "POST") for p in APPROVED_POST}
    assert ("/portfolio/analyze", "GET") in ops and ("/replay/agents/{date}", "GET") in ops


# ------------------------------------------------------------------ live job runner
def test_live_run_reports_real_progress_and_a_validated_trace(c, fake):
    jid = c.post("/agents/run", json={"date": "2020-02-21"}).json()["job_id"]
    j = _wait(c, jid)
    assert j["status"] == "done", j["error"]
    starts = [e["agent"] for e in j["events"] if e["type"] == "agent_start"]
    assert starts == ["graph", "macro", "news", "synthesis"]
    rec = j["result"]
    assert "historical data" in j["notice"] and "not real-time" in j["notice"]
    for s in rec["specialists"]:
        for cl in s["claims"]:
            f = next(x for x in rec["facts"] if x["id"] == cl["fact_id"])
            assert cl["text"] == f["text"]
            assert f["available"] <= "2020-02-21 21:00" or f["available"] <= "2020-02-21"


def test_live_run_never_changes_the_score_or_alert(c, fake):
    before = c.get("/replay/snapshot/2020-02-21").json()["model"]
    _wait(c, c.post("/agents/run", json={"date": "2020-02-21"}).json()["job_id"])
    assert c.get("/replay/snapshot/2020-02-21").json()["model"] == before


def test_cancel_stops_a_running_job_and_busy_is_refused(c, fake):
    fake["slow"] = 2.0
    jid = c.post("/agents/run", json={"date": "2020-02-21"}).json()["job_id"]
    assert c.post("/agents/run", json={"date": "2020-02-21"}).status_code == 409
    c.post(f"/agents/jobs/{jid}/cancel")
    assert _wait(c, jid)["status"] == "cancelled"


def test_job_timeout(c, fake, monkeypatch):
    from riskcausal_api import agents_live

    monkeypatch.setattr(agents_live, "JOB_TIMEOUT", 0.3)
    fake["slow"] = 2.0
    assert _wait(c, c.post("/agents/run", json={"date": "2020-02-21"}).json()["job_id"])["status"] == "timeout"


def test_model_unavailable_is_an_error_not_a_fake_answer(c, monkeypatch):
    from riskcausal_agents.backends import OllamaBackend
    from riskcausal_api import agents_live

    agents_live._jobs.clear()
    monkeypatch.setattr(agents_live, "_backend_factory", lambda: OllamaBackend(host="http://127.0.0.1:9"))
    assert c.get("/agents/status").json()["available"] is False
    j = _wait(c, c.post("/agents/run", json={"date": "2020-02-21"}).json()["job_id"])
    assert j["status"] == "error" and "unavailable" in j["error"] and j["result"] is None


def test_live_run_with_portfolio_adds_portfolio_facts(c, fake):
    j = _wait(c, c.post("/agents/run", json={"date": "2020-03-16", "portfolio": "SPY:60,IEF:40"}).json()["job_id"])
    assert j["status"] == "done", j["error"]
    src = [f["source"] for f in j["result"]["facts"]]
    assert "portfolio:vol_63d" in src and any(s.startswith("portfolio:risk_share:") for s in src)
    assert c.post("/agents/run", json={"date": "2020-03-16", "portfolio": "SPY:70"}).status_code == 422


# ------------------------------------------------------------------ Portfolio Lens
def test_parse_validates_weights():
    from riskcausal_api.portfolio import parse

    assert parse("spy:60, IEF:40") == {"SPY": 0.6, "IEF": 0.4}
    for bad in ("SPY:60", "SPY:60,XXX:40", "SPY:50,SPY:50", "SPY:110,IEF:-10", ""):
        with pytest.raises(ValueError):
            parse(bad)


def test_euler_contributions_sum_to_100_percent():
    from riskcausal_api.portfolio import euler

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)) @ rng.normal(size=(4, 4)), columns=list("abcd"))
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))
    rc = euler(X.cov(), w)
    assert rc.sum() == pytest.approx(1.0)
    assert float(w @ X.cov() @ w) == pytest.approx(sum(rc * float(w @ X.cov() @ w)))


def test_portfolio_analysis_is_point_in_time():
    """Changing every return after D must not change anything computed for D."""
    from riskcausal_api.portfolio import analyze, returns

    R = returns()
    d = "2020-02-21"
    net = {"edges": [], "window_end": d, "previous_window_end": d}
    a = analyze("SPY:50,TLT:30,GLD:20", d, R=R, net=net)
    R2 = R.copy()
    R2.loc[R2.index > pd.Timestamp(d)] = 0.5
    b = analyze("SPY:50,TLT:30,GLD:20", d, R=R2, net=net)
    assert a["metrics"] == b["metrics"] and a["holdings"] == b["holdings"] and a["data_through"] == d


def test_var_and_drawdown_match_direct_calculation():
    from riskcausal_api.portfolio import analyze, returns

    R = returns()
    d = "2020-03-16"
    a = analyze("SPY:60,IEF:40", d, R=R, net={"edges": [], "window_end": d, "previous_window_end": d})
    p = (R.loc[:d, ["SPY", "IEF"]].dropna() @ pd.Series({"SPY": 0.6, "IEF": 0.4})).iloc[-252:]
    assert a["metrics"]["var95_hist"] == pytest.approx(-np.quantile(p, 0.05) * 100, abs=0.01)
    v = (1 + p).cumprod()
    assert a["metrics"]["max_drawdown_252d"] == pytest.approx(float((v / v.cummax() - 1).min()) * 100, abs=0.01)
    assert sum(h["risk_share"] for h in a["holdings"]) == pytest.approx(100, abs=0.05)
    assert a["metrics"]["vol_63d"] > a["metrics"]["vol_252d"]          # March 2020 stress shows in the short window


def test_portfolio_endpoints(c):
    u = c.get("/portfolio/universe").json()
    assert len(u["etfs"]) == 25 and len(u["examples"]) >= 3 and "EXPLORATORY" in u["label"]
    for ex in u["examples"]:
        assert c.get("/portfolio/analyze", params={"spec": ex["spec"], "date": "2020-02-21"}).status_code == 200
    a = c.get("/portfolio/analyze", params={"spec": "XLF:40,KRE:30,HYG:30", "date": "2023-03-13"}).json()
    assert a["data_through"] == "2023-03-13" and sorted(a["network"]["nodes"]["XLF"]) == ["KRE", "XLF"]
    assert c.get("/portfolio/analyze", params={"spec": "SPY:50", "date": "2020-02-21"}).status_code == 422
    assert c.get("/portfolio/analyze", params={"spec": "HYG:100", "date": "2006-01-03"}).status_code in (404, 422)


def test_portfolio_facts_have_rule_based_labels():
    from riskcausal_agents.facts import level_label
    from riskcausal_api.portfolio import analyze, portfolio_facts

    a = analyze("SPY:60,IEF:40", "2020-03-16")
    fs = portfolio_facts(a)
    vol = next(f for f in fs if f.source == "portfolio:vol_63d")
    assert vol.label == level_label(a["metrics"]["vol_63d_pct"]) and vol.label not in vol.prompt
    top = next(f for f in fs if f.kind == "share")
    assert top.label == "largest contributor" and top.source.endswith(a["holdings"][0]["ticker"])
