"""M8 product API: historical replay is point-in-time and reports the locked results exactly."""

import json

import numpy as np
import pandas as pd
import pytest

from riskcausal.config import PROJECT_ROOT
from riskcausal_api.store import dashboard_config

RUN = PROJECT_ROOT / "outputs" / "runs" / dashboard_config()["runs"]["primary"]
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="M5 locked run not present")


@pytest.fixture(scope="module")
def c():
    from fastapi.testclient import TestClient

    from riskcausal_api.main import app

    return TestClient(app)


def test_pit_percentile_ignores_the_future():
    from riskcausal_api.replay import pit_percentile

    idx = pd.bdate_range("2010-01-01", periods=800)
    s = pd.Series(np.random.default_rng(0).normal(size=800), index=idx)
    a = pit_percentile(s, years=1, min_hist=50)
    s2 = s.copy()
    s2.iloc[500:] += 100
    b = pit_percentile(s2, years=1, min_hist=50)
    pd.testing.assert_series_equal(a.iloc[:500], b.iloc[:500])


def test_series_end_and_snapshot_never_return_later_dates(c):
    d = "2020-02-24"
    s = c.get("/replay/series", params={"end": d}).json()
    assert max(s["dates"]) <= d
    snap = c.get(f"/replay/snapshot/{d}").json()
    assert snap["date"] == d and max(snap["sii"]["spark_dates"]) <= d
    assert all(a["start"] <= d and a["end"] <= d for a in snap["recent_alerts"])
    assert snap["graph_window_end"] < d
    assert c.get("/replay/snapshot/2020-02-22").json()["date"] == "2020-02-21"     # weekend -> last session before


def test_regime_is_the_filtered_probability_not_hindsight_labels(c):
    lf = pd.read_parquet(RUN / "labels_filtered_test.parquet")
    snap = c.get("/replay/snapshot/2020-03-02").json()
    assert snap["regime"]["p_stress"] == pytest.approx(float(lf.loc["2020-03-02", "p_stress"]))
    s = c.get("/replay/series", params={"start": "2020-01-02", "end": "2020-01-31"}).json()
    assert "retrospective_stress" in s and "regime_p_stress" in s      # hindsight only under an explicit name


def test_snapshot_values_equal_saved_artifacts(c):
    P = pd.read_parquet(RUN / "predictions.parquet")
    F = pd.read_parquet(RUN / "features.parquet")
    d = "2020-02-24"
    snap = c.get(f"/replay/snapshot/{d}").json()
    m = P[(P.method == "bayes_logit:combined") & (P.date == d)].iloc[0]
    sii = P[(P.method == "inst_composite") & (P.date == d)].iloc[0]
    assert snap["model"]["score"] == pytest.approx(m.score) and snap["model"]["alert"] == bool(m.alert)
    assert snap["sii"]["value"] == pytest.approx(sii.score)
    assert snap["vix"]["value"] == pytest.approx(F.loc[d, "vix"]) and snap["nfci"]["value"] == pytest.approx(F.loc[d, "nfci"])
    assert snap["status"]["level"] == ("alert" if m.alert else snap["status"]["level"])


def test_status_rule_is_documented_and_consistent(c):
    for d in ("2019-06-03", "2020-02-24", "2021-06-01", "2024-08-05"):
        s = c.get(f"/replay/snapshot/{d}").json()
        st = s["status"]
        if s["model"]["alert"]:
            assert st["level"] == "alert"
        elif st["unusual"]:
            assert st["level"] == "unusual" and all(u["percentile"] > 80 for u in st["unusual"])
        else:
            assert st["level"] == "normal"
        assert "Not a forecast" in st["rule"]


def test_network_uses_published_window_and_matches_graph_cache(c):
    n = c.get("/replay/network/2020-02-24", params={"lag": 4}).json()
    assert n["window_end"] < "2020-02-24" and n["previous_window_end"] < n["window_end"]
    gdir = json.loads((RUN / "graph_cache.json").read_text())["dir"]
    z = np.load(f"{gdir}/varlingam.npz")
    ends = list(pd.DatetimeIndex(z["ends"]).strftime("%Y-%m-%d"))
    np.testing.assert_allclose(n["B0"], z["B"][ends.index(n["window_end"]), 0].round(4))
    assert all(e["change"] in ("appeared", "disappeared", "sign flipped", "strengthened", "weakened") for e in n["edges"])
    assert "not proven causal" in n["caveat"]


def test_agents_news_is_before_cutoff_or_explicitly_unavailable(c):
    a = c.get("/replay/agents/2020-01-10").json()
    cutoff = pd.Timestamp(a["cutoff_utc"])
    news = a["agents"]["news"]["evidence"]
    assert news and all(pd.Timestamp(i["available"]) <= cutoff for i in news)
    later = c.get("/replay/agents/2023-03-08").json()
    assert later["agents"]["news"]["evidence"] == [] and later["agents"]["news"]["abstained"]
    assert "not neutral" in later["agents"]["news"]["unavailable"][0]
    assert any("missing data, not neutral" in b["text"] for b in later["synthesis"]["briefing"])


def test_llm_draft_is_unverified_and_flagged(c):
    a = c.get("/replay/agents/2020-01-10").json()
    d = a["llm_draft"]
    assert d["label"].startswith("UNVERIFIED") and d["status"].startswith("unverified draft")
    assert any("unfaithful" in f for f in d["flags"])
    assert len(a["llm_draft_available_for"]) == 24
    assert c.get("/replay/agents/2020-02-24").json()["llm_draft"] is None


def test_research_equals_locked_metrics(c):
    r = c.get("/replay/research").json()
    M = json.loads((RUN / "metrics.json").read_text())
    rows = {m["method"]: m for m in r["primary"]["data"]["methods"]}
    for m, v in M["by_rule"][M["primary_rule"]]["common_span_metrics"].items():
        assert rows[m]["roc_auc"] == v["day"]["roc_auc"]
        assert r["lead_times"][m] == [x for x in v["event"]["lead_times"] if x is not None]
    assert round(rows["vix"]["roc_auc"], 3) == 0.721 and round(rows["inst_composite"]["roc_auc"], 3) == 0.601
    assert "did not give earlier or better warning" in r["conclusion"]
    assert r["reproducibility"]["status"] == "bit-identical"


def test_ui_index_is_not_cached_and_routes_stay_get_only(c):
    from riskcausal_api.main import app

    # every route, including those of included routers (FastAPI >= 0.162 nests them out of app.routes)
    ops = {(path, m.upper()) for path, item in app.openapi()["paths"].items() for m in item}
    assert len(ops) > 20
    assert {o for o in ops if o[1] != "GET"} == {("/agents/run", "POST"), ("/agents/jobs/{jid}/cancel", "POST")}
    if (PROJECT_ROOT / "frontend" / "dist" / "index.html").exists():
        r = c.get("/")
        assert r.status_code == 200 and "Structural Instability Monitor" in r.text
        assert r.headers.get("cache-control") == "no-cache"
