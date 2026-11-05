"""M7 integration: the read-only API serves exactly the locked M5 results (and nothing else)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskcausal.config import PROJECT_ROOT
from riskcausal_api import describe
from riskcausal_api.store import MERGE_GAP, dashboard_config

PRIMARY = dashboard_config()["runs"]["primary"]
RUN = PROJECT_ROOT / "outputs" / "runs" / PRIMARY
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="M5 locked run not present")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from riskcausal_api.main import app

    return TestClient(app)


def test_api_is_get_only_and_local():
    from riskcausal.config import load_config
    from riskcausal_api.main import app

    # every route, including those of included routers (FastAPI >= 0.162 nests them out of app.routes)
    ops = {(path, m.upper()) for path, item in app.openapi()["paths"].items() for m in item}
    assert len(ops) > 20
    assert {o for o in ops if o[1] != "GET"} == {("/agents/run", "POST"), ("/agents/jobs/{jid}/cancel", "POST")}
    assert dashboard_config()["api"]["host"] == "127.0.0.1" and load_config("full").api.host == "127.0.0.1"
    src = (PROJECT_ROOT / "src" / "riskcausal_api" / "main.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+riskcausal\.pipeline", src, re.MULTILINE)
    assert "backends" not in src                               # never touches the LLM
    import riskcausal.pipeline as pl
    assert MERGE_GAP == pl.MERGE_GAP


def test_primary_run_is_m5_locked_audited_and_reproducible(client):
    r = client.get("/runs").json()
    prim = next(x for x in r["runs"] if x["role"] == "primary")
    assert prim["run_id"] == "20261030T225627Z_full_c1cccc3aa2_m4_selected" and prim["audit_passed"]
    assert prim["synthetic"] is False
    assert r["reproducibility_of_primary"]["status"] == "bit-identical"


def test_metrics_equal_saved_metrics_and_cis(client):
    out = client.get(f"/runs/{PRIMARY}/metrics").json()
    assert out["meta"]["kind"] == "research_result"
    M = json.loads((RUN / "metrics.json").read_text())
    C = json.loads((RUN / "ci.json").read_text())
    saved = M["by_rule"][M["primary_rule"]]["common_span_metrics"]
    rows = {x["method"]: x for x in out["data"]["methods"]}
    assert set(rows) == set(saved)
    for m, v in saved.items():
        assert rows[m]["roc_auc"] == v["day"]["roc_auc"] and rows[m]["pr_auc"] == v["day"]["pr_auc"]
        assert rows[m]["n_detected"] == v["event"]["n_detected"] and rows[m]["n_onsets"] == v["event"]["n_onsets"]
        assert rows[m]["roc_ci95"] == C["methods"][m]["roc"]["ci95"]
    d = {x["comparison"]: x for x in out["data"]["differences"]}
    assert d["bayes_logit:combined - bayes_logit:controls"]["delta_roc"] == \
        C["differences"]["bayes_logit:combined - bayes_logit:controls"]["roc"]["point"]
    assert round(rows["vix"]["roc_auc"], 3) == 0.721 and round(rows["bayes_logit:instability"]["roc_auc"], 3) == 0.438


def test_timeline_equals_saved_predictions(client):
    P = pd.read_parquet(RUN / "predictions.parquet")
    for m in ("vix", "bayes_logit:combined"):
        t = client.get(f"/runs/{PRIMARY}/timeline", params={"method": m}).json()["data"]
        g = P[P["method"] == m].sort_values("date")
        assert t["dates"] == [str(d.date()) for d in g["date"]]
        np.testing.assert_array_equal(np.array(t["score"], float), g["score"].to_numpy(float))
        assert t["alert"] == g["alert"].astype(bool).tolist()


def test_graph_snapshot_matches_cached_graph_and_uses_lagged_window(client):
    g = client.get(f"/runs/{PRIMARY}/graph/2020-02-20").json()
    assert g["meta"]["kind"] == "illustrative" and "not causality" in g["data"]["caveat"]
    d = g["data"]
    assert pd.Timestamp(d["window_end"]) < pd.Timestamp("2020-02-20")
    gdir = Path(json.loads((RUN / "graph_cache.json").read_text())["dir"])
    z = np.load(gdir / "varlingam.npz")
    w = list(pd.DatetimeIndex(z["ends"]).date.astype(str)).index(d["window_end"])
    np.testing.assert_allclose(np.array(d["B0"]), z["B"][w, 0].round(4))
    assert all(abs(e["coef"]) > d["edge_threshold"] for e in d["edges"])


def test_evidence_is_point_in_time_with_deterministic_descriptions(client):
    e = client.get(f"/runs/{PRIMARY}/evidence/bayes_logit:combined/2024-12-19").json()
    assert e["meta"]["kind"] == "evidence"
    items = {i["id"].split("@")[0]: i for i in e["data"]["items"]}
    baa = items["macro:BAA10Y"]
    assert baa["values"]["pct"] < 20 and "smaller-than-usual" in baa["description"]   # low spread != high risk
    assert "change, not the level" in items["macro:vix_chg_5d"]["description"]
    cutoff = pd.Timestamp(e["data"]["cutoff_utc"])
    for i in e["data"]["items"]:
        a = pd.Timestamp(i["available"])
        a = a.tz_localize("America/New_York").replace(hour=16).tz_convert("UTC") if a.tzinfo is None else a
        assert a <= cutoff
    assert any("2020-06-11" in n for n in e["data"]["coverage_notes"])              # news gap stated, not neutral
    assert all(x.startswith("[retrospective") for x in e["data"]["method_reliability_retrospective"])


def test_llm_drafts_are_labelled_unverified_flagged_or_suppressed(client):
    lst = client.get(f"/runs/{PRIMARY}/explanations").json()
    assert lst["meta"]["kind"] == "unverified_draft" and len(lst["data"]["drafts"]) == 24
    for d in lst["data"]["drafts"]:
        x = client.get(f"/runs/{PRIMARY}/explanations/{d['alert_id']}").json()["data"]
        assert x["label"].startswith("UNVERIFIED")
        assert x["status"].startswith(("unverified draft", "suppressed"))
        hc = x["hand_check"]["R2_faithful_claims"]
        k, n = (int(v) for v in hc.split("/"))
        if k < n:
            assert x["status"] == "unverified draft — flagged" and any("unfaithful" in f for f in x["flags"])
        if x["status"].startswith("suppressed"):
            assert x["claims"] == []
    flagged = client.get(f"/runs/{PRIMARY}/explanations/bayes_logit:combined@2024-12-19").json()["data"]
    assert flagged["status"].endswith("flagged")


def test_synthetic_run_is_refused(client, monkeypatch):
    from riskcausal_api import store

    r = store.runs()[PRIMARY]
    monkeypatch.setattr(type(r), "synthetic", property(lambda self: True))
    assert client.get(f"/runs/{PRIMARY}/metrics").status_code == 409


@pytest.mark.parametrize("before,after,kind", [(0.0, 0.42, "appeared"), (0.88, 0.0, "disappeared"),
                                               (0.41, -0.48, "sign flipped"), (-0.14, -0.65, "strengthened"),
                                               (0.16, 0.11, "weakened"), (0.0, -0.05, "absent")])
def test_edge_change_classification(before, after, kind):
    c = describe.edge_change("SPY", "IWM", before, after, tol=0.05)
    assert c["change"] == kind and "SPY → IWM" in c["text"] and "not a causal effect" in c["text"]


def test_indicator_descriptions_cannot_invert_meaning():
    assert "elevated" in describe.indicator("BAA10Y", 3.5, 95) and "benign" in describe.indicator("BAA10Y", 1.4, 2)
    assert "inverted" in describe.indicator("T10Y2Y", -0.3, 3) and "flattening" in describe.indicator("T10Y2Y", 0.9, 1)
    assert "fell by 2.00" in describe.indicator("vix_chg_5d", -2.0, 10)
    assert "tighter" in describe.indicator("NFCI", 0.2, 90) and "looser" in describe.indicator("NFCI", -0.6, 15)
