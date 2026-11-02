"""M7 integration: every dashboard page renders from the locked M5 run, local-only and without telemetry."""

import tomllib

import pytest
from streamlit.testing.v1 import AppTest

from riskcausal.config import PROJECT_ROOT
from riskcausal_api.store import dashboard_config

RUN = PROJECT_ROOT / "outputs" / "runs" / dashboard_config()["runs"]["primary"]
APP = str(PROJECT_ROOT / "dashboard" / "app.py")
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="M5 locked run not present")


def test_streamlit_is_local_and_has_no_telemetry():
    cfg = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["address"] == "127.0.0.1" and cfg["browser"]["gatherUsageStats"] is False


def test_client_refuses_non_local_api():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))
    from client import ApiClient

    with pytest.raises(ValueError, match="must be local"):
        ApiClient("http://example.com:8000")


@pytest.mark.parametrize("page", ["Overview", "Stress timeline", "Model comparison", "Graph snapshot",
                                  "Alert explorer", "Exploratory: banking stress"])
def test_page_renders_without_errors(page, monkeypatch):
    monkeypatch.delenv("RISKCAUSAL_API_URL", raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_overview_shows_locked_m5_numbers():
    at = AppTest.from_file(APP, default_timeout=120).run()
    vals = {m.label: m.value for m in at.metric}
    assert vals["VIX level — ROC-AUC"] == "0.721"
    assert vals["Graph instability only (Bayes) — ROC-AUC"] == "0.438"
    assert vals["Adding instability to controls (Bayes) — ΔROC-AUC"] == "-0.050"
    text = " ".join(m.value for m in at.markdown)
    assert "bit-identical" in text and "passed" in text


def test_alert_explorer_marks_drafts_unverified():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Alert explorer").run()
    at.selectbox[1].set_value("2024-12-19").run()
    assert not at.exception
    labels = [e.value for e in at.error] + [w.value for w in at.warning]
    assert any("UNVERIFIED LLM DRAFT" in v for v in labels)
    md = " ".join(m.value for m in at.markdown)
    assert "smaller-than-usual yield premium" in md                         # deterministic Baa description
