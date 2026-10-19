from riskcausal.config import PROJECT_ROOT
from riskcausal.data.manifest import write_manifest


def test_manifest_reports_failures_and_pending_vintages(tmp_path):
    report = {
        "snapshot": "2026-01-01", "study_start": "2005-01-01", "fred_api_key_present": False,
        "etfs": [{"ticker": "AAA", "name": "x", "status": "FAILED", "error": "boom"}],
        "fred": [{"series_id": "NFCI", "name": "n", "freq": "W", "revised": True, "status": "ok",
                  "first_date": "1971-01-08", "covers_study_start": True,
                  "alfred": "PENDING — FRED_API_KEY not set (free key required)"}],
        "fnspid": {"status": "FAILED", "error": "offline"},
        "gdelt": {"status": "FAILED", "error": "offline"},
    }
    out = tmp_path / "M.md"
    write_manifest(report, out, PROJECT_ROOT / "data" / "x.json")
    text = out.read_text()
    assert "ETF downloads failed: ['AAA']" in text
    assert "lacking point-in-time vintages: ['NFCI']" in text
    assert "Do not edit by hand" in text
