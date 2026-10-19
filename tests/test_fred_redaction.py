from riskcausal.data.ingest_fred import redact


def test_api_key_is_redacted_from_messages():
    msg = "400 Client Error for url: https://api.stlouisfed.org/fred/series/observations?series_id=NFCI&api_key=abc123def&file_type=json"
    out = redact(msg)
    assert "abc123def" not in out
    assert "api_key=<redacted>" in out
