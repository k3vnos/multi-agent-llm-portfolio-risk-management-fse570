"""Rows 1, 2, 9 on a tiny synthetic raw snapshot (tests only): ragged listings are never back-filled, controls and
graph rows use release-based availability, and revised series come from vintages."""

import numpy as np
import pandas as pd
import pytest

from riskcausal.config import load_config
from riskcausal.data.calendar import nyse_sessions
from riskcausal.data.panel import build_panel, processed_dir
from riskcausal.registry import write_json

SNAP = "2099-01-01"


@pytest.fixture()
def built(tmp_path):
    raw = tmp_path / "raw"
    write_json(raw / "latest_snapshot.json", {"snapshot": SNAP})
    s = nyse_sessions("2005-01-01", "2006-12-31")
    rng = np.random.default_rng(0)
    for t, start in [("SPY", "2005-01-03"), ("TLT", "2005-01-03"), ("GLD", "2005-06-01")]:   # GLD listed late
        idx = s[s >= start]
        (raw / "yfinance" / SNAP).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx).to_parquet(
            raw / "yfinance" / SNAP / f"{t}.parquet")
    (raw / "fred" / SNAP).mkdir(parents=True)
    (raw / "fred_releases" / SNAP).mkdir(parents=True)
    for sid in ("VIXCLS", "BAA10Y"):
        pd.DataFrame({sid: 20 + rng.normal(size=len(s)).cumsum()}, index=s).to_parquet(raw / "fred" / SNAP /
                                                                                      f"{sid}.parquet")
    tracked = s[s >= "2006-01-01"]
    pd.DataFrame({"first_release": tracked}, index=tracked).to_parquet(raw / "fred_releases" / SNAP /
                                                                       "VIXCLS.parquet")
    rel = pd.Series(tracked, index=tracked)
    pos = s.get_indexer(tracked)
    rel[:] = s[np.minimum(pos + 1, len(s) - 1)]                        # BAA10Y posted the next NYSE session
    rel.loc["2006-06-01":"2006-06-30"] = pd.Timestamp("2006-08-01")     # outage
    rel.to_frame("first_release").to_parquet(raw / "fred_releases" / SNAP / "BAA10Y.parquet")
    (raw / "alfred" / SNAP).mkdir(parents=True)
    weeks = pd.date_range("2005-01-07", "2006-12-29", freq="W-FRI")
    pd.DataFrame({"date": weeks, "realtime_start": weeks + pd.Timedelta(days=5),
                  "realtime_end": pd.Timestamp("2262-04-11"), "value": rng.normal(size=len(weeks))}).to_parquet(
        raw / "alfred" / SNAP / "NFCI_vintages.parquet")
    cfg = load_config("dev", overrides={
        "data_dir": str(tmp_path), "start_date": "2005-01-01", "etfs": ["SPY", "TLT", "GLD"],
        "fred_series": ["VIXCLS", "BAA10Y", "NFCI"],
        "graph_nodes": {"etfs": ["SPY", "TLT", "GLD"], "macro": ["VIXCLS", "BAA10Y"]}})
    build_panel(cfg)
    d = processed_dir(cfg)
    return (cfg, pd.read_parquet(d / "returns.parquet"), pd.read_parquet(d / "graph_avail.parquet")["available"],
            pd.read_parquet(d / "controls_pit.parquet"), s)


def test_late_listing_is_not_back_filled(built):
    _, rets, _, _, _ = built
    assert rets["GLD"].loc[:"2005-06-01"].isna().all()
    assert rets["GLD"].first_valid_index() == pd.Timestamp("2005-06-02")


def test_graph_rows_wait_for_release_and_outage(built):
    _, _, av, _, s = built
    lag = pd.Series(s.get_indexer(pd.DatetimeIndex(av.reindex(s))) - np.arange(len(s)), index=s)
    assert (lag.loc["2006-02-01":"2006-05-31"] == 2).all()           # BAA10Y next-day release -> 2 sessions
    assert (lag.loc["2005-02-01":"2005-12-23"] == 3).all()           # untracked -> fallback 3 sessions
    jun = av.loc["2006-06-01":"2006-06-30"]
    assert (jun >= pd.Timestamp("2006-08-02")).all()                 # outage rows wait for publication


def test_controls_use_only_published_values(built):
    _, _, _, ctrl, _ = built
    for sid in ("VIXCLS", "BAA10Y", "NFCI"):
        od = ctrl[f"{sid}__obs_date"].dropna()
        assert (od < od.index).all()
    od = ctrl["BAA10Y__obs_date"].loc["2006-06-02":"2006-06-30"].dropna()
    assert (od < pd.Timestamp("2006-06-01")).all()                   # outage values unknown during June
    aug = ctrl["BAA10Y__obs_date"].loc["2006-07-06":"2006-08-31"]
    assert aug.is_monotonic_increasing                               # late June release never displaces July
    assert (aug >= pd.Timestamp("2006-07-01")).all()
    nf = ctrl["NFCI__obs_date"].dropna()
    assert ((nf.index - nf).dt.days >= 6).all()                      # Friday obs, released Wednesday, used Thursday
