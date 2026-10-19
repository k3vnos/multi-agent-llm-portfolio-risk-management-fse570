"""PIT macro access (LEAKAGE_AUDIT rows 1–2): synthetic vintages with a later revision."""

import pandas as pd

from riskcausal.data.calendar import nyse_sessions
from riskcausal.data.pit import latest_value_per_vintage, pit_from_observations, pit_from_vintages

FAR = pd.Timestamp("2262-04-11")


def _vintages():
    # Week-1 obs (2023-03-03) first released Wed 2023-03-08 as 1.0, revised to 9.0 on 2023-03-15.
    # Week-2 obs (2023-03-10) first released 2023-03-15 as 2.0.
    return pd.DataFrame([
        {"date": pd.Timestamp("2023-03-03"), "realtime_start": pd.Timestamp("2023-03-08"),
         "realtime_end": pd.Timestamp("2023-03-14"), "value": 1.0},
        {"date": pd.Timestamp("2023-03-03"), "realtime_start": pd.Timestamp("2023-03-15"),
         "realtime_end": FAR, "value": 9.0},
        {"date": pd.Timestamp("2023-03-10"), "realtime_start": pd.Timestamp("2023-03-15"),
         "realtime_end": FAR, "value": 2.0},
    ])


def test_latest_value_per_vintage_uses_value_as_published():
    lv = latest_value_per_vintage(_vintages()).set_index("vintage")
    assert lv.loc["2023-03-08", "value"] == 1.0          # not the later revision 9.0
    assert lv.loc["2023-03-15", "obs_date"] == pd.Timestamp("2023-03-10")
    assert lv.loc["2023-03-15", "value"] == 2.0


def test_pit_respects_release_lag_and_never_backfills():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    pit = pit_from_vintages(_vintages(), sessions, lag=1)
    assert pd.isna(pit.loc["2023-03-08", "value"])        # released that day -> usable next session only
    assert pit.loc["2023-03-09", "value"] == 1.0
    assert pit.loc["2023-03-15", "value"] == 1.0          # 2023-03-15 release not yet usable
    assert pit.loc["2023-03-16", "value"] == 2.0
    assert pit["value"].loc[:"2023-03-08"].isna().all()   # nothing before first release
    assert (pit["available"].dropna() <= pit.index[pit["available"].notna()]).all()


def test_revised_value_never_appears_before_its_vintage():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    pit = pit_from_vintages(_vintages(), sessions, lag=1)
    assert 9.0 not in set(pit["value"].loc[:"2023-03-15"].dropna())


def test_observation_series_lagged_one_session():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    s = pd.Series([10.0, 11.0], index=pd.DatetimeIndex(["2023-03-09", "2023-03-10"]))
    pit = pit_from_observations(s, sessions, lag=1)
    assert pit.loc["2023-03-10", "value"] == 10.0         # Friday's value not usable on Friday
    assert pit.loc["2023-03-13", "value"] == 11.0         # usable Monday
