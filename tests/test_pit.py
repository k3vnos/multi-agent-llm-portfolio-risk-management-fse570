"""PIT macro access (LEAKAGE_AUDIT rows 1–2): synthetic vintages with a later revision."""

import pandas as pd

from riskcausal.data.calendar import nyse_sessions
from riskcausal.data.pit import (
    latest_value_per_vintage,
    pit_from_observations,
    pit_from_vintages,
    release_availability,
)

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


def test_release_based_availability_for_daily_series():
    """Row 2 (M3): FRED posts daily series after the close; usable the session after the first-release date."""
    sessions = nyse_sessions("2023-03-01", "2023-04-30")
    obs = pd.DatetimeIndex(["2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09"])
    rel = pd.Series(pd.DatetimeIndex(["2023-03-06",       # same day  -> next session
                                      "2023-03-08",       # +1 day    -> 2 sessions after obs
                                      "2023-03-24"]),     # outage    -> weeks later
                    index=obs[:3])                        # 03-09 untracked -> fallback
    av = release_availability(obs, rel, sessions, lag=1, fallback_lag=3)
    assert list(av) == [pd.Timestamp("2023-03-07"), pd.Timestamp("2023-03-09"), pd.Timestamp("2023-03-27"),
                        pd.Timestamp("2023-03-14")]
    assert (av > obs).all()


def test_release_availability_never_earlier_than_lag_floor():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    obs = pd.DatetimeIndex(["2023-03-10"])
    rel = pd.Series(pd.DatetimeIndex(["2023-03-03"]), index=obs)     # bogus early release date
    assert release_availability(obs, rel, sessions, lag=1, fallback_lag=3)[0] == pd.Timestamp("2023-03-13")


def test_observation_series_with_release_dates():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    s = pd.Series([10.0, 11.0], index=pd.DatetimeIndex(["2023-03-09", "2023-03-10"]))
    rel = pd.Series(pd.DatetimeIndex(["2023-03-10", "2023-03-13"]), index=s.index)
    pit = pit_from_observations(s, sessions, lag=1, first_release=rel, fallback_lag=3)
    assert pd.isna(pit.loc["2023-03-10", "value"])
    assert pit.loc["2023-03-13", "value"] == 10.0          # Friday obs posted Monday -> usable Tuesday
    assert pit.loc["2023-03-14", "value"] == 11.0


def test_late_old_observation_does_not_displace_newer_known_value():
    sessions = nyse_sessions("2023-03-01", "2023-03-31")
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2023-03-06", "2023-03-08"]))
    rel = pd.Series(pd.DatetimeIndex(["2023-03-20", "2023-03-08"]), index=s.index)   # 03-06 published late
    pit = pit_from_observations(s, sessions, lag=1, first_release=rel, fallback_lag=3, max_staleness_days=None)
    assert pit.loc["2023-03-09", "value"] == 2.0
    assert pit.loc["2023-03-21", "value"] == 2.0          # not reverted to the older 1.0
