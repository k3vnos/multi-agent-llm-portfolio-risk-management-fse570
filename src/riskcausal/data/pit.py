"""Point-in-time (PIT) access: the single place that answers "what was known about series X at end of day t".

Rules (LEAKAGE_AUDIT rows 1–2):
- Revised series (ALFRED): at session t we use the most recent observation in the latest vintage whose release
  date r satisfies session(r + lag) <= t. Before the first vintage the value is missing — never back-filled.
- Non-revised daily series: observation for date d becomes usable at the first NYSE session strictly after the
  date FRED first published it (ALFRED initial-release date; FRED posts most daily series after the 4pm close),
  and never earlier than `lag` sessions after d. Where ALFRED does not track the first release (VIXCLS before
  2010-11, BAA10Y/T10Y2Y before 2014) a conservative fallback of `fallback_lag` sessions is used
  (LEAKAGE_AUDIT row 2; the fallback covers >= 99% of tracked releases outside documented outages).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskcausal.data.calendar import add_bdays


def latest_value_per_vintage(vint: pd.DataFrame) -> pd.DataFrame:
    """For each ALFRED release date, the newest observation (date, value) contained in that vintage.

    `vint` columns: date, realtime_start, realtime_end, value (one row per value-validity interval).
    """
    first_release = vint.groupby("date")["realtime_start"].min()
    vintages = pd.DatetimeIndex(sorted(vint["realtime_start"].unique()))
    # newest observation date first released on or before each vintage date
    newest = first_release.groupby(first_release).apply(lambda s: s.index.max()).sort_index().cummax()
    obs_for_vintage = newest.reindex(vintages, method="ffill")
    by_date = {d: g for d, g in vint[vint["date"].isin(set(obs_for_vintage.dropna()))].groupby("date")}
    out = []
    for r, obs_date in obs_for_vintage.items():
        if pd.isna(obs_date):
            continue
        g = by_date[obs_date]
        rows = g[(g["realtime_start"] <= r) & (g["realtime_end"] >= r)]
        if rows.empty:  # observation withdrawn in this vintage — skip rather than guess
            continue
        out.append({"vintage": pd.Timestamp(r), "obs_date": obs_date,
                    "value": float(rows.sort_values("realtime_start")["value"].iloc[-1])})
    return pd.DataFrame(out)


def _asof_on_sessions(avail: pd.DataFrame, sessions: pd.DatetimeIndex, max_staleness_days: int | None) -> pd.DataFrame:
    avail = avail.dropna(subset=["available"]).sort_values(["available", "obs_date"])
    # A late-published old observation (e.g. after an outage) must not displace a newer one already known:
    # keep only rows that advance the newest observation date known so far.
    newest_before = avail["obs_date"].cummax().shift(1)
    avail = avail[newest_before.isna() | (avail["obs_date"] > newest_before)]
    left = pd.DataFrame({"session": sessions})
    m = pd.merge_asof(left, avail, left_on="session", right_on="available", direction="backward")
    m = m.set_index("session")
    if max_staleness_days is not None:
        stale = (m.index - m["obs_date"]).dt.days > max_staleness_days
        m.loc[stale, "value"] = float("nan")
    m.index.name = "date"
    return m[["value", "obs_date", "available"]]


def pit_from_vintages(vint: pd.DataFrame, sessions: pd.DatetimeIndex, lag: int,
                      max_staleness_days: int | None = None) -> pd.DataFrame:
    lv = latest_value_per_vintage(vint)
    lv["available"] = add_bdays(pd.DatetimeIndex(lv["vintage"]), lag, sessions)
    return _asof_on_sessions(lv, sessions, max_staleness_days)


def release_availability(obs_dates: pd.DatetimeIndex, first_release: pd.Series | None,
                         sessions: pd.DatetimeIndex, lag: int, fallback_lag: int) -> pd.DatetimeIndex:
    """First session at whose close each observation is known (see module docstring)."""
    obs_dates = pd.DatetimeIndex(obs_dates)
    floor = add_bdays(obs_dates, lag, sessions)
    fallback = add_bdays(obs_dates, fallback_lag, sessions)
    if first_release is None:
        return fallback
    rel = pd.DatetimeIndex(first_release.reindex(obs_dates))
    tracked = ~rel.isna()
    after = add_bdays(pd.DatetimeIndex(rel[tracked]), 1, sessions)
    out = fallback.to_series(index=obs_dates)
    out[tracked] = pd.DatetimeIndex(np.maximum(after.values, floor[tracked].values))
    return pd.DatetimeIndex(out.values)


def pit_from_observations(s: pd.Series, sessions: pd.DatetimeIndex, lag: int,
                          max_staleness_days: int | None = 10, first_release: pd.Series | None = None,
                          fallback_lag: int | None = None) -> pd.DataFrame:
    """Non-revised series: value for obs date d usable from its release-based availability session.

    Without `first_release`/`fallback_lag` the plain rule "lag-th session after d" applies (used in tests and
    for series whose publication is not delayed).
    """
    s = s.dropna()
    df = pd.DataFrame({"obs_date": s.index, "value": s.values})
    if fallback_lag is None:
        df["available"] = add_bdays(pd.DatetimeIndex(s.index), lag, sessions)
    else:
        df["available"] = release_availability(pd.DatetimeIndex(s.index), first_release, sessions, lag,
                                               fallback_lag)
    return _asof_on_sessions(df, sessions, max_staleness_days)
