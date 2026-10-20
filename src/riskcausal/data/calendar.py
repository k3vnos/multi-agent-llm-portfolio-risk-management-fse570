"""NYSE trading calendar — the single daily timeline every series is aligned to."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=8)
def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index.normalize(), name="date")


def add_bdays(dates: pd.DatetimeIndex, n: int, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map each date to the n-th NYSE session strictly after it (n>=1), or to itself/next session if n==0."""
    pos = sessions.searchsorted(dates, side="right" if n >= 1 else "left") + max(n - 1, 0)
    out = pd.DatetimeIndex([sessions[p] if p < len(sessions) else pd.NaT for p in pos])
    return out
