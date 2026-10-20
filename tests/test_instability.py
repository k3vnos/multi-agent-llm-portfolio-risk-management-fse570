import numpy as np
import pandas as pd

from riskcausal.causal.varlingam_rolling import RollingGraphs
from riskcausal.features.instability import to_daily, varlingam_instability, window_availability


def _graphs():
    B = np.zeros((3, 2, 2, 2))
    B[1, 0, 1, 0] = 0.5            # window 1 adds edge 0->1
    B[2, 0, 1, 0] = 0.5
    B[2, 0, 0, 1] = 0.3            # window 2 adds edge 1->0
    ends = pd.DatetimeIndex(["2020-01-02", "2020-01-09", "2020-01-16"])
    return RollingGraphs(ends, ["a", "b"], B, np.ones(3, int), np.array([[0, 1], [0, 1], [1, 0]]))


def test_shd_and_frobenius_on_toy_graphs():
    f = varlingam_instability(_graphs(), edge_threshold=0.05)
    assert list(f["vl_shd_b0"]) == [1.0, 1.0]
    np.testing.assert_allclose(f["vl_frob_b0"], [0.5, 0.3])
    assert list(f["vl_order_change"]) == [0.0, 2.0]
    assert f.index[0] == pd.Timestamp("2020-01-09")


def test_to_daily_uses_window_only_from_its_availability_session():
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-31"))
    f = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(["2020-01-09", "2020-01-16"]))
    avail = pd.Series(pd.DatetimeIndex(["2020-01-10", "2020-01-17"]), index=f.index)
    d = to_daily(f, sessions, avail)
    assert pd.isna(d.loc["2020-01-09", "x"])       # not usable on the window-end day
    assert d.loc["2020-01-10", "x"] == 1.0
    assert d.loc["2020-01-16", "x"] == 1.0
    assert d.loc["2020-01-17", "x"] == 2.0


def test_late_publication_inside_a_window_delays_it_and_all_later_windows():
    """A value published late (e.g. the 2016-17 Moody's outage) blocks every window that contains it."""
    rows = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=30))
    row_avail = pd.Series(rows + pd.offsets.BDay(1), index=rows)
    row_avail.iloc[12] = rows[25]                   # row 12 only published at row 25
    ends = rows[[9, 14, 19, 24, 29]]
    av = window_availability(ends, row_avail, window=5)
    assert av.iloc[0] == rows[10]                   # rows 4..9 all on time -> next session
    assert av.iloc[1] == rows[25] and av.iloc[2] == rows[25]   # windows containing row 12 wait for it
    assert av.iloc[3] == rows[25]                   # running max: later window depends on earlier ones
    assert av.iloc[4] == rows[29] + pd.offsets.BDay(1)
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=40))
    f = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=ends)
    d = to_daily(f, sessions, av)
    assert d.loc[rows[24], "x"] == 1.0              # stale: only window 0 available during the outage
    assert d.loc[rows[25], "x"] == 4.0              # then jumps to the latest available window


def test_window_availability_nat_means_never():
    rows = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10))
    row_avail = pd.Series(rows + pd.offsets.BDay(1), index=rows)
    row_avail.iloc[7] = pd.NaT
    av = window_availability(rows[[4, 9]], row_avail, window=3)
    assert not pd.isna(av.iloc[0]) and pd.isna(av.iloc[1])
