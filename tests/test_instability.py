import numpy as np
import pandas as pd

from riskcausal.causal.varlingam_rolling import RollingGraphs
from riskcausal.features.instability import to_daily, varlingam_instability


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


def test_to_daily_lags_availability():
    sessions = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-31"))
    f = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(["2020-01-09", "2020-01-16"]))
    d = to_daily(f, sessions, lag=1)
    assert pd.isna(d.loc["2020-01-09", "x"])       # not usable on the window-end day
    assert d.loc["2020-01-10", "x"] == 1.0
    assert d.loc["2020-01-16", "x"] == 1.0
    assert d.loc["2020-01-17", "x"] == 2.0
