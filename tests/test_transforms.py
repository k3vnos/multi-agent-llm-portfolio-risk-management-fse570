"""Row 4: every transform and feature is trailing — no centred windows, back-fills or full-sample statistics."""

import re

import numpy as np
import pandas as pd
import pytest

from riskcausal.baselines.simple import corr_var_score
from riskcausal.config import PROJECT_ROOT
from riskcausal.data.transforms import log_returns, realized_vol, rolling_avg_corr
from riskcausal.features.controls import control_features
from riskcausal.features.instability import add_smoothed

FORBIDDEN = [r"center\s*=\s*True", r"\.bfill\(", r"backfill", r"method\s*=\s*['\"]bfill", r"\.shift\(\s*-",
             r"interpolate\("]


def test_no_lookahead_operations_in_core_source():
    hits = []
    for py in (PROJECT_ROOT / "src" / "riskcausal").rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if any(re.search(p, line) for p in FORBIDDEN):
                hits.append(f"{py.name}:{i}: {line.strip()}")
    assert not hits, hits


def _ret(n=400, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=pd.bdate_range("2020-01-01", periods=n),
                        columns=list("ABCD"))


def _perturb_after(df, i):
    out = df.copy()
    out.iloc[i:] = out.iloc[i:] * 9 + 0.05
    return out


@pytest.mark.parametrize("fn", [
    lambda r: realized_vol(r["A"], 21),
    lambda r: rolling_avg_corr(r, 63),
    lambda r: corr_var_score(r, 63),
    lambda r: log_returns(np.exp(r.cumsum())),
    lambda r: control_features(r, pd.DataFrame({"VIXCLS": 20 + r["B"].cumsum() * 100}), "A", 21),
    lambda r: add_smoothed(r, 4),
])
def test_features_at_t_unchanged_when_future_rows_change(fn):
    r = _ret()
    cut = 250
    a, b = fn(r), fn(_perturb_after(r, cut))
    pd.testing.assert_frame_equal(pd.DataFrame(a).iloc[:cut], pd.DataFrame(b).iloc[:cut])
    assert not pd.DataFrame(a).iloc[cut + 70:].equals(pd.DataFrame(b).iloc[cut + 70:])
