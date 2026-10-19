"""Fold geometry (LEAKAGE_AUDIT rows 7–8). Training-row and fitting checks are added in M3."""

import itertools

import pandas as pd

from riskcausal.data.calendar import nyse_sessions
from riskcausal.validation.walkforward import make_folds


def test_folds_are_chronological_with_embargo_and_purge_room():
    s = nyse_sessions("2005-01-01", "2026-10-16")
    folds = make_folds(s, 2011, 2026, embargo=5, val_years=2)
    assert [f.test_year for f in folds] == list(range(2011, 2027))
    for f in folds:
        assert s.get_loc(f.test_start) - s.get_loc(f.train_end) == 6         # 5-session embargo between
        assert f.inner_end < f.val_start <= f.train_end < f.test_start <= f.test_end
        assert s.get_loc(f.val_start) - s.get_loc(f.inner_end) == 6
        assert f.test_start.year == f.test_end.year == f.test_year


def test_test_blocks_do_not_overlap():
    s = nyse_sessions("2005-01-01", "2026-10-16")
    folds = make_folds(s, 2011, 2026, embargo=5)
    for a, b in itertools.pairwise(folds):
        assert a.test_end < b.test_start
    assert pd.Timestamp(folds[0].train_end).year == 2010
