"""Chronological walk-forward folds (E13). No random splits anywhere.

For test year Y:
  test        = sessions in year Y
  train_end   = last session that is `embargo` sessions before the test start
  val         = the last `val_years` years of the training period (inner validation, for thresholds)
  inner_end   = last session `embargo` sessions before the validation start
Targets need labels h sessions ahead; they are computed from label sets that end at train_end / inner_end, so
the last h rows of each training sample have no target (purge) automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Fold:
    k: int
    test_year: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    inner_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self).items()}


def make_folds(sessions: pd.DatetimeIndex, first_test_year: int, last_test_year: int, embargo: int,
               val_years: int = 2) -> list[Fold]:
    folds = []
    for k, year in enumerate(range(first_test_year, last_test_year + 1)):
        test = sessions[sessions.year == year]
        if len(test) == 0:
            continue
        i0 = sessions.get_loc(test[0])
        train_end = sessions[i0 - embargo - 1]
        val_start_target = pd.Timestamp(year - val_years, 1, 1)
        val_start = sessions[sessions >= val_start_target][0]
        j0 = sessions.get_loc(val_start)
        inner_end = sessions[j0 - embargo - 1]
        folds.append(Fold(k, year, test[0], test[-1], train_end, val_start, inner_end))
    return folds
