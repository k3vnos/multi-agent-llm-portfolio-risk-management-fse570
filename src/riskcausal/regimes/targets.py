"""From stress probabilities to regime states, stress onsets and the prediction target.

state_t = 1 (stress) if P(stress) > 0.5, after removing stress runs shorter than `min_stress_days`.
onset   = first stress day of a run preceded by at least `min_calm_days` calm days.
target  y_t = 1 if calm at t and an onset occurs in (t, t+h]; 0 if calm and none; NaN if t is in stress or
          the horizon runs past the end of the labelled sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _runs(x: np.ndarray):
    """Yield (start, end_inclusive, value) for runs of equal values."""
    if len(x) == 0:
        return
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            yield start, i - 1, x[start]
            start = i


def states_from_prob(p: pd.Series, min_stress_days: int) -> pd.Series:
    s = np.array((p > 0.5).astype(int).to_numpy(), copy=True)
    for a, b, v in list(_runs(s)):
        if v == 1 and (b - a + 1) < min_stress_days:
            s[a:b + 1] = 0
    return pd.Series(s, index=p.index, name="stress")


def onsets(states: pd.Series, min_calm_days: int) -> pd.Series:
    s = states.to_numpy()
    on = np.zeros(len(s), dtype=int)
    calm_run = 0
    for i in range(len(s)):
        if s[i] == 1:
            if i > 0 and s[i - 1] == 0 and calm_run >= min_calm_days:
                on[i] = 1
            calm_run = 0
        else:
            calm_run += 1
    return pd.Series(on, index=states.index, name="onset")


def make_target(states: pd.Series, onset: pd.Series, h: int) -> pd.Series:
    on = onset.to_numpy()
    n = len(on)
    y = np.full(n, np.nan)
    st = states.to_numpy()
    for i in range(n):
        if st[i] == 1 or i + h >= n:
            continue
        y[i] = float(on[i + 1: i + h + 1].any())
    return pd.Series(y, index=states.index, name="target")


def label_set(p_stress: pd.Series, min_stress_days: int, min_calm_days: int, h: int) -> pd.DataFrame:
    st = states_from_prob(p_stress, min_stress_days)
    on = onsets(st, min_calm_days)
    return pd.DataFrame({"p_stress": p_stress, "stress": st, "onset": on, "target": make_target(st, on, h)})
