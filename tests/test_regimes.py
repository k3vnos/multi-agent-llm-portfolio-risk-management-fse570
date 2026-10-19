"""Markov labels and target construction (LEAKAGE_AUDIT row 6)."""

import numpy as np
import pandas as pd

from riskcausal.regimes.markov import filtered_stress_prob, fit_ms
from riskcausal.regimes.targets import make_target, onsets, states_from_prob


def _two_regime_series(seed=0):
    rng = np.random.default_rng(seed)
    regime = np.r_[np.zeros(300), np.ones(100), np.zeros(300), np.ones(100)].astype(int)
    y = np.where(regime == 1, rng.normal(1.0, 0.3, len(regime)), rng.normal(-1.0, 0.15, len(regime)))
    return pd.Series(y, index=pd.bdate_range("2010-01-01", periods=len(y))), regime


def test_ms_recovers_simulated_regimes():
    y, regime = _two_regime_series()
    f = fit_ms(y, 2, seed=0)
    assert f.regime_means[f.stress_regime] > 0.5
    acc = ((f.smoothed > 0.5).astype(int).to_numpy() == regime).mean()
    assert acc > 0.95


def test_filtered_probs_do_not_use_future_data():
    y, _ = _two_regime_series()
    f = fit_ms(y.iloc[:500], 2, seed=0)
    p1 = filtered_stress_prob(y, f)
    y2 = y.copy()
    y2.iloc[600:] = 5.0                                   # change the future
    p2 = filtered_stress_prob(y2, f)
    np.testing.assert_allclose(p1.iloc[:600], p2.iloc[:600])


def test_short_stress_runs_removed_and_onset_needs_calm_history():
    p = pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    st = states_from_prob(p, min_stress_days=5)
    assert st.iloc[3:5].sum() == 0 and st.iloc[10:16].sum() == 6
    on = onsets(st, min_calm_days=5)
    assert on.tolist().index(1) == 10 and on.sum() == 1
    assert onsets(st, min_calm_days=11).sum() == 0


def test_target_marks_calm_days_before_onset_and_purges_end():
    st = pd.Series([0, 0, 0, 0, 1, 1, 0, 0])
    on = pd.Series([0, 0, 0, 0, 1, 0, 0, 0])
    y = make_target(st, on, h=2)
    assert y.tolist()[:4] == [0.0, 0.0, 1.0, 1.0]
    assert np.isnan(y.iloc[4]) and np.isnan(y.iloc[5])   # in stress
    assert np.isnan(y.iloc[6]) and np.isnan(y.iloc[7])   # horizon past the labelled sample


def test_ms_fit_is_deterministic_across_calls():
    y, _ = _two_regime_series(seed=3)
    a, b = fit_ms(y, 2, seed=7), fit_ms(y, 2, seed=7)
    assert a.llf == b.llf
    np.testing.assert_array_equal(a.smoothed.to_numpy(), b.smoothed.to_numpy())
