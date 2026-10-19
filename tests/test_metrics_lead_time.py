import numpy as np
import pandas as pd

from riskcausal.evaluation.metrics import alert_episodes, choose_threshold, event_metrics


def test_alert_episodes_merge_small_gaps():
    a = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1])
    assert alert_episodes(a, merge_gap=1) == [(1, 4), (8, 8)]
    assert alert_episodes(a, merge_gap=0) == [(1, 2), (4, 4), (8, 8)]


def test_lead_time_and_false_alarm_by_hand():
    n = 60
    stress = np.zeros(n, int)
    stress[30:40] = 1
    onset = np.zeros(n, int)
    onset[30] = 1
    alert = np.zeros(n, bool)
    alert[24:27] = True          # 6 sessions before the onset -> detected, lead 6
    alert[50] = True             # calm, no onset follows -> false alarm
    alert[35] = True             # inside stress -> neither
    m = event_metrics(alert, stress, onset, h=10, merge_gap=0, n_years=1.0)
    assert m["n_onsets"] == 1 and m["n_detected"] == 1 and m["lead_times"] == [6]
    assert m["n_true_episodes"] == 1 and m["n_false_alarms"] == 1 and m["n_in_stress_episodes"] == 1
    assert m["precision"] == 0.5 and m["false_alarms_per_year"] == 1.0


def test_alert_after_onset_is_not_a_detection():
    n = 40
    onset = np.zeros(n, int)
    onset[20] = 1
    stress = np.zeros(n, int)
    stress[20:] = 1
    alert = np.zeros(n, bool)
    alert[20] = True
    m = event_metrics(alert, stress, onset, h=10, merge_gap=0, n_years=1.0)
    assert m["n_detected"] == 0 and m["recall"] == 0.0


def test_threshold_respects_false_alarm_budget():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=504)
    score = pd.Series(rng.normal(size=504), index=idx)
    zeros = pd.Series(0, index=idx)
    thr, info = choose_threshold(score, zeros, zeros, h=10, merge_gap=5, max_fa_per_year=4)
    alert = (score >= thr).to_numpy()
    m = event_metrics(alert, zeros.to_numpy(), zeros.to_numpy(), 10, 5, 504 / 252)
    assert info["status"] == "ok" and m["false_alarms_per_year"] <= 4
