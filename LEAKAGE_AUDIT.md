# LEAKAGE_AUDIT

Every way future information could enter a historical prediction, the guard against it, and the test that
enforces the guard. A row is **closed** only when its test exists and passes.

| # | Risk | Guard | Enforcing test | Status |
|---|---|---|---|---|
| 1 | Revised macro values (PIT) | value at t = newest obs in latest ALFRED vintage released ≥ `lag` sessions before t; missing before first vintage (NFCI: 2011-05-25) | `tests/test_pit.py` (synthetic revision) | closed (M2) |
| 2 | Release-time lag | +1 NYSE session on every macro control value; graph windows are aligned by observation date and **every graph-derived feature is lagged +1 session** (`GRAPH_FEATURE_LAG`) | `tests/test_calendar.py`, `tests/test_pit.py`, `tests/test_instability.py::test_to_daily_lags_availability` | closed (M2) |
| 3 | Rolling windows peeking ahead | graph window ending t uses rows ≤ t, standardised within window | `tests/test_causal.py::test_rolling_windows_only_use_past_rows` (perturb-future); end-to-end feature perturbation test in M3 | partial |
| 4 | Centered / backward-filled transforms | no `center=True`, no `bfill`, no full-sample stats | `tests/test_transforms.py` | open |
| 5 | Scaling fit on test data | scalers fit on train fold only | `tests/test_walkforward_no_leakage.py` | open |
| 6 | Markov labels using future data | inner-train labels fit on ≤ inner_end, outer-train/validation labels on ≤ train_end; `label_eval` used only for scoring; filtered labels as sensitivity | `tests/test_regimes.py::test_filtered_probs_do_not_use_future_data`; pipeline-level check in M3 | partial |
| 7 | Target horizon overlapping test block | labels end at train_end so last h targets are NaN (purge) + 5-session embargo | `tests/test_regimes.py::test_target_marks_calm_days_before_onset_and_purges_end`, `tests/test_walkforward_no_leakage.py` | partial (fold geometry; row-level check in M3) |
| 8 | Threshold/hyperparameter tuning on test | inner validation split only | `tests/test_walkforward_no_leakage.py` | open |
| 9 | Ragged panel / back-filled pre-listing data | series enters at first trade date | `tests/test_panel.py` | open |
| 10 | yfinance adjusted prices | documented: adjustment uses later splits/dividends; affects price levels, log returns include dividends | doc only | documented |
| 11 | LLM hindsight | agents never feed predictions; retrieval bounded `≤ alert date` | `tests/agents/test_retrieval.py` | open |

## Notes
- **GBM early stopping is disabled**: scikit-learn's early stopping holds out a *random* subset of training rows.
- **Thresholds** for every method are chosen on the inner validation span (last 2 training years) by one fixed
  rule — lowest threshold with ≤ 4 false-alarm episodes/yr — never on test data.
- **Instability composite** is standardised with statistics from rows ≤ inner_end only.
