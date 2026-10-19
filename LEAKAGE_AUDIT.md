# LEAKAGE_AUDIT

Every way future information could enter a historical prediction, the guard against it, and the test that
enforces the guard. A row is **closed** only when its test exists and passes.

| # | Risk | Guard | Enforcing test | Status |
|---|---|---|---|---|
| 1 | Revised macro values (PIT) | value at t = latest ALFRED vintage with `realtime_start ≤ t − lag` | `tests/test_pit.py` | open |
| 2 | Release-time lag | +1 NYSE session on every macro value | `tests/test_calendar.py::test_add_bdays_moves_friday_release_to_monday`, `tests/test_pit.py` | partial |
| 3 | Rolling windows peeking ahead | graph/feature at t uses rows ≤ t | perturb-future test in `tests/test_rolling_no_lookahead.py` | open |
| 4 | Centered / backward-filled transforms | no `center=True`, no `bfill`, no full-sample stats | `tests/test_transforms.py` | open |
| 5 | Scaling fit on test data | scalers fit on train fold only | `tests/test_walkforward_no_leakage.py` | open |
| 6 | Markov labels using future data | fold-local MS fit for training labels; `label_eval` only in `evaluation/` | `tests/test_markov.py`, import-boundary test | open |
| 7 | Target horizon overlapping test block | purge h + embargo | `tests/test_walkforward_no_leakage.py` | open |
| 8 | Threshold/hyperparameter tuning on test | inner validation split only | `tests/test_walkforward_no_leakage.py` | open |
| 9 | Ragged panel / back-filled pre-listing data | series enters at first trade date | `tests/test_panel.py` | open |
| 10 | yfinance adjusted prices | documented: adjustment uses later splits/dividends; affects price levels, log returns include dividends | doc only | documented |
| 11 | LLM hindsight | agents never feed predictions; retrieval bounded `≤ alert date` | `tests/agents/test_retrieval.py` | open |
