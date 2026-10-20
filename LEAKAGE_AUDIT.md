# LEAKAGE_AUDIT

Every way future information could enter a historical prediction, the guard against it, and the test that
enforces the guard. A row is **closed** only when its test exists and passes. Two layers of enforcement:
- **unit / end-to-end tests** (`pytest`) on synthetic data with known answers, including "perturb the future"
  tests that change everything after a cut-off and require every earlier prediction, threshold and fitted model
  to stay identical;
- a **real-data audit** (`src/riskcausal/audit.py`) that runs at the end of every pipeline run and writes
  `outputs/runs/<id>/audit.json` (violations are counted, never assumed away).

| # | Risk | Guard | Enforcing test / check | Status |
|---|---|---|---|---|
| 1 | Revised macro values (PIT) | value at t = newest obs in latest ALFRED vintage released ≥ `lag` sessions before t; missing before first vintage (NFCI: 2011-05-25) | `tests/test_pit.py` (synthetic revision); `tests/test_panel.py::test_controls_use_only_published_values`; audit: *control value used at t: observation dated before t and published by t* | closed (M3) |
| 2 | Publication timing of daily series | per-series rule (`configs/universe.yaml`): **market_close** series (VIX, Treasury yields) usable obs + 1 session; **release** series (BAA10Y, AAA10Y, HY OAS) usable the session after FRED's ALFRED first-release date, fallback 3 sessions where untracked; a graph window is usable only once every row in it is published; a late-published old observation never displaces a newer known one | `tests/test_pit.py::test_release_based_availability_for_daily_series`, `…never_earlier_than_lag_floor`, `…late_old_observation_does_not_displace_newer_known_value`; `tests/test_instability.py::test_late_publication_inside_a_window_delays_it_and_all_later_windows`; `tests/test_panel.py::test_graph_rows_wait_for_release_and_outage`; audit: *every FRED value inside a feature's source window was published by t (ALFRED dates)* | closed (M3) — **leak found and fixed**, see finding A |
| 3 | Rolling windows peeking ahead | graph window ending t uses rows ≤ t, standardised within window; feature at t comes from a window ending strictly before t | `tests/test_causal.py::test_rolling_windows_only_use_past_rows`; end-to-end perturbation tests in `tests/test_walkforward_no_leakage.py`; audit: *graph feature at t comes from a window ending strictly before t* | closed (M3) |
| 4 | Centered / backward-filled transforms | trailing operations only: no `center=True`, `bfill`, negative shifts, interpolation | `tests/test_transforms.py` (source scan + perturb-future test for every transform and control feature) | closed (M3) |
| 5 | Scaling fit on test data | model scalers fit on the training rows of the fold; instability composite standardised on rows ≤ inner_end | `tests/test_walkforward_no_leakage.py::test_test_block_features_never_reach_thresholds_or_scaling` (mutation-checked: planting a test-block scaling leak makes it fail) | closed (M3) |
| 6 | Markov labels using future data | inner-train labels fit on ≤ inner_end, outer-train/validation labels on ≤ train_end; `label_eval` is not a parameter of `run_walkforward` and is built only after every prediction exists | `tests/test_regimes.py::test_filtered_probs_do_not_use_future_data`; `tests/test_walkforward_no_leakage.py::test_labels_after_train_end_never_reach_that_fold` (mutation-checked: fitting training labels through the test year makes it fail), `::test_walkforward_cannot_see_retrospective_labels` (AST check) | closed (M3) |
| 7 | Target horizon overlapping test block | labels end at train_end so the last h targets are NaN (purge) + 5-session embargo | `tests/test_regimes.py::test_target_marks_calm_days_before_onset_and_purges_end`; fold-geometry tests; audit: *fold geometry … with embargo*, *every prediction lies inside its own fold's test block* | closed (M3) |
| 8 | Threshold/hyperparameter tuning on test | thresholds chosen on the inner validation span only, under every alert rule; one threshold per (method, fold, rule) | `tests/test_walkforward_no_leakage.py::test_test_block_features_never_reach_thresholds_or_scaling`, `::test_perturbing_everything_after_a_fold_leaves_earlier_folds_unchanged`; audit: *one threshold per (method, fold, rule)* | closed (M3) |
| 9 | Ragged panel / back-filled pre-listing data | ETF prices reindexed to NYSE sessions without fill; returns start the session after the first close | `tests/test_panel.py::test_late_listing_is_not_back_filled`; audit: *no ETF return on or before its first traded close* | closed (M3) |
| 10 | yfinance adjusted prices | documented: adjustment uses later splits/dividends; affects price levels, log returns include dividends | doc only | documented |
| 11 | LLM hindsight | agents never feed predictions: the core package cannot import the optional layers; retrieval bound `≤ alert date` to be tested in M6 | `tests/test_walkforward_no_leakage.py::test_core_never_imports_optional_agent_or_api_layers`; `tests/agents/test_retrieval.py` in M6 | partial (structural guard closed; retrieval test when agents exist) |

## M3 findings

**A. BAA10Y publication timing (leak, fixed).** ALFRED initial-release dates (`riskcausal fetch-releases`;
`data/raw/fred_releases/<snapshot>/summary.json`) show, for tracked observations:

| series | tracked from | released same session | +1 session | > 1 session late | max |
|---|---|---|---|---|---|
| VIXCLS | 2010-11-22 | 3,020 | 965 | 0.3% | 3 |
| T10Y2Y | 2014-01-27 | 1,820 | 1,328 | 1.1% | 4 |
| BAA10Y | 2014-01-27 | 6 | 2,953 | **6.9%** | **113** (Dec 2016–Mar 2017 Moody's outage) |

The M2 rule (every daily FRED value usable one session after its date) therefore let graph windows use BAA10Y
values before FRED had published them (routinely by one session; for ~3 months in 2016–17 by up to 113
sessions). Fixed by the per-series rule in row 2. After the fix, graph rows become usable 2 sessions (2,986
rows) or 3 sessions (2,369 rows) after their date, 109 rows later (the outage); during the outage the features
stay on the last fully published window (max staleness 169 calendar days, at 2017-03-22). VarLiNGAM features
change on 1,548 of 5,210 sessions relative to M2. VIX and Treasury series are *not* delayed by FRED's posting
time: they are public market data at the close, and delaying them would handicap the strongest baseline.

**B. Out-of-order publication in as-of joins (latent bug, fixed).** When an old observation is published after
a newer one (outage back-fill), the as-of join selected the most recently *released* row, which would revert a
control to a stale value. It now keeps the newest *observation* among those published. Effect on the dev
controls: VIXCLS differs from M2 on 8 of 5,482 sessions (sessions after NYSE holidays on which FRED carries a
holiday-dated VIX value; both values were public before the decision date).

**C. Reproducibility (fixed).** The panel's session range ran to today's date, so rebuilding on another day
added an empty session. It now ends at the snapshot's last market day.

**D. Version control (fixed).** The unanchored `.gitignore` pattern `data/` also matched `src/riskcausal/data/`,
so the data package (calendar, ingest, PIT, panel, verification) was missing from commits `cffd744` and
`1bd4292`. Patterns are now anchored (`/data/`, `/outputs/`) and the package is committed (`0d05429`). The M2
runs' recorded git sha therefore does not pin the data-package code they used.

## Notes
- **GBM early stopping is disabled**: scikit-learn's early stopping holds out a *random* subset of training rows.
- **Thresholds** for every method are chosen on the inner validation span (last 2 training years). Two rules are
  computed from the same validation scores and both are reported: the H0 rule (≤ 4 false-alarm episodes/yr) and
  the approved post-hoc M3 rule (also alerts on ≤ 10% of validation calm days).
- **Instability composite** is standardised with statistics from rows ≤ inner_end only.
- **Validation truth** for thresholds = labels fit on data ≤ train_end; onsets just after train_end are unknown
  at that point, so a late-validation alert can be counted as a false alarm (conservative bias, not leakage).
