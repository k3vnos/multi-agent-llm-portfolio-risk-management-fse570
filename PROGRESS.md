# PROGRESS

## M0 — Scaffold — 2026-10-18
**Completed:** `COST_POLICY.md`; local git repo (`main`, no remote); `pyproject.toml`; Python 3.12 venv with free
OSS deps; YAML configs (`base`, `universe`, `profiles/dev`, `profiles/full`) loaded into typed pydantic models;
JSON-lines logging; run registry (config snapshot, git sha, `synthetic` flag); `typer` CLI; `.env.example`;
`Makefile`; README stub; `LEAKAGE_AUDIT.md`, `EXPERIMENT_LOG.md`.
**Tests:** `.venv/bin/pytest` → 14 passed (cost policy, config, NYSE calendar, manifest renderer). `ruff check` clean.
**Deviations from plan:**
- `uv` is not installed, so the project uses `python3.12 -m venv` + `pip` instead of installing new tooling.
- Optional layers live in `src/riskcausal_agents/` and `src/riskcausal_api/` (not top-level `agents/`, `api/`)
  to avoid generic package-name collisions. The core package still never imports them.
**Next:** M1.

## M1 — Dataset verification — 2026-10-18
**Completed:** `riskcausal verify-data --profile full` downloaded every candidate source and generated
`DATASET_MANIFEST.md` from `data/raw/verification_2026-10-18.json`. Raw snapshot stored under
`data/raw/{yfinance,fred}/2026-10-18/` with SHA256 per file.
**Actual results (from the verification report):**
- ETFs: 25/25 downloaded; 20 available from 2005-01-01; 0 missing NYSE sessions in every series.
  Late listings: KRE 2006-06-22, HYG 2007-04-11, DBC 2006-02-06, USO 2006-04-10, UUP 2007-03-01.
- FRED: 10/10 series downloaded. **ICE BofA HY OAS (BAMLH0A0HYM2) only from 2023-10-17** on FRED — confirms
  ambiguity E3; BAA10Y (from 1986) and AAA10Y (from 1983) cover the study period.
- ALFRED vintages for revised series (NFCI, UNRATE, STLFSI4): **not downloaded — no FRED API key set.**
  The public CSV endpoint ignores `vintage_date` (tested: it returned current values), so the free API key is
  required for point-in-time data.
- FNSPID: public, not gated, 2 news CSVs (5.73 GB and 23.23 GB). Byte-range samples show dates from 2004 to
  2023-12; full coverage of Feb–Mar 2020 / Mar 2023 is **not yet confirmed** (needs the full download, M6).
- GDELT 2.0: endpoint reachable.
**Tests:** 14 passed.
**Unresolved problems:**
- No FRED API key → NFCI (feature and baseline) cannot be used point-in-time yet.
- FNSPID `licence` field is empty on Hugging Face; licence terms to be checked before redistribution
  (research use of a public dataset is not affected).
**Decisions needed (gate before M2):**
1. Provide a free FRED API key in `.env` (or approve running M2 without NFCI until it is available).
2. Approve credit-spread substitute: BAA10Y instead of HY OAS for the graph and features.
3. Approve full-profile graph sample starting 2007-04-11 (when HYG and UUP both exist); dev profile is
   unaffected (all six dev ETFs exist from 2004-11-18).
**Next:** M2 thin end-to-end prototype on the dev profile.

### M1 addendum — 2026-10-18
FRED API key added by the team. ALFRED verified without exposing the key: messages are passed through
`ingest_fred.redact()` (test added), and the saved report contains only `fred_api_key_present: true`.
Vintages: NFCI 800 (from 2011-05-25), UNRATE 441 (from 1990-02-02), STLFSI4 206 (from 2022-11-10).
Decisions approved: BAA10Y replaces HY OAS (methodological difference documented in IMPLEMENTATION_PLAN §H);
full-profile start 2007-04-11.

## M2 — Thin end-to-end prototype (dev profile) — 2026-10-18
**Completed (all on real data, no mocks):**
- PIT panel: 6 ETFs (SPY, TLT, LQD, GLD, XLF, IEF), VIXCLS, BAA10Y, T10Y2Y (graph nodes, first differences),
  NFCI from ALFRED vintages; ADF+KPSS report (only non-stationary case: ΔT10Y2Y in 2005–2009 per KPSS).
  Example of why PIT matters: NFCI as known on 2020-03-20 = −0.48; today's revised value for that week = +0.031.
- Rolling VarLiNGAM (W=250, step 5, BIC lag ≤5, pruned): 1,047 windows, 2005-12-28 → 2026-10; rolling Granger
  (same lags, BH-FDR) on the same windows. Graph features lagged +1 session.
- Instability features (SHD, Frobenius ΔB0/Δlag, edge count, causal-order change, 4-window means) + Granger
  instability; controls (VIX, ΔVIX, 21-day realised vol, 63-day average correlation).
- Markov-switching labels (2 regimes on log 21-day realised vol of SPY) with fold-local fits; retrospective
  `label_eval` for scoring only; filtered-label sensitivity.
- Walk-forward: 16 annual test folds 2011–2026, 5-session embargo, inner validation = last 2 training years.
  Bayesian logit (PyMC NUTS, max R-hat 1.014) and GBM × {instability, granger, controls, combined};
  pre-specified instability composite; baselines VIX, NFCI (PIT), rolling-correlation VaR.
- Evaluation, figures (2020, 2023, full span) and generated `metrics.md` per run.

**Tests:** `.venv/bin/pytest` → **36 passed** (0 failed); `ruff check` clean.
**Reproducibility:** two consecutive runs from the same snapshot (`20261019T050503Z_dev_17c0c6c46d`,
`20261019T060334Z_dev_17c0c6c46d`) → identical `metrics.json` (excluding runtime), predictions and labels.
**Runtime:** 492 s for the full pipeline with cached graphs (run 4). Run 5 reports 3,507 s because the Mac slept
mid-run.

**Actual results** (run `20261019T060334Z_dev_17c0c6c46d`, common span 2012-01-03 → 2026-10-16 where all methods
are scored; truth = retrospective labels; day-level metrics on calm days; base rate 0.098):

| method | ROC-AUC | PR-AUC |
|---|---|---|
| VIX level | 0.735 | 0.300 |
| Bayes logit — controls | 0.699 | 0.249 |
| GBM — combined | 0.672 | 0.201 |
| Bayes logit — combined | 0.659 | 0.198 |
| GBM — controls | 0.640 | 0.173 |
| GBM — instability | 0.565 | 0.147 |
| rolling-corr VaR | 0.511 | 0.105 |
| NFCI (PIT) | 0.489 | 0.095 |
| Bayes logit — instability | 0.477 | 0.152 |
| Bayes logit — Granger | 0.471 | 0.102 |
| GBM — Granger | 0.460 | 0.098 |
| instability composite | 0.451 | 0.098 |

What these numbers do and do not show (H0 settings, dev profile, no confidence intervals yet):
- In this prototype the VarLiNGAM/Granger instability features carry **no clear out-of-sample signal** for
  stress onsets (ROC-AUC 0.45–0.57), and adding them to the controls did not raise ROC-AUC (Bayes: 0.659 combined
  vs 0.699 controls). This is a prototype observation, not a conclusion — robustness (M4) and CIs are pending.
- In the 2020 case figure the instability composite rises **after** the 2020-02-21 onset, not before.
- **Event-level metrics (recall/precision/lead) are not interpretable yet** — see problem 1.

**Problems found and fixed during M2:**
- Label construction crashed on a read-only array under pandas copy-on-write → fixed, test added.
- statsmodels' Markov `search_reps` was non-reproducible and sometimes converged to a far worse optimum
  (log-likelihood −490 vs −248 on 2005–2008) → replaced with deterministic seeded multi-start, test added.
  The two earlier runs are superseded (EXPERIMENT_LOG).

**Unresolved problems:**
1. **Threshold rule (E14) is degenerate.** "Lowest threshold with ≤ 4 false-alarm *episodes*/yr" lets alerts
   merge into long episodes: every method alerts on 37–56% of test days, recall looks high for that reason, and
   median lead sits at the cap h=10. The same pattern is visible on validation data. Because I have now seen
   test results, changing the rule is post-hoc and must be decided explicitly and logged.
2. **Label definition.** The 2-regime model labels 40.7% of days as "stress" (calm ≈ 9.7% vol, stress ≈ 21.6%),
   giving 34 onsets since 2005. **No onset occurs in March 2023**: the high-volatility regime from 2022 had not
   ended, so SVB is not a calm→stress transition under H0 labels. 3 regimes is already in the pre-registered
   robustness grid (§F4).
3. NFCI baseline is scored only from 2012 (point-in-time vintages start 2011-05-25 and the threshold needs two
   years of validation data).

**Decisions needed:**
1. Approve a revised alert rule for M3/M4, e.g. add an alert-day budget (alerts on ≤ 10% of validation calm
   days) on top of the false-alarm cap, recorded in EXPERIMENT_LOG as a post-hoc change made after M2 test
   results. ROC/PR-AUC (threshold-free) stay the primary comparison either way.
2. Whether 3-regime labels should be added to the H0 robustness grid (already listed) or promoted earlier.

**Next:** M3 — leakage audit & hardening (close remaining LEAKAGE_AUDIT rows with end-to-end tests).

## M3 — Leakage audit & hardening — 2026-10-19
**Completed:**
- Every LEAKAGE_AUDIT row now has an enforcing test (rows 1–9 closed; row 10 documented; row 11 structural guard
  closed, retrieval test due with the agents in M6). New: end-to-end "perturb the future" walk-forward tests
  (changing data after a cut-off leaves every earlier prediction, threshold and fitted model identical),
  mutation-checked by planting two leaks (test-block scaling; labels fit through the test year) — both caught.
- Real-data audit (`audit.json`) runs at the end of every pipeline run: 7 checks, recomputed from raw data.
- **Leak found and fixed (finding A):** ALFRED first-release dates show FRED posts BAA10Y one session after its
  date, 6.9% of observations later still, and up to 113 sessions late during the Dec 2016–Mar 2017 Moody's
  outage. M2's "usable one session later" rule let graph windows use BAA10Y before publication. Now:
  vendor series use FRED release dates (+1 session; 3-session fallback before ALFRED tracking), market-close
  series (VIX, Treasury yields) keep obs + 1 session, and a graph window is usable only once all its rows are
  published. VIX was deliberately not delayed by FRED's posting time (that would handicap the strongest baseline).
- Other fixes: out-of-order publication in as-of joins (B); panel session range tied to today's date (C);
  `.gitignore` excluded `src/riskcausal/data/` from every commit, and ruff skipped it too (D) — now versioned
  (commit 0d05429) and linted.
- Approved post-hoc alert rule `m3` (≤ 4 false-alarm episodes/yr **and** ≤ 10% of validation calm days) computed
  next to the original H0 rule from the same validation scores — no refits; M2 runs preserved unchanged.
- Separate 3-regime experiment (`configs/experiments/regimes3.yaml`); validation-fold predictions now saved
  (median-over-folds metrics) for the M4 selection; graph noise-floor diagnostic (`diagnose-graphs`).

**Tests:** `.venv/bin/pytest` → **60 passed** (0 failed; 36 at M2); `ruff check src tests` clean (now including
`src/riskcausal/data/`).
**Runs (2, both reusing cached graphs):** `20261020T035004Z_dev_24a61f320e` (2 regimes, 520 s) and
`20261020T040849Z_dev_05b453a740_regimes3` (3 regimes, 1,625 s). Both audits PASSED: 0 violations across fold
geometry (16 folds), 47,652 prediction placements, 384 thresholds, 5,225 graph-feature dates, 15,675
FRED-publication checks, 20,317 control values, 6 ETF listings.

**Actual results — 2 regimes, run 24a61f320e** (common span 2012-01-03 → 2026-10-16, truth = retrospective
labels, 24 onsets, calm-day base rate 0.098):

| method | ROC-AUC (M2 → M3) | PR-AUC | recall / precision / FA per yr / calm-day alert share — **H0 rule** | same — **M3 rule** |
|---|---|---|---|---|
| VIX level | 0.735 → 0.735 | 0.300 | 0.88 / 0.32 / 2.17 / 0.30 | 0.63 / 0.36 / 1.69 / 0.12 |
| Bayes — controls | 0.699 → 0.699 | 0.249 | 0.71 / 0.38 / 1.56 / 0.22 | 0.67 / 0.33 / 1.97 / 0.16 |
| Bayes — combined | 0.659 → 0.667 | 0.202 | 0.67 / 0.41 / 1.29 / 0.28 | 0.58 / 0.37 / 1.63 / 0.14 |
| GBM — combined | 0.672 → 0.653 | 0.169 | 0.83 / 0.30 / 2.37 / 0.38 | 0.58 / 0.33 / 1.83 / 0.15 |
| GBM — controls | 0.640 → 0.640 | 0.172 | 0.96 / 0.39 / 2.24 / 0.28 | 0.67 / 0.33 / 2.30 / 0.14 |
| GBM — instability | 0.565 → 0.611 | 0.140 | 0.83 / 0.39 / 1.83 / 0.46 | 0.29 / 0.27 / 1.08 / 0.12 |
| rolling-corr VaR | 0.511 → 0.511 | 0.105 | 0.50 / 0.78 / 0.14 / 0.43 | 0.21 / 0.60 / 0.14 / 0.17 |
| Bayes — Granger | 0.471 → 0.494 | 0.111 | 0.58 / 0.34 / 1.29 / 0.47 | 0.21 / 0.38 / 0.34 / 0.19 |
| NFCI (PIT) | 0.489 → 0.489 | 0.095 | 0.50 / 0.40 / 0.20 / 0.38 | 0.38 / 0.67 / 0.14 / 0.20 |
| GBM — Granger | 0.460 → 0.484 | 0.100 | 0.63 / 0.31 / 1.83 / 0.51 | 0.17 / 0.18 / 1.22 / 0.12 |
| Bayes — instability | 0.477 → 0.475 | 0.138 | 0.50 / 0.29 / 1.49 / 0.35 | 0.21 / 0.15 / 1.49 / 0.10 |
| instability composite | 0.451 → 0.452 | 0.099 | 0.54 / 0.33 / 1.49 / 0.46 | 0.17 / 0.22 / 0.95 / 0.12 |

Inner-validation medians over 16 folds (ROC-AUC) rank the same way: VIX 0.710, Bayes controls 0.673, Bayes
combined 0.670, … instability composite 0.409.

What this shows (dev profile, H0 graph settings, no confidence intervals yet):
- The leakage fix changed only graph-feature timing; controls and baselines are unchanged. Graph-based ROC-AUCs
  moved by up to ±0.05 (GBM instability 0.565 → 0.611) from a one-to-two-session timing shift — a sign of how
  noisy these features are, not of a real gain.
- **The M3 alert rule works as intended:** calm-day alert shares drop from 22–51% to 10–20% on test, and the
  event-level numbers become interpretable. Under it, graph-instability methods detect 4–7 of 24 onsets with
  precision 0.15–0.27, versus 15–16 for VIX / Bayes-controls. Event metrics are still on 24 onsets and have no CIs.
- **No evidence so far that directed-dependence instability gives earlier warning** than VIX: graph-only methods
  are at or below chance by ROC-AUC (0.45–0.61), and adding them to controls does not improve on controls alone.
  In the 2020 case the composite again rises only after the 2020-02-21 onset.
- **Noise floor (Risk 2):** two bootstrap refits of the *same* 250-day window disagree on a median of 20.7 of 72
  contemporaneous edges; consecutive windows differ by 4. Only 9% (SHD) / 14% (Frobenius) of window-to-window
  changes exceed the same-window resampling noise. Single-window graphs are imprecise at W = 250 with 9 nodes, so
  much of the "instability" signal is estimation noise. (Block resampling distorts lags, so the floor is approximate.)

**3-regime experiment (run 05b453a740, exploratory):** regimes 8.6% / 14.0% / 25.6% vol; stress (top regime only)
26.7% of days, 22 onsets in total, 14 in the common span (base rate 0.048). ROC-AUC: VIX 0.765, Bayes controls
0.753, GBM controls 0.691, Bayes combined 0.678, GBM combined 0.643, NFCI 0.538, rolling-corr VaR 0.476,
GBM instability 0.459, composite 0.445, GBM Granger 0.426, Bayes instability 0.425, Bayes Granger 0.396. The
3-regime model fits much better (full-sample log-likelihood 347.5 vs −1421.9), but **it also has no onset in
March 2023**: 2022 sits in the top regime, and the next onset is 2024-08-05. The conclusion about instability
features is unchanged under these labels (they are below chance).

**Unresolved problems:**
1. March 2023 (SVB) is not a calm→stress transition under either label definition on SPY realised vol. Showing
   it as a case study needs a different, pre-specified event definition (e.g. regional-bank / credit stress
   series in the full profile, KRE), not tuning of the SPY labels.
2. Event-level metrics rest on 14–24 onsets; block-bootstrap CIs (M4) are needed before any comparison claim.
3. Graph estimates are noisy at W = 250 (noise-floor result). The M4 grid (W ∈ {126, 250, 500}, edge threshold)
   should be judged on validation folds only, as pre-registered.
4. The M2 runs' git sha does not pin the data-package code (finding D). M3 runs use commit 0d05429 (recorded as
   `-dirty` because docs/behaviour-neutral lint edits happened during the run).

**Decisions needed before M4:**
1. Which label definition is primary going into M4: keep 2 regimes as primary with 3 regimes as a
   pre-registered alternative, chosen by the validation-only rule (proposed), or switch now.
2. Whether to pre-register a separate credit/banking stress event definition so March 2023 can be evaluated.
3. Approve the M4 robustness grid (`configs/experiments/robustness.yaml`, validation folds only) and DCC-GARCH.

**Next:** M4 — robustness on validation folds, DCC-GARCH baseline, block-bootstrap CIs, Week-8 decision memo.

## M4 — Robustness on validation data, DCC-GARCH, confidence intervals, decision gate — 2026-10-20
**Completed:**
- **Pre-registration first:** `configs/experiments/robustness.yaml` committed (7464cfa) before any M4 code ran —
  label-selection rule, graph grid, target sensitivity, CI method, DCC-GARCH spec, banking-stress definitions.
- **Validation-only selection** (`riskcausal m4-select`, run `20261020T050131Z_dev_84f1d5ecaf_m4select`):
  - Label model: 3 regimes beat 2 on validation predictive log-likelihood in 15 of 16 folds (median −0.200 →
    +0.057 per obs; label agreement 0.964 → 0.983). **2 regimes stays primary** (team decision); 3 regimes is
    reported as the validation-selected secondary.
  - Graph grid (W {126, 250, 500} × edge {0.03, 0.05, 0.10}): **W=500, edge 0.05** selected under both label sets.
    Best instability validation PR-AUC 0.246 vs VIX 0.411 (2 regimes).
  - Target sensitivity (h 5/20, min stress 3/10; validation, descriptive): instability-only stays below
    controls and VIX in every variant except h=20, where all methods converge (base rate 0.24).
- **DCC-GARCH baseline** (Engle 2002; GARCH(1,1)+DCC(1,1) QML, refit per fold ≤ inner_end / ≤ train_end; scores
  `dcc_var`, `dcc_corr`).
- **CIs:** paired stationary block bootstrap (mean block 63, 2,000 reps) + Jeffreys intervals, computed from
  saved predictions (`riskcausal ci <run>`), for every M3/M4 run — no refits.
- **Locked test runs** (scored once): primary (selected config, 2 regimes) and pre-registered secondary (3 regimes).
- **Exploratory banking/credit stress** (kept separate from the SPY target): B1 KRE-volatility target (full
  walk-forward), B2 descriptive BAA10Y regime check.
- Week-8 decision memo: `DECISION_MEMO_M4.md`.
- Efficiency: Markov fits cached on disk (identical results); validation grid with fast GBM/MAP-logit
  (9 configs × 2 label sets in ~2 min); W=126/500 graph builds 62 s / 84 s; CIs from saved predictions.

**Tests:** `.venv/bin/pytest` → **67 passed** (0 failed; 60 at M3); `ruff check src tests` clean. New: DCC
no-lookahead and parameter recovery, Markov cache identity, predictive log-likelihood ignores later data,
bootstrap/Jeffreys, validation selection never reads after `train_end`; walk-forward perturbation tests now also
perturb DCC-GARCH inputs.

**Actual results — locked primary** (`20261020T071838Z_dev_89c4b14759_m4_selected`, 2012-01-03 → 2026-10-16,
24 onsets, audit PASSED). ROC-AUC [95% CI]; event metrics under the M3 alert rule:

| method | ROC-AUC [95% CI] | PR-AUC | detected | precision |
|---|---|---|---|---|
| VIX level | 0.735 [0.651, 0.815] | 0.300 | 15/24 | 0.36 |
| Bayes — controls | 0.699 [0.616, 0.780] | 0.249 | 16/24 | 0.33 |
| GBM — controls | 0.640 [0.560, 0.714] | 0.172 | 16/24 | 0.33 |
| Bayes — combined | 0.608 [0.500, 0.715] | 0.131 | 13/24 | 0.33 |
| GBM — combined | 0.598 [0.501, 0.691] | 0.150 | 10/24 | 0.26 |
| DCC-GARCH VaR | 0.571 [0.472, 0.676] | 0.162 | 13/24 | 0.36 |
| Bayes — Granger | 0.553 [0.453, 0.653] | 0.110 | 5/24 | 0.24 |
| rolling-corr VaR | 0.511 [0.399, 0.631] | 0.105 | 5/24 | 0.60 |
| GBM — instability | 0.495 [0.398, 0.594] | 0.095 | 4/24 | 0.22 |
| Bayes — instability | 0.490 [0.382, 0.597] | 0.093 | 7/24 | 0.70 |
| NFCI (PIT) | 0.489 [0.383, 0.598] | 0.095 | 9/24 | 0.67 |
| DCC correlation | 0.474 [0.363, 0.586] | 0.109 | 8/24 | 0.44 |
| GBM — Granger | 0.459 [0.357, 0.562] | 0.090 | 5/24 | 0.19 |
| instability composite | 0.403 [0.298, 0.511] | 0.081 | 5/24 | 0.27 |

Paired differences (95% CI): every graph-based method minus VIX excludes 0 (worse). Instability added to
controls: Bayes **−0.090 [−0.185, −0.010]** (significantly worse), GBM −0.042 [−0.124, 0.042]. Bayes-controls
vs VIX −0.037 [−0.097, 0.017] (not distinguishable). Posterior recall (5/50/95%): Bayes controls 0.67/0.71/0.79,
combined 0.50/0.54/0.62, instability 0.21/0.29/0.33.

**Secondary — 3-regime labels, same settings** (`…64c063c907_m4_selected+regimes3_label`, 14 onsets in span):
VIX 0.765 [0.637, 0.882], Bayes controls 0.753, Bayes combined 0.689, DCC VaR 0.600, Bayes instability 0.537,
GBM instability 0.506, composite 0.421. Combined − controls: Bayes −0.065 [−0.199, 0.054]. Validation had shown
combined > controls for this config (PR-AUC 0.358 vs 0.281); it did not carry over to test.

**Exploratory banking stress** (B1, `…ab1b3e098a_m4_selected+banking_kre`; KRE vol regimes 18.6% / 41.0%,
stress 36% of days, 23 onsets in span): **2023-03-09 (SVB) is an onset under this target.** ROC-AUC: Bayes
controls 0.737, VIX 0.732, Bayes combined 0.723, DCC VaR 0.651, NFCI 0.649, instability-only 0.62, composite
0.453; combined − controls −0.015 [−0.085, 0.065]. In the 10 sessions before 2023-03-09, under the M3 rule only
DCC correlation (10 days) and GBM-Granger (5 days) were alerting — one event, not evidence. B2 (BAA10Y level,
descriptive): Markov flags an onset on 2023-03-15, but the measure is crude (58% of all days "stress").

**Conclusion at the decision gate (`DECISION_MEMO_M4.md`):** on the dev profile, no configuration — H0,
validation-selected, 3-regime labels, or the banking-stress target — shows directed-dependence instability
beating VIX or adding to the controls; several differences are significantly negative. Recommendation: report a
rigorous negative result, plus one locked full-profile run (M5) as the generality check.

**Problems found and fixed in M4:** KRE target missing from the panel (first run failed; target now computed from
the raw snapshot); run provenance recorded the git sha at run end, not start (fixed in 58253a6 — the three M4
runs record 58253a6 but used 252cc7c / 0ce3cdd, logged in EXPERIMENT_LOG).

**Unresolved:** event-level metrics rest on 14–24 onsets (wide Jeffreys intervals); the W=500 window drops the
2011 test fold (features start later), so M4 runs cover 2012–2026, as did the M3 common span; the
validation-selected config did worse on test than H0 (within noise).

**Next:** see "What remains before the multi-agent system and dashboard" in the M4 summary; M5 needs your go-ahead.

## M5 — Full-profile locked evaluation — 2026-10-30
**Pre-flight (before running):** configuration frozen (no config change since the M4 overlay 4c12305; selection
rules committed in 7464cfa before M4 code); selection code reads nothing after each fold's train_end (tested);
working tree clean, no remote. Node-set conflict (brief said "10-node graph incl. HYG and KRE"; frozen
`full.yaml` has 10 ETFs incl. HYG, no KRE, + 3 macro = 13 nodes) → team chose the **frozen list**; KRE covered by
the exploratory banking-stress target. Hardware: Apple M2, 24 GB RAM, 466 GB free.

**Completed:**
- ALFRED first-release dates for all 7 non-revised daily FRED series of the full profile (HY OAS tracked only
  from 2023-11; used in no feature). Revised series (NFCI, UNRATE, STLFSI4) use the M1 vintages unchanged.
- Full panel: 25 ETFs, 13 graph nodes (SPY, IWM, EFA, EEM, XLF, TLT, LQD, HYG, GLD, UUP, ΔVIX, ΔBAA10Y, ΔT10Y2Y);
  graph rows complete from HYG's listing (2007-04), matching the approved start; W=500 → 882 windows.
- One locked evaluation with the frozen M4 settings (W=500, edge 0.05, 2 regimes, M3 alert rule), plus a
  bit-for-bit reproducibility check and the exploratory KRE target as a secondary run.

**Reproducibility problem found and fixed:** the first M5 run and its re-run were *not* bit-identical. Bayesian
logit draws (instability/combined feature sets only) differed: PyTensor's numba backend produced a different MCMC
realisation in the first process with a cold compile cache (first run after the macOS 15.7 → 27.0 upgrade) than
in later processes. Diagnosed step by step (inputs, layout, threads, C cache, numba cache) and confirmed by
cold/warm experiments. Effect: ROC-AUC ≤ 0.003, no CI or significance call changed. Fix: Bayes compiles with the
deterministic C backend (8a61dcb) + a cold/warm-cache test (mutation-checked). Both earlier M5 runs are superseded
and kept. The locked run was then repeated with a cold and a warm cache → **bit-identical**.
Also: re-fetching release dates once overwrote the dev files — verified harmless (rebuilt M3/M4 features
identical); `fetch-releases` no longer overwrites. corr_var differs from M4's saved values at ≤ 1.4e-17 after the
OS upgrade (ranks identical).

**Tests:** `.venv/bin/pytest` → **70 passed** (67 at M4); `ruff check src tests` clean.
**Leakage audit (locked run):** PASSED, 0 violations — 13 folds, 45,038 prediction placements, 364 thresholds,
4,400 graph-feature dates, 13,200 FRED-publication checks, 43,980 control values, 25 ETF listings.
**Resources:** locked run 518 s wall (588 s CPU), peak memory 0.78 GB; graph build 218 s; disk: data 23 MB,
outputs < 60 MB.

**Actual results — locked M5** (`20261030T225627Z_full_c1cccc3aa2_m4_selected`, test 2014-01-02 → 2026-10-16,
21 onsets, calm-day base rate 0.103; 95% paired block-bootstrap CIs; event metrics under the M3 rule with
Jeffreys intervals):

| method | ROC-AUC [95% CI] | PR-AUC | detected (recall 95% CI) | precision |
|---|---|---|---|---|
| Bayes — controls | 0.721 [0.628, 0.808] | 0.279 | 14/21 [0.45, 0.84] | 0.34 |
| VIX level | 0.721 [0.624, 0.807] | 0.293 | 14/21 [0.45, 0.84] | 0.37 |
| DCC-GARCH VaR | 0.692 [0.598, 0.768] | 0.240 | 14/21 [0.45, 0.84] | 0.41 |
| Bayes — combined | 0.671 [0.569, 0.769] | 0.224 | 14/21 [0.45, 0.84] | 0.45 |
| GBM — combined | 0.620 [0.515, 0.720] | 0.178 | 13/21 | 0.35 |
| GBM — controls | 0.613 [0.514, 0.711] | 0.208 | 13/21 | 0.33 |
| instability composite | 0.601 [0.467, 0.722] | 0.157 | 3/21 [0.04, 0.33] | 0.27 |
| Bayes — Granger | 0.543 [0.418, 0.657] | 0.147 | 7/21 | 0.19 |
| GBM — instability | 0.518 [0.416, 0.624] | 0.114 | 4/21 [0.07, 0.39] | 0.21 |
| GBM — Granger | 0.517 [0.386, 0.644] | 0.136 | 3/21 | 0.21 |
| rolling-corr VaR | 0.517 [0.396, 0.633] | 0.106 | 4/21 | 0.50 |
| NFCI (PIT) | 0.477 [0.359, 0.599] | 0.098 | 9/21 | 0.67 |
| DCC correlation | 0.466 [0.354, 0.578] | 0.090 | 4/21 | 0.38 |
| Bayes — instability | 0.438 [0.334, 0.548] | 0.109 | 4/21 [0.07, 0.39] | 0.18 |

Paired ROC-AUC differences [95% CI]: Bayes instability − VIX −0.283 [−0.437, −0.109]; GBM instability − VIX
−0.204 [−0.335, −0.060]; composite − VIX −0.121 [−0.304, 0.059]; DCC-GARCH VaR − VIX −0.030 [−0.117, 0.046];
Bayes controls − VIX 0.000 [−0.067, 0.068]. Instability added to controls: Bayes −0.050 [−0.115, 0.013],
GBM +0.007 [−0.059, 0.083].

**Exploratory banking stress on the full profile** (`20261030T231758Z_full_ff27ad9508_m4_selected+banking_kre`,
23 onsets): 2023-03-09 (SVB) is an onset; Bayes controls 0.738, VIX 0.736, DCC VaR 0.649, instability-only
0.59, composite 0.542; combined − controls Bayes −0.057 [−0.130, 0.015]. In the 10 sessions before SVB only DCC
correlation was alerting (one event; anecdotal).

**What M5 shows:** the dev-profile conclusion generalises to the 25-ETF / 13-node full profile. Directed-
dependence instability features do not beat VIX (instability-only models significantly worse) and do not add to
the controls (differences not distinguishable from 0, point estimates negative for Bayes). The composite does
better here (0.601) than on dev (0.403) but is not distinguishable from VIX and detects 3 of 21 onsets.
With the 25-ETF covariance, DCC-GARCH VaR becomes competitive with VIX (−0.030, CI includes 0).

**Limitations:** test span 2014–2026 only (W=500 + HYG listing push the first test fold to 2014), 21 onsets;
event-level intervals wide; NFCI vintages from 2011-05; FRED release dates untracked before 2014 (3-session
fallback); results conditional on SPY-volatility Markov labels (retrospective truth); one dataset/period;
VarLiNGAM assumptions (linearity, non-Gaussian independent shocks, no hidden confounders) are not tested here.

**Dashboard dataset:** M5 locked run passes all audits and the reproducibility check → primary dashboard
dataset (M4 dev run kept as reference).

**Next:** local LLM install (approved; hardware check passed), minimal FNSPID subset (size check first), M6.
