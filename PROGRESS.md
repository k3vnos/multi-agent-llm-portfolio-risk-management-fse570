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
