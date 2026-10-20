# Week-8 decision memo (M4) — 2026-10-20

Every number below is copied from saved artifacts: `outputs/runs/<run_id>/{metrics.json, ci.json, selection.json}`.
Dev profile only (6 ETFs + VIX/BAA10Y/T10Y2Y graph nodes). Settings were selected on inner-validation spans
using the rules pre-registered in `configs/experiments/robustness.yaml` (commit 7464cfa), before any M4 test
scoring.

## Research question
Does instability in rolling VarLiNGAM directed-dependence estimates give earlier warning of transitions into a
stressed regime than VIX, NFCI, rolling-correlation VaR and DCC-GARCH?

## Evidence

**1. Locked test result (run `20261020T071838Z_dev_89c4b14759_m4_selected`; W=500, edge 0.05, 2 regimes;
2012-01-03 → 2026-10-16; 24 onsets; 95% paired block-bootstrap CIs).**

| method | ROC-AUC [95% CI] | Δ vs VIX [95% CI] |
|---|---|---|
| VIX level | 0.735 [0.651, 0.815] | — |
| Bayes logit, controls | 0.699 [0.616, 0.780] | −0.037 [−0.097, 0.017] |
| Bayes logit, instability + controls | 0.608 [0.500, 0.715] | −0.127 [−0.243, −0.024] |
| DCC-GARCH VaR | 0.571 [0.472, 0.676] | −0.164 [−0.264, −0.067] |
| GBM, instability | 0.495 [0.398, 0.594] | −0.241 [−0.366, −0.123] |
| Bayes logit, instability | 0.490 [0.382, 0.597] | −0.245 [−0.393, −0.097] |
| instability composite | 0.403 [0.298, 0.511] | −0.332 [−0.468, −0.190] |

Adding instability features to the control model: Bayes **−0.090 [−0.185, −0.010]** (significantly worse),
GBM −0.042 [−0.124, 0.042].

**2. The same conclusion holds under every alternative we pre-registered or ran:**
- H0 settings (W=250; M3 run `…24a61f320e`): every instability method significantly below VIX; combined − controls
  Bayes −0.032 [−0.064, 0.000], GBM +0.013 [−0.055, 0.090].
- 3-regime labels (validation-selected label model; runs `…05b453a740_regimes3`, `…64c063c907_…regimes3_label`):
  instability-only ROC-AUC 0.43–0.54; combined − controls not positive (−0.065 [−0.199, 0.054] Bayes).
- Validation grid (9 graph configs × 2 label sets): instability-only validation PR-AUC never reaches VIX
  (best 0.246 vs 0.411 under 2 regimes). Under 2 regimes, adding instability lowered validation PR-AUC versus
  controls in all 9 configs. Under 3 regimes it raised it for W=500 (0.358 vs 0.281) — **that gain did not carry
  over to test**.
- Exploratory banking-stress target (KRE volatility, run `…ab1b3e098a_…banking_kre`): instability-only 0.62
  ROC-AUC vs VIX 0.73 and Bayes-controls 0.74; combined − controls −0.015 [−0.085, 0.065].

**3. Why (diagnostics).** Single-window graphs are imprecise: bootstrap refits of the *same* 250-day window
disagree on ~21 of 72 contemporaneous edges, and only 9–14% of window-to-window changes exceed that noise
(`diagnose-graphs`). The validation-selected window (W=500) is the longest in the grid, consistent with noise
reduction mattering most — but it still did not produce test-period signal.

**4. Selection did not transfer.** Validation preferred W=500 over H0 (W=250), yet on test the instability models
did worse with W=500 (GBM instability 0.495 vs 0.611 at H0). With 24 test onsets and CIs of ±0.1, differences of
this size between configurations are within noise.

## Options (proposal §D decision gate)
1. **Continue unchanged** — not supported: no configuration shows a positive, CI-supported effect.
2. **Lengthen windows / average methods** — partially tested already (W=500, Granger, composite, GBM); none helps.
   A further search would be a fishing expedition on 24 onsets.
3. **Pivot to explaining a negative result (recommended).** Report a rigorous, leakage-audited null: under
   point-in-time data and validation-only selection, VarLiNGAM-instability features do not provide earlier or
   better warning than VIX on this sample, and the estimated graphs are too noisy at feasible window lengths.
   Keep M5 (full profile, 10 graph nodes incl. HYG/KRE, locked protocol, one run) as the generality check the
   proposal promises — it could still overturn the dev-profile result, and it is the dataset the dashboard and
   agents should present.

## Recommendation
Option 3, with one locked M5 full-profile run using the M4-selected settings (no new tuning on test), then the
optional explanation layer framed honestly: the agents explain *what the evidence showed at alert time*
(VIX/controls, graph changes, date-bounded news), not that the graph gave early warning.
