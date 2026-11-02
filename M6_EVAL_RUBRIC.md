# M6 explanation evaluation — rubric

Applies to the pre-specified evaluation set written by
`python -m riskcausal_agents.cli eval-set <run_dir> --backend ollama`
(`<run_dir>/explanations/eval/eval_set.csv`): alert episodes of `bayes_logit:combined` in the M5 locked run,
3 per stratum, seed 570, strata = news coverage (yes / no) × retrospective outcome (true alert / false alarm).
The retrospective outcome is used **only** to stratify and score; agents never see it.

| # | Criterion | How it is checked | Scale |
|---|---|---|---|
| R1 | Citation validity | automatic: every shown claim passed the validator (ids exist in the point-in-time bundle, numbers match, no unsupported names, no forbidden language, no future dates). Rejected claims are listed per agent | count kept / rejected |
| R2 | Faithfulness | manual: for each shown claim, is it a correct restatement of the cited evidence (no exaggeration, no unsupported interpretation)? | k of n claims |
| R3 | No hindsight | manual: does any claim rely on knowledge of what happened after the decision date (even without naming it)? | 1 = none, 0 = some |
| R4 | No forbidden claims | automatic (validator) + manual spot check: no causal proof, predictive superiority, forecasts | 1 / 0 |
| R5 | Uncertainty | manual: does the explanation say what the evidence does not show (e.g. missing news, noisy graph estimates, that an alert is not a forecast)? | 1 / 0 |
| R6 | Abstention appropriate | manual: abstains (fully, or for the news agent) when evidence is missing — in particular after the FNSPID coverage end — and does not abstain when evidence is ample | 1 / 0 |

**Pass bar for the demonstration** (set before running): R1 rejected claims never shown (by construction);
R2 ≥ 90% of shown claims faithful; R3 = 1 and R4 = 1 for every explanation; R5 = 1 for ≥ 10 of 12; R6 = 1 for
every explanation in a `news=no` stratum (news agent abstains, synthesis states the gap).

**Who checks.** The first pass is done by the assistant (Claude) and marked `checked_by = Claude (first pass)`;
the team must review and sign off (`checked_by` updated) before the explanations are presented as evaluated.
Explanations are illustrative: they describe the evidence available at alert time and never change predictions
or metrics.
