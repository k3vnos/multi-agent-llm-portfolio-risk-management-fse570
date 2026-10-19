# EXPERIMENT_LOG

Every model/configuration run, including discarded ones. "Split" says which data the result was computed on
(`inner-validation` or `outer-test`). Outer-test results are produced once per locked configuration.

| date | run_id | config hash | split | what changed | outcome |
|---|---|---|---|---|---|
| 2026-10-18 | (failed, no run dir) | 17c0c6c46d | — | first M2 run, H0 settings | crashed in label construction (read-only numpy array under pandas copy-on-write); fixed in `regimes/targets.py`, regression test added |
| 2026-10-18 | 20261019T044804Z_dev_17c0c6c46d | 17c0c6c46d | outer-test (2011–2026) | first complete M2 prototype run, all H0 settings | completed in 333 s. Diagnostic: every method alerts on 44–59% of test days under the H0 threshold rule → event-level recall/precision/lead not interpretable; see PROGRESS M2. **No setting was changed in response to these test results.** |
| 2026-10-18 | 20261019T045451Z_dev_17c0c6c46d | 17c0c6c46d | outer-test | reproducibility re-run of the run above, same code/data | **not reproducible**: test scores identical, but fold-0 thresholds differed. Cause: statsmodels `search_reps` Markov fit non-deterministic and sometimes stuck in a far worse optimum (llf −490 vs −248 on 2005–2008). Fixed with deterministic seeded multi-start (`regimes/markov.py`), test added. Both runs superseded. |
