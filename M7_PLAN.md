# M7 plan — read-only local API + dashboard (not started)

Status: **plan only**. M7 starts after the team reviews the M6 evaluation (`M6_EVAL_RUBRIC.md`).

## Primary dataset
- **M5 locked run** `20261030T225627Z_full_c1cccc3aa2_m4_selected` — passed all 7 leakage-audit checks (0 violations)
  and the bit-for-bit reproducibility check, so it is the primary dashboard dataset.
- M4 dev run `20261020T071838Z_dev_89c4b14759_m4_selected` is kept as a reference dataset (selector in the UI).
- Exploratory KRE banking-stress run shown only on a page labelled **Exploratory**.

## Architecture (free, local, read-only)
```
outputs/runs/<run_id>/ (saved artifacts)  ──►  FastAPI on 127.0.0.1:8000 (read-only)  ──►  Streamlit on 127.0.0.1:8501
data/interim/<graph cache>/varlingam.npz        no writes, no model calls                   calls the API only
<run>/explanations/*.json (precomputed)
```
- `src/riskcausal_api/` (FastAPI + uvicorn; free, MIT/BSD): binds `api.host` = 127.0.0.1 (config validator already
  refuses non-local hosts). Every response carries `run_id`, `git_sha`, `synthetic` and a `kind` field:
  `research_result` (from metrics.json / ci.json) or `illustrative` (explanations, case figures).
- The API never runs the pipeline and never calls the LLM: explanations are precomputed by
  `python -m riskcausal_agents.cli` and served as files. The dashboard therefore works with Ollama stopped.
- Refuses to serve a run whose `provenance.json` says `synthetic: true` as a result, or whose `audit.json` failed
  (served only with a red "audit failed" banner).

## Endpoints
| endpoint | returns | kind |
|---|---|---|
| `GET /runs` | run ids, profile, git sha, audit status, reproducibility status | metadata |
| `GET /runs/{id}/metrics` | per-method ROC/PR-AUC with 95% CIs, event metrics with Jeffreys intervals, both alert rules | research_result |
| `GET /runs/{id}/timeline?method=` | dates, score, threshold, alert flag, retrospective stress/onset (flagged retrospective) | research_result |
| `GET /runs/{id}/graph/{date}` | the source window used at that date (end date), B0 edges above the edge threshold, changes vs previous window — "estimated directed dependence" | illustrative |
| `GET /runs/{id}/alerts?method=` | alert episodes with start/end and retrospective outcome | research_result |
| `GET /runs/{id}/explanations/{alert_id}` | validated claims + cited evidence, uncertainty, abstention, coverage notes, retrospective reliability block, rejected-claim count | illustrative |

## Dashboard pages (Streamlit, local only — never Streamlit Community Cloud)
1. **Overview** — research question, headline negative result with CIs, links to PROGRESS/decision memo.
2. **Stress timeline** — target-series volatility with retrospective stress shading (labelled retrospective),
   scores and thresholds per method, alert markers; small multiples, no dual axes.
3. **Model comparison** — ROC-AUC/PR-AUC with 95% CI intervals vs VIX; combined − controls differences.
4. **Graph snapshot** — pick a date → the window used at that date, edges (estimated directed dependence),
   changes vs previous window, noise-floor caveat.
5. **Alert explorer** — pick an alert → explanation claims each with clickable cited evidence (with
   availability timestamps), uncertainty, abstentions, "news coverage ends 2020-06-11" banner, retrospective
   reliability block, and the illustrative-only disclaimer.
6. **Exploratory: banking stress (KRE)** — separate page with an "exploratory" banner.

## Tests (before demo)
- API binds 127.0.0.1 only; every route is GET; no route imports `riskcausal.pipeline` or the agents' backends.
- Every response has `kind` and `run_id`; failed-audit or synthetic runs are flagged/refused.
- Explanation endpoint returns only validated claims; every cited id resolves to evidence available ≤ cutoff.
- Snapshot tests on the M5 run (counts/metrics equal metrics.json values).

## Dependencies to install (free, PyPI, local)
`fastapi`, `uvicorn`, `httpx` (tests), `streamlit` — the existing `[api]` / `[dashboard]` extras. No cloud, no
deployment, no remote. Estimated work: 1–2 sessions.
