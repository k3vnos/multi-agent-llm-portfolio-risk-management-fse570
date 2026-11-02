# COST_POLICY — zero additional spending, local-only

This project must cost **nothing** beyond the team's existing subscriptions. These rules override every other
document in the repo (including `IMPLEMENTATION_PLAN.md`).

## 1. Zero additional spending
- Claude Code is used only through the existing **Claude Pro** subscription. A Pro subscription does **not**
  include Anthropic API access, and nothing in this project assumes it does.
- Project code makes **no** requests to the Anthropic API or any other paid LLM API. No paid-LLM SDK
  (`anthropic`, `openai`, `cohere`, `mistralai`, `google-generativeai`, …) is a dependency.
- No access to, or changes in, Anthropic billing, API keys, payment methods or subscriptions.
- No cloud resources, no deployments (AWS, Azure, GCP, Streamlit Community Cloud, Heroku, …), no paid
  services, trials or purchases.
- Only free, open-source libraries, publicly accessible datasets, and software running on this Mac.
- **If any component turns out to require payment, work stops and the team is asked.** Nothing proceeds
  automatically.

## 2. LLM agents are free and local
- The optional explanation agents use a **local** open-weight model served by **Ollama** on this Mac.
- Ollama is installed and models are pulled **only with explicit team approval** (multi-GB downloads).
- The default is still `DisabledBackend`; the local model is used only when `--backend ollama` is passed (`agents.enabled: false`): interfaces, prompts, retrieval,
  validation and tests exist, but no model is called.
- The Ollama backend refuses any host that is not `localhost` / `127.0.0.1` / `::1`. There is **no fallback**
  to any other provider: if the local model is unavailable, the agent step fails with a clear error.

## 3. Everything stays local
- All development and runs happen on this Mac.
- Local `git` only. No GitHub repository creation, no `git remote add`, no `git push`, no `gh` commands.
  The team uploads to GitHub themselves later.
- The FastAPI service and dashboard bind to `127.0.0.1` only.

## 4. Cost verification log
Every component is checked to be free for this use **before** installation/connection.

| Component | Cost for this use | Verified | Notes |
|---|---|---|---|
| Homebrew Python 3.12 + `venv` + `pip` | Free (PSF licence) | 2026-10-18 — already installed | used instead of `uv` (not installed) |
| PyPI libraries in `pyproject.toml` (pandas, numpy, scipy, pyarrow, statsmodels, lingam, pymc, arviz, arch, scikit-learn, yfinance, fredapi, requests, pydantic, pyyaml, typer, matplotlib, pandas_market_calendars, huggingface_hub, duckdb, pytest, ruff) | Free, open source | 2026-10-18 | installed into the local `.venv` |
| Yahoo Finance via `yfinance` | Free, no key | 2026-10-18 | public endpoints, rate-limited; research use |
| FRED graph CSV endpoint (`fredgraph.csv`) | Free, no key | 2026-10-18 | used for current-vintage series |
| FRED / ALFRED API | Free | 2026-10-18 | requires a **free** API key from fred.stlouisfed.org (no billing); stored in `.env` |
| FNSPID on Hugging Face | Free public dataset | checked in M1 | public download; disk space only |
| GDELT | Free public data | optional, checked in M1 | |
| Ollama 0.40.13 (Homebrew core formula, MIT) | Free, local | 2026-10-30 — installed with team approval | no background service registered; started manually, listening on 127.0.0.1:11434 only (verified with lsof) |
| qwen2.5:7b (Q4_K_M, 7.6B, digest 845dbda0ea48) | Free, Apache-2.0 | 2026-10-30 — pulled with team approval | 4.4 GB disk; ~35 tokens/s on the M2; used only by the optional explanation layer |
| FNSPID `Stock_news/All_external.csv` (streamed once) | Free public dataset | 2026-11-01 — team approved streaming the 5.7 GB file | not stored whole: 5.73 GB streamed and filtered on the fly (SHA256 recorded); 82 MB subset kept |
| FastAPI 0.162 (MIT), uvicorn 0.54 / httpx 0.28 (BSD-3), Streamlit 1.66 (Apache-2.0), Altair 6.3 (BSD-3) | Free, open source | 2026-11-02 — installed (team approved M7) | both servers bind 127.0.0.1 (verified with lsof); Streamlit usage statistics disabled and "Deploy" button hidden (`.streamlit/config.toml`, tested); never a hosted service |

## 5. Defaults that enforce this policy
- `configs/base.yaml`: `agents.enabled: false`, `agents.backend: disabled`, `api.host: 127.0.0.1`, `deploy: none`.
- `.env.example` lists only `FRED_API_KEY` and `OLLAMA_HOST=http://localhost:11434` — no paid-API key fields.
- `tests/test_cost_policy.py` fails if a paid-LLM SDK appears in `pyproject.toml`, if any source file imports
  one, or if config defaults point to a non-local host or enable deployment.
