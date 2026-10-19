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
