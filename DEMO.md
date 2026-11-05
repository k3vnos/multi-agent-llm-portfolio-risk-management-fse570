# Structural Instability Monitor — running and demonstrating the product

## Start, open, stop
```bash
./run.sh start      # builds the UI on first run (~5 s), starts the local server, opens http://127.0.0.1:8000
./run.sh stop       # stop it
./run.sh restart    # after a reboot, a code change, or if anything looks stuck
./run.sh status     # running? which URL?
./run.sh open       # open the app in your default browser again
./run.sh test       # backend + frontend unit tests, and browser tests if the server is running
```
Shortcuts: `make app`, `make app-stop`, `make app-test`. Everything binds to **127.0.0.1** only, reads saved research
outputs only, and makes no network calls (no paid APIs, no cloud). Nothing needs to run in the background after a
reboot: run `./run.sh start` again.

**Live AI agents.** `./run.sh start` also starts the local model server (Ollama, 127.0.0.1:11434) if it is
installed and not running; `./run.sh status` says whether it is up. The button **▶ Ask the agents** (Simple: *Ask
the AI agents* and *My portfolio*; Advanced: *AI agents* and *Portfolio Lens*) runs the four agents on Qwen 2.5 7B for
the replay date — about 30–70 s on this Mac, with real progress and a **Cancel** button. Without Ollama (or with
`NO_OLLAMA=1 ./run.sh start`) the button is disabled and the saved explanation for alert dates is shown instead.

**First time on a new machine:** `make install` (Python environment), and Node.js 20+ installed; `./run.sh start`
installs the UI's npm packages automatically the first time.

## Two modes
- **Simple** (default, for presenting): *Briefing · Relationships · My portfolio · Ask the AI agents · What we
  found*. Plain English, large type; technical terms live in the **?** tooltips and the "Show …" expanders.
- **Advanced** (toggle at the top): the complete technical dashboard — *Monitor · Timeline · Relationship network ·
  Alerts · Portfolio Lens · AI agents · Research & validation*. Switching keeps the date (and the portfolio) and
  opens the matching page.

## 4–6 minute demonstration (Simple Mode)
Click **▶ Guided demo** on any Simple page — seven plain-language steps:

1. **What this app is** — a replay of past data, answering three questions: what is happening, why the model did or
   did not alert, what evidence supports it.
2. **COVID-19, 24 Feb 2020** — score above the alert line → alert; market mood stressed. Try *Play forward*.
3. **The idea we tested** — purple (our relationship-change index) vs orange (the VIX fear gauge): if the idea
   worked, purple would rise first. It didn't reliably.
4. **Relationships** — only the 5 biggest changes by default; click an arrow to read it in words.
5. **My portfolio, 16 Mar 2020** (exploratory) — the *Classic 60/40* example: volatility (unusually high), last
   month, bad-day loss, worst fall, which holding carries the risk (stocks; bonds offset part of it) and the changed
   links around the holdings. Everything is computed from prices up to that close only.
6. **Ask the AI agents** — press **▶ Ask the agents (≈1 minute)** with *Include my portfolio* ticked. Watch the four
   steps progress (Graph → Macro → News → Synthesis). The trace shows what each agent was given, what passed the
   checker (✓), what was rejected and why (✗), who had no data, and the combined summary.
7. **What we found** — the indicator did not improve on the fear gauge; numbers, uncertainty and limitations.

**Full journey to show live:** *My portfolio* → *Classic 60/40* → *COVID-19 crash* → set the date to 16 Mar 2020 →
read the risk cards → *Ask the AI agents* → **▶ Ask the agents** → read the trace.

## Advanced Mode walkthrough
Click **▶ Guided demo** (top right). Each step opens a real page on a real date:

1. **Monitor** (31 Dec 2019) — SII, market regime as estimated then, risk status (documented rule), relationship
   turnover; hover any **?** for a plain-English explanation.
2. **A real stress episode** (24 Feb 2020, COVID-19) — status turns to *Model alert*; press **▶ Play**.
3. **Timeline** — point-in-time percentiles, raw small multiples, model score vs threshold; *Show what happened
   next* reveals hindsight (clearly marked). The SII rose *with*, not before, the VIX.
4. **Relationship network** — estimated directed dependence; toggles; click an asset or an arrow.
5. **Portfolio Lens** (16 Mar 2020, 60/40) — all metrics incl. CVaR, normal-model VaR, co-movement shift, Euler risk
   shares, network links touching the holdings, and the exact definitions.
6. **AI agents** — **▶ Ask the agents** runs the v4 pipeline live; the trace shows each agent's numbered inputs (numbers
   only, no labels), confirmed claims, rejected claims with reasons, model notes that passed the checks, abstentions
   and the synthesis (confirmed, or rejected → rule-based summary). Below: the rule-based evidence view and the
   superseded M6 free-text draft.
7. **Research & validation** — the locked result, CIs, comparisons, leakage audit, reproducibility.

Optional extras: **Alerts** → pick any alert → its evidence and, separately, the hindsight outcome;
**Regional banking stress · Mar 2023** button → the exploratory KRE episode (news unavailable after June 2020 is
shown as missing, not neutral).

## Troubleshooting
| symptom | fix |
|---|---|
| `Port 8000 is in use by another program` | another app uses the port: `PORT=8010 ./run.sh start` (then open http://127.0.0.1:8010) |
| `Python environment missing` | `make install` |
| `Node.js/npm not found` | install Node 20+ (free), then `./run.sh start` |
| `UI build failed` | `./run.sh rebuild`; details in `outputs/logs/app.log` |
| blank page or "Could not load this view" | `./run.sh restart`, then reload the browser tab (⌘⇧R) |
| page shows old content after an update | `./run.sh rebuild` |
| "local model not running" on the agents panel | `ollama serve` (or `./run.sh restart`); check `curl 127.0.0.1:11434/api/version`. Saved explanations still show for alert dates |
| an agent run takes long or hangs | press **Cancel**; each run stops by itself after 300 s |
| `No saved runs in outputs/runs` | the app replays saved research outputs; restore `outputs/` (it is not in git) |

Logs: `outputs/logs/app.log`. The app never modifies research artifacts.
