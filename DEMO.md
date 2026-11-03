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

**First time on a new machine:** `make install` (Python environment), and Node.js 20+ installed; `./run.sh start`
installs the UI's npm packages automatically the first time.

## 3–5 minute demonstration
Click **▶ Guided demo** (top right). Each step opens a real page on a real date:

1. **Monitor** (31 Dec 2019) — what the dashboard shows: SII, market regime as estimated then, risk status (documented
   rule), relationship turnover; hover any **?** for a plain-English explanation.
2. **A real stress episode** (24 Feb 2020, COVID-19) — status turns to *Model alert*, the regime estimate flips to
   *stressed*. Press **▶ Play** in the replay bar to step forward week by week; **❚❚ Pause** to stop.
3. **Timeline** — SII, VIX and NFCI as point-in-time percentiles plus raw small multiples and the model's score vs
   threshold. Click **Show what happened next** to reveal hindsight (clearly marked). Point out that the SII rose
   *with*, not before, the VIX.
4. **Relationship network** — estimated directed dependence among 10 asset classes and 3 macro factors. Toggle
   *Changes / Current links*, *vs 1 week / vs 1 month*, *Network / Matrix*; click an asset or an arrow.
5. **AI analysis** (10 Jan 2020) — Graph, Macro and News agents (headlines published before the 4pm close), the
   deterministic synthesis briefing with citations, and the LLM draft shown as **UNVERIFIED** with its hand-check
   flags (held-out faithfulness 73% < 90% target).
6. **Research & validation** — the locked result: instability measures did not beat the VIX; CIs, paired
   comparisons, lead times, exploratory banking stress, leakage audit, bit-identical reproducibility.

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
| `No saved runs in outputs/runs` | the app replays saved research outputs; restore `outputs/` (it is not in git) |

Logs: `outputs/logs/app.log`. The app never modifies research artifacts.
