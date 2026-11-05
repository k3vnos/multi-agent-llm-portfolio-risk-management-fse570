# M10 — v4 structured agents: design and evaluation

Model: **qwen2.5:7b** via local Ollama 0.40.13 (digest `845dbda0ea48…`), temperature 0, seed 570, JSON-schema
constrained decoding. No new or larger model was downloaded. All outputs are in `outputs/agents_v4/` (the locked
M5 run directory and the M6 explanation sets are only read).

## What changed vs M6 (free text, 73% held-out faithfulness)
| | M6 (`agents.py`, kept for the locked M6 evaluation) | M10 v4 (`facts.py`, `structured.py`) |
|---|---|---|
| model input | evidence sentences incl. label words | numbered facts F1…Fn with **numbers only** (no "high/low" words) |
| model output | free-text claims citing evidence ids | `{fact_id, reading, label, note}`; fact id and label are **enums** in the JSON schema, one alternative per fact so each fact can only take its own label vocabulary |
| validation | ids, numbers, names, dates, forbidden phrases | all of M6, plus: label must equal the label computed from the saved numbers; note checked for direction vs label polarity, change-over-time words on a level fact, correlation claims, risk-level claims, forecasts, causal/effect wording, investment advice; news theme must have a keyword in the headline |
| text shown to users | the model's sentences (marked unverified) | the fact's **rule-rendered sentence** for every confirmed claim; a model note only if it passed every check |
| synthesis | free text | key facts (enum of confirmed facts), overall class (enum) and missing agents, each checked against deterministic values; on failure the rule-based summary is shown |

Label rules: level = high (>80th pct) / low (<20th) / typical / unknown; 5-day VIX change = little changed (|Δ|<0.5)
/ rose / fell / "sharply" if its percentile is >80 / <20; links = appeared / disappeared / sign flipped /
strengthened / weakened (threshold 0.05); alert = above / below line; portfolio risk share = largest contributor /
above / in line with / below its weight (±5 points).

## Protocol
1. **Tuning** only on the 12 M6 tuning alerts (seed 570). Round 1 (`tune_r1`, prompt m10-v4-r1): label accuracy
   78% (81/104). Changes: explicit percentile band examples, edge facts written as "(no link)/(link)", 5-day VIX
   threshold 0.5 points, "reading" field before the label, headline-only notes, word-start news keywords.
   Round 2 (`tune`, **m10-v4-r2**): 82% (87/106). Prompts **frozen** at m10-v4-r2.
2. **Fresh held-out set**, never used by M6 or v4 development, scored **once**: all 23 alert episodes not in either
   M6 set, plus 12 random no-alert days (seed 572) to exercise the "below alert line" path → 35 dates
   (`outputs/agents_v4/heldout/`). The 12 M6 held-out alerts were not reused because M6 development looked at them.
3. **All alerts**: the frozen pipeline on all 47 alert-episode start dates (`outputs/agents_v4/all/`), shown in the app.

## Held-out results (35 dates, 117 model calls)
| measure | result |
|---|---|
| label accuracy **before** validation (all claims the model asserted) | **80.5%** (239/297) |
| — fresh alert dates (23) | 86.5% (179/207) |
| — no-alert days (12) | 66.7% (60/90) |
| — by fact kind | level 126/151 · change 26/32 · link 43/58 · alert 10/13 · news theme 34/43 |
| claims rejected by the validator | 58 (57 label mismatches, 1 duplicate) — none reached a user |
| statements shown as fact | rule-rendered from confirmed facts → **faithful by construction** |
| model notes that passed the automatic checks, hand-checked (first pass, Claude; pending team review) | **97/106 = 91.5%** faithful |
| — specialist notes | **96/96** |
| — synthesis notes | **1/10** → synthesis notes are **no longer displayed as statements** (decided after this held-out check; scores above are unchanged) |
| notes rejected automatically | 14 (direction 6, numbers 4, semantics 3, names 1) |
| synthesis outputs confirmed (key facts, overall class, missing agents all correct) | 33/35 |
| news agent abstentions | 23/35 dates (all no-coverage dates: FNSPID ends 2020-06-11; none model-initiated) |
| dates with no confirmed fact | 0 |

**Honest reading.** The model alone is right about 4 times in 5 when it reads a number into a label (worse on quiet
days). Because every label is checked against the saved numbers and the displayed sentence is rendered from the
fact, those errors are caught rather than shown. The remaining risk lies in free-text notes. Specialist notes were
all faithful on the held-out set. Synthesis notes mostly were not, so they are hidden. Hand checks are a first pass
by the assistant and need team review. The M6 figure (73%) measured free-text claims; v4's figures measure label
choices and short notes, so the two are not directly comparable.

## Known limitations
- 7B model: label accuracy is below 90%; live runs on no-alert days confirm fewer facts.
- News "themes" are validated by keyword presence, which is a heuristic (e.g. a headline mentioning both a rally
  and a drop matches "market decline").
- The causal-wording check matches "effect" but not "effects"; the hidden synthesis notes were the cases it missed.
- Notes copied from headlines may contain the headline's own causal phrasing; they are attributed to the headline.
- Portfolio facts in live runs were not part of the held-out evaluation (they use the same label rules and checks).
