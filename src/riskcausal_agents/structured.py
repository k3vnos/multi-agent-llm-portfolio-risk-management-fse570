"""v4 structured multi-agent pipeline (M10): Graph, Macro and News specialists, then Synthesis.

Differences from the M6 free-text agents (`agents.py`, kept unchanged for the locked M6 evaluation):
- each agent receives numbered facts (F1, F2, …) built deterministically from point-in-time evidence, with the raw
  numbers but WITHOUT the label words;
- decoding is constrained by a JSON schema: the model can only pick fact ids from its own list and labels from a
  fixed vocabulary (Ollama structured outputs);
- the validator compares every asserted label with the label computed from the saved numbers, and checks the
  optional short note for direction, change-over-time, correlation, causal, risk-level, forecast, number, name and
  date errors;
- the statement shown to a user is always the fact's rendered sentence; a model note is shown only if it passed
  every check, and is always marked as model-written.
The pipeline never feeds back into scores, alerts or metrics.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime

from riskcausal_agents.backends import LLMBackend
from riskcausal_agents.evidence import Bundle
from riskcausal_agents.facts import (
    ALERT_LABELS,
    CHANGE_LABELS,
    EDGE_LABELS,
    LEVEL_LABELS,
    NEWS_LABELS,
    OVERALL_LABELS,
    POLARITY,
    SHARE_LABELS,
    UNUSUAL,
    Fact,
    bundle_facts,
    label_ok,
    number,
    overall_label,
)
from riskcausal_agents.validator import _num_ok, _numbers, forbidden_hits, names_in

PROMPT_VERSION = "m10-v4-r2"
AGENTS = ("graph", "macro", "news")
MAX_CLAIMS = 4
NOTE_WORDS = 25

LABEL_RULES = f"""Labelling rules (apply them to the numbers exactly):
- level facts: the label depends ONLY on the percentile, never on whether the value seems large or small.
  "high" only if percentile > 80; "low" only if percentile < 20; any percentile from 20 to 80 is "typical"
  (examples: 85 -> high, 79 -> typical, 25 -> typical, 21 -> typical, 12 -> low); "unknown" if no history.
  Allowed: {LEVEL_LABELS}.
- change facts (5-day VIX change): "little changed" if |change| < 0.5 points; otherwise "rose"/"fell" by the sign, and
  "rose sharply" only if the change's percentile > 80, "fell sharply" only if it is < 20. Allowed: {CHANGE_LABELS}.
- link facts: a link exists when |coefficient| is above the threshold. "appeared" (absent before, present now),
  "disappeared" (present before, absent now), "sign flipped" (present both times, opposite signs), otherwise
  "strengthened"/"weakened" by whether |coefficient| grew or shrank (examples: +0.00 -> -0.54 appeared;
  +0.40 -> +0.02 disappeared; +0.30 -> -0.20 sign flipped; -0.20 -> -0.35 strengthened). Allowed: {EDGE_LABELS}.
- alert fact: "above alert line" if score >= alert line, else "below alert line". Allowed: {ALERT_LABELS}.
- portfolio share facts: "largest contributor" for the holding with the biggest risk share; otherwise "above its
  weight" / "below its weight" if its risk share is more than 5 points above / below its portfolio weight, else
  "in line with its weight". Allowed: {SHARE_LABELS}.
- news facts: the theme of the headline, one of {NEWS_LABELS}; pick a theme only if the headline is about it."""

NOTE_RULES = f"""The optional "note" (at most {NOTE_WORDS} words, or empty) may only restate the chosen fact in plain words; for a
headline, only copy words from the headline itself (or leave it empty).
It must NOT: say what caused anything; forecast or say what will happen; mention correlation; call risk high or low;
describe a change over time for a level fact; add numbers, names, events or dates that are not in the fact."""

ROLE = {
    "graph": "You are the Graph agent. Your facts describe the estimated directed-dependence network among assets "
             "(a noisy statistical estimate, not causality).",
    "macro": "You are the Macro agent. Your facts are market and macro-financial indicators exactly as known at the "
             "market close of the decision date, plus the warning model's alert status.",
    "news": "You are the News agent. Your facts are headlines published before the 4pm New York cutoff.",
}

UP = r"\b(rose|rise[sn]?|rising|increas\w*|higher|up|jump\w*|surg\w*|spik\w*|climb\w*|elevated|widen\w*|grew|grow\w*|strong\w*|tighten\w*|gain\w*|rall\w*)\b"
DOWN = r"\b(fell|fall\w*|declin\w*|lower|down|drop\w*|dip\w*|eas\w*|narrow\w*|shrank|shrink\w*|weak\w*|loosen\w*|subdued|slid|slump\w*|plung\w*|tumbl\w*)\b"
CHANGE_WORDS = r"\b(rose|rise[sn]?|rising|fell|fall(s|ing)?|increas\w*|decreas\w*|jump\w*|drop\w*|climb\w*|declin\w*|surg\w*|grew|shrank|since|compared (with|to)|than (last|before|the previous))\b"
SEMANTIC = [
    (r"\bcorrelat\w*", "correlation claim (not supported by the facts)"),
    (r"\b(high|low|elevated|increased|reduced|significant|major|serious)[- ](level of )?risk\b|\brisk (is|was|were|are|looked|appeared) (high|low|elevated)\b|\b(dangerous|risky|safe)\b",
     "risk-level claim"),
    (r"\b(will|would likely|is likely to|are likely to|expected to|could lead|may lead|might lead|going to|poised to|set to)\b",
     "forecast language"),
    (r"\b(effect|impact|influenc\w*|affect\w*|trigger\w*|spark\w*|fuel\w*|amid|in response to|as a result)\b",
     "causal or effect language"),
    (r"\b(buy|sell|hold|should|recommend\w*|advis\w*|reallocat\w*|hedg\w*)\b", "investment advice"),
]


class Stop(Exception):
    pass


def fact_line(f: Fact) -> str:
    return f"- {f.id} [{f.kind}]: {f.prompt}"


def schema_for(facts: list[Fact]) -> dict:
    """One alternative per fact, so each fact id can only be paired with the label vocabulary of its kind. The
    `reading` field (the number the label rests on) is generated before the label."""
    alts = [{"type": "object", "properties": {
        "fact_id": {"type": "string", "enum": [f.id]}, "reading": {"type": "string"},
        "label": {"type": "string", "enum": f.allowed}, "note": {"type": "string"}},
        "required": ["fact_id", "reading", "label", "note"]} for f in facts]
    return {"type": "object", "properties": {
        "claims": {"type": "array", "maxItems": MAX_CLAIMS, "items": {"anyOf": alts}},
        "abstain": {"type": "boolean"}, "abstain_reason": {"type": "string"}},
        "required": ["claims", "abstain", "abstain_reason"]}


def check_note(note: str, facts: list[Fact], decision_date: str) -> list[str]:
    """Semantic checks on a model-written note about `facts` (one fact for specialists; several for synthesis)."""
    note = (note or "").strip()
    if not note:
        return []
    r = []
    if len(note.split()) > NOTE_WORDS:
        r.append("note: too long")
    r += [f"note language: {w}" for w in forbidden_hits(note)]
    r += [f"note semantics: {why}" for pat, why in SEMANTIC if re.search(pat, note, re.IGNORECASE)]
    pool = [x for f in facts for x in _numbers(f.text + " " + f.prompt)]
    pool += [float(v) for f in facts for v in f.values.values() if isinstance(v, int | float)]
    bad = [x for x in _numbers(note) if not _num_ok(x, pool)]
    if bad:
        r.append(f"note numbers: {bad} not in the fact")
    src = " ".join(f.text + " " + f.prompt for f in facts).lower()
    missing = sorted(n for n in names_in(note) if n.lower() not in src)
    if missing:
        r.append(f"note names: {missing} not in the fact (possible hindsight)")
    for m in re.finditer(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", note):
        if m.group(0) > decision_date:
            r.append(f"note dates: {m.group(0)} is after the decision date")
    if len(facts) == 1:
        f = facts[0]
        up, down = bool(re.search(UP, note, re.IGNORECASE)), bool(re.search(DOWN, note, re.IGNORECASE))
        if f.kind == "news":
            for w in re.findall(UP + "|" + DOWN, note, re.IGNORECASE):
                w = w if isinstance(w, str) else next(x for x in w if x)
                if w.lower()[:4] not in f.text.lower():
                    r.append(f"note direction: '{w}' is not in the headline")
                    break
        else:
            pol = POLARITY.get(f.label, "none")
            if f.kind in ("level", "share", "alert") and re.search(CHANGE_WORDS, note, re.IGNORECASE):
                r.append("note semantics: describes a change over time, but the fact is a level")
            elif (up and pol in ("down", "none")) or (down and pol in ("up", "none")):   # "flip" allows both
                r.append(f"note direction: contradicts the evidence ({f.label})")
    return r


def validate_specialist(out: dict, facts: list[Fact], decision_date: str) -> dict:
    by = {f.id: f for f in facts}
    claims = out.get("claims") if isinstance(out.get("claims"), list) else []
    kept, rejected, seen = [], [], set()
    for c in claims[:MAX_CLAIMS + 2]:
        c = c if isinstance(c, dict) else {}
        fid, lab, note = str(c.get("fact_id", "")), str(c.get("label", "")), str(c.get("note", "") or "")
        f = by.get(fid)
        reasons = []
        if f is None:
            reasons.append(f"citation: unknown fact id {fid!r}")
        elif fid in seen:
            reasons.append("duplicate fact")
        elif lab not in f.allowed:
            reasons.append(f"label: {lab!r} is not a valid label for a {f.kind} fact")
        elif not label_ok(f, lab):
            truth = f.label if f.kind != "news" else (", ".join(f.values.get("themes", [])) or "no theme keywords")
            reasons.append(f"label mismatch: model said {lab!r}, the numbers say {truth!r}")
        rec = {"fact_id": fid, "label": lab, "note": note, "source": f.source if f else None}
        if reasons:
            rejected.append({**rec, "reasons": reasons})
            continue
        seen.add(fid)
        nr = check_note(note, [f], decision_date)
        kept.append({**rec, "kind": f.kind, "text": f.text, "true_label": f.label,
                     "note": "" if nr else note, **({"note_rejected": {"text": note, "reasons": nr}} if nr else {})})
    return {"claims": kept, "rejected": rejected,
            "abstain": not kept, "model_abstained": bool(out.get("abstain")) and not claims,
            "abstain_reason": (str(out.get("abstain_reason", ""))[:300] if not kept else "")}


def _call(backend: LLMBackend, system: str, prompt: str, schema: dict, ctl) -> tuple[dict, dict]:
    t0 = time.time()
    out, meta = backend.complete_json(system, prompt, schema=schema, should_cancel=ctl.cancelled,
                                      on_token=ctl.tokens)
    meta["wall_secs"] = round(time.time() - t0, 2)
    return out, meta


class Control:
    """Progress + cancellation hooks for one run (no-ops by default; the API job runner overrides them)."""

    def __init__(self, emit=None, cancelled=None):
        self._emit, self._cancelled, self.current = emit, cancelled, None

    def emit(self, **ev):
        if self._emit:
            self._emit(ev)

    def cancelled(self) -> bool:
        return bool(self._cancelled and self._cancelled())

    def tokens(self, n: int):
        if self._emit and n % 8 == 0:
            self._emit({"type": "tokens", "agent": self.current, "tokens": n})


def run_specialist(role: str, facts: list[Fact], b: Bundle, backend: LLMBackend, ctl: Control,
                   context: str = "") -> dict:
    mine = [f for f in facts if f.agent == role]
    base = {"role": role, "inputs": [f.to_json() for f in mine]}
    notes = [n for n in b.notes if n.startswith(role)]
    if not mine or all(f.kind == "alert" for f in mine):
        reason = "; ".join(notes) or f"no {role} evidence available for this date"
        return {**base, "claims": [], "rejected": [], "abstain": True, "model_abstained": False,
                "abstain_reason": reason, "called_model": False}
    system = f"{ROLE[role]}\n{LABEL_RULES}\n{NOTE_RULES}\nUse only the facts listed. Do not use outside knowledge."
    prompt = (f"Decision date: {b.date} (market close). {context}\n"
              f"Pick the 1 to {MAX_CLAIMS} facts that best describe conditions on this date, most important first, and "
              f"give each the label its numbers support. In \"reading\" write the number the label depends on and "
              f"where it falls, BEFORE choosing the label, e.g. \"percentile 78: between 20 and 80\", \"change -0.96: "
              f"|change| >= 0.5, percentile 31\", \"no link -> link\". If none of the facts is informative, set abstain "
              f"to true.\n"
              f"Facts:\n" + "\n".join(fact_line(f) for f in mine))
    ctl.current = role
    out, meta = _call(backend, system, prompt, schema_for(mine), ctl)
    v = validate_specialist(out, mine, b.date)
    return {**base, **v, "called_model": True, "system": system, "prompt": prompt, "raw_output": out, "meta": meta}


def synthesis_schema(valid_ids: list[str]) -> dict:
    return {"type": "object", "properties": {
        "key_facts": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": valid_ids}},
        "overall": {"type": "string", "enum": OVERALL_LABELS},
        "missing_agents": {"type": "array", "items": {"type": "string", "enum": list(AGENTS)}},
        "note": {"type": "string"}}, "required": ["key_facts", "overall", "missing_agents", "note"]}


def deterministic_synthesis(b: Bundle, facts: list[Fact], valid: list[Fact], abstained: list[str]) -> str:
    alert = next((f for f in facts if f.kind == "alert"), None)
    parts = [alert.text.split(";")[0] + "." if alert else ""]
    ov = overall_label(valid)
    parts.append(f"Of the {len(valid)} facts the agents selected and the validator confirmed, the readings were {ov}."
                 if valid else "No agent produced a validated fact.")
    if abstained:
        parts.append("Not covered: " + ", ".join(f"{a} agent" for a in abstained) + " (missing data, not 'calm').")
    parts.append("This describes conditions observable at the close; it is not a forecast.")
    return " ".join(p for p in parts if p)


def run_synthesis(b: Bundle, facts: list[Fact], specialists: list[dict], backend: LLMBackend, ctl: Control,
                  context: str = "") -> dict:
    by = {f.id: f for f in facts}
    valid = [by[c["fact_id"]] for s in specialists for c in s["claims"]]
    abstained = [s["role"] for s in specialists if s["abstain"]]
    truth = overall_label(valid)
    base = {"role": "synthesis", "inputs": {"validated_facts": [f.id for f in valid], "abstained": abstained},
            "fallback_text": deterministic_synthesis(b, facts, valid, abstained), "true_overall": truth}
    if len(valid) < 2:
        return {**base, "abstain": True, "called_model": False, "validated": False, "claims": [], "rejected": [],
                "abstain_reason": f"only {len(valid)} validated fact(s); at least 2 are needed for a synthesis"}
    alert = next((f for f in facts if f.kind == "alert"), None)
    system = ("You are the Synthesis agent. You receive only facts that other agents selected and a validator "
              "confirmed. Choose up to 3 key facts, classify the evidence overall, and list which agents had no "
              "evidence. overall rule: count the facts whose label is unusual (high, low, rose sharply, fell sharply, "
              "appeared, disappeared, sign flipped, above alert line, largest contributor, above its weight); news "
              "facts are not counted; 'mostly unusual' if that share is >= 60%, 'mostly typical' if <= 30%, else "
              f"'mixed'.\n{NOTE_RULES}")
    def mark(f: Fact) -> str:
        return "news (not counted)" if f.kind == "news" else ("unusual" if f.label in UNUSUAL else "not unusual")

    n_unusual = sum(f.label in UNUSUAL for f in valid if f.kind != "news")
    n_counted = sum(f.kind != "news" and f.label != "unknown" for f in valid)
    lines = [f"- {f.id} ({f.agent}) label={f.label or 'news'} [{mark(f)}]: {f.text}" for f in valid]
    prompt = (f"Decision date: {b.date}. {context}\n" + (f"Alert status: {alert.label} ({alert.prompt}).\n" if alert else "")
              + "Validated facts:\n" + "\n".join(lines)
              + f"\nCounted facts: {n_counted}; unusual: {n_unusual}."
              + "\nAgents with no evidence: " + (", ".join(abstained) if abstained else "none"))
    ctl.current = "synthesis"
    out, meta = _call(backend, system, prompt, synthesis_schema([f.id for f in valid]), ctl)
    reasons = []
    keys = [k for k in (out.get("key_facts") or []) if k in by][:3]
    if not keys:
        reasons.append("no key facts selected")
    if out.get("overall") != truth:
        reasons.append(f"overall mismatch: model said {out.get('overall')!r}, the validated facts give {truth!r}")
    if set(out.get("missing_agents") or []) != set(abstained):
        reasons.append(f"missing-agents mismatch: model said {sorted(out.get('missing_agents') or [])}, "
                       f"actual {sorted(abstained)}")
    note = str(out.get("note", "") or "")
    nr = check_note(note, [by[k] for k in keys] or valid, b.date)
    ok = not reasons
    text = None
    if ok:
        text = (f"{alert.text.split(';')[0]}. " if alert else "") + \
               f"Taken together, the validated evidence was {truth}. Key facts: " + \
               " ".join(by[k].text for k in keys if by[k].kind != "alert") + \
               (f" Not covered: {', '.join(a + ' agent' for a in abstained)} (missing data, not 'calm')." if abstained else "")
    return {**base, "called_model": True, "validated": ok, "abstain": False, "key_facts": keys,
            "overall": out.get("overall"), "missing_agents": out.get("missing_agents"), "text": text,
            "note": "" if (nr or not ok) else note, **({"note_rejected": {"text": note, "reasons": nr}} if nr else {}),
            "rejected": ([{"reasons": reasons, "output": {k: out.get(k) for k in ("key_facts", "overall", "missing_agents")}}]
                         if reasons else []),
            "system": system, "prompt": prompt, "raw_output": out, "meta": meta}


def explain(b: Bundle, backend: LLMBackend, edge_threshold: float, extra_facts: list[Fact] | None = None,
            context: str = "", model_digest: str | None = None, ctl: Control | None = None) -> dict:
    """Run Graph, Macro and News specialists, then Synthesis, on one decision date. Returns the full trace."""
    ctl = ctl or Control()
    facts = number(bundle_facts(b, edge_threshold) + list(extra_facts or []))
    t0 = time.time()
    specialists = []
    for role in AGENTS:
        if ctl.cancelled():
            raise Stop("cancelled")
        ctl.emit(type="agent_start", agent=role, n_inputs=sum(f.agent == role for f in facts))
        s = run_specialist(role, facts, b, backend, ctl, context)
        specialists.append(s)
        ctl.emit(type="agent_done", agent=role, called_model=s["called_model"], abstain=s["abstain"],
                 n_valid=len(s["claims"]), n_rejected=len(s["rejected"]), secs=s.get("meta", {}).get("wall_secs"))
    if ctl.cancelled():
        raise Stop("cancelled")
    ctl.emit(type="agent_start", agent="synthesis", n_inputs=sum(len(s["claims"]) for s in specialists))
    syn = run_synthesis(b, facts, specialists, backend, ctl, context)
    ctl.emit(type="agent_done", agent="synthesis", called_model=syn["called_model"], abstain=syn["abstain"],
             validated=syn.get("validated"), secs=syn.get("meta", {}).get("wall_secs"))
    return {"alert_id": f"{b.method}@{b.date}", "run_id": b.run_id, "decision_date": b.date, "cutoff_utc": b.cutoff_utc,
            "method": b.method, "backend": backend.name, "model_digest": model_digest, "prompt_version": PROMPT_VERSION,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "secs": round(time.time() - t0, 1),
            "facts": [f.to_json() for f in facts], "coverage_notes": b.notes, "specialists": specialists,
            "synthesis": syn,
            "disclaimer": ("Agent output explains saved historical evidence. Shown statements are rendered from the "
                           "facts; model notes appear only if they passed validation. Nothing here changes the risk "
                           "score or alert status. Graph links are estimated directed dependence, not causality.")}


def summarise(rec: dict) -> dict:
    """Counts used by the evaluation tables."""
    sp = rec["specialists"]
    asserted = [c for s in sp for c in s["claims"]] + [c for s in sp for c in s["rejected"]]
    mism = [c for s in sp for c in s["rejected"] if any(r.startswith(("label", "citation")) for r in c["reasons"])]
    notes = [c for s in sp for c in s["claims"] if c.get("note")]
    nrej = [c for s in sp for c in s["claims"] if c.get("note_rejected")]
    syn = rec["synthesis"]
    return {"calls": sum(s["called_model"] for s in sp) + int(syn["called_model"]), "asserted": len(asserted),
            "label_correct": len(asserted) - len(mism), "valid_claims": sum(len(s["claims"]) for s in sp),
            "notes_kept": len(notes), "notes_rejected": len(nrej) + int(bool(syn.get("note_rejected"))),
            "synthesis_called": syn["called_model"], "synthesis_valid": bool(syn.get("validated")),
            "abstained": [s["role"] for s in sp if s["abstain"]],
            "model_abstained": [s["role"] for s in sp if s.get("model_abstained")]}
