"""News, macro and graph specialist agents + a synthesis agent.

Each specialist sees only its own slice of the point-in-time bundle and must cite evidence ids. A specialist
with no evidence abstains without calling the model. The synthesis agent sees only the specialists' *validated*
claims (with the evidence they cite) and the coverage notes, never the raw data, labels or evaluation results.
Every agent output passes through `validator.validate_output`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from riskcausal_agents.backends import LLMBackend
from riskcausal_agents.evidence import Bundle, Evidence
from riskcausal_agents.validator import allowed_for, validate_output

MIN_CLAIMS_FOR_SYNTHESIS = 2
PROMPT_VERSION = "m6-v2"

RULES = """Rules (all mandatory):
- Use ONLY the evidence items listed below. Do not use any other knowledge, including what happened later.
- Every claim must cite one or more evidence ids from the list, copied exactly, e.g. "macro:VIXCLS@2020-02-21"
  (no brackets, no shortening).
- Copy numbers exactly as they appear in the cited evidence. Do not compute new numbers.
- Do not name any event, company, person or institution that is not written in the cited evidence.
- Describe, do not explain causes: never say something caused, drove, led to or was due to something else.
- Graph items are ESTIMATED DIRECTED DEPENDENCE from a statistical model, not causal effects.
- Never say the model or any indicator predicts, outperforms or reliably warns. Never forecast.
- If the evidence is too thin to say anything useful, set "abstain": true and explain why in "abstain_reason".
Return JSON: {"claims": [{"text": "<= 40 words", "evidence_ids": ["..."]}], "uncertainty": "<= 40 words on
what the evidence does NOT show", "abstain": false, "abstain_reason": ""}"""

ROLES = {
    "news": ("news", {"news"}, ("You summarise what financial news published BEFORE the decision cutoff said. "
             "Report themes in the headlines; do not judge whether they mattered.")),
    "macro": ("macro", {"macro", "alert"}, ("You describe the market and macro-financial indicators exactly as "
              "they were known at the decision date, including how unusual they were (percentiles).")),
    "graph": ("graph", {"graph"}, ("You describe how the estimated directed-dependence graph among assets changed. "
              "Stress that these are noisy statistical estimates, not causal relationships.")),
}


def _evidence_block(items: list[Evidence]) -> str:
    return "\n".join(f"- id: {e.id}\n  fact: {e.text}" for e in items)


def run_specialist(role: str, bundle: Bundle, backend: LLMBackend) -> dict:
    kind_label, kinds, brief = ROLES[role]
    allowed = allowed_for(bundle, kinds)
    if not any(e.kind != "alert" for e in allowed.values()):
        notes = [n for n in bundle.notes if n.startswith(role)] or [f"{role}: no {role} evidence available"]
        return {"role": role, "claims": [], "rejected": [], "abstain": True, "abstain_reason": "; ".join(notes),
                "uncertainty": "", "called_model": False}
    system = f"You are the {kind_label} analyst in a risk-alert explanation team. {brief}\n{RULES}"
    prompt = (f"Decision date: {bundle.date} (alert issued at the market close; cutoff {bundle.cutoff_utc}).\n"
              f"Alerting method: {bundle.method}.\nEvidence:\n{_evidence_block(list(allowed.values()))}")
    out, meta = backend.complete_json(system, prompt)
    v = validate_output(out, allowed, bundle.date)
    return {"role": role, **v, "called_model": True, "system": system, "prompt": prompt, "meta": meta}


def run_synthesis(bundle: Bundle, specialists: list[dict], backend: LLMBackend) -> dict:
    valid = [c for s in specialists for c in s["claims"]]
    if len(valid) < MIN_CLAIMS_FOR_SYNTHESIS:
        return {"role": "synthesis", "claims": [], "rejected": [], "abstain": True, "called_model": False,
                "abstain_reason": f"insufficient validated evidence ({len(valid)} claim(s); "
                                  f"need {MIN_CLAIMS_FOR_SYNTHESIS})", "uncertainty": ""}
    cited = {i for c in valid for i in c["evidence_ids"]} | {i for i, e in bundle.items.items() if e.kind == "alert"}
    allowed = {i: bundle.items[i] for i in cited}
    findings = "\n".join(f"- ({s['role']}) {c['text']} {c['evidence_ids']}" for s in specialists for c in s["claims"])
    abst = "\n".join(f"- ({s['role']}) abstained: {s['abstain_reason']}" for s in specialists if s["abstain"])
    system = ("You are the synthesis analyst. Combine the specialists' validated findings into a short, neutral "
              "explanation of what the evidence showed when the alert was issued. Mention which evidence is "
              f"missing. 2 to 4 claims.\n{RULES}")
    prompt = (f"Decision date: {bundle.date}. Alerting method: {bundle.method}.\nValidated findings:\n{findings}\n"
              f"{abst}\nCoverage notes:\n" + "\n".join(f"- {n}" for n in bundle.notes) +
              f"\nEvidence you may cite:\n{_evidence_block(list(allowed.values()))}")
    out, meta = backend.complete_json(system, prompt)
    v = validate_output(out, allowed, bundle.date)
    if not v["claims"]:
        v["abstain"] = True
        v["abstain_reason"] = v["abstain_reason"] or "no synthesis claim survived validation"
    return {"role": "synthesis", **v, "called_model": True, "system": system, "prompt": prompt, "meta": meta}


def explain(bundle: Bundle, backend: LLMBackend, model_digest: str | None = None) -> dict:
    specialists = [run_specialist(r, bundle, backend) for r in ("news", "macro", "graph")]
    synthesis = run_synthesis(bundle, specialists, backend)
    return {
        "alert_id": f"{bundle.method}@{bundle.date}", "run_id": bundle.run_id, "decision_date": bundle.date,
        "cutoff_utc": bundle.cutoff_utc, "backend": backend.name, "model_digest": model_digest,
        "prompt_version": PROMPT_VERSION, "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "explanation": {"claims": synthesis["claims"], "uncertainty": synthesis.get("uncertainty", ""),
                        "abstain": synthesis["abstain"], "abstain_reason": synthesis.get("abstain_reason", "")},
        "coverage_notes": bundle.notes,
        "method_reliability_retrospective": bundle.reliability,
        "disclaimer": ("Illustrative explanation of saved evidence. Graph items are estimated directed dependence, "
                       "not causality. Explanations never influence predictions or metrics."),
        "specialists": specialists, "synthesis": synthesis, "bundle": bundle.to_json(),
    }


def save(record: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / (record["alert_id"].replace(":", "_").replace("@", "__") + ".json")
    p.write_text(json.dumps(record, indent=2, default=str))
    return p
