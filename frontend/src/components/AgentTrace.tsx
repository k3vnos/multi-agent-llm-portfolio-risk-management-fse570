import { useState, type ReactNode } from "react";
import type { Fact, V4Claim, V4Record, V4Specialist, V4Synthesis } from "../api";
import { nice } from "../format";

// Visual trace of one v4 multi-agent run: what each specialist was given, what it asserted, what the validator
// confirmed or rejected (and why), who abstained, and the synthesis. Every confirmed statement shown is the fact's
// rule-rendered sentence; a model-written note appears only if it passed validation and is labelled as such.

export const AGENT_COLOR: Record<string, string> = { graph: "#9085e9", macro: "#d95926", news: "#199e70", synthesis: "#e6e8eb" };
export const AGENT_NAME: Record<string, string> = { graph: "Graph agent", macro: "Macro agent", news: "News agent", synthesis: "Synthesis agent" };
const AGENT_ROLE: Record<string, string> = {
  graph: "estimated relationships between assets", macro: "market & economic indicators, alert status",
  news: "headlines before the 4pm New York close", synthesis: "combines only the confirmed facts",
};
const UP = new Set(["high", "rose sharply", "rose", "appeared", "strengthened", "above alert line", "largest contributor", "above its weight"]);
const DOWN = new Set(["low", "fell", "fell sharply", "disappeared", "weakened", "below alert line", "below its weight"]);

export function LabelChip({ label }: { label: string }) {
  const c = UP.has(label) ? "border-warn/50 text-warn" : DOWN.has(label) ? "border-neg/50 text-[#8fbaf2]" : "border-line text-muted";
  return <span className={`inline-flex shrink-0 items-center rounded border px-1.5 py-px text-[11px] font-medium ${c}`}>{label}</span>;
}

function plainReason(r: string): string {
  return r.replace(/^label mismatch: model said '([^']+)', the numbers say '([^']+)'$/, "the model said “$1”, but the numbers say “$2”")
    .replace(/^note /, "note — ");
}

export function AgentTrace({ rec, compact = false }: { rec: V4Record; compact?: boolean }) {
  const facts = Object.fromEntries(rec.facts.map((f) => [f.id, f]));
  const portfolio = rec.facts.filter((f) => f.source.startsWith("portfolio:")).length;
  return (
    <div className="space-y-3" data-testid="agent-trace">
      <FlowStrip rec={rec} portfolio={portfolio} compact={compact} />
      <div className="grid gap-3 lg:grid-cols-3">
        {rec.specialists.map((s) => <SpecialistCard key={s.role} s={s} compact={compact} />)}
      </div>
      <SynthesisCard syn={rec.synthesis} facts={facts} compact={compact} />
      <p className={`${compact ? "text-[14px]" : "text-[11.5px]"} text-dim`}>{rec.disclaimer}</p>
    </div>
  );
}

function FlowStrip({ rec, portfolio, compact }: { rec: V4Record; portfolio: number; compact: boolean }) {
  const s = (role: string) => rec.specialists.find((x) => x.role === role)!;
  const step = (title: ReactNode, sub: ReactNode, color: string, testid?: string) => (
    <div className="min-w-0 flex-1 rounded-lg border bg-panel2 px-3 py-2" style={{ borderColor: color }} data-testid={testid}>
      <div className={`${compact ? "text-[15px]" : "text-[13px]"} font-semibold text-ink`}>{title}</div>
      <div className={`${compact ? "text-[13px]" : "text-[11px]"} text-muted`}>{sub}</div>
    </div>
  );
  const arrow = <div className="hidden items-center px-1 text-xl text-dim md:flex" aria-hidden>→</div>;
  const nConfirmed = rec.specialists.reduce((a, x) => a + x.claims.length, 0);
  const nRejected = rec.specialists.reduce((a, x) => a + x.rejected.length, 0);
  return (
    <div className="flex flex-col gap-2 md:flex-row md:items-stretch" data-testid="agent-flow">
      {step("1 · Evidence", `${rec.facts.length} facts known at the close${portfolio ? `, incl. ${portfolio} about your portfolio` : ""}`, "#3a4352")}
      {arrow}
      <div className="flex min-w-0 flex-[2] flex-col gap-1.5">
        {(["graph", "macro", "news"] as const).map((r) => {
          const x = s(r);
          return (
            <div key={r} className="flex items-center gap-2 rounded-md border border-line bg-panel2 px-2 py-1 text-[12.5px]" data-testid={`flow-${r}`}>
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: AGENT_COLOR[r] }} />
              <span className="font-medium text-ink">{AGENT_NAME[r]}</span>
              <span className="truncate text-muted">{x.abstain ? (x.called_model ? "ran, nothing confirmed" : "abstained — no data") : `ran · ${x.claims.length} confirmed`}{x.rejected.length ? ` · ${x.rejected.length} rejected` : ""}</span>
            </div>
          );
        })}
      </div>
      {arrow}
      {step("2 · Validator", `${nConfirmed} confirmed · ${nRejected} rejected`, "#5d6673")}
      {arrow}
      {step("3 · Synthesis", rec.synthesis.validated ? "model synthesis confirmed" : rec.synthesis.called_model ? "model synthesis rejected → rule-based summary" : "not enough confirmed facts", "#e6e8eb")}
    </div>
  );
}

function SpecialistCard({ s, compact }: { s: V4Specialist; compact: boolean }) {
  const [showInputs, setShowInputs] = useState(false);
  const color = AGENT_COLOR[s.role];
  const t = compact ? "text-[15px]" : "text-[12.5px]";
  return (
    <section className="rounded-xl border bg-panel p-3" style={{ borderColor: `${color}88` }} data-testid={`trace-${s.role}`}>
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full" style={{ background: color }} />
        <span className={`${compact ? "text-[17px]" : "text-[14px]"} font-semibold text-ink`}>{AGENT_NAME[s.role]}</span>
        {s.called_model && s.meta?.wall_secs !== undefined && <span className="ml-auto font-mono text-[11px] text-dim">{s.meta.wall_secs}s · {s.meta.eval_count ?? "?"} tokens</span>}
      </div>
      <div className={`${compact ? "text-[13px]" : "text-[11px]"} text-muted`}>{AGENT_ROLE[s.role]} · given {s.inputs.length} fact{s.inputs.length === 1 ? "" : "s"}</div>
      {s.abstain && (
        <div className={`mt-2 rounded-md border border-warn/40 bg-warn/10 p-2 ${t} text-warn`} data-testid={`abstain-${s.role}`}>
          <b>Abstained.</b> {s.called_model ? (s.model_abstained ? `The model declined: ${s.abstain_reason || "no reason given"}` : "None of its statements passed the validator.") : nice(s.abstain_reason.replace(/^news: /, ""))}
        </div>
      )}
      {s.claims.length > 0 && (
        <ul className={`mt-2 space-y-2 ${t}`}>
          {s.claims.map((c) => <Confirmed key={c.fact_id} c={c} compact={compact} />)}
        </ul>
      )}
      {s.rejected.length > 0 && (
        <details className="mt-2 rounded-md border border-crit/30 bg-crit/5 p-2" open={!compact}>
          <summary className={`cursor-pointer ${compact ? "text-[14px]" : "text-[12px]"} font-medium text-[#ff9c9c]`}>✗ {s.rejected.length} statement{s.rejected.length === 1 ? "" : "s"} rejected by the validator (never shown as fact)</summary>
          <ul className={`mt-1 space-y-1 ${compact ? "text-[13px]" : "text-[11.5px]"} text-muted`} data-testid={`rejected-${s.role}`}>
            {s.rejected.map((r, i) => <li key={i}><span className="font-mono text-dim">{r.fact_id}</span> “{r.label}” — {(r.reasons || []).map(plainReason).join("; ")}</li>)}
          </ul>
        </details>
      )}
      {!compact && s.inputs.length > 0 && (
        <div className="mt-2">
          <button type="button" className="text-[11.5px] text-accent hover:underline" onClick={() => setShowInputs((x) => !x)} aria-expanded={showInputs}>
            {showInputs ? "Hide" : "Show"} the {s.inputs.length} inputs the model saw (numbers only, no labels)</button>
          {showInputs && <ul className="mt-1 space-y-0.5 font-mono text-[10.5px] text-dim">{s.inputs.map((f: Fact) => <li key={f.id}>{f.id} [{f.kind}] {nice(f.prompt)}</li>)}</ul>}
        </div>
      )}
    </section>
  );
}

function Confirmed({ c, compact }: { c: V4Claim; compact: boolean }) {
  return (
    <li data-testid="confirmed-claim">
      <div className="flex items-start gap-2"><span className="mt-0.5 text-good" aria-label="confirmed">✓</span>
        <div className="min-w-0"><LabelChip label={c.label} /> <span className="text-ink/90">{nice(c.text || "")}</span>
          {c.note && <div className={`mt-0.5 ${compact ? "text-[13px]" : "text-[11.5px]"} italic text-muted`}>Model note (passed checks): “{c.note}”</div>}
          {!compact && c.note_rejected && <div className="mt-0.5 text-[11px] text-dim">Model note removed — {c.note_rejected.reasons.map(plainReason).join("; ")}</div>}
          {!compact && <div className="font-mono text-[10px] text-dim">{c.fact_id} · {c.source}</div>}
        </div>
      </div>
    </li>
  );
}

function SynthesisCard({ syn, facts, compact }: { syn: V4Synthesis; facts: Record<string, Fact>; compact: boolean }) {
  const ok = syn.validated && syn.text;
  return (
    <section className="rounded-xl border border-ink/30 bg-panel2 p-4" data-testid="trace-synthesis">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`${compact ? "text-[17px]" : "text-[14px]"} font-semibold text-ink`}>Synthesis agent</span>
        {ok ? <span className="rounded border border-good/50 px-1.5 py-px text-[11px] text-[#6ee06e]">✓ confirmed by the validator</span>
          : syn.called_model ? <span className="rounded border border-crit/50 px-1.5 py-px text-[11px] text-[#ff9c9c]">✗ model synthesis rejected — showing the rule-based summary</span>
            : <span className="rounded border border-warn/50 px-1.5 py-px text-[11px] text-warn">not run — {syn.abstain_reason}</span>}
        {syn.meta?.wall_secs !== undefined && <span className="ml-auto font-mono text-[11px] text-dim">{syn.meta.wall_secs}s</span>}
      </div>
      <p className={`mt-2 ${compact ? "text-[17px]" : "text-[13.5px]"} leading-relaxed text-ink`} data-testid="synthesis-text">{nice(ok ? syn.text! : syn.fallback_text)}</p>
      {ok && syn.note && <p className={`mt-1 ${compact ? "text-[14px]" : "text-[12px]"} italic text-muted`}>Model note (passed checks): “{syn.note}”</p>}
      {!ok && syn.rejected.length > 0 && <p className={`mt-1 ${compact ? "text-[13px]" : "text-[11.5px]"} text-muted`}>Why rejected: {syn.rejected[0].reasons.join("; ")}</p>}
      {!compact && ok && syn.key_facts && <div className="mt-1 font-mono text-[10px] text-dim">key facts: {syn.key_facts.map((k) => `${k} (${facts[k]?.source})`).join(", ")}</div>}
    </section>
  );
}
