import { useState } from "react";
import { api, type Agents as AgentsT, type EvidenceItem } from "../api";
import { ReplayBar, useData } from "../components/Shell";
import { Button, Card, ErrorState, Info, Loading, Pill, Unavailable } from "../components/ui";
import { longDate, methodName, nice } from "../format";
import { useAsync } from "../hooks";
import { useRoute } from "../state";
import { NewsCard } from "./Alerts";

export default function Agents() {
  const { date } = useRoute();
  const { meta } = useData();
  const a = useAsync(() => api.agents(date, meta.monitored_model), [date]);
  return (
    <div className="space-y-4">
      <ReplayBar />
      <Pipeline />
      {a.error ? <ErrorState message={a.error} /> : !a.data ? <Loading label="Gathering point-in-time evidence…" /> : <AgentViews a={a.data} />}
    </div>
  );
}

function Pipeline() {
  const box = (t: string, s: string, c: string) => (
    <div className="min-w-[150px] flex-1 rounded-lg border bg-panel2 px-3 py-2" style={{ borderColor: c }}><div className="text-[13px] font-semibold text-ink">{t}</div><div className="text-[11px] text-muted">{s}</div></div>
  );
  return (
    <Card title="Multi-agent evidence team" info="Each specialist agent only sees its own slice of the evidence that was available at the 4pm New York close of the replay date. The synthesis agent combines their validated findings. The reliable default shown here is deterministic; LLM-written drafts are shown separately as unverified.">
      <div className="flex flex-wrap items-stretch gap-2 text-center md:flex-nowrap">
        <div className="flex flex-1 flex-col gap-2">{box("Graph agent", "estimated relationship changes", "#9085e9")}{box("Macro agent", "market & economic indicators", "#d95926")}{box("News agent", "headlines before the cutoff", "#199e70")}</div>
        <div className="flex items-center px-2 text-2xl text-dim" aria-hidden>→</div>
        <div className="flex flex-1 items-center">{box("Synthesis agent", "evidence briefing with citations, uncertainty and abstention", "#e6e8eb")}</div>
        <div className="flex items-center px-2 text-2xl text-dim" aria-hidden>→</div>
        <div className="flex flex-1 items-center">{box("Validator", "checks ids, numbers, names, dates, forbidden claims", "#5d6673")}</div>
      </div>
    </Card>
  );
}

function AgentViews({ a }: { a: AgentsT }) {
  return (
    <>
      <div className="grid gap-4 xl:grid-cols-3">
        <AgentCard title="Graph agent" role={a.agents.graph.role} color="#9085e9" items={a.agents.graph.evidence} />
        <AgentCard title="Macro agent" role={a.agents.macro.role} color="#d95926" items={a.agents.macro.evidence} />
        <NewsAgentCard a={a} />
      </div>
      <Card title="Synthesis agent · evidence briefing" subtitle={`as of the close of ${longDate(a.date)} (cutoff ${a.cutoff_utc.slice(0, 16)} UTC)`}
        right={<Pill tone="good">deterministic · reliable default</Pill>}>
        <ol className="space-y-2.5 text-[13px] leading-relaxed" data-testid="synthesis">
          {a.synthesis.briefing.map((b, i) => (
            <li key={i} className="flex gap-3"><span className="mt-0.5 font-mono text-xs text-dim">{i + 1}</span>
              <div><span className="text-ink/90">{nice(b.text)}</span>
                {b.evidence_ids.length > 0 && <div className="mt-0.5 flex flex-wrap gap-1">{b.evidence_ids.slice(0, 6).map((id) => <span key={id} className="font-mono text-[10px] text-dim">[{id}]</span>)}</div>}</div></li>
          ))}
        </ol>
      </Card>
      <LlmDraft a={a} />
      <Card title="Method reliability" subtitle="retrospective — NOT known on the replay date">
        <ul className="list-disc space-y-1 pl-5 text-xs text-muted">{a.reliability_retrospective.map((r) => <li key={r}>{r.replace(/^\[retrospective[^\]]*\] /, "")}</li>)}</ul>
      </Card>
    </>
  );
}

function AgentCard({ title, role, color, items }: { title: string; role: string; color: string; items: EvidenceItem[] }) {
  return (
    <Card title={<span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />{title}</span>} subtitle={role}>
      {items.length === 0 ? <Unavailable>Abstained: no evidence of this kind was available.</Unavailable> : (
        <ul className="space-y-2.5 text-[12.5px] leading-relaxed">
          {items.map((i) => (
            <li key={i.id}><div className="text-[11px] font-medium text-muted">{nice(i.label)}</div><div className="text-ink/90">{nice(i.description)}</div>
              <div className="font-mono text-[10px] text-dim">{i.id} · available {i.available.slice(0, 16)}</div></li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function NewsAgentCard({ a }: { a: AgentsT }) {
  return (
    <div>
      {a.agents.news.abstained && <div className="mb-2 rounded-lg border border-warn/40 bg-warn/10 p-3 text-xs text-warn"><b>News agent abstained.</b> Missing news is not treated as neutral sentiment.</div>}
      <NewsCard a={a} title="News agent" />
    </div>
  );
}

function LlmDraft({ a }: { a: AgentsT }) {
  const { setDate } = useRoute();
  const [open, setOpen] = useState(false);
  const d = a.llm_draft;
  return (
    <Card title="LLM-written draft" subtitle="local qwen2.5:7b via Ollama · generated offline for 24 evaluated alerts"
      info="In the project's held-out hand check, 73% of LLM claims were faithful to their evidence (below the 90% target). Drafts are therefore never shown as conclusions; automatic validation catches citation, number, name, date and forbidden-language errors but not all misreadings.">
      {!d ? (
        <div className="space-y-2 text-sm text-muted">
          <p>No LLM draft exists for {methodName(a.method)} on {longDate(a.date)}. The deterministic briefing above is the explanation.</p>
          <div className="flex flex-wrap items-center gap-1.5 text-xs"><span>Dates with evaluated drafts:</span>
            {a.llm_draft_available_for.map((x) => <Button key={x} onClick={() => setDate(x)}>{x}</Button>)}</div>
        </div>
      ) : (
        <div data-testid="llm-draft">
          <div className={`mb-3 rounded-md border px-3 py-2 text-sm ${d.flags.length ? "border-crit/50 bg-crit/10 text-[#ff9c9c]" : "border-warn/50 bg-warn/10 text-warn"}`}>
            <b>{d.label}</b><div className="mt-1 text-xs">Status: {d.status} · evaluation set: {d.evaluation_set}</div>
            {d.flags.map((f) => <div key={f} className="mt-0.5 text-xs">⚠ {f}</div>)}
          </div>
          <div className="mb-3 grid gap-2 md:grid-cols-3">
            {Object.entries(d.specialists).map(([role, s]) => (
              <div key={role} className="rounded-md border border-line bg-panel2 p-2 text-xs"><div className="font-medium capitalize text-ink">{role} agent (LLM)</div>
                <div className="text-muted">{s.abstain ? `abstained — ${s.abstain_reason.slice(0, 110)}` : `${s.claims.length} validated claim(s), ${s.n_rejected} rejected by the validator`}</div></div>
            ))}
          </div>
          <Button onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"} unverified draft text</Button>
          {open && (
            <div className="mt-3 space-y-2 rounded-md border border-dashed border-line p-3 text-[13px]">
              {d.claims.length === 0 ? <p className="text-muted">Suppressed: no claim passed validation.</p> : d.claims.map((c, i) => (
                <p key={i} className="text-ink/80">“{c.text}” <span className="font-mono text-[10px] text-dim">{c.evidence_ids.join(", ")}</span></p>
              ))}
              {d.uncertainty && <p className="text-xs text-muted">Draft uncertainty: {d.uncertainty}</p>}
            </div>
          )}
          <div className="mt-2 flex items-center gap-1 text-[11px] text-dim"><Info text="Hand check by the assistant (first pass), pending team review." /> Hand-check findings are attached to every draft.</div>
        </div>
      )}
    </Card>
  );
}
