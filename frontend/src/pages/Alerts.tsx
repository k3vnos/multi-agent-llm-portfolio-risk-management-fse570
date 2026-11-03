import { useMemo, useState } from "react";
import { api, type Agents, type Episode } from "../api";
import { useData } from "../components/Shell";
import { Button, Card, ErrorState, Info, LevelBadge, Loading, Pill, Unavailable } from "../components/ui";
import { fmt, longDate, methodName, parseNews, nice } from "../format";
import { G } from "../glossary";
import { useAsync } from "../hooks";
import { useRoute } from "../state";

const METHODS = ["bayes_logit:combined", "vix", "inst_composite", "bayes_logit:controls", "dcc_var"];
const OUTCOME: Record<string, { text: string; tone: "crit" | "muted" | "warn" }> = {
  true_alert: { text: "Stress followed within 10 days", tone: "crit" },
  false_alarm: { text: "No stress followed (false alarm)", tone: "muted" },
  in_stress: { text: "Issued during an existing stress period", tone: "warn" },
};

export default function Alerts() {
  const { meta } = useData();
  const { alert, go } = useRoute();
  const [method, setMethod] = useState(alert?.split("@")[0] || meta.monitored_model);
  const runs = useAsync(() => api.runs(), []);
  const primary = runs.data?.runs.find((r) => r.role === "primary")?.run_id;
  const eps = useAsync(() => (primary ? api.episodes(primary, method) : Promise.resolve(null)), [primary, method]);
  const list = useMemo(() => (eps.data ? [...eps.data.data.episodes].reverse() : []), [eps.data]);
  const selected = alert && alert.startsWith(method) ? alert.split("@")[1] : list[0]?.start;

  return (
    <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
      <Card title="Historical alerts" info="Every alert episode issued in the locked out-of-sample evaluation (2014–2026). Consecutive alert days within 5 sessions are merged into one episode.">
        <label className="mb-3 block text-xs text-muted">Method
          <select value={method} onChange={(e) => { setMethod(e.target.value); go("alerts", { alert: null }); }} aria-label="Alert method"
            className="mt-1 block w-full rounded-md border border-line bg-panel2 px-2 py-1.5 text-sm text-ink [color-scheme:dark]">
            {METHODS.map((m) => <option key={m} value={m}>{methodName(m)}</option>)}
          </select>
        </label>
        {eps.error ? <ErrorState message={eps.error} /> : !eps.data ? <Loading /> : (
          <ul className="max-h-[720px] space-y-1.5 overflow-y-auto pr-1" data-testid="alert-list">
            {list.map((e) => (
              <li key={e.start} ref={selected === e.start ? (el) => el?.scrollIntoView({ block: "nearest" }) : undefined}>
                <button type="button" onClick={() => go("alerts", { date: e.start, alert: `${method}@${e.start}` })}
                  className={`w-full rounded-lg border px-3 py-2 text-left text-xs ${selected === e.start ? "border-accent bg-accent/15" : "border-line bg-panel2 hover:border-dim"}`}>
                  <div className="flex items-center justify-between"><span className="font-mono text-ink">{longDate(e.start)}</span><span className="text-muted">peak {fmt(e.peak_score, 3)}</span></div>
                  <div className="mt-0.5 text-muted">{e.start === e.end ? "single day" : `until ${longDate(e.end)}`}</div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>
      {selected ? <AlertDetail method={method} date={selected} episode={list.find((e) => e.start === selected)} /> : <Card title="Alert detail"><Unavailable>Select an alert.</Unavailable></Card>}
    </div>
  );
}

function AlertDetail({ method, date, episode }: { method: string; date: string; episode?: Episode }) {
  const { go } = useRoute();
  const ag = useAsync(() => api.agents(date, method), [date, method]);
  const snap = useAsync(() => api.snapshot(date), [date]);
  const net = useAsync(() => api.network(date), [date]);
  if (ag.error) return <ErrorState message={ag.error} />;
  if (!ag.data || !snap.data || !net.data) return <Card title="Alert detail"><Loading /></Card>;
  const a = ag.data;
  const alertItem = a.agents.macro.evidence.find((i) => i.kind === "alert");
  return (
    <div className="space-y-4" data-testid="alert-detail">
      <Card title={`Alert · ${longDate(date)}`} subtitle={methodName(method)}
        right={<div className="flex flex-wrap gap-1.5"><Button onClick={() => go("timeline", { date })}>Timeline →</Button><Button onClick={() => go("network", { date })}>Network →</Button><Button onClick={() => go("agents", { date })}>AI analysis →</Button></div>}>
        <div className="grid gap-4 md:grid-cols-3">
          <Stat label="Alert score" value={fmt(alertItem?.values.score, 3)} note={`threshold ${fmt(alertItem?.values.threshold, 3)} (chosen on earlier validation data)`} />
          <div><div className="text-[10.5px] uppercase tracking-[0.12em] text-muted">Conditions that day</div><div className="mt-1.5"><LevelBadge level={snap.data.status.level} /></div>
            <div className="mt-1 text-xs text-muted">VIX {fmt(snap.data.vix.value, 1)} · NFCI {fmt(snap.data.nfci.value, 2)} · regime {snap.data.regime.state ?? "—"}</div></div>
          <Stat label="Structural changes" value={`${net.data.counts.appeared + net.data.counts.disappeared}`} note={`links appeared/disappeared vs previous week (${net.data.counts["sign flipped"]} sign flips)`} />
        </div>
        <p className="mt-3 text-xs leading-relaxed text-muted">{alertItem?.description}</p>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <EvidenceCard title="Macroeconomic & market evidence" items={a.agents.macro.evidence.filter((i) => i.kind === "macro")} />
        <EvidenceCard title="Structural (graph) evidence" items={a.agents.graph.evidence} />
      </div>
      <NewsCard a={a} />
      <Card title="Retrospective outcome" info={G.alertOutcome} className="border-warn/30">
        <div className="mb-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">Known only with hindsight — not available when the alert was issued.</div>
        {episode ? <div className="flex flex-wrap items-center gap-2 text-sm"><Pill tone={OUTCOME[episode.outcome_retrospective]?.tone ?? "muted"}>{OUTCOME[episode.outcome_retrospective]?.text ?? episode.outcome_retrospective}</Pill>
          <span className="text-xs text-muted">Judged with the full-sample stress labels used in the evaluation.</span></div> : <Unavailable>Outcome unavailable.</Unavailable>}
      </Card>
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return <div><div className="text-[10.5px] uppercase tracking-[0.12em] text-muted">{label}</div><div className="mt-1 font-mono text-2xl text-ink">{value}</div><div className="mt-0.5 text-xs text-muted">{note}</div></div>;
}

export function EvidenceCard({ title, items }: { title: string; items: Agents["agents"]["macro"]["evidence"] }) {
  return (
    <Card title={title}>
      {items.length === 0 ? <Unavailable>No evidence of this kind for this date.</Unavailable> : (
        <ul className="space-y-2.5 text-[12.5px] leading-relaxed">
          {items.map((i) => (
            <li key={i.id}><span className="text-ink/90">{nice(i.description)}</span>
              <div className="mt-0.5 font-mono text-[10.5px] text-dim">{i.id} · available {i.available.slice(0, 16)}</div></li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function NewsCard({ a, title = "News evidence" }: { a: Agents; title?: string }) {
  const n = a.agents.news;
  return (
    <Card title={title} subtitle="published before 4pm New York (FNSPID)"
      info="Retrieved from the public FNSPID dataset, which ends on 11 June 2020. Only articles published before the market close of the decision date (7-day lookback) can appear. No headline is generated.">
      {n.evidence.length === 0 ? (
        <div className="rounded-md border border-dashed border-warn/50 p-3 text-sm text-warn" data-testid="news-unavailable">
          News evidence unavailable for this date. {n.unavailable[0]?.replace(/^news: /, "") ?? ""}
        </div>
      ) : (
        <ul className="space-y-2 text-[12.5px]">
          {n.evidence.map((i) => {
            const p = parseNews(i.source_text);
            return (
              <li key={i.id} className="rounded-md border border-line bg-panel2 px-3 py-2"><div className="text-ink/90">{p.title}</div>
                <div className="mt-0.5 font-mono text-[10.5px] text-dim">{p.time} · {p.publisher} · {p.ticker} · {i.id}</div></li>
            );
          })}
        </ul>
      )}
      <div className="mt-2 flex items-center gap-1 text-[11px] text-dim"><Info text="Ranking is a transparent keyword score for market-stress terms, newest first among ties." /> Up to 8 most stress-relevant headlines.</div>
    </Card>
  );
}
