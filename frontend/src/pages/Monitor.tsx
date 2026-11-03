import { useState } from "react";
import { api, type Network, type Snapshot } from "../api";
import { Legend, Spark, StressChart, type Range } from "../components/charts";
import { NetworkGraph, NetworkLegend, material, type Mode, type Selection } from "../components/NetworkGraph";
import { ReplayBar, useData } from "../components/Shell";
import { Button, Card, ErrorState, Info, Label, LevelBadge, Loading, Pill } from "../components/ui";
import { band, fmt, longDate, methodName, ordinal, signed, nice } from "../format";
import { G } from "../glossary";
import { useAsync } from "../hooks";
import { useRoute } from "../state";

export default function Monitor() {
  const { date } = useRoute();
  const snap = useAsync(() => api.snapshot(date), [date]);
  return (
    <div className="space-y-4">
      <HowToRead />
      {snap.error ? <ErrorState message={snap.error} /> : snap.data ? <Kpis s={snap.data} /> : <Loading />}
      <MainChart />
      <div className="grid gap-4 lg:grid-cols-[1.25fr_1fr]">
        <MiniNetwork />
        {snap.data ? <RecentAlerts s={snap.data} /> : <Card title="Alerts"><Loading /></Card>}
      </div>
      <WhatMoved />
      <BriefingTeaser />
    </div>
  );
}

function HowToRead() {
  const [open, setOpen] = useState(true);
  return (
    <section className="rounded-xl border border-line border-l-accent bg-panel" style={{ borderLeftWidth: 3 }}>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-4 py-3 text-left text-[13px] font-semibold text-ink" aria-expanded={open}>
        <span className="text-muted">{open ? "▾" : "▸"}</span> How to read this dashboard
      </button>
      {open && (
        <div className="grid gap-5 px-4 pb-4 text-[12.5px] leading-relaxed text-muted md:grid-cols-4">
          <p><b className="text-ink">The idea.</b> Normal risk models measure how assets move <i>together</i>. This system
            estimates which asset moves <i>first</i> — a web of directed relationships re-estimated every week — and tracks how
            much that web changes.</p>
          <p><b className="text-ink">The number.</b> The Structural Instability Index (SII) summarises how much those
            relationships changed recently. High means the market's internal wiring is rearranging itself.</p>
          <p><b className="text-ink">The claim tested.</b> "Instability rises <i>before</i> volatility does." The locked
            evaluation (2014–2026) did <b className="text-ink">not</b> support it: the SII did not warn earlier or better than
            the VIX. See <i>Research &amp; validation</i>.</p>
          <p><b className="text-ink">What it is not.</b> Not a trading signal, not a forecast, not live data. You are replaying
            what was observable at past dates; arrows are statistical dependence, not proven cause and effect.</p>
        </div>
      )}
    </section>
  );
}

function Kpis({ s }: { s: Snapshot }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" data-testid="kpis">
      <div className="rounded-xl border border-line bg-panel p-4">
        <Label info={G.sii}>Structural instability (SII)</Label>
        <div className="mt-1.5 flex items-baseline gap-2">
          <span className="font-mono text-[30px] font-semibold leading-none text-sii" data-testid="kpi-sii">{fmt(s.sii.value, 2)}</span>
          <span className="text-xs text-muted">{ordinal(s.sii.percentile)} percentile</span>
        </div>
        <div className="mt-1 text-xs text-muted">{s.sii.change_21d === null ? "—" : `${signed(s.sii.change_21d)} over 21 sessions`}</div>
        <Spark values={s.sii.spark} />
      </div>
      <div className="rounded-xl border border-line bg-panel p-4">
        <Label info={G.regime}>Market regime (as estimated then)</Label>
        <div className="mt-2">
          {s.regime.state === null ? <span className="text-sm text-muted">Unavailable</span> : (
            <span className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-sm ${s.regime.state === "stressed" ? "border-crit/50 bg-crit/15 text-[#ff8a8a]" : "border-good/50 bg-good/10 text-[#5bd65b]"}`}>
              ● {s.regime.state === "stressed" ? "Stressed" : "Calm"}
            </span>
          )}
          <span className="ml-2 text-xs text-muted">{s.regime.p_stress === null ? "" : `${Math.round(s.regime.p_stress * 100)}% stress probability`}</span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <div><span className="flex items-center gap-1 text-muted">VIX <Info text={G.vix} /></span><span className="font-mono text-ink">{fmt(s.vix.value, 1)}</span> <span className="text-muted">({band(s.vix.percentile)})</span></div>
          <div><span className="flex items-center gap-1 text-muted">NFCI <Info text={G.nfci} /></span><span className="font-mono text-ink">{fmt(s.nfci.value, 2)}</span> <span className="text-muted">({band(s.nfci.percentile)})</span></div>
        </div>
      </div>
      <div className={`rounded-xl border bg-panel p-4 ${s.status.level === "alert" ? "border-crit/50" : s.status.level === "unusual" ? "border-warn/40" : "border-line"}`} data-testid="kpi-status">
        <Label info={G.status}>Risk status</Label>
        <div className="mt-2"><LevelBadge level={s.status.level} /></div>
        <p className="mt-2 text-xs leading-relaxed text-muted">{s.status.summary}</p>
        {s.status.unusual.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">{s.status.unusual.map((u) => <Pill key={u.indicator} tone="warn">{u.indicator} · {ordinal(u.percentile)}</Pill>)}</div>
        )}
      </div>
      <div className="rounded-xl border border-line bg-panel p-4">
        <Label info={G.turnover}>Relationship turnover</Label>
        <div className="mt-1.5 flex items-baseline gap-2">
          <span className="font-mono text-[30px] font-semibold leading-none text-ink">{fmt(s.edge_turnover.value, 0)}</span>
          <span className="text-xs text-muted">links changed this week</span>
        </div>
        <div className="mt-1 text-xs text-muted">{ordinal(s.edge_turnover.percentile)} percentile · 4-week avg {fmt(s.edge_turnover.avg_4_windows, 1)}</div>
        <div className="mt-3 text-xs text-muted">Model score <span className="font-mono text-ink">{fmt(s.model.score, 3)}</span> vs threshold <span className="font-mono text-ink">{fmt(s.model.threshold, 3)}</span> <Info text={G.model} /></div>
      </div>
    </div>
  );
}

function MainChart() {
  const { series } = useData();
  const { date, setDate } = useRoute();
  const [range, setRange] = useState<Range>("1Y");
  return (
    <Card title="Instability against conventional stress measures"
      subtitle="Each line = percentile of its own previous 5 years (point-in-time), so all three share one axis"
      info="Lines end at the replay date: nothing after that date is used or shown. Click the chart to move the replay date."
      right={<div className="flex gap-1">{(["6M", "1Y", "3Y", "5Y", "Max"] as Range[]).map((r) => <Button key={r} active={range === r} onClick={() => setRange(r)}>{r}</Button>)}</div>}>
      <Legend />
      <StressChart series={series} date={date} range={range} onPick={setDate} />
      <div className="mt-3"><ReplayBar /></div>
    </Card>
  );
}

function MiniNetwork() {
  const { date, go } = useRoute();
  const [mode, setMode] = useState<Mode>("change");
  const [sel, setSel] = useState<Selection>(null);
  const net = useAsync(() => api.network(date), [date]);
  return (
    <Card title="Estimated relationship structure" info={G.edges}
      subtitle={net.data ? `window ending ${longDate(net.data.window_end)} vs previous week` : undefined}
      right={<div className="flex gap-1"><Button active={mode === "change"} onClick={() => setMode("change")}>Change</Button><Button active={mode === "current"} onClick={() => setMode("current")}>Current</Button></div>}>
      {net.error ? <ErrorState message={net.error} /> : !net.data ? <Loading /> : (
        <div className="grid gap-3 md:grid-cols-[1fr_210px]">
          <div><div className="mx-auto max-w-[460px]"><NetworkGraph net={net.data} mode={mode} selection={sel} onSelect={setSel} size={440} /></div><NetworkLegend mode={mode} /></div>
          <Inspector net={net.data} sel={sel} onOpen={() => go("network")} />
        </div>
      )}
    </Card>
  );
}

export function Inspector({ net, sel, onOpen }: { net: Network; sel: Selection; onOpen?: () => void }) {
  if (!sel) {
    const c = net.counts;
    return (
      <div className="space-y-2 text-xs text-muted">
        <p>{net.n_active} estimated links active. Since the previous window: <b className="text-ink">{c.appeared}</b> appeared,
          <b className="text-ink"> {c.disappeared}</b> disappeared, <b className="text-ink">{c["sign flipped"]}</b> flipped sign.</p>
        <p>Click an asset or an arrow to inspect it.</p>
        {onOpen && <Button onClick={onOpen}>Open full network →</Button>}
      </div>
    );
  }
  if (sel.type === "node") {
    const n = net.nodes.find((x) => x.id === sel.id)!;
    const out = net.edges.filter((e) => e.source === n.id && e.active).sort((a, b) => Math.abs(b.after) - Math.abs(a.after));
    const inc = net.edges.filter((e) => e.target === n.id && e.active).sort((a, b) => Math.abs(b.after) - Math.abs(a.after));
    return (
      <div className="space-y-2 text-xs" data-testid="inspector">
        <div className="text-sm font-semibold text-ink">{nice(n.id)}</div><div className="text-muted">{n.name}</div>
        <div><span className="text-muted">Helps explain:</span> {out.length ? out.map((e) => `${nice(e.target)} ${signed(e.after)}`).join(", ") : "none"}</div>
        <div><span className="text-muted">Explained by:</span> {inc.length ? inc.map((e) => `${nice(e.source)} ${signed(e.after)}`).join(", ") : "none"}</div>
        <div className="text-muted">{n.changed} material change(s) this window.</div>
      </div>
    );
  }
  const e = net.edges.find((x) => x.id === sel.id)!;
  return (
    <div className="space-y-2 text-xs" data-testid="inspector">
      <div className="text-sm font-semibold text-ink">{nice(e.source)} → {nice(e.target)}</div>
      <Pill tone="accent">{e.change}</Pill>
      <p className="leading-relaxed text-muted">{nice(e.text)}</p>
      <div className="font-mono text-ink">{signed(e.before)} → {signed(e.after)}</div>
    </div>
  );
}

function RecentAlerts({ s }: { s: Snapshot }) {
  const { go } = useRoute();
  return (
    <Card title="Alerts" subtitle="Issued on or before the replay date" info="Alerts are model outputs, not guaranteed predictions. Whether stress actually followed is shown only in the Alerts page, separated as hindsight.">
      {s.recent_alerts.length === 0 ? <p className="text-sm text-muted">No alerts issued up to this date.</p> : (
        <ul className="space-y-2" data-testid="recent-alerts">
          {s.recent_alerts.map((a) => (
            <li key={`${a.method}${a.start}`}>
              <button type="button" onClick={() => go("alerts", { date: a.start, alert: `${a.method}@${a.start}` })}
                className="w-full rounded-lg border border-line bg-panel2 p-3 text-left hover:border-dim" style={{ borderLeft: `3px solid ${a.ongoing ? "#d03b3b" : "#5d6673"}` }}>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-mono text-muted">{longDate(a.start)}</span>
                  {a.ongoing ? <Pill tone="crit">active</Pill> : <Pill>{a.start === a.end ? "1 day" : `until ${longDate(a.end)}`}</Pill>}
                </div>
                <div className="mt-1 text-[13px] font-medium text-ink">{methodName(a.method)}</div>
                <div className="mt-0.5 text-xs text-muted">peak score {fmt(a.peak_score, 3)} · threshold {fmt(a.threshold, 3)}</div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function WhatMoved() {
  const { date } = useRoute();
  const net = useAsync(() => api.network(date, 4), [date]);
  const rows = net.data ? net.data.edges.filter(material).slice(0, 8) : [];
  const max = Math.max(0.01, ...rows.map((e) => Math.abs(e.after - e.before)));
  return (
    <Card title="What moved this month" subtitle={net.data ? `window ending ${longDate(net.data.window_end)} vs ${longDate(net.data.previous_window_end)} (4 windows earlier)` : undefined}
      info="The largest changes in estimated relationships over about one month. Descriptions are generated deterministically from the numbers.">
      {net.error ? <ErrorState message={net.error} /> : !net.data ? <Loading /> : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-xs">
            <thead><tr className="text-[10.5px] uppercase tracking-wider text-dim"><th className="py-2 font-medium">Relationship</th><th className="font-medium">What changed</th><th className="text-right font-medium">Prev</th><th className="text-right font-medium">Now</th><th className="pl-4 font-medium">Change</th></tr></thead>
            <tbody>
              {rows.map((e) => {
                const d = e.after - e.before;
                return (
                  <tr key={e.id} className="border-t border-line">
                    <td className="py-2 pr-3 font-mono text-ink">{nice(e.source)} → {nice(e.target)}</td>
                    <td className="pr-3 text-muted">{e.change}{e.change === "appeared" || e.change === "sign flipped" ? ` (now ${e.after > 0 ? "positive" : "negative"})` : ""}</td>
                    <td className="text-right font-mono text-muted">{signed(e.before)}</td>
                    <td className="text-right font-mono text-ink">{signed(e.after)}</td>
                    <td className="pl-4"><div className="flex items-center gap-2"><span className={`w-12 text-right font-mono ${d >= 0 ? "text-[#ff8a8a]" : "text-[#6fa8f0]"}`}>{signed(d)}</span>
                      <span className="h-1.5 rounded-full" style={{ width: `${(Math.abs(d) / max) * 120}px`, background: d >= 0 ? "#e66767" : "#3987e5" }} /></div></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function BriefingTeaser() {
  const { date, go } = useRoute();
  const { meta } = useData();
  const a = useAsync(() => api.agents(date, meta.monitored_model), [date]);
  return (
    <Card title="Evidence briefing" subtitle="Synthesis agent · deterministic, every sentence traceable to point-in-time evidence"
      right={<Button onClick={() => go("agents")}>Open AI analysis →</Button>}>
      {a.error ? <ErrorState message={a.error} /> : !a.data ? <Loading /> : (
        <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-ink/90" data-testid="briefing">
          {a.data.synthesis.briefing.map((b, i) => <li key={i}>{nice(b.text)}</li>)}
        </ul>
      )}
    </Card>
  );
}
