import { useState } from "react";
import { api, type Network } from "../../api";
import { CHANGE_COLOR, NetworkGraph, material, type Mode, type Selection } from "../../components/NetworkGraph";
import { Details, SCard, SimpleLoading, SimpleReplay } from "../../components/simple";
import { ErrorState } from "../../components/ui";
import { longDate, signed } from "../../format";
import { G } from "../../glossary";
import { useAsync } from "../../hooks";
import { edgePlain, NAMES, networkSummary, short } from "../../plain";
import { useRoute } from "../../state";

const TOP = 5;

export default function Relations() {
  const { date, go } = useRoute();
  const [all, setAll] = useState(false);
  const [mode, setMode] = useState<Mode>("change");
  const [sel, setSel] = useState<Selection>(null);
  const net = useAsync(() => api.network(date), [date]);
  const n = net.data;
  const top = n ? n.edges.filter(material).slice(0, TOP) : [];
  return (
    <div className="space-y-5">
      <SimpleReplay />
      <SCard kicker="How assets influence each other" title="Which relationships changed this week?" info={G.edges}
        right={<div className="flex flex-wrap gap-2" role="group" aria-label="Network view">
          <Toggle on={!all} onClick={() => setAll(false)}>Top {TOP} changes</Toggle>
          <Toggle on={all && mode === "change"} onClick={() => { setAll(true); setMode("change"); }}>All changes</Toggle>
          <Toggle on={all && mode === "current"} onClick={() => { setAll(true); setMode("current"); }}>All current connections</Toggle>
        </div>}>
        {net.error ? <ErrorState message={net.error} /> : !n ? <SimpleLoading /> : (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,560px)_1fr]">
            <div>
              <NetworkGraph net={n} mode={all ? mode : "change"} selection={sel} onSelect={setSel} size={540} limit={all ? undefined : TOP} />
              <Legend mode={all ? mode : "change"} />
            </div>
            <div className="space-y-4" data-testid="relations-explanation">
              <div className="rounded-xl border border-line bg-panel2 p-4">
                <div className="text-[13px] font-semibold uppercase tracking-[0.14em] text-muted">How to read this</div>
                <p className="mt-1">Each circle is a type of investment or an economic signal. An arrow from A to B means that, in our
                  statistical model, moves in A help explain same-day moves in B. It is an <b>estimate from the past two years of
                  data</b>, not proof that A causes B.</p>
              </div>
              <p data-testid="network-summary">{networkSummary(n)}</p>
              {sel ? <Inspect n={n} sel={sel} /> : (
                <div>
                  <div className="mb-2 font-semibold">{all ? "Biggest changes" : `The ${TOP} biggest changes (shown on the graph)`}</div>
                  <ol className="space-y-2" data-testid="top-changes">
                    {top.map((e, i) => (
                      <li key={e.id} className="flex gap-3"><span className="w-5 shrink-0 text-right font-mono text-muted">{i + 1}.</span>
                        <button type="button" className="text-left hover:text-accent" onClick={() => setSel({ type: "edge", id: e.id })}>
                          {edgePlain(e).replace(/^the/, "The")}</button></li>
                    ))}
                  </ol>
                  <p className="mt-3 text-[15px] text-muted">Click a circle or an arrow to see details.</p>
                </div>
              )}
              <p className="text-[15px] text-warn">These estimates are noisy: re-estimating the same period on slightly different data changes many links, so treat single changes with caution.</p>
              <button type="button" onClick={() => go("network")} className="text-[15px] font-medium text-accent hover:underline">Open the full technical network view →</button>
            </div>
          </div>
        )}
        {n && <Details label="Technical details"><p className="text-muted">VarLiNGAM contemporaneous matrix (B0) estimated on the 500 trading days ending {longDate(n.window_end)}, compared with the window ending {longDate(n.previous_window_end)}. A link exists when |coefficient| &gt; {n.edge_threshold}. {n.caveat}</p></Details>}
      </SCard>
    </div>
  );
}

function Toggle({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button type="button" aria-pressed={on} onClick={onClick}
    className={`rounded-lg border px-3 py-1.5 text-[14px] ${on ? "border-accent bg-accent/20 text-ink" : "border-line bg-panel2 text-muted hover:text-ink"}`}>{children}</button>;
}

function Legend({ mode }: { mode: Mode }) {
  const items = mode === "current"
    ? [["Move together", "#e66767"], ["Move opposite", "#3987e5"]]
    : [["New link", CHANGE_COLOR.appeared], ["Got stronger", CHANGE_COLOR.strengthened], ["Got weaker", CHANGE_COLOR.weakened],
       ["Reversed", CHANGE_COLOR["sign flipped"]], ["Disappeared (dashed)", CHANGE_COLOR.disappeared]];
  return (
    <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[14px] text-muted">
      {items.map(([t, c]) => <span key={t} className="flex items-center gap-2"><span className="h-1 w-6 rounded" style={{ background: c }} />{t}</span>)}
    </div>
  );
}

function Inspect({ n, sel }: { n: Network; sel: NonNullable<Selection> }) {
  if (sel.type === "edge") {
    const e = n.edges.find((x) => x.id === sel.id)!;
    return (
      <div className="rounded-xl border border-accent/40 bg-accent/10 p-4" data-testid="simple-inspector">
        <div className="font-semibold">{short(e.source)} → {short(e.target)}</div>
        <p className="mt-1">{edgePlain(e).replace(/^the/, "The")}</p>
        <p className="mt-1 text-[14px] text-muted">Estimated strength: {signed(e.before)} last week → {signed(e.after)} now.</p>
      </div>
    );
  }
  const out = n.edges.filter((e) => e.source === sel.id && e.active);
  const inc = n.edges.filter((e) => e.target === sel.id && e.active);
  return (
    <div className="rounded-xl border border-accent/40 bg-accent/10 p-4" data-testid="simple-inspector">
      <div className="font-semibold">{NAMES[sel.id] || sel.id}</div>
      <p className="mt-1">Helps explain {out.length ? out.map((e) => short(e.target)).join(", ") : "nothing this week"}.</p>
      <p className="mt-1">Is explained by {inc.length ? inc.map((e) => short(e.source)).join(", ") : "nothing this week"}.</p>
    </div>
  );
}
