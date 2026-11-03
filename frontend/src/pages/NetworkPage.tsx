import { useMemo, useState } from "react";
import { api, type Network } from "../api";
import { NetworkGraph, NetworkLegend, material, type Mode, type Selection } from "../components/NetworkGraph";
import { ReplayBar, useData } from "../components/Shell";
import { Button, Card, ErrorState, Loading } from "../components/ui";
import { longDate, signed, nice } from "../format";
import { G } from "../glossary";
import { useAsync } from "../hooks";
import { useRoute } from "../state";
import { Inspector } from "./Monitor";

export default function NetworkPage() {
  const { date, go } = useRoute();
  const { meta } = useData();
  const [mode, setMode] = useState<Mode>("change");
  const [lag, setLag] = useState(1);
  const [view, setView] = useState<"graph" | "matrix">("graph");
  const [sel, setSel] = useState<Selection>(null);
  const net = useAsync(() => api.network(date, lag), [date, lag]);
  return (
    <div className="space-y-4">
      <ReplayBar />
      <Card title="Financial relationship network" info={G.edges}
        subtitle={net.data ? `estimation window ending ${longDate(net.data.window_end)} (${meta.graph_window} trading days) vs window ending ${longDate(net.data.previous_window_end)}` : undefined}
        right={<div className="flex flex-wrap gap-1">
          <Button active={view === "graph"} onClick={() => setView("graph")}>Network</Button>
          <Button active={view === "matrix"} onClick={() => setView("matrix")}>Matrix</Button>
          <span className="mx-1" />
          <Button active={mode === "change"} onClick={() => setMode("change")}>Changes</Button>
          <Button active={mode === "current"} onClick={() => setMode("current")}>Current links</Button>
          <span className="mx-1" />
          <Button active={lag === 1} onClick={() => setLag(1)}>vs 1 week</Button>
          <Button active={lag === 4} onClick={() => setLag(4)}>vs 1 month</Button>
        </div>}>
        <ExplainBox />
        {net.error ? <ErrorState message={net.error} /> : !net.data ? <Loading /> : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
            <div>
              {view === "graph" ? <><NetworkGraph net={net.data} mode={mode} selection={sel} onSelect={setSel} size={620} /><NetworkLegend mode={mode} /></>
                : <Matrix net={net.data} mode={mode} />}
            </div>
            <div className="space-y-4">
              <div className="rounded-lg border border-line bg-panel2 p-3"><Inspector net={net.data} sel={sel} /></div>
              <ChangeList net={net.data} onPick={(id) => setSel({ type: "edge", id })} />
              <p className="text-[11px] leading-relaxed text-dim">{net.data.caveat}</p>
              <div className="flex flex-wrap gap-2"><Button onClick={() => go("timeline")}>Timeline on this date →</Button><Button onClick={() => go("agents")}>Graph agent evidence →</Button></div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

function ExplainBox() {
  return (
    <div className="mb-3 grid gap-3 rounded-lg border border-line bg-panel2 p-3 text-xs leading-relaxed text-muted md:grid-cols-3">
      <p><b className="text-ink">Circles</b> are monitored assets and economic indicators (outline colour = asset class). The three Δ nodes are daily changes in VIX, the Baa credit spread and the yield curve.</p>
      <p><b className="text-ink">An arrow A → B</b> means that, in the statistical model, same-day moves in A help explain moves in B. Thicker = stronger. It is an <i>estimate</i>, not proof that A causes B.</p>
      <p><b className="text-ink">Changes</b> compare this week's estimation window with an earlier one: links that appeared, disappeared, flipped sign or changed strength by more than 0.1.</p>
    </div>
  );
}

function ChangeList({ net, onPick }: { net: Network; onPick: (id: string) => void }) {
  const rows = net.edges.filter(material).slice(0, 10);
  return (
    <div>
      <div className="mb-1.5 text-[10.5px] font-medium uppercase tracking-[0.12em] text-muted">Largest changes</div>
      <ul className="space-y-1 text-xs" data-testid="change-list">
        {rows.map((e) => (
          <li key={e.id}><button type="button" onClick={() => onPick(e.id)} className="flex w-full items-center justify-between rounded px-2 py-1 text-left hover:bg-panel2">
            <span className="font-mono text-ink">{nice(e.source)} → {nice(e.target)}</span><span className="text-muted">{e.change} {signed(e.before)}→{signed(e.after)}</span>
          </button></li>
        ))}
        {rows.length === 0 && <li className="text-muted">No material changes.</li>}
      </ul>
    </div>
  );
}

function Matrix({ net, mode }: { net: Network; mode: Mode }) {
  const n = net.node_order;
  const M = useMemo(() => (mode === "current" ? net.B0 : net.B0.map((row, i) => row.map((v, j) => v - net.B0_previous[i][j]))), [net, mode]);
  const lim = Math.max(0.3, ...M.flat().map(Math.abs));
  const col = (v: number) => {
    const t = Math.min(1, Math.abs(v) / lim);
    const [r, g, b] = v >= 0 ? [230, 103, 103] : [57, 135, 229];
    return `rgba(${r},${g},${b},${0.08 + 0.85 * t})`;
  };
  return (
    <div className="overflow-x-auto" data-testid="matrix">
      <p className="mb-2 text-xs text-muted">{mode === "current" ? "Contemporaneous coefficients B0" : "Change in coefficients vs the earlier window"}: column = source (effect of), row = target (on). Red positive, blue negative.</p>
      <table className="border-separate border-spacing-[2px] text-[10px]">
        <thead><tr><th />{n.map((c) => <th key={c} className="h-16 w-9 align-bottom font-mono font-normal text-muted"><div className="origin-bottom-left -rotate-45 whitespace-nowrap">{nice(c)}</div></th>)}</tr></thead>
        <tbody>{n.map((r, i) => (
          <tr key={r}><th className="pr-2 text-right font-mono font-normal text-muted">{nice(r)}</th>
            {n.map((c, j) => <td key={c} title={`${c} → ${r}: ${i === j ? "—" : signed(M[i][j])}`} className="h-7 w-9 rounded-sm text-center font-mono text-ink/80"
              style={{ background: i === j ? "transparent" : col(M[i][j]) }}>{i === j || Math.abs(M[i][j]) < 0.05 ? "" : M[i][j].toFixed(1)}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}
