import { api, type MethodRow } from "../../api";
import { Details, SCard, SimpleLoading } from "../../components/simple";
import { ErrorState } from "../../components/ui";
import { G } from "../../glossary";
import { useAsync } from "../../hooks";
import { findings } from "../../plain";
import { useRoute } from "../../state";

export default function Findings() {
  const { go } = useRoute();
  const r = useAsync(() => api.research(), []);
  if (r.error) return <ErrorState message={r.error} />;
  if (!r.data) return <SimpleLoading />;
  const f = findings(r.data);
  const bars: [string, string, MethodRow, string][] = [
    ["Fear gauge (VIX) — the standard benchmark", "benchmark", f.vix, "#d95926"],
    ["Warning model with standard market data only", "", f.controls, "#d95926"],
    ["Warning model with market data + our indicator", "", f.combined, "#9085e9"],
    ["Our relationship-change index on its own (SII)", "", f.sii, "#9085e9"],
    ["Warning model with our indicator only", "", f.instOnly, "#9085e9"],
  ];
  return (
    <div className="space-y-5">
      <SCard kicker="What we found" title="Our experimental indicator did not improve on the standard fear gauge" className="border-crit/40">
        <p className="text-[18px]" data-testid="finding-plain">We asked: <i>do changes in how assets influence each other give an earlier warning of market stress
          than the fear gauge (VIX)?</i> On {f.onsets} stress episodes from {f.span[0].slice(0, 4)} to {f.span[1].slice(0, 4)} — tested with only the
          information available at the time, settings fixed in advance, and scored once — <b>the answer was no</b>.</p>
      </SCard>
      <SCard title="How well each signal picked out the days before stress" info={G.rocauc}>
        <p className="mb-4 text-muted">0.5 = no better than a coin flip; higher is better; 1.0 = perfect. Below 0.5 is worse than chance. Thin white lines show the 95% range of uncertainty.</p>
        <div className="space-y-4" data-testid="finding-bars">
          {bars.map(([label, tag, m, color]) => <Bar key={label} label={label} tag={tag} m={m} color={color} />)}
          <div className="grid grid-cols-[minmax(0,330px)_1fr_60px] gap-4 text-[13px] text-dim"><span /><div className="flex justify-between"><span>0.3</span><span>0.5 coin flip</span><span>0.7</span><span>0.9</span></div><span /></div>
        </div>
      </SCard>
      <div className="grid gap-5 lg:grid-cols-2">
        <SCard title="What this means">
          <ul className="list-disc space-y-2 pl-6">
            <li>The fear gauge did at least as well as every version of our indicator.</li>
            <li>Using our indicator alone was clearly worse than the fear gauge (difference {f.instVsVix.delta_roc.toFixed(2)}, uncertainty range {f.instVsVix.roc_ci95[0].toFixed(2)} to {f.instVsVix.roc_ci95[1].toFixed(2)}).</li>
            <li>Adding our indicator to standard market data changed the score by {f.addInst.delta_roc.toFixed(2)} (range {f.addInst.roc_ci95[0].toFixed(2)} to {f.addInst.roc_ci95[1].toFixed(2)}) — no improvement.</li>
            <li>A negative result is still a result: the system shows how to test such an idea without hindsight.</li>
          </ul>
        </SCard>
        <SCard title="Important limitations">
          <ul className="list-disc space-y-2 pl-6">
            <li>Only {f.onsets} stress episodes in the test period, so uncertainty is wide.</li>
            <li>The stress periods are defined with hindsight from S&amp;P 500 price swings.</li>
            <li>This is a historical replay, not live monitoring; news data ends in June 2020.</li>
            <li>The relationships are statistical estimates, not proven cause and effect.</li>
          </ul>
        </SCard>
      </div>
      <Details label="How was this tested? (one paragraph)">
        <p>Each year from 2014 to 2026 was predicted using only earlier data. Settings were chosen on earlier validation years and
          frozen before the final test. An automated audit checked that no future information leaked in (0 violations), and
          re-running the whole evaluation reproduced it exactly.</p>
      </Details>
      <button type="button" onClick={() => go("research")} className="text-[16px] font-medium text-accent hover:underline">See all technical results (Advanced Mode) →</button>
    </div>
  );
}

function Bar({ label, tag, m, color }: { label: string; tag: string; m: MethodRow; color: string }) {
  const x = (v: number) => `${((v - 0.3) / 0.6) * 100}%`;
  const v = m.roc_auc ?? 0;
  return (
    <div className="grid grid-cols-[minmax(0,330px)_1fr_60px] items-center gap-4" data-method={m.method}>
      <span className="text-[16px]">{label}{tag && <span className="ml-2 rounded border border-good/50 px-1.5 text-[12px] text-[#6ee06e]">{tag}</span>}</span>
      <div className="relative h-7 rounded bg-panel2">
        <div className="absolute inset-y-0 border-l border-dashed border-dim" style={{ left: x(0.5) }} />
        <div className="absolute inset-y-1 left-0 rounded" style={{ width: x(v), background: color, opacity: 0.85 }} />
        {m.roc_ci95 && <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-ink" style={{ left: x(m.roc_ci95[0]), width: `calc(${x(m.roc_ci95[1])} - ${x(m.roc_ci95[0])})` }} />}
      </div>
      <span className="text-right font-mono text-[17px]">{v.toFixed(2)}</span>
    </div>
  );
}
