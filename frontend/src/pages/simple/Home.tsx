import { useMemo } from "react";
import { api } from "../../api";
import { C, StressChart, window } from "../../components/charts";
import { useData } from "../../components/Shell";
import { Details, SCard, ScoreGauge, SimpleLoading, SimpleReplay, ToneChip } from "../../components/simple";
import { ErrorState, LevelBadge } from "../../components/ui";
import { longDate } from "../../format";
import { G } from "../../glossary";
import { useAsync } from "../../hooks";
import { evidenceRows, findings, headline, whatIsHappening, whyAlert } from "../../plain";
import { useRoute } from "../../state";

export default function Home() {
  const { date, go } = useRoute();
  const snap = useAsync(() => api.snapshot(date), [date]);
  const ag = useAsync(() => api.agents(date), [date]);
  const net = useAsync(() => api.network(date), [date]);
  const s = snap.data;
  return (
    <div className="space-y-5">
      <SimpleReplay />
      {snap.error ? <ErrorState message={snap.error} /> : !s ? <SimpleLoading /> : (
        <div className="grid gap-5 xl:grid-cols-3" data-testid="three-questions">
          <SCard kicker="1 · What is happening?" title={<span className="flex flex-col items-start gap-2"><LevelBadge level={s.status.level} />{headline(s)}</span>} info={G.status}>
            <ul className="space-y-2">{whatIsHappening(s).map((t) => <li key={t}>{t}</li>)}</ul>
          </SCard>
          <SCard kicker="2 · Why alert or no alert?" title={s.model.alert ? "Score above the alert line" : "Score below the alert line"} info={G.model}>
            <ScoreGauge score={s.model.score ?? 0} threshold={s.model.threshold ?? 0} alert={s.model.alert} />
            <p data-testid="why-alert">{whyAlert(s).main}</p>
            <p className="mt-2 text-[15px] text-muted">{whyAlert(s).how}</p>
            <p className="mt-2 text-[15px] text-warn">{whyAlert(s).caveat}</p>
          </SCard>
          <SCard kicker="3 · What evidence supports it?" title="What the data showed that day">
            <ul className="space-y-3" data-testid="evidence-rows">
              {evidenceRows(s, ag.data, net.data).map((r) => (
                <li key={r.title} className="flex gap-3"><div className="min-w-0 flex-1"><div className="font-semibold text-ink">{r.title}</div><div className="text-[15px] text-ink/80">{r.text}</div></div>{r.tone && <span className="self-start"><ToneChip tone={r.tone} /></span>}</li>
              ))}
            </ul>
            <button type="button" onClick={() => go("explain")} className="mt-4 text-[15px] font-medium text-accent hover:underline">See the full AI explanation →</button>
          </SCard>
        </div>
      )}
      <YearChart />
      <FindingTeaser onOpen={() => go("findings")} />
    </div>
  );
}

function YearChart() {
  const { series } = useData();
  const { date, setDate } = useRoute();
  const rows = useMemo(() => window(series, date, "1Y"), [series, date]);
  const last = rows.at(-1);
  const alerts = rows.filter((r) => r.alert !== null).length;
  const stressed = rows.filter((r) => (r.p ?? 0) > 0.5).length;
  return (
    <SCard kicker="The past 12 months" title="Fear gauge vs. relationship-change index"
      info="Both lines show where each measure stood compared with its own previous 5 years (0 = lowest, 100 = highest), using only data available on each day. Click the chart to jump to a date.">
      <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
        <div>
          <div className="mb-2 flex flex-wrap gap-5 text-[14px] text-muted">
            <span className="flex items-center gap-2"><span className="h-1 w-6 rounded" style={{ background: C.vix }} />Fear gauge (VIX)</span>
            <span className="flex items-center gap-2"><span className="h-1 w-6 rounded" style={{ background: C.sii }} />Relationship-change index (our indicator)</span>
            <span className="flex items-center gap-2"><span className="h-3 w-5 rounded-sm" style={{ background: "rgba(208,59,59,0.35)" }} />Stressed mood (judged at the time)</span>
            <span className="flex items-center gap-2"><span className="text-[#ff6b6b]">▲</span>Model alert</span>
          </div>
          <StressChart series={series} date={date} range="1Y" simple height={320} onPick={setDate} />
        </div>
        <div className="space-y-3 text-[16px]" data-testid="chart-explanation">
          <div className="text-[13px] font-semibold uppercase tracking-[0.14em] text-muted">What this chart shows</div>
          <p>On {longDate(rows.at(-1)!.date)} the fear gauge was at the <b>{fmtPct(last?.vix_pct)}</b> and our relationship-change index at the <b>{fmtPct(last?.sii_pct)}</b> of their past-5-year ranges.</p>
          <p>In these 12 months the warning model alerted on <b>{alerts}</b> of {rows.length} trading days, and the market mood was judged stressed on <b>{stressed}</b> days.</p>
          <p className="text-muted">If our indicator gave early warnings, the purple line would tend to rise before the orange one. In the full evaluation it did not do this reliably.</p>
        </div>
      </div>
    </SCard>
  );
}

const fmtPct = (p: number | null | undefined) => (p === null || p === undefined ? "—" : `${Math.round(p)}th percentile`);

function FindingTeaser({ onOpen }: { onOpen: () => void }) {
  const r = useAsync(() => api.research(), []);
  if (!r.data) return null;
  const f = findings(r.data);
  return (
    <SCard kicker="What our research found" title="Our experimental indicator did not beat the standard fear gauge" className="border-crit/40">
      <p>Tested on {f.onsets} stress episodes between {f.span[0].slice(0, 4)} and {f.span[1].slice(0, 4)}, using only information available at the time, the fear gauge (VIX) ranked risky days at least as well as every version of our relationship-change indicator. Models using the indicator alone did clearly worse, and adding it to standard market data did not improve the warning model.</p>
      <Details label="Why this still matters">
        <p>A clearly documented negative result is a valid finding: the project shows how to test such an idea honestly (no look-ahead, settings fixed before testing, confidence intervals), and a diagnostic suggested that this kind of graph-based signal is very noisy at the data sizes available.</p>
      </Details>
      <button type="button" onClick={onOpen} className="mt-4 text-[15px] font-medium text-accent hover:underline">See the result explained →</button>
    </SCard>
  );
}
