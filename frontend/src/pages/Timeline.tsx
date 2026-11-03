import { useState } from "react";
import { C, Legend, RawPanel, StressChart, type Range } from "../components/charts";
import { ReplayBar, useData } from "../components/Shell";
import { Button, Card, Info, Pill } from "../components/ui";
import { longDate } from "../format";
import { G } from "../glossary";
import { useRoute } from "../state";

export default function Timeline() {
  const { meta, series } = useData();
  const { date, setDate, go } = useRoute();
  const [range, setRange] = useState<Range>("1Y");
  const [hindsight, setHindsight] = useState(false);
  const events = meta.events;
  const ev = events.find((e) => e.focus === date || e.replay_start === date);
  return (
    <div className="space-y-4">
      <ReplayBar />
      <div className="grid gap-4 xl:grid-cols-[1fr_300px]">
        <Card title="Historical timeline" subtitle={`up to ${longDate(date)}${hindsight ? " + 3 months of hindsight" : ""}`}
          right={<div className="flex flex-wrap items-center gap-1">
            {(["6M", "1Y", "3Y", "5Y", "Max"] as Range[]).map((r) => <Button key={r} active={range === r} onClick={() => setRange(r)}>{r}</Button>)}
            <span className="mx-1" />
            <Button active={hindsight} onClick={() => setHindsight((h) => !h)} title="Show what happened after the replay date (not available at the time)">
              {hindsight ? "Hide hindsight" : "Show what happened next"}
            </Button>
          </div>}>
          {hindsight && (
            <div className="mb-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
              Hindsight view: the 3 months after the replay date and retrospective stress periods are shown for context only —
              none of this was known on {longDate(date)}.
            </div>
          )}
          <Legend hindsight={hindsight} />
          <StressChart series={series} date={date} range={range} hindsight={hindsight} height={320} onPick={setDate} />
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <RawPanel series={series} date={date} range={range} hindsight={hindsight} field="sii" color={C.sii} label="SII (graph-instability composite)" unit="z-score vs training years" />
            <RawPanel series={series} date={date} range={range} hindsight={hindsight} field="vix" color={C.vix} label="VIX level" unit="index points" />
            <RawPanel series={series} date={date} range={range} hindsight={hindsight} field="nfci" color={C.nfci} label="Chicago Fed NFCI (as published then)" unit="0 = average" />
            <RawPanel series={series} date={date} range={range} hindsight={hindsight} field="score" color="#e6e8eb" thresholdField label="Monitored model: alert score vs threshold" unit="probability" />
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-xs">
            <Button onClick={() => go("network")}>See the network on this date →</Button>
            <Button onClick={() => go("agents")}>See the evidence on this date →</Button>
          </div>
        </Card>
        <Card title="Historical episodes" info="Stress onsets identified in the project's evaluation (retrospective labels). Selecting one moves the replay date; everything shown for that date remains point-in-time.">
          {ev?.note && <div className="mb-3 rounded-md border border-accent/40 bg-accent/10 p-2 text-xs text-ink">{ev.note}</div>}
          <ul className="max-h-[760px] space-y-1.5 overflow-y-auto pr-1" data-testid="event-list">
            {events.map((e) => (
              <li key={e.id}>
                <button type="button" disabled={!e.available} onClick={() => e.focus && setDate(e.focus)}
                  className={`w-full rounded-lg border px-3 py-2 text-left text-xs disabled:cursor-not-allowed disabled:opacity-50 ${date === e.focus ? "border-accent bg-accent/15" : "border-line bg-panel2 hover:border-dim"}`}>
                  <div className="flex items-center justify-between gap-2"><span className="font-medium text-ink">{e.label}</span>
                    {e.target === "exploratory" ? <Pill tone="accent">exploratory</Pill> : !e.available ? <Pill>unavailable</Pill> : null}</div>
                  <div className="mt-0.5 text-muted">{e.available ? `onset ${longDate(e.focus!)}` : e.note}</div>
                </button>
              </li>
            ))}
          </ul>
          <p className="mt-3 flex items-start gap-1.5 text-[11px] leading-relaxed text-dim"><Info text={G.alertOutcome} /> Episode dates come from hindsight labels; they are used here only to navigate.</p>
        </Card>
      </div>
    </div>
  );
}
