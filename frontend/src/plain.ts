// Plain-English sentences for Simple Mode. Every sentence is a fixed template filled with saved numbers — no new
// conclusions are generated here. Technical names stay available in tooltips and Advanced Mode.
import type { Agents, Edge, EvidenceItem, Network, Research, Snapshot } from "./api";
import { parseNews } from "./format";

export type Tone = "high" | "low" | "normal" | "unknown";

/** Same 20th/80th-percentile bands as the backend rule (previous 5 years, data up to that date). */
export const tone = (p: number | null | undefined): Tone =>
  p === null || p === undefined ? "unknown" : p > 80 ? "high" : p < 20 ? "low" : "normal";

export const toneWord = (p: number | null | undefined) =>
  ({ high: "unusually high", low: "unusually low", normal: "normal", unknown: "no comparison available" })[tone(p)];

const r = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "—" : x.toFixed(d));

export const NAMES: Record<string, string> = {
  SPY: "US large companies (SPY)", IWM: "US small companies (IWM)", EFA: "Developed-market stocks (EFA)",
  EEM: "Emerging-market stocks (EEM)", XLF: "US banks & financials (XLF)", TLT: "Long-term US government bonds (TLT)",
  LQD: "High-quality corporate bonds (LQD)", HYG: "Riskier corporate bonds (HYG)", GLD: "Gold (GLD)",
  UUP: "US dollar (UUP)", d_VIXCLS: "the fear gauge (daily change in VIX)",
  d_BAA10Y: "corporate borrowing costs (daily change in Baa spread)", d_T10Y2Y: "the yield curve (daily change)",
};
export const short = (id: string) => (NAMES[id] || id).replace(/ \(.*\)$/, "");

const INDICATOR: Record<string, string> = {
  "Structural instability (SII)": "our relationship-change index", "VIX level": "the fear gauge",
  "5-day VIX change": "the weekly jump in the fear gauge", "NFCI financial conditions": "borrowing-conditions index",
  "Graph edge turnover": "the number of relationship changes",
};

export function headline(s: Snapshot): string {
  return { alert: "The warning model raised an alert", unusual: "No alert — but some readings were unusual",
           normal: "No alert — conditions looked normal" }[s.status.level];
}

/** Q1 — what is happening (as it looked at the close of the replay date). */
export function whatIsHappening(s: Snapshot): string[] {
  const out: string[] = [];
  if (s.regime.state)
    out.push(`A model of recent S&P 500 price swings put the market mood at "${s.regime.state === "stressed" ? "stressed" : "calm"}" (${Math.round((s.regime.p_stress ?? 0) * 100)}% chance of a stressed period).`);
  out.push(`The fear gauge (VIX) stood at ${r(s.vix.value)} — ${toneWord(s.vix.percentile)} compared with the previous 5 years.`);
  if (s.nfci.value !== null)
    out.push(`Borrowing conditions were ${s.nfci.value > 0 ? "tighter" : "looser"} than the long-run average (Chicago Fed index ${r(s.nfci.value, 2)}).`);
  if (s.status.unusual.length)
    out.push("Unusually high for the past 5 years: " + s.status.unusual.map((u) => `${INDICATOR[u.indicator] || u.indicator} (${u.percentile}th percentile)`).join("; ") + ".");
  return out;
}

/** Q2 — why the model did or did not alert: the score, the cut-off and the comparison. */
export function whyAlert(s: Snapshot): { main: string; how: string; caveat: string } {
  const sc = s.model.score ?? 0, th = s.model.threshold ?? 0;
  return {
    main: s.model.alert
      ? `Its score was ${sc.toFixed(2)}, above its alert line of ${th.toFixed(2)} — so it raised an alert.`
      : `Its score was ${sc.toFixed(2)}, below its alert line of ${th.toFixed(2)} — so no alert.`,
    how: "The score is the model's estimate (0 to 1) of the chance that a stressed period starts within the next two weeks. "
       + "It combines the relationship-change index with standard market indicators. The alert line was fixed using earlier "
       + "years only, to keep false alarms rare.",
    caveat: "An alert is a warning sign, not a prediction. In our tests this model was not more accurate than the fear gauge alone.",
  };
}

export interface EvidenceRow { title: string; text: string; tone: Tone | null; more?: string }

/** The more unusual of two readings (used when a level is normal but its weekly change is not). */
const extreme = (a: Tone, b: Tone): Tone => (a === "high" || b === "high" ? "high" : a === "low" || b === "low" ? "low" : a === "unknown" ? b : a);

/** Q3 — the supporting evidence, one plain sentence per source. */
export function evidenceRows(s: Snapshot, a: Agents | null, net: Network | null): EvidenceRow[] {
  const rows: EvidenceRow[] = [];
  const chg = s.vix.change_5d;
  rows.push({ title: "Fear gauge", tone: extreme(tone(s.vix.percentile), tone(s.vix.change_5d_percentile)),
    text: `VIX ${r(s.vix.value)} (${toneWord(s.vix.percentile)})` + (chg !== null ? `; it ${chg >= 0 ? "rose" : "fell"} ${Math.abs(chg).toFixed(1)} points over the week (${toneWord(s.vix.change_5d_percentile)} for a weekly move).` : ".") });
  rows.push({ title: "Borrowing conditions", tone: tone(s.nfci.percentile),
    text: s.nfci.value === null ? "Not available for this date." : `${s.nfci.value > 0 ? "Tighter" : "Looser"} than average; ${toneWord(s.nfci.percentile)} for the past 5 years.` });
  const baa = a?.agents.macro.evidence.find((i) => i.id.startsWith("macro:BAA10Y"));
  if (baa) {
    const p = baa.values.pct;
    rows.push({ title: "Corporate credit", tone: tone(p),
      text: `Lenders charged medium-quality companies ${baa.values.value.toFixed(2)} percentage points more than the US government — ${tone(p) === "high" ? "more than usual (credit looked stressed)" : tone(p) === "low" ? "less than usual (credit looked calm)" : "a usual amount"}.` });
  }
  rows.push({ title: "Relationship changes", tone: tone(s.edge_turnover.percentile),
    text: `${r(s.edge_turnover.value, 0)} estimated links between assets changed this week (${toneWord(s.edge_turnover.percentile)}).`
      + (net && net.edges[0] ? ` Biggest: ${edgePlain(net.edges[0])}` : "") });
  rows.push(newsRow(a));
  return rows;
}

export function newsRow(a: Agents | null): EvidenceRow {
  if (!a) return { title: "News", tone: null, text: "Loading…" };
  const n = a.agents.news.evidence;
  if (!n.length)
    return { title: "News", tone: null,
      text: "No news available for this date — our news dataset ends in June 2020. This means missing data, not 'no news'." };
  return { title: "News", tone: null, text: `${n.length} headlines published in the week before the market close, e.g. “${headlineOf(n[0])}”` };
}

/** Headline only (no summary), shortened for display. */
export const headlineOf = (i: EvidenceItem, max = 120) => {
  const t = parseNews(i.source_text).title;
  return t.length > max ? `${t.slice(0, max).replace(/\s+\S*$/, "")}…` : t;
};

const CHANGE: Record<string, string> = {
  appeared: "is a new link", disappeared: "has disappeared", "sign flipped": "reversed direction",
  strengthened: "got stronger", weakened: "got weaker",
};

/** One edge in words: "Moves in A now help explain same-day moves in B — a new link (estimated)." */
export function edgePlain(e: Edge): string {
  const dir = (e.change === "disappeared" ? e.before : e.after) >= 0 ? "the same direction" : "the opposite direction";
  return `the link from ${short(e.source)} to ${short(e.target)} ${CHANGE[e.change] || "changed"}`
    + (e.change === "disappeared" ? "." : ` (they tend to move in ${dir}).`);
}

/** Plain summary of a network snapshot. */
export function networkSummary(n: Network): string {
  const c = n.counts;
  return `Compared with the previous week, ${c.appeared} links appeared and ${c.disappeared} disappeared` +
    (c["sign flipped"] ? `, and ${c["sign flipped"]} reversed direction` : "") + `; ${n.n_active} links are active now.`;
}

/** Main research result in plain language, read from the saved locked metrics. */
export function findings(res: Research) {
  const row = (m: string) => res.primary.data.methods.find((x) => x.method === m)!;
  const d = (k: string) => res.primary.data.differences.find((x) => x.comparison === k)!;
  return {
    vix: row("vix"), controls: row("bayes_logit:controls"), combined: row("bayes_logit:combined"),
    sii: row("inst_composite"), instOnly: row("bayes_logit:instability"),
    addInst: d("bayes_logit:combined - bayes_logit:controls"), instVsVix: d("bayes_logit:instability - vix"),
    onsets: row("vix").n_onsets, span: res.primary.data.span,
  };
}
