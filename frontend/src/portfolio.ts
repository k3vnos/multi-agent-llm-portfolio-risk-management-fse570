// Portfolio Lens helpers (exploratory). The URL form is "SPY:60,IEF:40" (percent weights that total 100).
import type { PortfolioAnalysis } from "./api";

export interface Row { ticker: string; weight: number }

export function parseSpec(spec: string | null | undefined): Row[] {
  if (!spec) return [];
  return spec.split(",").filter(Boolean).map((p) => {
    const [t, w] = p.split(":");
    return { ticker: (t || "").trim().toUpperCase(), weight: Number(w) };
  });
}

export const formatSpec = (rows: Row[]) => rows.filter((r) => r.ticker).map((r) => `${r.ticker}:${+r.weight.toFixed(2)}`).join(",");
export const total = (rows: Row[]) => Math.round(rows.reduce((a, r) => a + (Number.isFinite(r.weight) ? r.weight : 0), 0) * 100) / 100;

/** Problems that stop the portfolio from being analysed (empty list = valid). */
export function problems(rows: Row[], universe: string[]): string[] {
  const out: string[] = [];
  if (!rows.length) out.push("Add at least one holding.");
  const seen = new Set<string>();
  for (const r of rows) {
    if (!universe.includes(r.ticker)) out.push(`Choose a fund for every row.`);
    else if (seen.has(r.ticker)) out.push(`${r.ticker} is listed twice.`);
    seen.add(r.ticker);
    if (!(r.weight > 0 && r.weight <= 100)) out.push(`${r.ticker || "A holding"} needs a weight between 0 and 100%.`);
  }
  const t = total(rows);
  if (rows.length && Math.abs(t - 100) > 0.01) out.push(`Weights add up to ${t}% — they must total 100%.`);
  return [...new Set(out)];
}

/** Scale weights proportionally so they total exactly 100 (rounded to 0.1, remainder on the largest). */
export function normalise(rows: Row[]): Row[] {
  const t = total(rows);
  if (!t) return rows;
  const out = rows.map((r) => ({ ...r, weight: Math.round((r.weight / t) * 1000) / 10 }));
  const diff = Math.round((100 - total(out)) * 10) / 10;
  const i = out.reduce((best, r, k) => (r.weight > out[best].weight ? k : best), 0);
  out[i] = { ...out[i], weight: Math.round((out[i].weight + diff) * 10) / 10 };
  return out;
}

/** Plain-language summary of which holdings carry the risk (fixed templates over computed numbers). */
export function riskSentences(a: PortfolioAnalysis): string[] {
  const h = a.holdings;
  const out: string[] = [];
  if (h.length === 1) return [`All of the risk comes from ${h[0].ticker}, the only holding.`];
  const top = h[0];
  out.push(`${top.ticker} (${top.name}) is ${top.weight}% of the money but accounts for ${top.risk_share.toFixed(0)}% of the portfolio's day-to-day swings over the last 3 months.`);
  const offset = h.filter((x) => x.risk_share < 0);
  if (offset.length) out.push(`${offset.map((x) => x.ticker).join(" and ")} moved against the rest of the portfolio, offsetting part of the swings (negative risk share).`);
  const small = h.filter((x) => x.risk_share >= 0 && x.risk_share < x.weight - 5);
  if (small.length) out.push(`${small.map((x) => x.ticker).join(", ")} carried less risk than ${small.length === 1 ? "its" : "their"} share of the money.`);
  return out;
}
