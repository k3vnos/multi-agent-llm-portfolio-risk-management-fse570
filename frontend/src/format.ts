export const fmt = (x: number | null | undefined, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(d));
export const signed = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(d)}`);
export const pct = (p: number | null | undefined) => (p === null || p === undefined ? "—" : `${Math.round(p)}th`);
export const ordinal = (p: number | null | undefined) => {
  if (p === null || p === undefined) return "—";
  const n = Math.round(p), s = ["th", "st", "nd", "rd"], v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};
export const band = (p: number | null | undefined) =>
  p === null || p === undefined ? "no history" : p > 80 ? "high" : p < 20 ? "low" : "typical";
export const longDate = (d: string) =>
  new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
export const METHOD_NAME: Record<string, string> = {
  "bayes_logit:combined": "Bayesian model · instability + market controls",
  "bayes_logit:controls": "Bayesian model · market controls only",
  "bayes_logit:instability": "Bayesian model · instability only",
  "bayes_logit:granger": "Bayesian model · Granger instability",
  "gbm:combined": "Gradient boosting · instability + controls",
  "gbm:controls": "Gradient boosting · controls only",
  "gbm:instability": "Gradient boosting · instability only",
  "gbm:granger": "Gradient boosting · Granger instability",
  inst_composite: "Structural Instability Index (SII)",
  vix: "VIX level (baseline)",
  nfci: "Chicago Fed NFCI (baseline)",
  corr_var: "Rolling-correlation VaR (baseline)",
  dcc_var: "DCC-GARCH VaR (baseline)",
  dcc_corr: "DCC-GARCH correlation (baseline)",
};
export const methodName = (m: string) => METHOD_NAME[m] || m;

/** Split a saved news evidence line: `<time> UTC, <publisher>, ticker <T>: "<title>" — <summary>`. */
export function parseNews(text: string): { time: string; publisher: string; ticker: string; title: string; summary: string } {
  const m = text.match(/^(.+? UTC), (.*?), ticker (\S+): "(.*)"(?: — (.*))?$/s);
  if (!m) return { time: "", publisher: "", ticker: "", title: text, summary: "" };
  return { time: m[1], publisher: m[2], ticker: m[3], title: m[4], summary: m[5] || "" };
}

const NICE: Record<string, string> = { d_VIXCLS: "ΔVIX", d_BAA10Y: "ΔBaa spread", d_T10Y2Y: "ΔYield curve" };
/** Friendly names for the macro graph nodes, applied to ids and to any generated sentence. */
export const nice = (s: string) => s.replace(/d_VIXCLS|d_BAA10Y|d_T10Y2Y/g, (m) => NICE[m]);
