import { describe, expect, it } from "vitest";
import type { Series } from "./api";
import { monthTicks, runs, window } from "./components/charts";
import { material } from "./components/NetworkGraph";
import { band, nice, ordinal, parseNews } from "./format";

const dates = ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-28", "2020-03-02", "2020-03-03"];
const S: Series = {
  dates, sii: dates.map((_, i) => i), sii_pct: dates.map(() => 50), vix: dates.map(() => 15), vix_pct: dates.map(() => 40),
  nfci: dates.map(() => -0.5), nfci_pct: dates.map(() => 30), model_score: dates.map(() => 0.2), model_threshold: dates.map(() => 0.3),
  regime_p_stress: [0.1, 0.2, 0.7, 0.8, 0.2, 0.9, 0.9], model_alert: dates.map((_, i) => i === 3),
  vix_alert: dates.map(() => false), retrospective_stress: [0, 0, 1, 1, 1, 1, 1],
};

describe("point-in-time chart window", () => {
  it("never includes dates after the replay date unless hindsight is requested", () => {
    const w = window(S, "2020-02-04", "Max");
    expect(w.at(-1)!.date).toBe("2020-02-04");
    expect(w.every((r) => !r.future)).toBe(true);
    const h = window(S, "2020-02-04", "Max", true);
    expect(h.at(-1)!.date).toBe("2020-03-03");
    expect(h.filter((r) => r.future).map((r) => r.date)).toEqual(["2020-02-28", "2020-03-02", "2020-03-03"]);
  });
  it("snaps a weekend replay date to the previous session", () => {
    expect(window(S, "2020-02-01", "Max").at(-1)!.date).toBe("2020-01-31");
  });
  it("finds regime runs from the as-of-then probability", () => {
    const w = window(S, "2020-03-03", "Max");
    expect(runs(w, (r) => (r.p ?? 0) > 0.5)).toEqual([["2020-02-03", "2020-02-04"], ["2020-03-02", "2020-03-03"]]);
  });
  it("marks alerts only on alert days", () => {
    expect(window(S, "2020-03-03", "Max").filter((r) => r.alert !== null).map((r) => r.date)).toEqual(["2020-02-04"]);
  });
});

describe("axis ticks", () => {
  it("uses the first trading day of each month, no duplicates", () => {
    const t = monthTicks(dates);
    expect(t).toEqual(["2020-02-03", "2020-03-02"]);
    expect(new Set(t).size).toBe(t.length);
  });
});

describe("formatting", () => {
  it("parses saved news evidence lines", () => {
    const p = parseNews('2020-01-06 00:00 UTC, Benzinga Newsdesk, ticker BBD: "Banks trade lower" — summary text');
    expect(p).toEqual({ time: "2020-01-06 00:00 UTC", publisher: "Benzinga Newsdesk", ticker: "BBD", title: "Banks trade lower", summary: "summary text" });
  });
  it("names macro nodes in plain language", () => {
    expect(nice("SPY -> d_VIXCLS and d_BAA10Y")).toBe("SPY -> ΔVIX and ΔBaa spread");
  });
  it("labels percentile bands like the backend rule", () => {
    expect([band(85), band(50), band(10), band(null)]).toEqual(["high", "typical", "low", "no history"]);
    expect([ordinal(1), ordinal(22), ordinal(83), ordinal(11)]).toEqual(["1st", "22nd", "83rd", "11th"]);
  });
  it("flags material network changes only", () => {
    const e = { id: "a", source: "A", target: "B", text: "", active: true, sign: "positive" };
    expect(material({ ...e, before: 0.3, after: 0.35, change: "strengthened" })).toBe(false);
    expect(material({ ...e, before: 0.3, after: 0.45, change: "strengthened" })).toBe(true);
    expect(material({ ...e, before: 0.0, after: 0.06, change: "appeared" })).toBe(true);
  });
});
