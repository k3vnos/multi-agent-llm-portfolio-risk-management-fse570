import { describe, expect, it } from "vitest";
import type { PortfolioAnalysis } from "./api";
import { formatSpec, normalise, parseSpec, problems, riskSentences, total } from "./portfolio";

const U = ["SPY", "IEF", "TLT", "GLD"];

describe("portfolio editor", () => {
  it("round-trips the URL spec", () => {
    expect(parseSpec("SPY:60,IEF:40")).toEqual([{ ticker: "SPY", weight: 60 }, { ticker: "IEF", weight: 40 }]);
    expect(formatSpec(parseSpec("spy:60,IEF:40"))).toBe("SPY:60,IEF:40");
  });
  it("requires weights that total 100% and known, unique funds", () => {
    expect(problems(parseSpec("SPY:60,IEF:40"), U)).toEqual([]);
    expect(problems(parseSpec("SPY:60,IEF:30"), U).join()).toMatch(/total 100%/);
    expect(problems(parseSpec("SPY:50,SPY:50"), U).join()).toMatch(/twice/);
    expect(problems(parseSpec("XXX:100"), U).join()).toMatch(/Choose a fund/);
    expect(problems([], U).join()).toMatch(/at least one/);
  });
  it("normalises weights to exactly 100", () => {
    const n = normalise(parseSpec("SPY:1,IEF:1,GLD:1"));
    expect(total(n)).toBe(100);
    expect(n.map((r) => r.weight).sort()).toEqual([33.3, 33.3, 33.4]);
  });
  it("explains risk shares with fixed templates", () => {
    const a = { holdings: [
      { ticker: "SPY", name: "S&P 500", weight: 60, risk_share: 112, vol_63d: 10, return_21d: 1, node: "SPY", node_is_proxy: false },
      { ticker: "IEF", name: "7–10y Treasuries", weight: 40, risk_share: -12, vol_63d: 5, return_21d: 1, node: "TLT", node_is_proxy: true }] } as unknown as PortfolioAnalysis;
    const s = riskSentences(a);
    expect(s[0]).toContain("60% of the money but accounts for 112%");
    expect(s[1]).toContain("IEF moved against the rest");
  });
});
