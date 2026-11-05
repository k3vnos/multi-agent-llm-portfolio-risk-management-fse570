import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export const SIMPLE_PAGES = ["home", "relations", "portfolio", "explain", "findings"] as const;
export const ADVANCED_PAGES = ["monitor", "timeline", "network", "alerts", "lens", "agents", "research"] as const;
export const PAGES = [...SIMPLE_PAGES, ...ADVANCED_PAGES] as const;
export type Page = (typeof PAGES)[number];
export type Mode = "simple" | "advanced";

/** The mode is a property of the page, so every URL (including all existing advanced ones) is stable. */
export const modeOf = (p: Page): Mode => ((SIMPLE_PAGES as readonly string[]).includes(p) ? "simple" : "advanced");
/** Switching mode keeps the date and lands on the matching page. */
export const COUNTERPART: Record<Page, Page> = {
  home: "monitor", relations: "network", portfolio: "lens", explain: "agents", findings: "research",
  monitor: "home", timeline: "home", network: "relations", alerts: "explain", lens: "portfolio", agents: "explain",
  research: "findings",
};

/** `pf` is the exploratory Portfolio Lens portfolio, e.g. "SPY:60,IEF:40" — kept in the URL so it follows every page. */
interface Route { page: Page; date: string | null; alert: string | null; pf: string | null }

function parse(hash: string): Route {
  const [path, q] = hash.replace(/^#\/?/, "").split("?");
  const params = new URLSearchParams(q || "");
  const page = (PAGES as readonly string[]).includes(path) ? (path as Page) : "home";
  return { page, date: params.get("date"), alert: params.get("alert"), pf: params.get("pf") };
}

function format(r: Route): string {
  const p = new URLSearchParams();
  if (r.date) p.set("date", r.date);
  if (r.alert) p.set("alert", r.alert);
  if (r.pf) p.set("pf", r.pf);
  const q = p.toString().replace(/%3A/g, ":").replace(/%2C/g, ",");
  return `#/${r.page}${q ? `?${q}` : ""}`;
}

interface Ctx extends Omit<Route, "date"> {
  date: string;
  go: (page: Page, opts?: { date?: string; alert?: string | null; pf?: string | null }) => void;
  setDate: (d: string) => void;
  setPf: (pf: string | null) => void;
}

const RouteCtx = createContext<Ctx | null>(null);

export function RouteProvider({ children, defaultDate }: { children: ReactNode; defaultDate: string }) {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const date = route.date || defaultDate;
  const go = useCallback((page: Page, opts?: { date?: string; alert?: string | null; pf?: string | null }) => {
    window.location.hash = format({ page, date: opts?.date ?? date, alert: opts?.alert === undefined ? route.alert : opts.alert,
                                    pf: opts?.pf === undefined ? route.pf : opts.pf });
    window.scrollTo({ top: 0 });
  }, [date, route.alert, route.pf]);
  const setDate = useCallback((d: string) => {
    window.history.replaceState(null, "", format({ ...route, date: d }));
    setRoute((r) => ({ ...r, date: d }));
  }, [route]);
  const setPf = useCallback((pf: string | null) => {
    window.history.replaceState(null, "", format({ ...route, pf }));
    setRoute((r) => ({ ...r, pf }));
  }, [route]);
  const value = useMemo(() => ({ ...route, date, go, setDate, setPf }), [route, date, go, setDate, setPf]);
  return <RouteCtx.Provider value={value}>{children}</RouteCtx.Provider>;
}

export function useRoute(): Ctx {
  const c = useContext(RouteCtx);
  if (!c) throw new Error("useRoute outside RouteProvider");
  return c;
}
