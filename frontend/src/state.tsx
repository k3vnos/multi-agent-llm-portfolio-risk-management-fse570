import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export const PAGES = ["monitor", "timeline", "network", "alerts", "agents", "research"] as const;
export type Page = (typeof PAGES)[number];

interface Route { page: Page; date: string | null; alert: string | null }

function parse(hash: string): Route {
  const [path, q] = hash.replace(/^#\/?/, "").split("?");
  const params = new URLSearchParams(q || "");
  const page = (PAGES as readonly string[]).includes(path) ? (path as Page) : "monitor";
  return { page, date: params.get("date"), alert: params.get("alert") };
}

function format(r: Route): string {
  const p = new URLSearchParams();
  if (r.date) p.set("date", r.date);
  if (r.alert) p.set("alert", r.alert);
  const q = p.toString();
  return `#/${r.page}${q ? `?${q}` : ""}`;
}

interface Ctx extends Omit<Route, "date"> {
  date: string;
  go: (page: Page, opts?: { date?: string; alert?: string | null }) => void;
  setDate: (d: string) => void;
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
  const go = useCallback((page: Page, opts?: { date?: string; alert?: string | null }) => {
    window.location.hash = format({ page, date: opts?.date ?? date, alert: opts?.alert === undefined ? route.alert : opts.alert });
    window.scrollTo({ top: 0 });
  }, [date, route.alert]);
  const setDate = useCallback((d: string) => {
    window.history.replaceState(null, "", format({ ...route, date: d }));
    setRoute((r) => ({ ...r, date: d }));
  }, [route]);
  const value = useMemo(() => ({ ...route, date, go, setDate }), [route, date, go, setDate]);
  return <RouteCtx.Provider value={value}>{children}</RouteCtx.Provider>;
}

export function useRoute(): Ctx {
  const c = useContext(RouteCtx);
  if (!c) throw new Error("useRoute outside RouteProvider");
  return c;
}
