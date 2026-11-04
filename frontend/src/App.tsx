import { Component, lazy, Suspense, useState, type ReactNode } from "react";
import { api, type Meta, type Series } from "./api";
import { DemoTour } from "./components/DemoTour";
import { DataProvider, Footer, Header } from "./components/Shell";
import { ErrorState, Loading } from "./components/ui";
import { useAsync } from "./hooks";
import Monitor from "./pages/Monitor";
import { modeOf, RouteProvider, useRoute } from "./state";

/** Lazy page import that reloads once if a stale tab asks for a chunk from an older build. */
function page<T>(load: () => Promise<T>): () => Promise<T> {
  return () => load().catch((e) => {
    let reloaded = false;
    try { reloaded = sessionStorage.getItem("chunk-reload") === "1"; sessionStorage.setItem("chunk-reload", "1"); } catch { /* storage blocked */ }
    if (!reloaded) window.location.reload();
    throw e;
  });
}
const Timeline = lazy(page(() => import("./pages/Timeline")));
const NetworkPage = lazy(page(() => import("./pages/NetworkPage")));
const Alerts = lazy(page(() => import("./pages/Alerts")));
const Agents = lazy(page(() => import("./pages/Agents")));
const Research = lazy(page(() => import("./pages/Research")));
const SimpleHome = lazy(page(() => import("./pages/simple/Home")));
const SimpleRelations = lazy(page(() => import("./pages/simple/Relations")));
const SimpleExplain = lazy(page(() => import("./pages/simple/Explain")));
const SimpleFindings = lazy(page(() => import("./pages/simple/Findings")));

export default function App() {
  const d = useAsync(() => Promise.all([api.meta(), api.series()]), []);
  if (d.error) return <div className="mx-auto max-w-3xl p-8"><ErrorState message={d.error} /></div>;
  if (!d.data) return <div className="p-16"><Loading label="Loading the saved research outputs…" /></div>;
  const [meta, series] = d.data as [Meta, Series];
  return (
    <DataProvider value={{ meta, series }}>
      <RouteProvider defaultDate={meta.end}>
        <Shell />
      </RouteProvider>
    </DataProvider>
  );
}

function Shell() {
  const { page } = useRoute();
  const [tour, setTour] = useState<number | null>(null);
  const [tourMode, setTourMode] = useState(modeOf(page));
  return (
    <div className={`min-h-screen bg-bg font-sans text-ink ${modeOf(page) === "simple" ? "simple-mode" : ""}`}>
      <Header onTour={() => { setTourMode(modeOf(page)); setTour(0); }} />
      <main className="mx-auto max-w-[1400px] px-5 py-5">
        <Boundary key={page}><Suspense fallback={<Loading />}>
          {page === "home" && <SimpleHome />}
          {page === "relations" && <SimpleRelations />}
          {page === "explain" && <SimpleExplain />}
          {page === "findings" && <SimpleFindings />}
          {page === "monitor" && <Monitor />}
          {page === "timeline" && <Timeline />}
          {page === "network" && <NetworkPage />}
          {page === "alerts" && <Alerts />}
          {page === "agents" && <Agents />}
          {page === "research" && <Research />}
        </Suspense></Boundary>
      </main>
      <Footer />
      {tour !== null && <DemoTour mode={tourMode} step={tour} setStep={setTour} onClose={() => setTour(null)} />}
    </div>
  );
}

class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidMount() { try { sessionStorage.removeItem("chunk-reload"); } catch { /* storage blocked */ } }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="space-y-3">
        <ErrorState message={this.state.error.message} />
        <button type="button" onClick={() => window.location.reload()} className="rounded-md border border-line bg-panel2 px-3 py-1.5 text-xs text-ink">Reload page</button>
      </div>
    );
  }
}
