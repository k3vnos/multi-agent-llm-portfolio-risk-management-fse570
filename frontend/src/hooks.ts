import { useEffect, useState } from "react";

export interface Async<T> { data: T | null; error: string | null; loading: boolean }

/** Load data for a key; stale results from an older key are ignored so fast date changes stay consistent. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Async<T> {
  const [s, set] = useState<Async<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    set((p) => ({ ...p, loading: true, error: null }));
    fn().then((data) => live && set({ data, error: null, loading: false }))
        .catch((e: Error) => live && set({ data: null, error: e.message, loading: false }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}
