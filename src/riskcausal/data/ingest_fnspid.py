"""FNSPID (Dong, Fan & Peng, KDD'24) on Hugging Face — public, free. Used only by the explanation layer."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

REPO = "Zihan1004/FNSPID"
API_TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
API_INFO = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def repo_info() -> dict:
    r = requests.get(API_INFO, timeout=30)
    r.raise_for_status()
    js = r.json()
    return {"private": js.get("private"), "gated": js.get("gated"), "last_modified": js.get("lastModified"),
            "license": (js.get("cardData") or {}).get("license")}


def list_files() -> list[dict]:
    r = requests.get(API_TREE, timeout=30)
    r.raise_for_status()
    return [{"path": f["path"], "size_bytes": f.get("size")} for f in r.json() if f.get("type") == "file"]


def read_byte_range(path: str, start: int, length: int) -> bytes:
    r = requests.get(RESOLVE.format(path=path), headers={"Range": f"bytes={start}-{start + length - 1}"},
                     timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content


def sample_csv(path: str, nbytes: int = 4_000_000, offset: int = 0, header: list[str] | None = None) -> pd.DataFrame:
    """Parse complete CSV rows from a byte range (partial first/last lines discarded)."""
    raw = read_byte_range(path, offset, nbytes).decode("utf-8", errors="replace")
    if offset > 0:
        raw = raw.split("\n", 1)[1]
    raw = raw.rsplit("\n", 1)[0]
    if header is not None:
        return pd.read_csv(io.StringIO(raw), header=None, names=header, on_bad_lines="skip", engine="python")
    return pd.read_csv(io.StringIO(raw), on_bad_lines="skip", engine="python")


class _HashingReader(io.RawIOBase):
    """Wraps a streamed HTTP body: hashes and counts every byte while the CSV parser reads it."""

    def __init__(self, raw):
        import hashlib

        self.raw, self.sha, self.n = raw, hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        chunk = self.raw.read(len(b))
        if not chunk:
            return 0
        self.sha.update(chunk)
        self.n += len(chunk)
        b[:len(chunk)] = chunk
        return len(chunk)


KEEP = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Lsa_summary"]


def stream_subset(path: str, windows: list[tuple[pd.Timestamp, pd.Timestamp]], max_summary: int = 600) -> dict:
    """Stream one FNSPID CSV once (never stored whole) and keep rows whose UTC timestamp falls inside any
    [start, end] window. Returns the kept rows plus full-file coverage statistics and the stream's SHA256."""
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    starts = np.array([w[0].value for w in windows])
    ends = np.array([w[1].value for w in windows])
    order = np.argsort(starts)
    starts, ends = starts[order], np.maximum.accumulate(ends[order])
    r = requests.get(RESOLVE.format(path=path), stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()
    hr = _HashingReader(r.raw)
    text = io.TextIOWrapper(io.BufferedReader(hr, buffer_size=1 << 20), encoding="utf-8", errors="replace",
                            newline="")
    reader = csv.reader(text)
    header = next(reader)
    ix = {c: header.index(c) for c in KEEP}
    kept, n_rows, n_bad = [], 0, 0
    years: dict[str, int] = {}
    tmin = tmax = None
    for row in reader:
        n_rows += 1
        if len(row) != len(header):
            n_bad += 1
            continue
        ts = pd.to_datetime(row[ix["Date"]], utc=True, errors="coerce")
        if ts is pd.NaT or pd.isna(ts):
            n_bad += 1
            continue
        y = str(ts.year)
        years[y] = years.get(y, 0) + 1
        tmin = ts if tmin is None or ts < tmin else tmin
        tmax = ts if tmax is None or ts > tmax else tmax
        v = ts.value
        k = np.searchsorted(starts, v, side="right") - 1
        if k >= 0 and v <= ends[k]:
            rec = {c: row[ix[c]] for c in KEEP}
            rec["Lsa_summary"] = rec["Lsa_summary"][:max_summary]
            rec["published_utc"] = ts
            kept.append(rec)
    df = pd.DataFrame(kept)
    return {"rows": df, "n_rows_total": n_rows, "n_rows_unparsed": n_bad, "bytes": hr.n,
            "sha256": hr.sha.hexdigest(), "min_utc": str(tmin), "max_utc": str(tmax), "rows_per_year": years}


EXPLAINED_METHODS = ["bayes_logit:combined", "bayes_logit:controls", "vix", "inst_composite", "dcc_var"]


def evidence_dates(run_dir, methods: list[str] = EXPLAINED_METHODS) -> pd.DatetimeIndex:
    """Decision dates that need news evidence: start of every alert episode (primary rule, merged as in the
    evaluation) for the explained methods, plus every retrospective stress onset in the test span."""
    from pathlib import Path

    from riskcausal.evaluation.metrics import alert_episodes
    from riskcausal.pipeline import MERGE_GAP

    run_dir = Path(run_dir)
    P = pd.read_parquet(run_dir / "predictions.parquet")
    dates = set()
    for m in methods:
        g = P[P["method"] == m].sort_values("date")
        d = g["date"].to_numpy()
        for s, _ in alert_episodes(g["alert"].to_numpy(bool), MERGE_GAP):
            dates.add(pd.Timestamp(d[s]))
    L = pd.read_parquet(run_dir / "labels_eval.parquet")
    lo, hi = P["date"].min(), P["date"].max()
    dates |= {d for d in L.index[L["onset"] == 1] if lo <= d <= hi}
    return pd.DatetimeIndex(sorted(dates))


def news_windows(dates: pd.DatetimeIndex, lookback_days: int = 7) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[date - lookback, date + 1 day) in UTC. Deliberately wider than the cutoff: the exact point-in-time rule
    (published at or before 16:00 New York on the decision date) is enforced at retrieval, not here."""
    return [((d - pd.Timedelta(days=lookback_days)).tz_localize("UTC"), (d + pd.Timedelta(days=1)).tz_localize("UTC"))
            for d in dates]
