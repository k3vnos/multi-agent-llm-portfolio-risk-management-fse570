"""FRED current-vintage series (public CSV, no key) and ALFRED real-time vintages (free API key)."""

from __future__ import annotations

import io
import logging
import re
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"


def redact(text: str) -> str:
    """Strip API keys from any message before it is logged, saved or raised."""
    return re.sub(r"(api_key=)[^&\s'\"]+", r"\1<redacted>", str(text))


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = redact(e)
            log.warning("GET %s attempt %d/%d failed: %s", url.split("?")[0], attempt, retries, last)
            time.sleep(2 * attempt)
    raise RuntimeError(f"request failed: {last}") from None


def download_current(series_id: str) -> pd.Series:
    """Latest-vintage observations. Only point-in-time safe for series that are never revised."""
    r = _get(FREDGRAPH.format(sid=series_id))
    df = pd.read_csv(io.StringIO(r.text), na_values=["."])
    df.columns = ["date", series_id]
    s = df.set_index(pd.to_datetime(df["date"]))[series_id].astype(float)
    s.index.name = "date"
    return s


def download_first_releases(series_id: str, api_key: str, first_year: int, last_date: str) -> pd.Series:
    """Date each observation first appeared on FRED (ALFRED `output_type=4`), indexed by observation date.

    ALFRED caps a request at 2000 vintage dates, so the real-time axis is queried one year at a time. Years before
    ALFRED started tracking the series return 400 "does not exist in ALFRED" and are skipped; observations whose
    first release is untracked are simply absent from the result.
    """
    rows: list[dict] = []
    last_year = int(last_date[:4])
    for y in range(first_year, last_year + 1):
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "output_type": 4,
                  "realtime_start": f"{y}-01-01", "realtime_end": min(f"{y}-12-31", last_date),
                  "observation_start": f"{y - 1}-01-01", "observation_end": f"{y}-12-31"}
        r = requests.get(FRED_API, params=params, timeout=120)
        if r.status_code == 400 and "does not exist in ALFRED" in r.text:
            continue
        if r.status_code != 200:
            raise RuntimeError(f"ALFRED first-release request failed for {series_id} {y}: "
                               f"{r.status_code} {redact(r.text)[:200]}")
        rows.extend(o for o in r.json().get("observations", []) if o["value"] != ".")
        time.sleep(0.2)
    if not rows:
        raise RuntimeError(f"ALFRED has no tracked releases for {series_id}")
    df = pd.DataFrame(rows)
    first = pd.to_datetime(df["realtime_start"]).groupby(pd.to_datetime(df["date"])).min()
    first.index.name = "date"
    return first.rename("first_release")


def download_vintages(series_id: str, api_key: str, start: str = "1990-01-01") -> pd.DataFrame:
    """All ALFRED releases: one row per (observation date, realtime_start) with the value published then.

    Columns: date, realtime_start, realtime_end, value. A value is knowable from `realtime_start` onward.
    """
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "realtime_start": start, "realtime_end": "9999-12-31", "observation_start": start,
        "limit": 100000, "offset": 0,
    }
    rows: list[dict] = []
    while True:
        js = _get(FRED_API, params=params).json()
        obs = js.get("observations", [])
        rows.extend(obs)
        if len(obs) < params["limit"]:
            break
        params["offset"] += params["limit"]
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"ALFRED returned no vintages for {series_id}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("date", "realtime_start"):
        df[c] = pd.to_datetime(df[c])
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2262-04-11"))
    return df[["date", "realtime_start", "realtime_end", "value"]].dropna(subset=["value"])
