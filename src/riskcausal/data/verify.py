"""M1 dataset verification: actually download every candidate source and record what exists.

Writes the raw snapshot (data/raw/<source>/<snapshot>/...), a JSON verification report, and
DATASET_MANIFEST.md generated from that report. Nothing is assumed to exist; failures are recorded.
"""

from __future__ import annotations

import logging
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from riskcausal.config import PROJECT_ROOT, Config, fred_api_key
from riskcausal.data import ingest_fnspid, ingest_fred, ingest_yf
from riskcausal.data.calendar import nyse_sessions
from riskcausal.registry import sha256_file, write_json

log = logging.getLogger(__name__)
EARLIEST = "1993-01-01"  # request max history; report coverage relative to cfg.start_date


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}".splitlines()[0][:300]


def verify_etfs(cfg: Config, out_dir: Path) -> list[dict]:
    sessions = nyse_sessions(EARLIEST, pd.Timestamp.today().strftime("%Y-%m-%d"))
    rows = []
    for t in cfg.universe["etfs"]:
        rec: dict = {"ticker": t, "name": cfg.universe["etfs"][t]["name"]}
        try:
            df = ingest_yf.download_etf(t, start=EARLIEST)
            path = out_dir / f"{t}.parquet"
            df.to_parquet(path)
            first, last = df.index.min(), df.index.max()
            span_start = max(first, pd.Timestamp(cfg.start_date))
            expected = sessions[(sessions >= span_start) & (sessions <= last)]
            have = df.index[(df.index >= span_start)]
            close = df["close"]
            lr = np.log(close).diff()
            rec.update(
                status="ok", first_date=first.date(), last_date=last.date(),
                starts_by_study_start=bool(first <= pd.Timestamp(cfg.start_date) + pd.Timedelta(days=7)),
                n_rows_since_start=len(have), n_expected_sessions=len(expected),
                missing_session_rate=round(1 - len(expected.intersection(have)) / max(len(expected), 1), 5),
                extra_non_session_rows=len(have.difference(sessions)),
                nan_close=int(close.isna().sum()), zero_volume_days=int((df["volume"] == 0).sum()),
                max_abs_daily_logret=round(float(lr.abs().max()), 4),
                adjusted="auto_adjust=True (splits+dividends)",
                file=str(path.relative_to(PROJECT_ROOT)), sha256=sha256_file(path),
            )
        except Exception as e:  # noqa: BLE001
            rec.update(status="FAILED", error=_err(e))
            log.error("ETF %s failed: %s", t, e)
        rows.append(rec)
        log.info("ETF %-4s %s", t, rec["status"])
    return rows


def verify_fred(cfg: Config, out_dir: Path, alfred_dir: Path) -> list[dict]:
    key = fred_api_key()
    rows = []
    for sid, meta in cfg.universe["fred"].items():
        rec: dict = {"series_id": sid, "name": meta["name"], "freq": meta["freq"], "revised": meta["revised"]}
        try:
            s = ingest_fred.download_current(sid)
            path = out_dir / f"{sid}.parquet"
            s.to_frame().to_parquet(path)
            valid = s.dropna()
            since = s[s.index >= pd.Timestamp(cfg.start_date)]
            rec.update(
                status="ok", first_date=valid.index.min().date(), last_date=valid.index.max().date(),
                n_obs=int(valid.size),
                missing_rate_since_start=round(float(since.isna().mean()), 5) if len(since) else None,
                covers_study_start=bool(valid.index.min() <= pd.Timestamp(cfg.start_date)),
                file=str(path.relative_to(PROJECT_ROOT)), sha256=sha256_file(path),
            )
        except Exception as e:  # noqa: BLE001
            rec.update(status="FAILED", error=_err(e))
        # ALFRED vintages (needed for revised series)
        if not meta["revised"]:
            rec["alfred"] = "not needed (series not revised)"
        elif key is None:
            rec["alfred"] = "PENDING — FRED_API_KEY not set (free key required)"
        else:
            try:
                v = ingest_fred.download_vintages(sid, key)
                vpath = alfred_dir / f"{sid}_vintages.parquet"
                v.to_parquet(vpath)
                rec["alfred"] = {
                    "status": "ok", "n_rows": len(v), "n_vintages": int(v["realtime_start"].nunique()),
                    "first_vintage": v["realtime_start"].min().date(),
                    "last_vintage": v["realtime_start"].max().date(),
                    "file": str(vpath.relative_to(PROJECT_ROOT)), "sha256": sha256_file(vpath),
                }
            except Exception as e:  # noqa: BLE001
                rec["alfred"] = {"status": "FAILED", "error": _err(e)}
        rows.append(rec)
        log.info("FRED %-13s %s", sid, rec.get("status"))
    return rows


def verify_fnspid() -> dict:
    rec: dict = {"repo": ingest_fnspid.REPO}
    try:
        rec["info"] = ingest_fnspid.repo_info()
        rec["files"] = ingest_fnspid.list_files()
        sizes = {f["path"]: f["size_bytes"] for f in rec["files"]}
        samples = {}
        for path in ("Stock_news/All_external.csv", "Stock_news/nasdaq_exteral_data.csv"):
            if path not in sizes:
                samples[path] = {"status": "FAILED", "error": "file not listed"}
                continue
            head = ingest_fnspid.sample_csv(path, nbytes=3_000_000)
            cols = list(head.columns)
            date_col = next((c for c in cols if c.lower() == "date"), None)
            # Probe dates at evenly spaced byte offsets (a sample, not a full scan).
            probes = []
            for frac in np.linspace(0, 0.995, 21):
                off = int(sizes[path] * frac)
                try:
                    df = ingest_fnspid.sample_csv(path, nbytes=400_000, offset=off, header=cols if off else None)
                    d = pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed").dropna() if date_col else []
                    probes.append({"offset_frac": round(float(frac), 3), "n_rows": len(df),
                                   "min_date": str(d.min().date()) if len(d) else None,
                                   "max_date": str(d.max().date()) if len(d) else None,
                                   "tickers": sorted(map(str, df.get("Stock_symbol", pd.Series()).dropna()
                                                         .unique()))[:5]})
                except Exception as e:  # noqa: BLE001
                    probes.append({"offset_frac": round(float(frac), 3), "error": _err(e)})
            all_min = [p["min_date"] for p in probes if p.get("min_date")]
            all_max = [p["max_date"] for p in probes if p.get("max_date")]
            samples[path] = {
                "status": "ok", "size_gb": round(sizes[path] / 1e9, 2), "columns": cols,
                "sampled_min_date": min(all_min) if all_min else None,
                "sampled_max_date": max(all_max) if all_max else None,
                "probes": probes,
            }
        rec["samples"] = samples
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec.update(status="FAILED", error=_err(e), trace=traceback.format_exc()[-800:])
    return rec


def verify_gdelt() -> dict:
    url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return {"status": "ok", "endpoint": url, "sample": r.text.splitlines()[0][:160],
                "note": "GDELT 2.0 event/GKG files reachable; historical coverage from 2015-02 (GDELT 2.0)."}
    except Exception as e:  # noqa: BLE001
        return {"status": "FAILED", "endpoint": url, "error": _err(e)}


def run_verification(cfg: Config) -> Path:
    snap = date.today().isoformat()
    raw = cfg.data_path / "raw"
    dirs = {k: raw / k / snap for k in ("yfinance", "fred", "alfred")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    report = {
        "snapshot": snap, "study_start": cfg.start_date,
        "etfs": verify_etfs(cfg, dirs["yfinance"]),
        "fred": verify_fred(cfg, dirs["fred"], dirs["alfred"]),
        "fnspid": verify_fnspid(),
        "gdelt": verify_gdelt(),
        "fred_api_key_present": fred_api_key() is not None,
    }
    jpath = raw / f"verification_{snap}.json"
    write_json(jpath, report)
    write_json(raw / "latest_snapshot.json", {"snapshot": snap})
    from riskcausal.data.manifest import write_manifest
    write_manifest(report, PROJECT_ROOT / "DATASET_MANIFEST.md", jpath)
    return jpath


def render_latest(cfg: Config) -> Path:
    from riskcausal.data.manifest import write_manifest
    from riskcausal.registry import read_json

    raw = cfg.data_path / "raw"
    snap = read_json(raw / "latest_snapshot.json")["snapshot"]
    jpath = raw / f"verification_{snap}.json"
    write_manifest(read_json(jpath), PROJECT_ROOT / "DATASET_MANIFEST.md", jpath)
    return jpath
