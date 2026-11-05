"""Live "Ask the Agents" (M10): runs the v4 Graph/Macro/News/Synthesis agents on the local Ollama model for a
historical replay date. This is live AI processing of HISTORICAL data, not real-time market monitoring.

Routes (the only non-GET routes in the API; each only starts or stops a local model run — nothing is written to
the run directories and nothing changes a score, alert or metric):
  GET  /agents/status            Ollama reachable? model pulled?
  POST /agents/run               start a job {date, method?, portfolio?}; one job at a time
  GET  /agents/jobs/{id}         progress events + the full trace when finished
  POST /agents/jobs/{id}/cancel  stop the job (closes the model stream)
  GET  /agents/saved/{date}      the saved v4 explanation for an alert episode start date, if one exists
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from riskcausal_api.replay import _date, _explained, _primary, json_safe

router = APIRouter(prefix="/agents", tags=["agents (live, local LLM)"])

JOB_TIMEOUT = 300           # seconds for the whole four-agent run
CALL_TIMEOUT = 150          # seconds for one model call
NOTICE = ("Live AI processing of historical data: the local model reads the evidence that was available at the "
          "close of the selected past date. It is not real-time market monitoring, and it never changes the risk "
          "score or alert status.")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_backend_factory = None      # tests replace this with a fake backend


def _make_backend():
    from riskcausal_agents.backends import OllamaBackend

    if _backend_factory is not None:
        return _backend_factory()
    return OllamaBackend(timeout=CALL_TIMEOUT)


def _v4_dir() -> Path:
    from riskcausal.config import PROJECT_ROOT

    return PROJECT_ROOT / "outputs" / "agents_v4"


class RunRequest(BaseModel):
    date: str
    method: str | None = None
    portfolio: str | None = None


@router.get("/status")
def status() -> dict:
    b = _make_backend()
    ok, info = b.available() if hasattr(b, "available") else (True, "test backend")
    busy = any(j["status"] in ("queued", "running") for j in _jobs.values())
    return {"available": ok, "model": getattr(b, "model", b.name), "digest" if ok else "reason": info,
            "busy": busy, "notice": NOTICE,
            "fallback": "Saved evidence-based explanations and the deterministic briefing stay available."}


def _worker(job: dict) -> None:
    from riskcausal_agents.backends import Cancelled
    from riskcausal_agents.structured import Control, Stop, explain, summarise
    from riskcausal_api.main import _run_evidence

    t0 = time.time()

    def emit(ev: dict) -> None:
        ev = {**ev, "t": round(time.time() - t0, 1)}
        with _lock:
            if ev.get("type") == "tokens" and job["events"] and job["events"][-1].get("type") == "tokens":
                job["events"][-1] = ev                         # keep one rolling token counter, not thousands
            else:
                job["events"].append(ev)

    def cancelled() -> bool:
        if time.time() - t0 > JOB_TIMEOUT:
            job["timed_out"] = True
            return True
        return job["cancel"]

    job["status"] = "running"
    try:
        r = _primary()
        b = _run_evidence(r.run_id).bundle(job["method"], job["date"])
        extra, ctx = [], ""
        if job["portfolio"]:
            from riskcausal_api.portfolio import analyze, context_line, portfolio_facts

            emit({"type": "stage", "stage": "portfolio", "detail": "computing point-in-time portfolio facts"})
            pa = analyze(job["portfolio"], job["date"])
            extra, ctx = portfolio_facts(pa), context_line(pa)
        backend = _make_backend()
        digest = None
        if hasattr(backend, "available"):
            ok, digest = backend.available()
            if not ok:
                raise RuntimeError(f"local model unavailable: {digest}")
        emit({"type": "stage", "stage": "evidence", "detail": f"{len(b.items)} point-in-time evidence items loaded"})
        rec = explain(b, backend, float(r.config["graph"]["edge_threshold"]), extra_facts=extra, context=ctx,
                      model_digest=digest, ctl=Control(emit=emit, cancelled=cancelled))
        rec["summary"] = summarise(rec)
        rec["portfolio"] = job["portfolio"]
        job["result"], job["status"] = rec, "done"
    except (Stop, Cancelled):
        job["status"] = "timeout" if job.get("timed_out") else "cancelled"
    except Exception as e:  # noqa: BLE001 — reported to the UI; the saved explanations remain available
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"[:400]
    finally:
        job["finished"] = time.time()
        emit({"type": "end", "status": job["status"]})


@router.post("/run")
def run(req: RunRequest) -> dict:
    t = _date(req.date)
    method = req.method or _explained()
    if req.portfolio:
        from riskcausal_api.portfolio import parse

        try:
            parse(req.portfolio)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
    with _lock:
        if any(j["status"] in ("queued", "running") for j in _jobs.values()):
            raise HTTPException(409, "another agent run is in progress; cancel it or wait")
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "date": str(t.date()), "method": method, "portfolio": req.portfolio, "status": "queued",
               "events": [], "cancel": False, "created": time.time(), "result": None, "error": None}
        _jobs[jid] = job
        for old in sorted(_jobs, key=lambda k: _jobs[k]["created"])[:-20]:   # keep the last 20 jobs in memory
            _jobs.pop(old, None)
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    return {"job_id": jid, "status": "queued", "notice": NOTICE}


@router.get("/jobs/{jid}")
@json_safe
def job(jid: str) -> dict:
    j = _jobs.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job")
    with _lock:
        return {k: j[k] for k in ("id", "date", "method", "portfolio", "status", "error", "result")} | {
            "events": list(j["events"]), "elapsed": round((j.get("finished") or time.time()) - j["created"], 1),
            "notice": NOTICE}


@router.post("/jobs/{jid}/cancel")
def cancel(jid: str) -> dict:
    j = _jobs.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job")
    j["cancel"] = True
    return {"id": jid, "status": j["status"], "cancel_requested": True}


@router.get("/saved/{date}")
@json_safe
def saved(date: str) -> dict:
    """Saved v4 explanation (generated offline by `riskcausal_agents.cli v4-eval --which all`) for this date."""
    d = str(_date(date).date())
    p = _v4_dir() / "all" / f"{d}.json"
    avail = sorted(x.stem for x in (_v4_dir() / "all").glob("*.json")) if (_v4_dir() / "all").is_dir() else []
    return {"date": d, "available_for": avail, "record": json.loads(p.read_text()) if p.exists() else None}
