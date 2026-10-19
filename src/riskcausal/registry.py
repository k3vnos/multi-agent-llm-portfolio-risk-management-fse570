"""Run registry: every reported number lives in outputs/runs/<run_id>/ with its config and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from riskcausal.config import PROJECT_ROOT, Config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], check=False, cwd=PROJECT_ROOT, capture_output=True, text=True)
        sha = out.stdout.strip() or "uncommitted"
        dirty = subprocess.run(["git", "status", "--porcelain"], check=False, cwd=PROJECT_ROOT, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def new_run_dir(cfg: Config, tag: str = "") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{cfg.profile}_{cfg.hash()}" + (f"_{tag}" if tag else "")
    run_dir = cfg.output_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    write_json(run_dir / "provenance.json", {
        "run_id": run_id, "created_utc": stamp, "git_sha": git_sha(), "config_hash": cfg.hash(),
        "synthetic": False,
    })
    return run_dir


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, sort_keys=False))


def read_json(path: Path):
    return json.loads(Path(path).read_text())
