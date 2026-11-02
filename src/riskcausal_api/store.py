"""Read-only access to saved run artifacts for the API. Nothing here writes a file, runs the pipeline or calls an
LLM. Runs are listed in configs/dashboard.yaml."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from riskcausal.config import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
DASH_CFG = PROJECT_ROOT / "configs" / "dashboard.yaml"
MERGE_GAP = 5          # = riskcausal.pipeline.MERGE_GAP (copied to avoid importing the pipeline; equality tested)
REPRO_FILES = ["predictions.parquet", "predictions_val.parquet", "labels_eval.parquet", "features.parquet",
               "folds.json"]


def dashboard_config() -> dict:
    return yaml.safe_load(DASH_CFG.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Run:
    def __init__(self, run_id: str, role: str):
        self.run_id, self.role = run_id, role
        self.dir = RUNS_DIR / run_id
        if not self.dir.is_dir():
            raise FileNotFoundError(run_id)

    def _json(self, name: str):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    @cached_property
    def provenance(self) -> dict:
        return self._json("provenance.json")

    @cached_property
    def metrics(self) -> dict:
        return self._json("metrics.json")

    @cached_property
    def ci(self) -> dict | None:
        return self._json("ci.json")

    @cached_property
    def config(self) -> dict:
        return yaml.safe_load((self.dir / "config.yaml").read_text())

    @cached_property
    def predictions(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "predictions.parquet")

    @cached_property
    def labels(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "labels_eval.parquet")

    @cached_property
    def features(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "features.parquet")

    @cached_property
    def graphs(self) -> dict:
        gdir = Path(self._json("graph_cache.json")["dir"])
        z = np.load(gdir / "varlingam.npz", allow_pickle=False)
        return {"ends": pd.DatetimeIndex(z["ends"]), "nodes": [str(n) for n in z["nodes"]], "B": z["B"]}

    @property
    def audit_passed(self) -> bool:
        return bool((self.metrics or {}).get("audit", {}).get("passed"))

    @property
    def synthetic(self) -> bool:
        return bool(self.provenance.get("synthetic"))

    def header(self, kind: str) -> dict:
        return {"run_id": self.run_id, "role": self.role, "git_sha": self.provenance.get("git_sha"),
                "synthetic": self.synthetic, "audit_passed": self.audit_passed, "kind": kind}


@lru_cache(maxsize=1)
def runs() -> dict[str, Run]:
    cfg = dashboard_config()["runs"]
    return {rid: Run(rid, role) for role, rid in cfg.items() if role != "primary_twin"}


@lru_cache(maxsize=1)
def reproducibility() -> dict:
    """Primary run vs its re-run: SHA256 of every result file (computed live; nothing is written)."""
    cfg = dashboard_config()["runs"]
    a, b = RUNS_DIR / cfg["primary"], RUNS_DIR / cfg.get("primary_twin", "")
    if not b.is_dir():
        return {"status": "not checked", "files": {}}
    files = {f: _sha(a / f) == _sha(b / f) for f in REPRO_FILES}
    ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
    for m in (ma, mb):
        m.pop("runtime_secs", None)
    files["metrics.json (excl. runtime)"] = ma == mb
    return {"status": "bit-identical" if all(files.values()) else "DIFFERENT", "twin_run_id": cfg["primary_twin"],
            "files": files}
