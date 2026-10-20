"""Configuration: YAML (base + universe + profile) merged into typed pydantic models; secrets from env."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class PitCfg(BaseModel):
    macro_release_lag_bdays: int = 1
    daily_fallback_lag_bdays: int = 3


class GraphCfg(BaseModel):
    window: int = 250
    step: int = 5
    max_lag: int = 5
    lag_criterion: Literal["bic", "aic", "fixed"] = "bic"
    prune: bool = True
    edge_threshold: float = 0.05


class GrangerCfg(BaseModel):
    max_lag: int = 5
    alpha: float = 0.05


class RegimeCfg(BaseModel):
    vol_window: int = 21
    n_regimes: int = 2
    min_calm_days: int = 10
    min_stress_days: int = 5
    target_series: str | None = None   # exploratory alternative target (e.g. KRE); controls keep regime_series


class TargetCfg(BaseModel):
    horizon: int = 10


class WalkForwardCfg(BaseModel):
    min_train_years: int = 5
    test_block: Literal["year"] = "year"
    embargo_days: int = 5


class AlertCfg(BaseModel):
    max_false_alarms_per_year: float = 4
    max_alert_day_share: float | None = 0.10      # M3 post-hoc rule; None = original H0 rule only


class AgentsCfg(BaseModel):
    enabled: bool = False
    backend: Literal["disabled", "ollama"] = "disabled"
    model: str = "qwen2.5:7b"


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("host")
    @classmethod
    def _local_only(cls, v: str) -> str:
        if v not in LOCAL_HOSTS:
            raise ValueError(f"api.host must be local (COST_POLICY.md); got {v!r}")
        return v


class GraphNodes(BaseModel):
    etfs: list[str]
    macro: list[str]


class Config(BaseModel):
    seed: int = 570
    data_dir: str = "data"
    output_dir: str = "outputs"
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"
    profile: str = "dev"
    experiment: str | None = None
    etfs: list[str]
    fred_series: list[str]
    graph_nodes: GraphNodes
    control_features: list[str]
    regime_series: str = "SPY"
    universe: dict[str, Any]
    pit: PitCfg = PitCfg()
    graph: GraphCfg = GraphCfg()
    granger: GrangerCfg = GrangerCfg()
    regime: RegimeCfg = RegimeCfg()
    target: TargetCfg = TargetCfg()
    walkforward: WalkForwardCfg = WalkForwardCfg()
    alerts: AlertCfg = AlertCfg()
    agents: AgentsCfg = AgentsCfg()
    api: ApiCfg = ApiCfg()
    deploy: Literal["none"] = "none"

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def output_path(self) -> Path:
        return PROJECT_ROOT / self.output_dir

    def hash(self) -> str:
        """Stable short hash of the full configuration (used in run ids / cache keys)."""
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(profile: str = "dev", overrides: dict | None = None, config_dir: Path = CONFIG_DIR,
                experiment: str | None = None) -> Config:
    """base.yaml <- profiles/<profile>.yaml <- experiments/<experiment>.yaml <- overrides."""
    base = yaml.safe_load((config_dir / "base.yaml").read_text())
    universe = yaml.safe_load((config_dir / "universe.yaml").read_text())
    prof = yaml.safe_load((config_dir / "profiles" / f"{profile}.yaml").read_text())
    merged = _deep_merge(base, prof)
    if experiment:                      # comma-separated overlays, applied in order
        for name in experiment.split(","):
            exp = yaml.safe_load((config_dir / "experiments" / f"{name}.yaml").read_text())
            exp.pop("description", None)
            merged = _deep_merge(merged, exp)
        merged["experiment"] = experiment.replace(",", "+")
    if overrides:
        merged = _deep_merge(merged, overrides)
    if merged.get("etfs") == "all":
        merged["etfs"] = list(universe["etfs"])
    if merged.get("fred_series") == "all":
        merged["fred_series"] = list(universe["fred"])
    merged["universe"] = universe
    return Config(**merged)


def fred_api_key() -> str | None:
    """Free FRED key from .env / environment. Never logged."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key or None
