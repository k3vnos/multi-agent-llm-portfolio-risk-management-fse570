"""LLM backends. Local Ollama only (COST_POLICY.md): any non-local host is refused and there is no fallback to
another provider. `DisabledBackend` is the default and raises instead of calling anything."""

from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.parse import urlparse

import requests

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LLMBackend(Protocol):
    name: str

    def complete_json(self, system: str, prompt: str) -> tuple[dict, dict]:
        """Return (parsed JSON object, metadata such as model digest and token counts)."""


class AgentsDisabledError(RuntimeError):
    pass


class DisabledBackend:
    name = "disabled"

    def complete_json(self, system: str, prompt: str) -> tuple[dict, dict]:
        raise AgentsDisabledError("agents are disabled (agents.enabled: false); no model is called")


class OllamaBackend:
    """Ollama REST API on this machine. Deterministic decoding: temperature 0, fixed seed, JSON output."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str | None = None, seed: int = 570, timeout: int = 600,
                 num_ctx: int = 8192):
        host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if "://" not in host:
            host = "http://" + host
        if urlparse(host).hostname not in LOCAL_HOSTS:
            raise ValueError(f"Ollama host must be local (COST_POLICY.md); got {host!r}")
        self.base, self.model, self.seed, self.timeout, self.num_ctx = host.rstrip("/"), model, seed, timeout, num_ctx

    def model_digest(self) -> str:
        r = requests.get(f"{self.base}/api/tags", timeout=30)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m["name"] == self.model:
                return m["digest"]
        raise RuntimeError(f"model {self.model} not pulled locally")

    def complete_json(self, system: str, prompt: str) -> tuple[dict, dict]:
        body = {"model": self.model, "system": system, "prompt": prompt, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx}}
        r = requests.post(f"{self.base}/api/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        meta = {"model": self.model, "eval_count": js.get("eval_count"), "prompt_eval_count": js.get("prompt_eval_count"),
                "total_secs": round(js.get("total_duration", 0) / 1e9, 2), "raw": js.get("response", "")}
        try:
            return json.loads(js.get("response", "")), meta
        except json.JSONDecodeError:
            return {"claims": [], "abstain": True, "abstain_reason": "model output was not valid JSON"}, meta
