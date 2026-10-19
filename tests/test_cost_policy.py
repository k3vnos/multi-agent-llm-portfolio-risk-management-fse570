"""COST_POLICY.md enforcement: no paid-LLM SDKs, local-only defaults, no deployment."""

import re
from pathlib import Path

import pytest

from riskcausal.config import PROJECT_ROOT, load_config

PAID_SDKS = ["anthropic", "openai", "cohere", "mistralai", "google-generativeai", "google.generativeai",
             "langchain_openai", "langchain-anthropic", "groq", "together", "replicate"]


def test_no_paid_llm_sdk_in_dependencies():
    text = (PROJECT_ROOT / "pyproject.toml").read_text().lower()
    deps = re.findall(r'"([a-z0-9_.\-]+)\s*[<>=!~\[]', text)
    for sdk in PAID_SDKS:
        assert sdk not in deps, f"paid-LLM SDK {sdk!r} found in pyproject.toml"


def test_no_source_file_imports_paid_sdk():
    pat = re.compile(r"^\s*(import|from)\s+(" + "|".join(re.escape(s) for s in PAID_SDKS) + r")\b", re.MULTILINE)
    for py in list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "dashboard").rglob("*.py")):
        assert not pat.search(py.read_text()), f"{py} imports a paid-LLM SDK"


@pytest.mark.parametrize("profile", ["dev", "full"])
def test_defaults_are_local_and_free(profile):
    cfg = load_config(profile)
    assert cfg.agents.enabled is False
    assert cfg.agents.backend == "disabled"
    assert cfg.api.host in {"127.0.0.1", "localhost", "::1"}
    assert cfg.deploy == "none"


def test_non_local_api_host_rejected():
    with pytest.raises(ValueError):
        load_config("dev", overrides={"api": {"host": "0.0.0.0"}})


def test_env_example_has_no_paid_keys():
    text = (PROJECT_ROOT / ".env.example").read_text()
    keys = re.findall(r"^([A-Z_]+)=", text, re.MULTILINE)
    assert set(keys) <= {"FRED_API_KEY", "OLLAMA_HOST"}, keys
    assert "localhost" in text or "127.0.0.1" in text


def test_no_git_remote_configured():
    cfg = Path(PROJECT_ROOT / ".git" / "config")
    if cfg.exists():
        assert "[remote" not in cfg.read_text(), "local-only policy: no git remotes"
