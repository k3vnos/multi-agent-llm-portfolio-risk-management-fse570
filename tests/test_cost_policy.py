"""COST_POLICY.md enforcement: no paid-LLM SDKs, local-only defaults, no deployment."""

import re
import subprocess

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


# The one remote approved in COST_POLICY.md §3 (private team repository, authorised 2026-11-06). Any other remote
# fails this test until the policy names it.
APPROVED_REPO = "k3vnos/multi-agent-llm-portfolio-risk-management-fse570"
_REMOTE_URL = re.compile(r"^(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)"
                         r"(?P<repo>[^/]+/[^/]+?)(?:\.git)?/?$")


def remote_is_approved(url: str) -> bool:
    m = _REMOTE_URL.match(url.strip())
    return bool(m) and m["repo"].lower() == APPROVED_REPO.lower()


@pytest.mark.parametrize("url,ok", [
    (f"https://github.com/{APPROVED_REPO}.git", True),
    (f"https://github.com/{APPROVED_REPO}", True),
    (f"git@github.com:{APPROVED_REPO}.git", True),
    (f"ssh://git@github.com/{APPROVED_REPO}.git", True),
    ("https://github.com/someone-else/multi-agent-llm-portfolio-risk-management-fse570.git", False),
    (f"https://gitlab.com/{APPROVED_REPO}.git", False),
    (f"https://github.com/{APPROVED_REPO}-copy.git", False),
    ("/Users/someone/portfolio-risk-capstone", False),
])
def test_remote_url_matcher(url, ok):
    assert remote_is_approved(url) is ok


def test_only_approved_git_remote_configured():
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "remote", "-v"], capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git not available")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    urls = {line.split()[1] for line in out.stdout.splitlines() if line.strip()}
    bad = sorted(u for u in urls if not remote_is_approved(u))
    assert not bad, f"unapproved git remote(s) {bad}; only {APPROVED_REPO} is approved (COST_POLICY.md §3)"
