"""Provenance: the recorded git sha is the code state when the run started (M4 finding), not when it ended."""

import ast
import inspect
import json

import riskcausal.pipeline as pl
from riskcausal.config import load_config
from riskcausal.registry import new_run_dir


def test_new_run_dir_records_the_given_start_sha(tmp_path):
    cfg = load_config("dev", overrides={"output_dir": str(tmp_path)})
    d = new_run_dir(cfg, tag="t", sha="abc123-start")
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["git_sha"] == "abc123-start" and prov["synthetic"] is False


def test_run_pipeline_captures_sha_before_any_work():
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(pl))) if isinstance(n, ast.FunctionDef)
              and n.name == "run_pipeline")
    calls = {getattr(n.func, "id", getattr(n.func, "attr", "")): n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Call)}
    assert calls["git_sha"] < min(calls["build_panel"], calls["build_graphs"], calls["run_walkforward"])
    kw = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "new_run_dir")
    assert any(k.arg == "sha" for k in kw.keywords)
