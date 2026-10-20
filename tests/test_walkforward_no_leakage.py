"""Walk-forward leakage (LEAKAGE_AUDIT rows 5–8): fold geometry, plus end-to-end "perturb the future" tests on
synthetic data — changing anything after a cut-off must not change any prediction, threshold or fitted model
issued before it."""

import ast
import functools
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

import riskcausal.pipeline as pl
from riskcausal.config import load_config
from riskcausal.data.calendar import nyse_sessions
from riskcausal.regimes.markov import fit_ms
from riskcausal.validation.walkforward import make_folds


def test_folds_are_chronological_with_embargo_and_purge_room():
    s = nyse_sessions("2005-01-01", "2026-10-16")
    folds = make_folds(s, 2011, 2026, embargo=5, val_years=2)
    assert [f.test_year for f in folds] == list(range(2011, 2027))
    for f in folds:
        assert s.get_loc(f.test_start) - s.get_loc(f.train_end) == 6         # 5-session embargo between
        assert f.inner_end < f.val_start <= f.train_end < f.test_start <= f.test_end
        assert s.get_loc(f.val_start) - s.get_loc(f.inner_end) == 6
        assert f.test_start.year == f.test_end.year == f.test_year


def test_test_blocks_do_not_overlap():
    s = nyse_sessions("2005-01-01", "2026-10-16")
    folds = make_folds(s, 2011, 2026, embargo=5)
    for a, b in itertools.pairwise(folds):
        assert a.test_end < b.test_start
    assert pd.Timestamp(folds[0].train_end).year == 2010


# ------------------------------------------------------------------ end-to-end, synthetic (tests only)
MODELS = [("gbm", "controls"), ("gbm", "instability")]


def _synthetic(seed=0):
    sessions = nyse_sessions("2005-01-01", "2012-12-31")
    rng = np.random.default_rng(seed)
    n = len(sessions)
    reg, i = np.zeros(n, int), 0
    while i < n:
        L = int(rng.integers(30, 150))
        reg[i:i + L] = int(rng.integers(0, 2))
        i += L
    y = pd.Series(np.where(reg == 1, rng.normal(-1.3, 0.2, n), rng.normal(-2.3, 0.15, n)), index=sessions)
    cols = sorted(set(itertools.chain(*pl.FEATURE_SETS.values())) | set(pl.GC) | {f"{c}_ma4" for c in pl.GC})
    feats = pd.DataFrame(rng.normal(size=(n, len(cols))), index=sessions, columns=cols)
    feats["vix_level"] += 3 * pd.Series(reg, index=sessions).shift(-5, fill_value=0)   # some signal
    for c in ("vix", "nfci", "corr_var"):
        feats[c] = rng.normal(size=n)
    return feats, y


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("dev")
    fast_fit = functools.partial(fit_ms, n_starts=3)
    mp = pytest.MonkeyPatch()
    mp.setattr(pl, "fit_ms", fast_fit)
    feats, y = _synthetic()
    base = pl.run_walkforward(cfg, feats, y, models=MODELS)
    yield cfg, feats, y, base
    mp.undo()


def _fold_rows(P, k):
    return P[P["fold"] == k].reset_index(drop=True)


def test_perturbing_everything_after_a_fold_leaves_earlier_folds_unchanged(setup):
    cfg, feats, y, (P, V, info, _, _) = setup
    cut = pd.Timestamp(info[1]["test_end"])
    f2, y2 = feats.copy(), y.copy()
    f2.loc[f2.index > cut] = f2.loc[f2.index > cut] * 7 + 3
    y2.loc[y2.index > cut] = y2.loc[y2.index > cut] + 2
    P2, V2, info2, _, _ = pl.run_walkforward(cfg, f2, y2, models=MODELS)
    for k in (0, 1):
        pd.testing.assert_frame_equal(_fold_rows(P, k), _fold_rows(P2, k))
        pd.testing.assert_frame_equal(_fold_rows(V, k), _fold_rows(V2, k))
        assert info[k] == info2[k]
    assert not _fold_rows(P, 2)["score"].equals(_fold_rows(P2, 2)["score"])      # the perturbation did bite


def test_labels_after_train_end_never_reach_that_fold(setup):
    """Row 6/7: the regime series after train_end (which defines future targets) cannot change fold k at all."""
    cfg, feats, y, (P, V, info, _, _) = setup
    k = 2
    y2 = y.copy()
    y2.loc[y2.index > pd.Timestamp(info[k]["train_end"])] *= -1
    P2, V2, info2, _, _ = pl.run_walkforward(cfg, feats, y2, models=MODELS)
    pd.testing.assert_frame_equal(_fold_rows(P, k), _fold_rows(P2, k))
    pd.testing.assert_frame_equal(_fold_rows(V, k), _fold_rows(V2, k))     # validation labels too
    assert info[k]["methods"] == info2[k]["methods"]


def test_test_block_features_never_reach_thresholds_or_scaling(setup):
    """Rows 5/8: thresholds and the composite's standardisation come from validation/training rows only."""
    cfg, feats, y, (P, _, info, _, _) = setup
    k = 2
    ts, te = pd.Timestamp(info[k]["test_start"]), pd.Timestamp(info[k]["test_end"])
    f2 = feats.copy()
    blk = (f2.index >= ts) & (f2.index <= te)
    f2.loc[blk] = f2.loc[blk] * 10 - 4
    P2, _, info2, _, _ = pl.run_walkforward(cfg, f2, y, models=MODELS)
    a, b = _fold_rows(P, k), _fold_rows(P2, k)
    tcols = [c for c in a.columns if c.startswith("threshold")]
    pd.testing.assert_frame_equal(a[tcols], b[tcols])
    assert {m: v["thresholds"] for m, v in info[k]["methods"].items()} == \
        {m: v["thresholds"] for m, v in info2[k]["methods"].items()}


def test_both_alert_rules_are_reported_and_budget_rule_is_stricter(setup):
    _, _, _, (P, _, _, _, _) = setup
    assert {"threshold_h0", "threshold_m3", "alert_h0", "alert_m3"} <= set(P.columns)
    ok = np.isfinite(P["threshold_h0"]) & np.isfinite(P["threshold_m3"])
    assert (P.loc[ok, "threshold_m3"] >= P.loc[ok, "threshold_h0"] - 1e-12).all()


# ------------------------------------------------------------------ structural boundary (row 6)
def test_walkforward_cannot_see_retrospective_labels():
    assert "label_eval" not in inspect.signature(pl.run_walkforward).parameters
    tree = ast.parse(inspect.getsource(pl))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("run_walkforward", "_record", "_thresholds", "_fit_predict", "_labels"):
        names = {n.id for n in ast.walk(fns[name]) if isinstance(n, ast.Name)}
        assert "label_eval" not in names, name
    rp = fns["run_pipeline"]
    wf_line = min(n.lineno for n in ast.walk(rp) if isinstance(n, ast.Call) and getattr(n.func, "id", "") ==
                  "run_walkforward")
    le_line = min(n.lineno for n in ast.walk(rp) if isinstance(n, ast.Name) and n.id == "label_eval")
    assert le_line > wf_line                     # retrospective labels are built only after all predictions


def test_core_never_imports_optional_agent_or_api_layers():
    """Row 11: LLM explanations can never feed predictions — the core package cannot even import them."""
    import re

    from riskcausal.config import PROJECT_ROOT

    pat = re.compile(r"^\s*(from|import)\s+(riskcausal_agents|riskcausal_api|dashboard|ollama)\b", re.MULTILINE)
    for py in (PROJECT_ROOT / "src" / "riskcausal").rglob("*.py"):
        assert not pat.search(py.read_text()), py
