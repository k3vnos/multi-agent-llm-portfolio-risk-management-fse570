"""M4 components: DCC-GARCH baseline, Markov cache / predictive likelihood, validation-only selection,
bootstrap intervals."""

import numpy as np
import pandas as pd

from riskcausal.baselines.dcc_garch import dcc_scores, fit_dcc
from riskcausal.evaluation.bootstrap import jeffreys, stationary_bootstrap_indices
from riskcausal.regimes.markov import cached_fit_ms, fit_ms, predictive_loglik


def _garch_returns(n=1500, seed=0, alpha=0.08, beta=0.9):
    rng = np.random.default_rng(seed)
    k = 3
    C = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    Lc = np.linalg.cholesky(C)
    s2 = np.full(k, 1e-4)
    out = np.zeros((n, k))
    for t in range(n):
        z = Lc @ rng.normal(size=k)
        out[t] = np.sqrt(s2) * z
        s2 = 1e-6 + alpha * out[t] ** 2 + beta * s2
    return pd.DataFrame(out, index=pd.bdate_range("2010-01-01", periods=n), columns=list("ABC"))


def test_dcc_recovers_garch_persistence_and_correlation_level():
    r = _garch_returns()
    p = fit_dcc(r, r.index[-1])
    for g in p["garch"]:
        assert 0.8 < g[2] + g[3] < 1.0
    s = dcc_scores(r, p)
    assert (s["dcc_var"].dropna() > 0).all()
    assert 0.25 < s["dcc_corr"].median() < 0.55       # true mean off-diagonal correlation = 0.4


def test_dcc_scores_use_only_past_returns_and_training_parameters():
    r = _garch_returns()
    fit_end, cut = r.index[900], r.index[1000]
    p = fit_dcc(r, fit_end)
    r2 = r.copy()
    r2.iloc[1000:] *= 6
    assert fit_dcc(r2, fit_end)["a"] == p["a"]                          # parameters ignore rows > fit_end
    a, b = dcc_scores(r, p), dcc_scores(r2, fit_dcc(r2, fit_end))
    pd.testing.assert_frame_equal(a.loc[:cut].iloc[:-1], b.loc[:cut].iloc[:-1])
    assert not a.loc[cut:].equals(b.loc[cut:])


def _y(seed=0):
    rng = np.random.default_rng(seed)
    reg = np.r_[np.zeros(200), np.ones(80), np.zeros(200), np.ones(80)].astype(int)
    return pd.Series(np.where(reg == 1, rng.normal(1, .3, len(reg)), rng.normal(-1, .15, len(reg))),
                     index=pd.bdate_range("2012-01-02", periods=len(reg)))


def test_cached_fit_is_identical_and_reused(tmp_path):
    y = _y()
    a = cached_fit_ms(y, 2, 0, tmp_path, n_starts=3)
    assert len(list(tmp_path.glob("ms_*.npz"))) == 1
    b = cached_fit_ms(y, 2, 0, tmp_path, n_starts=3)
    c = fit_ms(y, 2, seed=0, n_starts=3)
    np.testing.assert_array_equal(a.params, c.params)
    np.testing.assert_array_equal(b.smoothed.to_numpy(), c.smoothed.to_numpy())
    assert b.stress_regime == c.stress_regime and b.llf == c.llf


def test_predictive_loglik_ignores_data_after_end_and_prefers_true_model():
    y = _y()
    fit2 = fit_ms(y.iloc[:300], 2, seed=0, n_starts=3)
    start, end = y.index[300], y.index[450]
    a = predictive_loglik(y, fit2, 2, start, end)
    y2 = y.copy()
    y2.iloc[451:] += 9
    assert predictive_loglik(y2, fit2, 2, start, end) == a
    fit1 = fit_ms(y.iloc[:300], 3, seed=0, n_starts=3)
    assert np.isfinite(predictive_loglik(y, fit1, 3, start, end))


def test_stationary_bootstrap_indices_are_blocky_and_complete():
    rng = np.random.default_rng(0)
    ix = stationary_bootstrap_indices(1000, 50, rng)
    assert len(ix) == 1000 and ix.min() >= 0 and ix.max() < 1000
    cont = np.mean((np.diff(ix) == 1) | (np.diff(ix) == -999))
    assert cont > 0.9                                   # mostly contiguous runs (mean block 50)


def test_jeffreys_interval():
    lo, hi = jeffreys(15, 24)
    assert lo < 15 / 24 < hi and 0.4 < lo < 0.45 and 0.79 < hi < 0.84
    assert jeffreys(0, 10)[0] == 0.0 and jeffreys(10, 10)[1] == 1.0
    assert jeffreys(0, 0) is None


def test_validation_selection_never_reads_after_train_end():
    """The M4 grid statistic for a fold must not change when anything after that fold's train_end changes."""
    from test_walkforward_no_leakage import _synthetic

    from riskcausal.config import load_config
    from riskcausal.validation.select import validate_config
    from riskcausal.validation.walkforward import make_folds

    cfg = load_config("dev")
    feats, y, _ = _synthetic()
    folds = make_folds(feats.index, 2010, 2012, embargo=5, val_years=2)
    fits = {f.k: (fit_ms(y.loc[:f.inner_end], 2, seed=0, n_starts=3), fit_ms(y.loc[:f.train_end], 2, seed=0,
                                                                              n_starts=3)) for f in folds}
    models = [("gbm", "instability"), ("map_logit", "combined")]
    a = validate_config(cfg, feats, folds[:1], fits, models)
    f2 = feats.copy()
    f2.loc[f2.index > folds[0].train_end] = f2.loc[f2.index > folds[0].train_end] * 9 - 2
    b = validate_config(cfg, f2, folds[:1], fits, models)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["method"]) >= {"gbm:instability", "map_logit:combined", "inst_composite", "vix"}
