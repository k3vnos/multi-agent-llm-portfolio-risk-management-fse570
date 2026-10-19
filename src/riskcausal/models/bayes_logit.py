"""Bayesian logistic regression (PyMC, NUTS). Posterior draws give a range for each day's probability and,
downstream, for the warning lead time (proposal §3 "Models")."""

from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)


class BayesLogit:
    name = "bayes_logit"

    def __init__(self, draws: int = 500, tune: int = 500, chains: int = 2, seed: int = 0,
                 prior_sd: float = 1.0, n_keep: int = 200):
        self.draws, self.tune, self.chains, self.seed = draws, tune, chains, seed
        self.prior_sd, self.n_keep = prior_sd, n_keep

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesLogit:
        import pymc as pm

        logging.getLogger("pymc").setLevel(logging.ERROR)
        with pm.Model():
            a = pm.Normal("a", 0.0, 2.5)
            b = pm.Normal("b", 0.0, self.prior_sd, shape=X.shape[1])     # features are standardised
            pm.Bernoulli("y", logit_p=a + pm.math.dot(X, b), observed=y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains, cores=1,
                                  random_seed=self.seed, progressbar=False, compute_convergence_checks=False)
        post = idata.posterior
        A = post["a"].to_numpy().reshape(-1)
        B = post["b"].to_numpy().reshape(-1, X.shape[1])
        rng = np.random.default_rng(self.seed)
        keep = rng.choice(len(A), size=min(self.n_keep, len(A)), replace=False)
        self.a_, self.b_ = A[keep], B[keep]
        try:
            import arviz as az
            r = az.rhat(idata)
            self.rhat_max_ = max(float(r[v].max()) for v in r.data_vars)
        except Exception:  # noqa: BLE001
            self.rhat_max_ = float("nan")
        return self

    def predict_draws(self, X: np.ndarray) -> np.ndarray:
        """(n_draws, n_rows) posterior probabilities."""
        z = self.a_[:, None] + self.b_ @ X.T
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_draws(X).mean(axis=0)

    def summary(self, names: list[str]) -> dict:
        return {"rhat_max": self.rhat_max_, "intercept_mean": float(self.a_.mean()),
                "coef_mean": dict(zip(names, map(float, self.b_.mean(0)))),
                "coef_q05": dict(zip(names, map(float, np.quantile(self.b_, 0.05, axis=0)))),
                "coef_q95": dict(zip(names, map(float, np.quantile(self.b_, 0.95, axis=0))))}
