"""Bayesian logit draws are bit-identical across processes regardless of PyTensor compile-cache state (M5 finding:
the numba backend gave a different realisation in a process with a cold cache)."""

import subprocess
import sys

import numpy as np

SCRIPT = """
import sys, numpy as np
from riskcausal.models.bayes_logit import BayesLogit
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 12)); X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=400)
y = (rng.random(400) < 1 / (1 + np.exp(-(X[:, 0] - 1.5)))).astype(int)
m = BayesLogit(draws=200, tune=200, seed=570).fit(X, y)
np.save(sys.argv[1], m.b_)
"""


def test_draws_identical_with_cold_and_warm_compile_cache(tmp_path):
    env_flags = f"base_compiledir={tmp_path / 'ptcache'}"
    outs = []
    for i in range(2):                      # 1st process: cold cache; 2nd: warm cache
        out = tmp_path / f"b{i}.npy"
        subprocess.run([sys.executable, "-c", SCRIPT, str(out)], check=True, capture_output=True,
                       env={**__import__("os").environ, "PYTENSOR_FLAGS": env_flags})
        outs.append(np.load(out))
    np.testing.assert_array_equal(outs[0], outs[1])
