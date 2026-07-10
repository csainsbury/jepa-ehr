"""Rung-1 probe statistical-core tests: cluster bootstrap, swap excess, CRPS skill, ridge,
metrics, and the load-bearing zero-mass randomized-PIT calibration (Pi R8 #5/#7)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung1_probes as P


class BootstrapTests(unittest.TestCase):
    def test_mean_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(0)
        vals = rng.normal(0.5, 0.1, size=400)
        clusters = np.repeat(np.arange(80), 5)
        r = P.cluster_bootstrap_ci(vals, clusters, n_boot=500, seed=0)
        self.assertLess(r["ci_lo"], r["point"])
        self.assertGreater(r["ci_hi"], r["point"])
        self.assertAlmostEqual(r["point"], vals.mean(), places=6)

    def test_paired_excess_null_is_zero(self) -> None:
        v = np.random.default_rng(1).normal(size=300)
        clusters = np.repeat(np.arange(60), 5)
        r = P.paired_excess_ci(v, v, clusters, n_boot=300, seed=0)   # true==swap
        self.assertAlmostEqual(r["point"], 0.0, places=9)
        self.assertLessEqual(r["ci_lo"], 0.0)

    def test_paired_excess_positive_when_true_beats_swap(self) -> None:
        n = 500
        true = np.ones(n) * 0.9
        swap = np.ones(n) * 0.4
        clusters = np.repeat(np.arange(100), 5)
        r = P.paired_excess_ci(true, swap, clusters, n_boot=300, seed=0)
        self.assertGreater(r["ci_lo"], 0.10)                         # clears the excess margin

    def test_ratio_skill(self) -> None:
        clusters = np.repeat(np.arange(60), 5)
        eq = P.ratio_skill_ci(np.ones(300), np.ones(300), clusters, n_boot=300, seed=0)
        self.assertAlmostEqual(eq["point"], 0.0, places=9)
        better = P.ratio_skill_ci(np.ones(300) * 0.5, np.ones(300), clusters, n_boot=300, seed=0)
        self.assertGreater(better["ci_lo"], 0.05)                    # cond half the marg loss


class ReadoutMetricTests(unittest.TestCase):
    def test_ridge_recovers_linear(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4)); w = np.array([1.0, -2.0, 0.5, 3.0])
        y = X @ w + 0.7
        W = P.ridge_fit(X, y, lam=1e-6)
        pred = P.ridge_predict(W, X)
        self.assertLess(np.mean((pred - y) ** 2), 1e-3)

    def test_analytic_multiset_residual_small(self) -> None:
        rng = np.random.default_rng(0)
        V, D = 30, 8
        E = rng.normal(size=(V, D))
        p = np.zeros(V); p[[2, 5, 9]] = [0.5, 0.3, 0.2]
        z = E.T @ p
        p_hat = P.analytic_multiset(E, z)
        self.assertLess(P.multiset_reconstruction_residual(E, z, p_hat), 1e-6)

    def test_exact_count_and_order(self) -> None:
        self.assertTrue(np.array_equal(P.exact_count_hits([3.1, 5.0], [3, 6]), [1.0, 0.0]))
        h = P.exact_order_hits([[1, 2, 3], [1, 2]], [[1, 2, 3], [2, 1]])
        self.assertTrue(np.array_equal(h, [1.0, 0.0]))

    def test_kendall_and_f1(self) -> None:
        self.assertGreater(P.kendall_tau_tie_aware([1, 2, 3], [1, 2, 3]), 0.99)
        self.assertLess(P.kendall_tau_tie_aware([1, 2, 3], [3, 2, 1]), -0.99)
        r = P.marginal_f1_jaccard([{1, 2, 3}], [{1, 2, 3}])
        self.assertAlmostEqual(r["set_jaccard"], 1.0)


class TimingTests(unittest.TestCase):
    def test_zero_mass_randomized_pit_is_calibrated(self) -> None:
        # Δt with a point mass at 0 (simultaneous events) + exponential tail. A correctly
        # specified hurdle + randomized PIT must yield ~Uniform => small KS-D (Pi R8 #7).
        rng = np.random.default_rng(0)
        def draw(n):
            z = rng.random(n) < 0.3                       # 30% simultaneous (Δt=0)
            dt = np.where(z, 0.0, rng.exponential(2.0, size=n))
            return dt
        model = P.fit_marginal_hurdle(draw(20000))        # fit on train
        pit = P.randomized_pit(draw(20000), model, seed=1)
        self.assertLess(P.ks_d_uniform(pit), 0.03)        # calibrated
        self.assertGreaterEqual(pit.min(), 0.0)
        self.assertLessEqual(pit.max(), 1.0)

    def test_misspecified_pit_fails_ks(self) -> None:
        rng = np.random.default_rng(0)
        model = P.fit_marginal_hurdle(rng.exponential(1.0, size=10000))
        pit = P.randomized_pit(rng.exponential(5.0, size=10000), model, seed=1)  # wrong scale
        self.assertGreater(P.ks_d_uniform(pit), 0.10)

    def test_ks_upper_ci_above_point(self) -> None:
        rng = np.random.default_rng(0)
        pit = rng.uniform(size=1000)
        clusters = np.repeat(np.arange(200), 5)
        r = P.ks_d_upper_ci(pit, clusters, n_boot=300, seed=0)
        self.assertGreaterEqual(r["ci_hi"], r["point"])

    def test_crps_conditional_beats_broad_marginal(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        cond = [rng.normal(y[i], 0.2, size=40) for i in range(200)]   # sharp, on-target
        marg = [rng.normal(0.0, 1.5, size=40) for _ in range(200)]    # broad
        c_cond = P.crps_rows(cond, y); c_marg = P.crps_rows(marg, y)
        clusters = np.repeat(np.arange(40), 5)
        skill = P.ratio_skill_ci(c_cond, c_marg, clusters, n_boot=300, seed=0)
        self.assertGreater(skill["ci_lo"], 0.05)


if __name__ == "__main__":
    unittest.main()
