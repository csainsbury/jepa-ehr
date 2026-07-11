"""Rung-2 sub-gate 4 continuous-time/multiplicity head training tests (synthetic): the multiplicity
head learns a context-dependent cluster size; the inter-cluster-time head yields calibrated PIT when
context carries the rate."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from clinical_jepa.arms.rung2.timing_head import (
        predict_intercluster_quantiles, predict_multiplicity_pmf,
        train_intercluster_time_head, train_multiplicity_head,
    )
    from clinical_jepa.eval.rung1_probes import ks_d_uniform


@unittest.skipUnless(HAS_TORCH, "torch required")
class TimingHeadTests(unittest.TestCase):
    def test_multiplicity_head_learns(self) -> None:
        rng = np.random.default_rng(0)
        n, D, m_max = 800, 10, 12
        z = rng.normal(size=(n, D)).astype(np.float32)
        size = 1 + (z[:, 0] > 0).astype(int) * rng.integers(1, 6, size=n)   # context-dependent size
        m = train_multiplicity_head(z, size, D, m_max=m_max, steps=250)
        pmf = predict_multiplicity_pmf(m, z)
        self.assertEqual(pmf.shape, (n, m_max))
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-4)
        # high-context rows predict a larger expected cluster size than low-context rows
        exp = (pmf * (np.arange(m_max) + 1)).sum(axis=1)
        self.assertGreater(exp[z[:, 0] > 0].mean(), exp[z[:, 0] <= 0].mean())

    def test_intercluster_time_pit_calibrated(self) -> None:
        rng = np.random.default_rng(0)
        n, D = 4000, 8
        z = rng.normal(size=(n, D)).astype(np.float32)
        scale = np.exp(0.5 * z[:, 0])                          # context-dependent rate
        gap = rng.exponential(scale)                          # strictly positive inter-cluster gaps
        head, qs = train_intercluster_time_head(z, gap, D, steps=300)
        q = predict_intercluster_quantiles(head, z)
        # randomized PIT via the predicted quantile grid
        rngp = np.random.default_rng(1)
        pit = np.empty(n)
        for i in range(n):
            lo = qs[np.searchsorted(q[i], gap[i], side="left") - 1] if np.searchsorted(q[i], gap[i], side="left") > 0 else 0.0
            hi_i = np.searchsorted(q[i], gap[i], side="right")
            hi = qs[min(hi_i, len(qs) - 1)] if hi_i < len(qs) else 1.0
            pit[i] = np.clip(lo + rngp.uniform() * max(hi - lo, 0.0), 0, 1)
        self.assertLess(ks_d_uniform(pit), 0.08)              # reasonably calibrated on synthetic


if __name__ == "__main__":
    unittest.main()
