"""Rung-2 sub-gate 2 count-head training tests (synthetic): both interfaces produce a proper count
PMF; a context that carries the count is learned; RPS-skill over a modal baseline is positive."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from clinical_jepa.arms.rung2.count_heads import predict_count_pmf, train_count_head
    from clinical_jepa.eval.rung2_count_interface import ranked_probability_score, rps_skill_vs_baseline


@unittest.skipUnless(HAS_TORCH, "torch required")
class CountHeadTests(unittest.TestCase):
    def _data(self, n=600, D=12, k_max=20, seed=0):
        rng = np.random.default_rng(seed)
        counts = rng.integers(0, k_max + 1, size=n)
        z = rng.normal(size=(n, D)).astype(np.float32)
        z[:, 0] = counts / k_max                              # context carries the count in dim 0
        return z, counts, D, k_max

    def test_interface_A_pmf_and_learns(self) -> None:
        z, counts, D, k_max = self._data()
        m = train_count_head(z, counts, D, k_max=k_max, steps=250)
        pmf = predict_count_pmf(m, z)
        self.assertEqual(pmf.shape, (len(counts), k_max + 1))
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-4)   # proper PMF
        # RPS-skill over the modal-count baseline is positive (context carries count)
        rps_model = ranked_probability_score(pmf, counts)
        modal = np.bincount(counts, minlength=k_max + 1).argmax()
        base = np.zeros((len(counts), k_max + 1)); base[:, modal] = 1.0
        rps_base = ranked_probability_score(base, counts)
        skill = rps_skill_vs_baseline(rps_model, rps_base, np.arange(len(counts)) % 60, n_boot=200)
        self.assertGreater(skill["ci_lo"], 0.0)

    def test_interface_B_via_target_channel(self) -> None:
        z, counts, D, k_max = self._data(seed=1)
        # B: the target representation carries log1p(count) in its last dim (the concat interface)
        targ = np.concatenate([z, np.log1p(counts)[:, None]], axis=1).astype(np.float32)
        m = train_count_head(z, counts, D, k_max=k_max, steps=250, target_latents=targ)
        pmf = predict_count_pmf(m, z)
        self.assertEqual(pmf.shape, (len(counts), k_max + 1))
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-4)
        self.assertEqual(m["interface"], "B")


if __name__ == "__main__":
    unittest.main()
