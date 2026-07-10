"""Rung-1 KS precision-simulation tests (Pi R8 #3): a well-powered cell certifies D*=0.025
under the 0.05 gate; a tiny cell cannot and is flagged NOT_EVALUABLE, never weakening KS."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval.rung1_precision_sim import _sample_ks_alternative, run_precision_sim
from clinical_jepa.eval.rung1_probes import ks_d_uniform


class PrecisionSimTests(unittest.TestCase):
    def test_alternative_has_target_ks_distance(self) -> None:
        # The synthetic alternative really sits ~D* from Uniform (population check).
        x = _sample_ks_alternative(200000, 0.025, np.random.default_rng(0))
        self.assertAlmostEqual(ks_d_uniform(x), 0.025, delta=0.004)

    def test_large_cell_certifies(self) -> None:
        r = run_precision_sim(n_intervals=40000, n_clusters=4000, reps=40, n_boot=200, seed=1)
        self.assertGreaterEqual(r["coverage"], 0.95)
        self.assertGreaterEqual(r["power"], 0.80)
        self.assertTrue(r["passes"])
        self.assertEqual(r["action"], "certifiable")

    def test_tiny_cell_cannot_certify(self) -> None:
        r = run_precision_sim(n_intervals=150, n_clusters=30, reps=40, n_boot=200, seed=1)
        self.assertLess(r["power"], 0.80)                    # under-powered at 150 intervals
        self.assertFalse(r["passes"])
        self.assertIn("NOT_EVALUABLE", r["action"])          # dropped, KS gate never weakened

    def test_gate_never_weakened(self) -> None:
        r = run_precision_sim(n_intervals=100, n_clusters=20, reps=20, n_boot=150, seed=2)
        self.assertEqual(r["gate"], 0.05)                    # the KS gate is a constant, not tuned


if __name__ == "__main__":
    unittest.main()
