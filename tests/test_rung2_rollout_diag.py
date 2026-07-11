"""Rung-2 sub-gate 1 rollout-diag tests: ambient-normalized drift, exposure gap, dispersion,
descriptive ρ_t, and the frozen-margin signature classifier incl. NOT_EVALUABLE without
transition semantics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung2_rollout_diag as R
from clinical_jepa.eval.rung2_contract import NOT_EVALUABLE


class DriftTests(unittest.TestCase):
    def test_drift_self_over_nn(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.standard_normal((20, 8))
        # ẑ close to own truth -> small d_self, larger d_nn -> ratio < 1
        ztrue = z + 0.01 * rng.standard_normal((20, 8))
        pool = rng.standard_normal((40, 8))
        d = R.drift_self_over_nn(z, ztrue, pool, np.arange(20), np.arange(100, 140))
        self.assertLess(np.median(d["d_self_over_nn"]), 1.0)   # own truth beats nearest wrong

    def test_exposure_gap_sign(self) -> None:
        g = R.exposure_gap(np.array([0.5, 0.4]), np.array([0.2, 0.1]))
        self.assertTrue(np.all(g > 0))                          # free-running drifts more


class DispersionTests(unittest.TestCase):
    def test_collapse_ratio_descriptive_small_when_pred_collapses(self) -> None:
        rng = np.random.default_rng(0)
        true_cloud = rng.standard_normal((50, 8))
        pred_collapsed = np.tile(true_cloud.mean(0), (50, 1)) + 1e-3 * rng.standard_normal((50, 8))
        rho = R.collapse_ratio_descriptive(pred_collapsed, true_cloud)
        self.assertLess(rho, 0.1)                               # descriptive: predicted cloud tiny

    def test_effective_rank(self) -> None:
        rng = np.random.default_rng(0)
        disp = R.population_dispersion(rng.standard_normal((100, 8)))
        self.assertGreater(disp["effective_rank"], 1.0)


class SignatureTests(unittest.TestCase):
    def test_not_evaluable_without_transition_semantics(self) -> None:
        s = R.classify_signature(dself_over_nn_point=0.3, exposure_gap_slope_lo=0.0,
                                 dself_slope_hi=0.0, transition_evaluable=False)
        self.assertEqual(s, NOT_EVALUABLE)                      # no recursive semantics -> no label

    def test_collapse_signature(self) -> None:
        s = R.classify_signature(dself_over_nn_point=0.95, exposure_gap_slope_lo=0.0,
                                 dself_slope_hi=0.0, transition_evaluable=True)
        self.assertEqual(s, "COLLAPSE_DOMINANT")

    def test_drift_signature(self) -> None:
        s = R.classify_signature(dself_over_nn_point=0.4, exposure_gap_slope_lo=0.05,
                                 dself_slope_hi=0.05, transition_evaluable=True)
        self.assertEqual(s, "DRIFT_DOMINANT")

    def test_healthy_signature(self) -> None:
        s = R.classify_signature(dself_over_nn_point=0.4, exposure_gap_slope_lo=0.0,
                                 dself_slope_hi=0.0, transition_evaluable=True)
        self.assertEqual(s, "HEALTHY")


if __name__ == "__main__":
    unittest.main()
