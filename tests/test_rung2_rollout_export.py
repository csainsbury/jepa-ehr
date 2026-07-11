"""Rung-2 sub-gate 1 rollout-export tests: path planning (recursive NOT_EVALUABLE for horizon-count-1
checkpoints), direct path emits drift but never exposure-gap, recursive runs only when transition-
trained."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung2_rollout_export as RE
from clinical_jepa.eval.rung2_contract import NOT_EVALUABLE


class PathPlanTests(unittest.TestCase):
    def test_recursive_not_evaluable_for_horizon_count_1_checkpoint(self) -> None:
        p = RE.plan_paths({"horizon_count": 1})               # no fixed_width_transition_trained flag
        self.assertTrue(p["direct"])
        self.assertFalse(p["recursive"])
        self.assertEqual(p["recursive_status"], NOT_EVALUABLE)

    def test_recursive_evaluable_only_when_transition_trained(self) -> None:
        self.assertTrue(RE.plan_paths({"fixed_width_transition_trained": True})["recursive"])


class DirectPathTests(unittest.TestCase):
    def test_direct_emits_drift_no_exposure_gap(self) -> None:
        rng = np.random.default_rng(0)
        n, D = 60, 8
        true = rng.standard_normal((n, D)); pred = true + 0.05 * rng.standard_normal((n, D))
        rows = RE.direct_horizon_metrics({90.0: pred}, {90.0: true}, {90.0: np.arange(n)}, source="SCID")
        self.assertEqual(len(rows), 1)
        self.assertIn("d_self_over_ambient_nn", rows[0])
        self.assertNotIn("exposure_gap", rows[0])              # direct path never carries exposure-gap

    def test_recursive_metrics_refused_without_transition_semantics(self) -> None:
        r = RE.recursive_transition_metrics({"horizon_count": 1}, source="SCID", window_days=90.0)
        self.assertEqual(r["status"], NOT_EVALUABLE)

    def test_recursive_metrics_run_with_transition_semantics(self) -> None:
        r = RE.recursive_transition_metrics({"fixed_width_transition_trained": True},
                                            dself_free=np.array([0.5, 0.4]), dself_tf=np.array([0.2, 0.1]),
                                            dself_over_nn_point=0.4, source="SCID", window_days=30.0)
        self.assertEqual(r["status"], "evaluable")
        self.assertIsNotNone(r["exposure_gap_mean"])
        self.assertIn(r["signature"], ("HEALTHY", "DRIFT_DOMINANT", "COLLAPSE_DOMINANT"))


if __name__ == "__main__":
    unittest.main()
