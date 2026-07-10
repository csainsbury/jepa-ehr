"""Rung-1 verdict-assembly tests: worst-primary-cell combination, MIMIC-2d exclusion,
scope enforcement, and nomination gating (Pi R7/R8)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung1_contract as C
from clinical_jepa.eval.rung1_verdict import (
    build_rung1_manifest, classify_timing_cell, evaluate_property,
)

_DECODABLE = {"m1_gate_ok": False, "m1_excess_lo": -1.0, "m2_gate_ok": True,
              "m2_excess_lo": 0.30, "m2_copy_ok": True, "evaluable": True, "precise": True}
_NOT = {"m1_gate_ok": False, "m1_excess_lo": -1.0, "m2_gate_ok": False,
        "m2_excess_lo": -1.0, "m2_copy_ok": True, "evaluable": True, "precise": True}


def _cell(source, W, base):
    return {"source": source, "window_days": W, "base_class": base}


class CombinationTests(unittest.TestCase):
    def test_worst_primary_cell_governs(self) -> None:
        cells = [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR), _cell("SCID", 365.0, C.NOT_DECODABLE)]
        ev = evaluate_property("count_concat", "count", cells)
        self.assertEqual(ev["combined_base_class"], C.NOT_DECODABLE)

    def test_mimic_2d_does_not_gate_primary(self) -> None:
        # 2 d is sensitivity: a failure there must not sink a passing primary band.
        cells = [_cell("MIMIC", 0.5, C.DECODABLE_NONLINEAR), _cell("MIMIC", 2.0, C.NOT_DECODABLE)]
        ev = evaluate_property("count_concat", "count", cells)
        self.assertEqual(ev["combined_base_class"], C.DECODABLE_NONLINEAR)
        self.assertEqual(len(ev["per_sensitivity_cell"]), 1)
        self.assertTrue(ev["can_nominate"])

    def test_all_not_evaluable(self) -> None:
        cells = [_cell("SCID", 90.0, C.NOT_EVALUABLE), _cell("SCID", 365.0, C.NOT_EVALUABLE)]
        ev = evaluate_property("count_concat", "count", cells)
        self.assertEqual(ev["combined_base_class"], C.NOT_EVALUABLE)


class ScopeEnforcementTests(unittest.TestCase):
    def test_arm_a_order_never_direct_never_nominates(self) -> None:
        cells = [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]
        ev = evaluate_property("mean_embed", "order", cells)
        self.assertEqual(ev["verdict"], C.STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY)
        self.assertFalse(ev["can_nominate"])

    def test_temporal_slot_order_is_coarse_not_exact(self) -> None:
        cells = [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR), _cell("SCID", 365.0, C.DECODABLE_NONLINEAR)]
        ev = evaluate_property("temporal_slot", "order", cells)
        self.assertEqual(ev["verdict"], C.COARSE_SLOT_DECODABLE)
        self.assertEqual(ev["information_scope"], C.SCOPE_COARSE_SLOT)
        self.assertTrue(ev["can_nominate"])

    def test_tap_timing_direct_ceiling(self) -> None:
        cells = [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]
        ev = evaluate_property("tap_concat", "timing", cells)
        self.assertEqual(ev["verdict"], C.DIRECT_TIMING_CEILING_DECODABLE)
        self.assertTrue(ev["can_nominate"])


class TimingClassifyTests(unittest.TestCase):
    def test_ks_pass_skill_fail_is_prior_masked(self) -> None:
        # marginal reproduction: KS ok but no conditional skill.
        c = classify_timing_cell({"evaluable": True, "precise": True, "ks_upper_ci": 0.03, "crps_skill_lo": 0.0})
        self.assertEqual(c, C.PRIOR_MASKED)

    def test_both_gates_pass(self) -> None:
        c = classify_timing_cell({"evaluable": True, "precise": True, "ks_upper_ci": 0.03, "crps_skill_lo": 0.08})
        self.assertEqual(c, C.DECODABLE_NONLINEAR)

    def test_underfloor(self) -> None:
        self.assertEqual(classify_timing_cell({"evaluable": False}), C.NOT_EVALUABLE)


class ManifestTests(unittest.TestCase):
    def test_manifest_structure_and_nomination_order(self) -> None:
        evals = [
            evaluate_property("mean_embed", "order", [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]),
            evaluate_property("mean_embed", "count", [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]),
            evaluate_property("count_concat", "count", [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]),
            evaluate_property("tap_concat", "timing", [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]),
            evaluate_property("temporal_slot", "order", [_cell("SCID", 90.0, C.DECODABLE_NONLINEAR)]),
        ]
        m = build_rung1_manifest(evals, run_config={"SCID": {"horizons": [90]}})
        self.assertEqual(m["rung1a"]["arm"], "mean_embed")
        self.assertFalse(m["test_access"])
        self.assertEqual(len(m["config_hash"]), 64)
        # incumbent (1a) never nominates; 1b arms do, in the frozen comparison order.
        arms = [n["arm"] for n in m["nominations"]]
        self.assertNotIn("mean_embed", arms)
        self.assertEqual(arms, sorted(arms, key=lambda a: C.ARM_COMPARISON_ORDER.index(a)))
        self.assertIn("tap_concat", arms)
        self.assertIn("temporal_slot", arms)


if __name__ == "__main__":
    unittest.main()
