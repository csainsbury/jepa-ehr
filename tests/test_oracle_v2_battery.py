"""Step-3 (rebuild, final) — control/ablation battery machinery.

Exercises the ablation ORIENTATION for each active D component at a small seed count (the full 25-seed
rate-based power run is step 4), plus the null and source-swap controls and the rate aggregation structure.
Slower (verifier-heavy); modest N.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_realism_v2_battery as bat
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU

_BATTERY_IMPL_ID = "a52aaf3f212c9048bb0afa0e0b95db06119292722a7dbacc2f4f820efe74cd69"
# Finding F3 (routed to Pi): the class-relabelling couplings cross-load onto the terminal S8_class check.
_S8_CROSSLOAD = {"cluster_size_mark_diversity", "length_class_mix"}


class BatteryIdentity(unittest.TestCase):
    def test_impl_identity_and_wilson(self) -> None:
        self.assertEqual(bat.battery_impl_identity(), _BATTERY_IMPL_ID)
        r = bat._rate([True] * 24 + [False])          # 24/25
        self.assertEqual(r["k"], 24)
        self.assertEqual(r["rate"], 0.96)
        self.assertEqual(len(r["ci95"]), 2)


class AblationOrientation(unittest.TestCase):
    def test_each_component_orientation(self) -> None:
        base = bat.default_base_sampler(n_each=600)     # local (a class attr would bind self)
        for comp in V2_D_COMPONENT_MENU:
            o = bat.component_ablation(comp, 1, base_sampler=base)
            self.assertTrue(o.A_fails_primary, f"{comp}: candidate_A must FAIL primary — {o.detail}")
            self.assertTrue(o.D_recovers, f"{comp}: candidate_D_recovery must pass full row — {o.detail}")
            if comp in _S8_CROSSLOAD:
                # finding F3 (routed to Pi): these class-relabelling couplings cross-load ONLY onto the
                # terminal S8_class check; the specificity failure must be S8_class and nothing else.
                self.assertFalse(o.A_specificity_ok, f"{comp}: expected the S8_class cross-loading")
                self.assertEqual(o.detail["A_fails_nonattr"], ["S8_class"], comp)
            else:
                self.assertTrue(o.A_specificity_ok, f"{comp}: candidate_A must pass non-attributed — {o.detail}")


class Controls(unittest.TestCase):
    def test_null_control_no_fail(self) -> None:
        base = bat.default_base_sampler(n_each=600)
        nc = bat.null_control(1, base_sampler=base)
        self.assertTrue(nc["ok"], f"null control FAILed: {nc['fails']}")

    def test_source_swap_fails_nondegenerate(self) -> None:
        ss = bat.source_swap_control(1)
        self.assertTrue(ss["fails_nondegenerate"], f"source-swap must fail a non-degenerate check: {ss['fails']}")


class RateStructure(unittest.TestCase):
    def test_rate_battery_shape(self) -> None:
        base = bat.default_base_sampler(n_each=600)
        rates = bat.rate_battery(["burst_timing"], [1, 2], base_sampler=base)
        row = rates["burst_timing"]
        self.assertEqual(row["n"], 2)
        for k in ("A_fails_primary_rate", "A_specificity_rate", "D_recovery_rate"):
            self.assertIn("ci95", row[k])


if __name__ == "__main__":
    unittest.main()
