"""Step-3 (rebuild, final) — fail-closed control/ablation battery machinery (Pi F3 fold).

Exercises the fail-closed ablation ORIENTATION for each active D component (PASS-only; NOT_EVALUABLE
non-passing) at a small seed count — the full 25-seed source-conjunction power run is step 4 — plus the null,
boundary-short, structural-zero and source-swap controls and the per-check rate aggregation. After the F3 fix
(centered S8 + position-balanced CSMD) all four components pass the full orientation. Slower (verifier-heavy).
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_realism_v2_battery as bat
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU

_BATTERY_IMPL_ID = "ee0da40525d484c6de49b262e0322f50643cec912464a87f49ff29fe873c3bb1"


class BatteryIdentity(unittest.TestCase):
    def test_impl_identity_and_wilson(self) -> None:
        self.assertEqual(bat.battery_impl_identity(), _BATTERY_IMPL_ID)
        self.assertTrue(bat.BATTERY_IMPL["fail_closed"].startswith("required check satisfied ONLY on PASS"))
        r = bat._rate([True] * 24 + [False])          # 24/25
        self.assertEqual(r["k"], 24)
        self.assertEqual(r["rate"], 0.96)
        self.assertEqual(len(r["ci95"]), 2)


class AblationOrientation(unittest.TestCase):
    def test_each_component_full_orientation(self) -> None:
        base = bat.default_base_sampler(n_each=600)      # local (a class attr would bind self)
        for comp in V2_D_COMPONENT_MENU:
            o = bat.component_ablation(comp, 1, base_sampler=base, source="MIMIC")
            self.assertTrue(o.A_fails_primary, f"{comp}: candidate_A must FAIL primary")
            self.assertTrue(o.A_specificity_ok,
                            f"{comp}: non-attributed specificity — fails {[k for k,v in o.A_specificity.items() if not v]}")
            self.assertTrue(o.D_recovers,
                            f"{comp}: D-recovery — fails {[k for k,v in o.D_status.items() if v!=bat.PASS]}")


class Controls(unittest.TestCase):
    def test_null_all_pass(self) -> None:
        base = bat.default_base_sampler(n_each=600)
        nc = bat.null_control(1, base_sampler=base)
        self.assertTrue(nc["all_pass"], f"null FAILs={nc['fails']} NE={nc['not_evaluable']}")

    def test_structural_zero_control(self) -> None:
        sz = bat.structural_zero_control(1, n_each=600)
        self.assertTrue(sz["zeros_absent"])
        self.assertTrue(sz["no_false_fail"])

    def test_source_swap_fails_nondegenerate(self) -> None:
        ss = bat.source_swap_control(1, n_each=600)
        self.assertTrue(ss["fails_nondegenerate"], f"source-swap fails: {ss['fails']}")


class RateAndForecast(unittest.TestCase):
    def test_rate_battery_per_check_shape(self) -> None:
        base = bat.default_base_sampler(n_each=600)
        rates = bat.rate_battery(["burst_timing"], [1], base_sampler=base, sources=("MIMIC",))
        row = rates["burst_timing"]["per_source"]["MIMIC"]
        self.assertIn("A_specificity_per_check", row)
        self.assertIn("ci95", row["D_recovery_rate"])
        self.assertIn("conjunction_A_fails_primary", rates["burst_timing"])

    def test_forecast(self) -> None:
        base = bat.default_base_sampler(n_each=600)
        f = bat.forecast(base, source="MIMIC")
        for k in ("n_sequences", "mean_length", "total_events", "est_secs_per_verifier_call"):
            self.assertIn(k, f)
        self.assertEqual(f["n_sequences"], 1800)


if __name__ == "__main__":
    unittest.main()
