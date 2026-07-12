"""Tests for the synthetic-recovery certification runner (Pi #3/#8). Safe-public; no governed work."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_certify as CR
from clinical_jepa.eval.rung2_contract import ORDER_UNLOCK_CHECKS


class FamilyCertificationTests(unittest.TestCase):
    def test_train_family_passes_all_u_checks(self) -> None:
        fc = CR.certify_family("T_latent_factor")
        self.assertEqual(set(fc.checks), set(ORDER_UNLOCK_CHECKS))
        self.assertTrue(fc.all_pass, fc.checks)
        self.assertLessEqual(fc.realized_alpha, 0.05)

    def test_no_h_family_still_certifies_via_context(self) -> None:
        # order in the no-h family IS present in context (exogenous), so recovery still passes;
        # what fails there is the h-SHORTCUT, covered in the end-to-end suite.
        fc = CR.certify_family("E_no_h_exogenous", seed=50)
        self.assertTrue(fc.checks["U1_order_recovery"] == "PASS")
        self.assertTrue(fc.checks["U2_null"] == "PASS")


class ConjunctionTests(unittest.TestCase):
    def test_synthetic_recovery_certifies_and_issues_no_governed_manifest(self) -> None:
        out = CR.certify_synthetic_recovery()
        self.assertTrue(out["synthetic_recovery_certified"], out)
        self.assertEqual(set(out["unlock_checks"]), set(ORDER_UNLOCK_CHECKS))
        self.assertFalse(out["governed_manifest_issued"])          # never issued by the runner
        self.assertLessEqual(out["reference_bounds_candidate"]["evaluator_realized_alpha"], 0.05)


if __name__ == "__main__":
    unittest.main()
