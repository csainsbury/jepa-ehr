"""Tests for the evaluator-plumbing smoke runner (Pi #3/#4/#8). Safe-public; no governed work.

The runner is deliberately NON-CERTIFYING (hand-coded reference predictors, not a T4 recipe).
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_certify as CR
from clinical_jepa.eval.oracle_metrics import clopper_pearson_upper
from clinical_jepa.eval.rung2_contract import ORACLE_FPR_UPPER_CI_MAX, ORDER_UNLOCK_CHECKS


class FamilyCertificationTests(unittest.TestCase):
    def test_train_family_passes_all_u_checks(self) -> None:
        fc = CR.certify_family("T_latent_factor")
        self.assertEqual(set(fc.checks), set(ORDER_UNLOCK_CHECKS))
        self.assertTrue(fc.all_pass, fc.checks)

    def test_no_h_family_still_recovers_via_context(self) -> None:
        # order in the no-h family IS present in context (exogenous), so recovery still passes;
        # what fails there is the h-SHORTCUT, covered in the end-to-end suite.
        fc = CR.certify_family("E_no_h_exogenous", seed=50)
        self.assertEqual(fc.checks["U1_order_recovery"], "PASS")
        self.assertEqual(fc.checks["U2_null"], "PASS")

    def test_offgrid_family_evaluated_on_its_own_cells(self) -> None:
        # Pi #5: the off-grid family must NOT be silently re-evaluated on the train grid.
        fc = CR.certify_family("E_offgrid_nonlinear", seed=50)
        self.assertEqual(fc.checks["U1_order_recovery"], "PASS")
        self.assertEqual(fc.checks["U3_monotone"], "PASS")   # >=3 off-grid points => real monotone check


class NullOCStudyTests(unittest.TestCase):
    def test_clopper_pearson_upper_is_a_real_upper_bound(self) -> None:
        # a couple of anchors: 0/n has a finite upper bound < 1; k=n is 1.
        self.assertLess(clopper_pearson_upper(0, 200), 0.02)
        self.assertEqual(clopper_pearson_upper(5, 5), 1.0)
        self.assertGreater(clopper_pearson_upper(3, 200), 3 / 200)   # upper > point estimate

    def test_independent_seed_null_fpr_upper_ci_within_gate(self) -> None:
        oc = CR.null_oc_study()
        self.assertGreaterEqual(oc.n_studies, 200)                  # >= ORACLE_N_NULL_SEEDS
        self.assertLessEqual(oc.upper_ci, ORACLE_FPR_UPPER_CI_MAX)   # gate on the UPPER bound
        self.assertTrue(oc.passes)


class SmokeRunnerTests(unittest.TestCase):
    def test_runner_is_non_certifying(self) -> None:
        out = CR.certify_evaluator_smoke()
        self.assertFalse(out["certifies_recipe"])                   # Pi #8: not a recipe certifier
        self.assertNotIn("synthetic_recovery_certified", out)       # the certifying claim is gone
        self.assertFalse(out["governed_manifest_issued"])
        self.assertTrue(out["evaluator_plumbing_smoke_ok"], out)
        self.assertEqual(set(out["unlock_checks"]), set(ORDER_UNLOCK_CHECKS))
        self.assertTrue(out["null_control_global_pass"])            # no declared family fails U2
        self.assertLessEqual(out["null_oc_study"]["upper_ci"], ORACLE_FPR_UPPER_CI_MAX)


if __name__ == "__main__":
    unittest.main()
