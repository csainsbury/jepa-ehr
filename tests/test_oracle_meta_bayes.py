"""Exact conditional R0 + context-Bayes π* validated against independent MC (Pi whole-pass #2/#3).

Also documents the REFERENCE-ONLY signal freeze: the mechanism signal is selected from the reference
bracket / evaluator OC (hidden-null at κ=0, detectable R_bayes−R0 at κmid, non-saturated high-κ ceiling),
NOT from candidate power.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_meta_bayes as B
from clinical_jepa.eval import oracle_meta_refs as RF
from clinical_jepa.eval.oracle_meta_gen import (
    HELDOUT_FAMILIES, KAPPA_MID, TRAIN_FAMILIES, generate_meta_cell,
)
from clinical_jepa.eval.rung2_contract import ORACLE_R_BAYES_MARGIN, ORACLE_R_BAYES_MC_TOL

ALL_FAMILIES = (*TRAIN_FAMILIES, *HELDOUT_FAMILIES)


class R0AndPiStarValidationTests(unittest.TestCase):
    def test_r0_matches_independent_high_precision_mc(self) -> None:
        for fam in ALL_FAMILIES:                       # Gaussian analytic / Markov mixture / Student-t MC
            self.assertLessEqual(B.reference_mc_error(fam, 0.60, which="r0"), ORACLE_R_BAYES_MC_TOL, fam)

    def test_pistar_matches_independent_per_sequence_mc(self) -> None:
        # Pi #1: every family's π* — including no-h (conditioned on the observed driver) and Student-t
        # (true per-sequence posterior) — validated against a GENUINELY DIFFERENT integrator ≤ the frozen
        # ORACLE_R_BAYES_MC_TOL (0.01), NOT the relaxed 0.03.
        for fam in ALL_FAMILIES:
            self.assertLessEqual(B.reference_mc_error(fam, 0.60, which="pistar"), ORACLE_R_BAYES_MC_TOL, fam)

    def test_student_pistar_importance_is_well_conditioned(self) -> None:
        # the Student-t posterior importance sampler must keep a high ESS (tight proposal), else the
        # ≤0.01 agreement above would be luck.
        self.assertGreater(B.student_pistar_ess(), 0.5 * B.STUDENT_PISTAR_NMC)

    def test_no_h_pistar_conditions_on_observed_driver(self) -> None:
        # Pi #1: ignoring observed_covariates (the exact driver) is a decisive estimand error; the
        # point-mass reference must differ sharply from the latent-inference Gaussian form at κ>0.
        c = generate_meta_cell("E_no_h_exogenous", 0.60, "orthogonal", 200, seed=5)
        exact = B.pi_star_pairwise(c, 0.60)                     # conditions on observed z
        latent = B._pistar_gaussian(c, 0.60)                    # WRONG: infers driver from noisy context
        self.assertGreater(np.abs(exact - latent).max(), 0.10)

    def test_reference_tables_are_valid_probabilities(self) -> None:
        c = generate_meta_cell("T_hmm_markov", 0.60, "orthogonal", 100, seed=3)
        r0 = RF.r0_pairwise("T_hmm_markov", 0.60, c.item_classes)
        pis = RF.r_bayes_probs(c)
        for p in (r0, pis):
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            self.assertTrue(np.allclose(p + np.transpose(p, (0, 2, 1)), 1.0, atol=1e-6))  # antisymmetric


class ReferenceOnlySignalTests(unittest.TestCase):
    def test_signal_selected_from_reference_bracket_not_candidate(self) -> None:
        """The reference bracket R_bayes − R0 (NO candidate) exhibits the declared OC: κ=0 hidden-null,
        detectable κmid, non-saturated high-κ ceiling — the basis for the frozen signal."""
        fam = HELDOUT_FAMILIES[0]

        def ceiling(k):
            c = generate_meta_cell(fam, k, "orthogonal", 2000, seed=11)
            b = RF.briers_from_probs(RF.r_bayes_probs(c), c)
            return RF.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=1)[1]

        self.assertLess(ceiling(0.0), ORACLE_R_BAYES_MARGIN)       # κ=0 hidden-null anchor
        self.assertGreater(ceiling(KAPPA_MID), 0.10)              # κmid detectable
        hi = ceiling(0.60)
        self.assertGreater(hi, 0.20)                             # nontrivial high-κ ceiling
        self.assertLess(hi, 0.95)                               # ... but NOT saturated


if __name__ == "__main__":
    unittest.main()
