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
from clinical_jepa.eval.rung2_contract import ORACLE_R_BAYES_MARGIN

ALL_FAMILIES = (*TRAIN_FAMILIES, *HELDOUT_FAMILIES)


class R0AndPiStarValidationTests(unittest.TestCase):
    def test_r0_matches_independent_high_precision_mc(self) -> None:
        for fam in ALL_FAMILIES:                       # Gaussian analytic / Markov mixture / Student-t MC
            self.assertLess(B.reference_mc_error(fam, 0.60, which="r0"), 0.02, fam)

    def test_pistar_matches_independent_per_sequence_mc(self) -> None:
        for fam in ALL_FAMILIES:                       # per-family posterior integration
            self.assertLess(B.reference_mc_error(fam, 0.60, which="pistar"), 0.03, fam)

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
