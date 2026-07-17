"""Regime-aware reference contract (Pi C=5 R0-defect ruling). Safe-public / synthetic.

A meta-cell interleaves POSITIVE rows (coupling κ) and NULL rows (no coupling). The Bayes content prior
differs by regime, so a single R0(κ_cell) masked to null rows afterwards is the WRONG null reference —
that was the load-bearing defect C=5 exposed. These tests pin the corrected contract and add a regression
in which a content-aware control exposes the old bug.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_meta_refs as R
from clinical_jepa.eval.oracle_meta_gen import generate_meta_cell
from clinical_jepa.eval.rung2_contract import ORACLE_EO1_SKILL_GATE, ORDER_SUPPORT_FLOOR


def _cell(kappa=0.60, n=3000, seed=11):
    return generate_meta_cell("E_no_h_exogenous", kappa, "orthogonal", n, seed=seed,
                              support_floor=ORDER_SUPPORT_FLOOR)


class RegimeApiFailClosedTests(unittest.TestCase):
    def test_regime_is_required_no_default(self) -> None:
        cell = _cell()
        probs = R.pairwise_probs(R.random_codebook_scores(cell, _rc_q(cell)))
        with self.assertRaises(TypeError):                    # omission on a mixed cell fails closed
            R.briers_from_probs(probs, cell)                  # type: ignore[call-arg]

    def test_unknown_regime_refused(self) -> None:
        cell = _cell()
        probs = R.pairwise_probs(R.random_codebook_scores(cell, _rc_q(cell)))
        with self.assertRaises(ValueError):
            R.briers_from_probs(probs, cell, regime="positives_only")   # deprecated/ambiguous name

    def test_mixture_requires_explicit_p_null_and_scores_all_rows(self) -> None:
        cell = _cell()
        probs = R.pairwise_probs(R.random_codebook_scores(cell, _rc_q(cell)))
        with self.assertRaises(ValueError):
            R.briers_from_probs(probs, cell, regime=R.REGIME_MIXTURE)    # no p_null -> refuse
        # with an explicit weight the mixture scores the UNION of positive and null rows (no regime mask)
        b_pos = R.briers_from_probs(probs, cell, regime=R.REGIME_POSITIVE)
        b_null = R.briers_from_probs(probs, cell, regime=R.REGIME_NULL)
        b_mix = R.briers_from_probs(probs, cell, regime=R.REGIME_MIXTURE, p_null=0.5)
        union = (b_pos[2] > 0) | (b_null[2] > 0)
        self.assertTrue(np.array_equal(b_mix[2] > 0, union))


class RegimeReferenceRegressionTests(unittest.TestCase):
    def test_null_rows_use_R0_zero_not_R0_kappa(self) -> None:
        """The regression: a content-aware order-blind control has near-zero skill on NULL rows against
        their OWN reference R0(0), but SPURIOUS positive skill if scored against R0(κ_cell) (the old bug).
        """
        cell = _cell(kappa=0.60)
        rc = R.random_codebook_scores(cell, _rc_q(cell))

        # CORRECTED: null regime -> R0(0) on null rows.
        b_null = R.briers_vs_r0(rc, cell, regime=R.REGIME_NULL)
        corrected_upper = -R.paired_skill_contrast(b_null[1], b_null[0], b_null[1], b_null[2],
                                                   base_seed=7, alpha=0.0025)[1]

        # OLD BUG reproduced by hand: score the SAME null rows against R0(κ_cell).
        r0_kappa = R.r0_pairwise(cell.family_id, cell.kappa, cell.item_classes)
        b_rec, b_r0k, npair = R.per_sequence_briers(R.pairwise_probs(rc), cell.true_order, r0_kappa)
        npair_null = np.where(cell.is_null, npair, 0)
        buggy_upper = -R.paired_skill_contrast(b_r0k, b_rec, b_r0k, npair_null,
                                              base_seed=7, alpha=0.0025)[1]

        self.assertGreater(buggy_upper, ORACLE_EO1_SKILL_GATE)        # the defect: false positive skill
        self.assertLess(corrected_upper, ORACLE_EO1_SKILL_GATE)      # corrected: control passes null honestly
        self.assertGreater(buggy_upper - corrected_upper, 0.1)       # the correction is large, not marginal

    def test_positive_and_null_regimes_score_disjoint_rows(self) -> None:
        cell = _cell()
        rc = R.random_codebook_scores(cell, _rc_q(cell))
        b_pos = R.briers_vs_r0(rc, cell, regime=R.REGIME_POSITIVE)
        b_null = R.briers_vs_r0(rc, cell, regime=R.REGIME_NULL)
        pos_rows = b_pos[2] > 0
        null_rows = b_null[2] > 0
        self.assertFalse((pos_rows & null_rows).any())               # disjoint
        self.assertTrue((pos_rows | null_rows).any())


def _rc_q(cell):
    from clinical_jepa.eval import oracle_unlock as U
    return U._frozen_control_quants(8)[1]


if __name__ == "__main__":
    unittest.main()
