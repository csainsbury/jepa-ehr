"""Variable-length order-restriction invariance — the load-bearing precondition for a variable-length
realism redesign (Option D, `docs/oracle-realism-redesign-options-v1.md`).

To emit variable-length sequences, the certified order must become a DETERMINISTIC RESTRICTION of the
canonical fixed-L ranking to the realized items. That is valid only if the oracle's reference order
probabilities are PAIRWISE-LOCAL: each pair's P(a<b | .) depends only on that pair's classes and the
context/driver posterior, never on which OTHER items are present. If so, restricting to a subset preserves
every surviving pair's probability and the induced sub-ranking, and the frozen invariant is untouched.

These tests lock that property so a future generator change cannot silently break it.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from clinical_jepa.eval.oracle_meta_bayes import pi_star_pairwise, r0_pairwise
from clinical_jepa.eval.oracle_meta_gen import (
    HELDOUT_FAMILIES, KAPPA_MID, L_ITEMS, TRAIN_FAMILIES, generate_meta_cell, invariant_hash,
)

ALL_FAMILIES = (*TRAIN_FAMILIES, *HELDOUT_FAMILIES)
_SUBSET = (0, 2, 3, 6)          # a variable-length restriction: keep 4 of L_ITEMS=8 items


def _restrict(cell, subset):
    """Return a cell restricted to ``subset`` item positions — every item-indexed (N, L, ...) array is
    sub-selected; per-sequence arrays (context, driver, covariates, is_null) are unchanged."""
    sub = list(subset)

    def r(a):
        if a is None:
            return None
        a = np.asarray(a)
        return a[:, sub] if (a.ndim >= 2 and a.shape[1] == L_ITEMS) else a

    return dataclasses.replace(cell, **{f.name: r(getattr(cell, f.name)) for f in dataclasses.fields(cell)})


class OrderRestrictionInvarianceTests(unittest.TestCase):
    def test_pairwise_references_are_restriction_invariant(self) -> None:
        n = 200
        for fam in ALL_FAMILIES:
            cell = generate_meta_cell(fam, KAPPA_MID, "orthogonal", n, seed=17)
            r0_full = r0_pairwise(fam, KAPPA_MID, cell.item_classes)      # (n, L, L)
            ps_full = pi_star_pairwise(cell, KAPPA_MID)                   # (n, L, L)
            rc = _restrict(cell, _SUBSET)
            r0_res = r0_pairwise(fam, KAPPA_MID, rc.item_classes)         # (n, |S|, |S|)
            ps_res = pi_star_pairwise(rc, KAPPA_MID)
            idx = np.ix_(range(n), _SUBSET, _SUBSET)
            # surviving pairs' order probabilities are IDENTICAL (item-pair-local; not a tolerance match)
            self.assertEqual(float(np.max(np.abs(r0_res - r0_full[idx]))), 0.0, f"{fam} r0")
            self.assertEqual(float(np.max(np.abs(ps_res - ps_full[idx]))), 0.0, f"{fam} pi_star")

    def test_induced_subranking_is_the_restriction_of_the_full_ranking(self) -> None:
        for fam in ALL_FAMILIES:
            cell = generate_meta_cell(fam, KAPPA_MID, "orthogonal", 200, seed=17)
            rc = _restrict(cell, _SUBSET)
            full_sub = np.argsort(np.argsort(cell.true_order[:, list(_SUBSET)], 1), 1)
            restricted = np.argsort(np.argsort(rc.true_order, 1), 1)
            self.assertTrue(np.array_equal(full_sub, restricted), fam)

    def test_restriction_does_not_move_the_invariant(self) -> None:
        before = invariant_hash()
        cell = generate_meta_cell(TRAIN_FAMILIES[0], KAPPA_MID, "orthogonal", 50, seed=1)
        _restrict(cell, _SUBSET)
        self.assertEqual(invariant_hash(), before)


if __name__ == "__main__":
    unittest.main()
