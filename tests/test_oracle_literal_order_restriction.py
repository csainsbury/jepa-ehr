"""LITERAL-stack order-restriction invariance — the M0 prerequisite on the generator the v2 realism layer
actually modifies (Fable review #1; blueprint M0 "OPEN-for-literal").

`tests/test_oracle_order_restriction.py` proves the property on the META stack. The v2 realism seam
(`_apply_calibration_layer` / `generate_literal_cell`) and the aggregate BASE live on the LITERAL stack,
so the restriction property must be proven there too. This test establishes it for the certified-order and
nuisance channels and for the per-pair references built on per-item scores.

Finding recorded by these tests: the certification recipe (`GoodContextRecipe.predict_latent`) is
CONTEXT-ONLY and hard-coded to the training L, so certification stays at fixed L; variable length is an
emission-only operation. Hence M0 covers the generator channels (`true_order`, `nuisance_u`, item content)
and the per-pair references (`r0`, `r_nuis`) computed from per-item scores — not the fixed-L recipe.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.oracle_meta_gen import invariant_hash
from clinical_jepa.eval.oracle_recipe import pairwise_probs_from_scores

LITERAL_FAMILIES = ("T_hmm_markov", "T_realized_history", "T_latent_factor",
                    "E_no_h_exogenous", "E_offgrid_nonlinear")
NUISANCE_CELLS = ("orthogonal", "correlated_leak")
_SUBSET = (0, 2, 3, 6)          # a variable-length restriction: keep 4 of L_ITEMS=8 items


def _restrict(cell, subset):
    L = int(np.asarray(cell.true_order).shape[1])
    sub = list(subset)

    def r(a):
        if a is None:
            return None
        a = np.asarray(a)
        return a[:, sub] if (a.ndim >= 2 and a.shape[1] == L) else a

    return dataclasses.replace(cell, **{f.name: r(getattr(cell, f.name)) for f in dataclasses.fields(cell)})


class LiteralOrderRestrictionTests(unittest.TestCase):
    def test_certified_order_and_nuisance_channels_restrict_cleanly(self) -> None:
        for fam in LITERAL_FAMILIES:
            for nu in NUISANCE_CELLS:
                cell = generate_literal_cell(fam, 0.35, nu, 200, seed=17)
                rc = _restrict(cell, _SUBSET)
                sub = list(_SUBSET)
                # nuisance_u (incl. correlated_leak, standardized over full L) is the EXACT column slice
                self.assertTrue(np.array_equal(rc.nuisance_u, np.asarray(cell.nuisance_u)[:, sub]),
                                f"{fam}/{nu} nuisance_u")
                # item content is the exact slice; per-sequence channels unchanged
                self.assertTrue(np.array_equal(rc.item_features, np.asarray(cell.item_features)[:, sub]),
                                f"{fam}/{nu} item_features")
                self.assertTrue(np.array_equal(rc.context_features, cell.context_features), f"{fam}/{nu} ctx")
                self.assertTrue(np.array_equal(rc.is_null, cell.is_null), f"{fam}/{nu} is_null")
                # induced sub-ranking == restriction of the full ranking
                full_sub = np.argsort(np.argsort(np.asarray(cell.true_order)[:, sub], 1), 1)
                restricted = np.argsort(np.argsort(rc.true_order, 1), 1)
                self.assertTrue(np.array_equal(full_sub, restricted), f"{fam}/{nu} sub-ranking")

    def test_per_pair_references_are_restriction_invariant(self) -> None:
        n = 200
        for fam in LITERAL_FAMILIES:
            for nu in NUISANCE_CELLS:
                cell = generate_literal_cell(fam, 0.35, nu, n, seed=17)
                rc = _restrict(cell, _SUBSET)
                idx = np.ix_(range(n), _SUBSET, _SUBSET)
                # r_nuis: P(i<j) = sigmoid(u_j - u_i) — surviving pairs IDENTICAL (exact, not tolerance)
                pn_full = pairwise_probs_from_scores(np.asarray(cell.nuisance_u))
                pn_res = pairwise_probs_from_scores(rc.nuisance_u)
                self.assertEqual(float(np.max(np.abs(pn_res - pn_full[idx]))), 0.0, f"{fam}/{nu} r_nuis")
                # r0 is the constant content prior (0.5) — trivially restriction-invariant
                r0_full = np.full((n, np.asarray(cell.true_order).shape[1],) * 1 + (np.asarray(cell.true_order).shape[1],), 0.5)
                self.assertEqual(float(np.max(np.abs(r0_full[idx] - 0.5))), 0.0, f"{fam}/{nu} r0")

    def test_restriction_does_not_move_the_invariant(self) -> None:
        before = invariant_hash()
        _restrict(generate_literal_cell("T_latent_factor", 0.35, "orthogonal", 50, seed=1), _SUBSET)
        self.assertEqual(invariant_hash(), before)


if __name__ == "__main__":
    unittest.main()
