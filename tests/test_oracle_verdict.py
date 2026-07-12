"""Whole-pass acceptance matrix (Pi 2nd-pass Phase 6). Safe-public; no governed work, no manifest."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_verdict as V
from clinical_jepa.eval.oracle_recipe import (
    ContextBlindRecipe, GoodContextRecipe, LabelPeekingRecipe, NegligibleEffectRecipe,
)


class AcceptanceMatrixTests(unittest.TestCase):
    def test_good_recipe_crosses_every_gate_and_output_is_candidate_only(self) -> None:
        r = V.certify_recipe_candidate(lambda: GoodContextRecipe())
        self.assertEqual(r.outcome, V.CERTIFIED_CANDIDATE)
        self.assertTrue(all(c.status == "PASS" for c in r.cells))
        # a CANDIDATE aggregate result — it can NEVER populate policy or issue a governed manifest.
        self.assertFalse(r.governed_manifest_issued)
        self.assertFalse(r.can_populate_policy)

    def test_context_blind_recipe_refuted(self) -> None:
        r = V.certify_recipe_candidate(lambda: ContextBlindRecipe())
        self.assertEqual(r.outcome, V.REFUTED)               # no context signal => fails U1 on every cell
        self.assertTrue(all(c.status == "FAIL" for c in r.cells))

    def test_label_peeking_recipe_refuted_at_boundary(self) -> None:
        r = V.certify_recipe_candidate(lambda: LabelPeekingRecipe())
        self.assertEqual(r.outcome, V.REFUTED)
        self.assertEqual(r.reason, "boundary_guard_failed")  # caught before any cell scoring

    def test_negligible_effect_recipe_refuted(self) -> None:
        # a nonzero-but-negligible synthetic effect (mis-specified additive model) must NOT certify.
        r = V.certify_recipe_candidate(lambda: NegligibleEffectRecipe())
        self.assertEqual(r.outcome, V.REFUTED)

    def test_support_starved_cells_are_not_evaluable(self) -> None:
        r = V.certify_recipe_candidate(lambda: GoodContextRecipe(), n=300)  # < ORDER_SUPPORT_FLOOR
        self.assertEqual(r.outcome, V.REFUTED)               # nothing evaluable
        self.assertTrue(all(c.status == V.NOT_EVALUABLE for c in r.cells))

    def test_hidden_null_cell_is_excluded_not_a_pass(self) -> None:
        # a kappa=0 held-out cell is a hidden null: excluded, never counted as a recipe pass.
        r = V.certify_recipe_candidate(lambda: GoodContextRecipe(), kappa=0.0)
        self.assertEqual(r.outcome, V.REFUTED)
        self.assertTrue(all(c.status in ("HIDDEN_NULL", V.NOT_EVALUABLE) for c in r.cells))

    def test_evaluator_ready_when_all_family_null_controls_hold(self) -> None:
        self.assertTrue(V.evaluator_ready())


if __name__ == "__main__":
    unittest.main()
