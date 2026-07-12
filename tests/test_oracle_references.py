"""Reference bracket / paired-contrast / E-O1 / U6 tests (Pi 2nd-pass Phase 4a). Safe-public."""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_references as RF
from clinical_jepa.eval.oracle_recipe import (
    ContextBlindRecipe, GoodContextRecipe, split_views,
)
from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.rung2_contract import ORACLE_EO1_SKILL_GATE, ORACLE_R_BAYES_MC_TOL


def _cell(fam="T_latent_factor", kappa=0.75, cell="orthogonal", seed=1, n=800):
    return generate_literal_cell(fam, kappa, cell, n, seed=seed)


def _fit_good(fam="T_latent_factor", kappa=0.75, seed=1):
    tr = _cell(fam, kappa, seed=seed + 7000, n=3000)   # INDEPENDENT of any eval cell (no contamination)
    r = GoodContextRecipe(); r.fit(split_views(tr), split_views(tr))
    return r


class EO1Tests(unittest.TestCase):
    def test_r0_skill_is_zero(self) -> None:
        c = _cell()
        self.assertAlmostEqual(float(np.nanmean(RF.eo1_r0(c))), 0.0, places=6)

    def test_recipe_beats_r0_on_positive_and_does_not_predict_null_order(self) -> None:
        c = _cell()
        r = _fit_good()
        pos = ~c.is_null
        recipe = RF.eo1_recipe(r, c)
        # positive sequences: real skill; null sequences: NOT positive (a confident context predictor
        # applied to a genuine null scores <= 0 — it certainly does not predict the null's order).
        self.assertGreater(float(np.nanmean(recipe[pos])), ORACLE_EO1_SKILL_GATE)
        self.assertLess(float(np.nanmean(recipe[~pos])), 0.05)


class PairedContrastTests(unittest.TestCase):
    def test_recipe_minus_r0_positive_lower_ci(self) -> None:
        c = _cell(); r = _fit_good()
        pos = ~c.is_null                               # positive gate scores NON-null sequences only
        pc = RF.paired_contrast(RF.restrict_to(RF.eo1_recipe(r, c), pos),
                                RF.restrict_to(RF.eo1_r0(c), pos))
        self.assertGreater(pc.lower_ci, ORACLE_EO1_SKILL_GATE)   # practical margin, not merely > 0
        self.assertGreater(pc.mean_diff, ORACLE_EO1_SKILL_GATE)

    def test_paired_contrast_of_identical_arms_is_zero(self) -> None:
        c = _cell(); r = _fit_good()
        v = RF.eo1_recipe(r, c)
        pc = RF.paired_contrast(v, v)                       # same arm => exactly zero difference
        self.assertAlmostEqual(pc.mean_diff, 0.0, places=9)
        self.assertAlmostEqual(pc.lower_ci, 0.0, places=9)


class NuisanceTests(unittest.TestCase):
    def test_r_nuis_loses_skill_in_orthogonal_captures_leak(self) -> None:
        orth = _cell(cell="orthogonal", seed=2)
        leak = _cell(cell="correlated_leak", seed=3)
        self.assertLess(float(np.nanmean(RF.eo1_r_nuis(orth))), 0.05)     # orthogonal: no order info
        self.assertGreater(float(np.nanmean(RF.eo1_r_nuis(leak))), 0.05)  # leak: captures some


class U6ControlTests(unittest.TestCase):
    def test_mean_embed_quantized_is_order_blind(self) -> None:
        c = _cell(); r = _fit_good()
        me = float(np.nanmean(RF.eo1_mean_embed_quantized(c)))
        recipe = float(np.nanmean(RF.eo1_recipe(r, c)))
        self.assertLess(me, recipe)                          # order-blind pooling loses to the recipe

    def test_random_codebook_passes_null_fails_positive(self) -> None:
        pos = _cell(kappa=0.75, seed=4)
        self.assertLess(float(np.nanmean(RF.eo1_random_codebook(pos))), ORACLE_EO1_SKILL_GATE)  # no real skill


class HiddenNullTests(unittest.TestCase):
    def test_positive_cell_not_hidden_null_true_null_is(self) -> None:
        pos = _cell(kappa=0.75, seed=5)
        self.assertFalse(RF.hidden_null_excluded(pos, margin=0.05))       # ceiling beats prior
        null = _cell(kappa=0.0, seed=6)                                   # kappa=0 => no context signal
        self.assertTrue(RF.hidden_null_excluded(null, margin=0.05))


if __name__ == "__main__":
    unittest.main()
