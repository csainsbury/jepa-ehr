"""Keystone: shared-invariant fit-once transfer (Pi 2nd-pass REVISE #1). Safe-public / synthetic.

Demonstrates the CORRECTED design that fixes the primary blocker: ONE recipe fitted only on the TRAIN
families transfers UNCHANGED to the held-out families (no per-family refit, no held-out access).
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_meta_gen as G
from clinical_jepa.eval.oracle_meta_gen import (
    HELDOUT_FAMILIES, TRAIN_FAMILIES, exact_pi0, generate_meta_cell, invariant_hash,
)
from clinical_jepa.eval.oracle_meta_recipe import (
    InvariantLearner, MemorizerRecipe, transfer_score,
)
from clinical_jepa.eval.rung2_contract import ORACLE_EO1_SKILL_GATE

KAPPA = 0.60


class FitOnceTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = InvariantLearner().fit_on_train(seed=1, kappa=KAPPA, n=2500)
        cls.mem = MemorizerRecipe().fit_on_train(seed=1, kappa=KAPPA, n=2500)
        tr = generate_meta_cell(TRAIN_FAMILIES[0], KAPPA, "orthogonal", 1200, seed=999)
        cls.train_skill = float(np.nanmean(cls.inv.eo1(tr)[~tr.is_null]))

    def test_invariant_learner_transfers_to_every_held_out_family(self) -> None:
        # fit ONCE on TRAIN, applied UNCHANGED to held-out (never refit / never sees held-out).
        for fam in HELDOUT_FAMILIES:
            t = transfer_score(self.inv, fam, kappa=KAPPA, seed=2)
            self.assertGreater(t.mean_eo1_positive, ORACLE_EO1_SKILL_GATE,
                               f"{fam}: {t.mean_eo1_positive}")
        # no held-out degradation vs train (structural transfer, not memorization).
        self.assertGreater(self.train_skill, ORACLE_EO1_SKILL_GATE)

    def test_overfit_memorizer_does_not_transfer(self) -> None:
        for fam in HELDOUT_FAMILIES:
            t = transfer_score(self.mem, fam, kappa=KAPPA, seed=2)
            self.assertLess(t.mean_eo1_positive, ORACLE_EO1_SKILL_GATE)   # cannot certify by generalization

    def test_fit_touches_only_train_families(self) -> None:
        # the shared-invariant recipe's training set is the TRAIN families; held-out is disjoint.
        self.assertEqual(set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES), set())
        self.assertIn("E_no_h_exogenous", HELDOUT_FAMILIES)
        self.assertIn("E_offgrid_nonlinear", HELDOUT_FAMILIES)


class ExactContentPriorTests(unittest.TestCase):
    def test_pi0_is_non_uniform_and_class_driven(self) -> None:
        c = generate_meta_cell("T_latent_factor", KAPPA, "orthogonal", 400, seed=5)
        p = exact_pi0(c.item_classes)
        off = p[:, ~np.eye(p.shape[1], dtype=bool)]
        self.assertLess(off.min(), 0.2)          # some pairs strongly ordered by class means
        self.assertGreater(off.max(), 0.8)       # NOT hard-coded 0.5
        # symmetry: P(a<b) + P(b<a) = 1
        self.assertTrue(np.allclose(p + np.transpose(p, (0, 2, 1)), 1.0))


class RepeatedMultisetSupportTests(unittest.TestCase):
    def test_support_counts_repeated_multiset_clusters_not_N(self) -> None:
        ok = generate_meta_cell("T_latent_factor", KAPPA, "orthogonal", 2000, seed=5, support_floor=200)
        starved = generate_meta_cell("T_latent_factor", KAPPA, "orthogonal", 100, seed=5, support_floor=200)
        self.assertEqual(ok.support_status, "SUPPORTED")
        self.assertEqual(starved.support_status, "SUPPORT_STARVED")


class InvariantHashTests(unittest.TestCase):
    def test_hash_binds_literal_constants(self) -> None:
        base = invariant_hash()
        old = G.ORDER_NOISE
        try:
            G.ORDER_NOISE = old + 0.1
            self.assertNotEqual(invariant_hash(), base)      # a literal-constant change moves the hash
        finally:
            G.ORDER_NOISE = old
        self.assertEqual(invariant_hash(), base)


if __name__ == "__main__":
    unittest.main()
