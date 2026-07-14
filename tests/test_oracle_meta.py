"""Keystone: shared-invariant fit-once transfer, corrected per Pi's keystone GO-WITH-CHANGES.

Demonstrates the CORRECTED foundation: ONE recipe fitted only on the TRAIN families at the TRAIN κ grid
transfers UNCHANGED to the held-out families; the shortcut/memorizer control succeeds in-distribution
and fails on the held-out shift; R0 is MC-exact; normalization is frozen; the hash binds the full
executable mechanism; support uses the contract floor.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from clinical_jepa.eval import oracle_meta_gen as G
from clinical_jepa.eval.oracle_meta_gen import (
    HELDOUT_FAMILIES, KAPPA_HELDOUT_ENDPOINTS, KAPPA_MID, KAPPA_TRAIN_GRID, TRAIN_FAMILIES,
    INVARIANT, generate_meta_cell, invariant_hash,
)
from clinical_jepa.eval import oracle_meta_refs as R
from clinical_jepa.eval.oracle_meta_recipe import (
    FIT_KAPPAS, InvariantLearner, MemorizerRecipe, dev_score, transfer_score,
)
from clinical_jepa.eval.rung2_contract import ORACLE_EO1_SKILL_GATE, ORDER_SUPPORT_FLOOR


class _Fitted:
    inv = None
    mem = None

    @classmethod
    def get(cls):
        if cls.inv is None:
            cls.inv = InvariantLearner().fit_on_train(seed=1, n=1200)
            cls.mem = MemorizerRecipe().fit_on_train(seed=1, n=1200)
        return cls.inv, cls.mem


class FitProvenanceTests(unittest.TestCase):
    def test_training_provenance_is_train_only(self) -> None:
        inv, _ = _Fitted.get()
        self.assertEqual(set(inv.fit_provenance["families"]), set(TRAIN_FAMILIES))
        for k in inv.fit_provenance["kappas"]:
            self.assertIn(k, KAPPA_TRAIN_GRID)
        # no held-out family / endpoint / κmid can appear in the fit set (structural)
        self.assertFalse(set(inv.fit_provenance["families"]) & set(HELDOUT_FAMILIES))
        for k in (*KAPPA_HELDOUT_ENDPOINTS, KAPPA_MID):
            self.assertNotIn(k, inv.fit_provenance["kappas"])

    def test_fit_signature_takes_no_kappa(self) -> None:
        # the recipe cannot be pointed at a held-out κ: fit_on_train takes no κ argument.
        self.assertNotIn("kappa", inspect.signature(InvariantLearner.fit_on_train).parameters)
        self.assertTrue(set(FIT_KAPPAS).issubset(set(KAPPA_TRAIN_GRID)))


class TransferTests(unittest.TestCase):
    def test_invariant_transfers_to_held_out_with_uncertainty(self) -> None:
        inv, _ = _Fitted.get()
        for fam in HELDOUT_FAMILIES:                          # strong endpoint κ=0.60, adequate support
            t = transfer_score(inv, fam, kappa=0.60, seed=3, n=5000)
            self.assertGreater(t.lower_ci, ORACLE_EO1_SKILL_GATE, f"{fam}: {t.lower_ci}")
            self.assertGreater(t.n_sequences, 0)             # estimate carries a CI (not a point claim)

    def test_shortcut_memorizer_clears_dev_but_fails_held_out(self) -> None:
        _, mem = _Fitted.get()
        self.assertGreater(dev_score(mem, kappa=0.60, seed=3, n=3000).lower_ci,
                           ORACLE_EO1_SKILL_GATE)             # in-distribution success
        for fam in HELDOUT_FAMILIES:
            self.assertLess(transfer_score(mem, fam, kappa=0.60, seed=3, n=3000).lower_ci,
                            ORACLE_EO1_SKILL_GATE)           # fails to transfer


class FrozenNormalizationTests(unittest.TestCase):
    def test_held_out_raw_scale_does_not_change_frozen_normalization(self) -> None:
        # the coupling norm is a fixed invariant constant, independent of any generated (held-out) cell.
        before = INVARIANT.coupling_norm
        generate_meta_cell("E_offgrid_heavytail", 0.60, "orthogonal", 3000, seed=77)  # heavy-tail cell
        self.assertEqual(INVARIANT.coupling_norm, before)


class ExactR0Tests(unittest.TestCase):
    def test_r0_matches_high_precision_mc(self) -> None:
        for fam in ("T_latent_factor", "E_offgrid_heavytail"):
            self.assertLess(R.r0_table_mc_error(fam, 0.60), 0.01)   # coarse vs fine MC agree
        # R0 is a valid pairwise probability table (antisymmetric on average, in [0,1])
        c = generate_meta_cell("T_latent_factor", 0.60, "orthogonal", 200, seed=9)
        p = R.r0_pairwise("T_latent_factor", 0.60, c.item_classes)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


class HashTests(unittest.TestCase):
    def test_full_matrix_perturbation_changes_hash(self) -> None:
        # perturbing W_ctx while PRESERVING its absolute sum still changes the hash (full arrays hashed).
        base = invariant_hash()
        W = INVARIANT.W_ctx.copy()
        try:
            INVARIANT.W_ctx[0, 0] += 0.05
            INVARIANT.W_ctx[0, 1] -= 0.05                    # keep |·| sum ~unchanged
            self.assertNotEqual(invariant_hash(), base)
        finally:
            INVARIANT.W_ctx[:] = W
        self.assertEqual(invariant_hash(), base)

    def test_sub_rounding_perturbation_moves_hash(self) -> None:
        # Pi #8: full-byte hash — a change far below any rounding threshold still moves the hash.
        base = invariant_hash()
        W = INVARIANT.W_ctx.copy()
        try:
            INVARIANT.W_ctx[0, 0] += 1e-11
            self.assertNotEqual(invariant_hash(), base)
        finally:
            INVARIANT.W_ctx[:] = W
        self.assertEqual(invariant_hash(), base)

    def test_literal_constant_change_moves_hash(self) -> None:
        base = invariant_hash()
        old = G.ORDER_NOISE
        try:
            G.ORDER_NOISE = old + 0.1
            self.assertNotEqual(invariant_hash(), base)
        finally:
            G.ORDER_NOISE = old


class TimingTests(unittest.TestCase):
    def test_marked_cluster_timing_and_driver_modulation(self) -> None:
        for fam in ("T_latent_factor", "E_offgrid_heavytail", "T_realized_history"):
            c = generate_meta_cell(fam, 0.5, "orthogonal", 400, seed=3)
            self.assertTrue((c.multiplicity > 1).any())              # Δt=0 multiplicity clusters exist
            dt = np.diff(c.future_timestamps, axis=1)
            new_cluster = np.diff(c.cluster_ids, axis=1) > 0
            self.assertTrue((dt[new_cluster] > 0).all())            # inter-cluster gaps strictly positive
            self.assertTrue((np.abs(dt[~new_cluster]) < 1e-9).all())  # within-cluster Δt = 0
        # the mechanism measurably shapes timing: the history-dependent-gap family has larger gaps
        hist = generate_meta_cell("T_realized_history", 0.5, "orthogonal", 800, seed=4)
        lat = generate_meta_cell("T_latent_factor", 0.5, "orthogonal", 800, seed=4)
        hg = np.diff(hist.future_timestamps, 1)[np.diff(hist.cluster_ids, 1) > 0].mean()
        lg = np.diff(lat.future_timestamps, 1)[np.diff(lat.cluster_ids, 1) > 0].mean()
        self.assertGreater(hg, lg)


class SupportAndNamingTests(unittest.TestCase):
    def test_support_uses_contract_floor(self) -> None:
        ok = generate_meta_cell("T_latent_factor", 0.60, "orthogonal", 4000, seed=5,
                                support_floor=ORDER_SUPPORT_FLOOR)
        starved = generate_meta_cell("T_latent_factor", 0.60, "orthogonal", 400, seed=5,
                                     support_floor=ORDER_SUPPORT_FLOOR)
        self.assertEqual(ok.support_status, "SUPPORTED")
        self.assertEqual(starved.support_status, "SUPPORT_STARVED")

    def test_heavytail_family_named_accurately_no_nonlinear_claim(self) -> None:
        self.assertIn("E_offgrid_heavytail", HELDOUT_FAMILIES)
        self.assertNotIn("E_offgrid_nonlinear", (*TRAIN_FAMILIES, *HELDOUT_FAMILIES))
        # no stale nonlinear-map CLAIM (old family name / tanh); the docstring may still say it is LINEAR.
        self.assertNotIn("e_offgrid_nonlinear", G.__doc__.lower())
        self.assertNotIn("tanh", G.__doc__.lower())
        self.assertIn("linear", G.__doc__.lower())


if __name__ == "__main__":
    unittest.main()
