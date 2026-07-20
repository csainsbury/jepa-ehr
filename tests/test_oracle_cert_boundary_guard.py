"""Step-2 M0 order-core boundary (Pi P-A + guard-integration condition; thread thr-20260719T155635Z).

Two additive, synthetic-only pieces, both proven here:

1. `assert_canonical_certification_cell` — the fail-hard guard that REJECTS a `RestrictedOrderCore` / emission
   mask / restricted or non-canonical-L object at EVERY public certification/reference/verdict entrypoint, and
   is a NO-OP on a valid canonical L=8 cell (valid-path output unmoved). Rejection, never the recipe reshape.
2. `restrict_order_core` / `RestrictedOrderCore` — the production order-restriction primitive that replaces
   the tautological `_restrict` proof fixture: recomputes `future_events`, takes the nuisance column as an
   EXACT slice (no re-standardization), and materializes NO emission fields (that is M2 behaviour).

Also pins the additive `v2_certification_boundary_hash` and re-asserts the dev-scaffold + frozen v1 identities
are unmoved by the guard.
"""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from clinical_jepa.eval import oracle_references as RF
from clinical_jepa.eval.oracle_recipe import GoodContextRecipe, split_views
from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.oracle_realism_v2 import (
    CANONICAL_CERT_L, CERTIFICATION_ENTRYPOINTS, CertificationBoundaryError, RestrictedOrderCore,
    assert_canonical_certification_cell, restrict_order_core, v2_certification_boundary_hash,
    realism_v2_schema_hash,
)

FAMILIES = ("T_hmm_markov", "T_realized_history", "T_latent_factor",
            "E_no_h_exogenous", "E_offgrid_nonlinear")
NUIS = ("orthogonal", "correlated_leak")
_V2_BOUNDARY_HASH = "b33c2d9f6324c84763ebb85fde8912dbf0b84e94b7ee366e4adb879ceb14e8e4"
_DEV_SCAFFOLD_HASH = "2a7405ddadcdfdf3261a2b18e149c5523298a73d3e765cde78b6611927377673"
_EMISSION_FIELDS = ("future_timestamps", "cluster_ids", "multiplicity", "future_multiset")


class RestrictedOrderCorePrimitive(unittest.TestCase):
    def test_recompute_slice_and_emission_free_over_L2_to_8(self) -> None:
        for fam in FAMILIES:
            for nu in NUIS:
                cell = generate_literal_cell(fam, 0.35, nu, 120, seed=17)
                full = np.asarray(cell.true_order)
                L = full.shape[1]
                # contiguous prefix + a non-contiguous subset, over realized lengths 2..8
                subsets = [tuple(range(k)) for k in range(2, L + 1)]
                subsets += [(0, 2, 3, 6), (1, 4, 7), (2, 5), (0, L - 1)]
                for sub in subsets:
                    core = restrict_order_core(cell, sub)
                    self.assertIsInstance(core, RestrictedOrderCore)
                    self.assertEqual(core.realized_length, len(sub))
                    # future_events RECOMPUTED, not sliced
                    want_fe = np.argsort(np.argsort(full[:, list(sub)], axis=1), axis=1)
                    self.assertTrue(np.array_equal(core.future_events_subset, want_fe), f"{fam}/{nu}/{sub} fe")
                    # nuisance is the EXACT column slice (incl. correlated_leak, standardized over full L)
                    self.assertTrue(np.array_equal(core.nuisance_subset,
                                                   np.asarray(cell.nuisance_u)[:, list(sub)]), f"{fam}/{nu}/{sub} u")
                    self.assertTrue(np.array_equal(core.s_true_subset, full[:, list(sub)]), f"{fam}/{nu}/{sub} s")
                    self.assertTrue(np.array_equal(core.item_subset,
                                                   np.asarray(cell.item_features)[:, list(sub)]), "item")
                    # NO emission field is materialized on the order core
                    for ef in _EMISSION_FIELDS:
                        self.assertFalse(hasattr(core, ef), f"emission field {ef} must be absent")

    def test_pair_sign_invariance_against_full_order(self) -> None:
        # surviving-pair precedence signs match the full canonical order (the non-tautological property)
        cell = generate_literal_cell("T_latent_factor", 0.35, "correlated_leak", 200, seed=3)
        full = np.asarray(cell.true_order)
        for sub in [(0, 2, 3, 6), (1, 4, 7), tuple(range(5))]:
            core = restrict_order_core(cell, sub)
            k = len(sub)
            for a in range(k):
                for b in range(a + 1, k):
                    full_sign = np.sign(full[:, sub[a]] - full[:, sub[b]])
                    sub_sign = np.sign(core.s_true_subset[:, a] - core.s_true_subset[:, b])
                    self.assertTrue(np.array_equal(full_sign, sub_sign), f"{sub} pair {a},{b}")
                    # recomputed ranks reproduce the same sign
                    rank_sign = np.sign(core.future_events_subset[:, a] - core.future_events_subset[:, b])
                    self.assertTrue(np.array_equal(rank_sign, sub_sign), f"{sub} rank pair {a},{b}")

    def test_subset_validation(self) -> None:
        cell = generate_literal_cell("T_hmm_markov", 0.35, "orthogonal", 20, seed=1)
        with self.assertRaises(ValueError):      # L=1 belongs to M0b
            restrict_order_core(cell, (3,))
        with self.assertRaises(ValueError):      # out of range
            restrict_order_core(cell, (0, 99))
        with self.assertRaises(ValueError):      # duplicate positions
            restrict_order_core(cell, (2, 2, 3))
        with self.assertRaises(TypeError):       # not a LiteralCell
            restrict_order_core(restrict_order_core(cell, (0, 1)), (0, 1))


class CertificationBoundaryGuard(unittest.TestCase):
    def _valid_cell(self):
        return generate_literal_cell("T_latent_factor", 0.6, "orthogonal", 60, seed=5)

    def test_noop_on_valid_canonical_cell(self) -> None:
        cell = self._valid_cell()
        self.assertIsNone(assert_canonical_certification_cell(cell, entrypoint="test"))
        # valid-path outputs remain finite/computable through the guarded entrypoints
        self.assertTrue(np.isfinite(np.nanmean(RF.eo1_r0(cell))))
        self.assertTrue(np.isfinite(np.nanmean(RF.eo1_r_nuis(cell))))
        r = GoodContextRecipe()
        train = generate_literal_cell("T_latent_factor", 0.6, "orthogonal", 400, seed=99)
        r.fit(split_views(train), split_views(train))
        self.assertTrue(np.isfinite(np.nanmean(RF.eo1_recipe(r, cell, seed=0))))

    def test_rejects_restricted_order_core_at_every_reference_entrypoint(self) -> None:
        cell = self._valid_cell()
        core = restrict_order_core(cell, (0, 2, 3, 6))
        r = GoodContextRecipe()
        train = generate_literal_cell("T_latent_factor", 0.6, "orthogonal", 400, seed=99)
        r.fit(split_views(train), split_views(train))
        calls = {
            "eo1_recipe": lambda: RF.eo1_recipe(r, core, seed=0),
            "eo1_r0": lambda: RF.eo1_r0(core),
            "eo1_r_nuis": lambda: RF.eo1_r_nuis(core),
            "eo1_r_bayes": lambda: RF.eo1_r_bayes(core, ref_n=50, seed=0),
            "eo1_mean_embed_quantized": lambda: RF.eo1_mean_embed_quantized(core),
            "eo1_random_codebook": lambda: RF.eo1_random_codebook(core, seed=0),
            "hidden_null_excluded": lambda: RF.hidden_null_excluded(core, margin=0.05, seed=0),
        }
        # every reference entrypoint declared in the boundary identity is exercised here
        ref_entrypoints = [e for e in CERTIFICATION_ENTRYPOINTS if not e.startswith("verdict")]
        self.assertEqual(set(calls), set(ref_entrypoints), "guard coverage must match declared entrypoints")
        for name, fn in calls.items():
            with self.assertRaises(CertificationBoundaryError, msg=f"{name} must reject a RestrictedOrderCore"):
                fn()

    def test_rejects_variable_length_cell_masquerading_as_literalcell(self) -> None:
        cell = self._valid_cell()
        sub = [0, 2, 3, 6]
        sliced = dataclasses.replace(
            cell,
            true_order=np.asarray(cell.true_order)[:, sub],
            nuisance_u=np.asarray(cell.nuisance_u)[:, sub],
            item_features=np.asarray(cell.item_features)[:, sub],
            future_events=np.asarray(cell.future_events)[:, sub],
        )
        with self.assertRaises(CertificationBoundaryError):
            assert_canonical_certification_cell(sliced, entrypoint="test")
        with self.assertRaises(CertificationBoundaryError):
            RF.eo1_r0(sliced)

    def test_rejects_non_cell_object(self) -> None:
        with self.assertRaises(CertificationBoundaryError):
            assert_canonical_certification_cell({"true_order": np.zeros((3, 8))}, entrypoint="test")

    def test_canonical_length_constant(self) -> None:
        self.assertEqual(CANONICAL_CERT_L, 8)


class V2BoundaryIdentity(unittest.TestCase):
    def test_boundary_hash_pinned_and_scaffold_unmoved(self) -> None:
        self.assertEqual(v2_certification_boundary_hash(), _V2_BOUNDARY_HASH)
        # the guard is ADDITIVE: the dev-scaffold schema identity must not move
        self.assertEqual(realism_v2_schema_hash(), _DEV_SCAFFOLD_HASH)

    def test_frozen_v1_identities_unmoved_by_guard(self) -> None:
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_aggregate_extract import (
            base_schema_hash, generator_fit_schema_hash, extraction_code_identity,
        )
        self.assertEqual(invariant_hash(),
                         "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0")
        self.assertEqual(base_schema_hash(),
                         "13a0b4dedfd1ec773f29f680b5e752b2d6c7111cbc57d396704e6e75a619c8be")
        self.assertEqual(generator_fit_schema_hash(),
                         "b6aa74e1fd3ddc0565957328c8c7f489c87dd372eef1321af9344aea147b180f")
        # extraction_code_identity hashes literal/meta/etc source bytes but NOT references/verdict/realism_v2,
        # so the guard wiring leaves it unmoved.
        self.assertEqual(extraction_code_identity(),
                         "fd4b30be8c63a072cdcf35523443abcff45be6f7c1d12f138baa1e695829e6a0")


if __name__ == "__main__":
    unittest.main()
