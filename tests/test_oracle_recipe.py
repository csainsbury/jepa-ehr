"""Candidate-recipe boundary tests (Pi 2nd-pass Phase 3). Safe-public; no governed work."""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_recipe as RC
from clinical_jepa.eval import oracle_registry as REG
from clinical_jepa.eval.oracle_contracts import DecoderSamplerSpec, SamplerSpec
from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell


def _cells(fam="T_latent_factor", kappa=0.75, seed=1, n=800):
    train = generate_literal_cell(fam, kappa, "orthogonal", n, seed=seed)
    ev = generate_literal_cell(fam, kappa, "orthogonal", n, seed=seed + 100)
    return train, ev


class BoundaryGuardTests(unittest.TestCase):
    def test_good_recipe_passes_both_guards(self) -> None:
        train, ev = _cells()
        r = RC.GoodContextRecipe()
        r.fit(RC.split_views(train), RC.split_views(ev))
        self.assertTrue(RC.assert_predictor_context_only(r, ev))
        self.assertTrue(RC.assert_labels_eval_only(RC.GoodContextRecipe(), train, ev))

    def test_context_blind_passes_context_only_guard(self) -> None:
        train, ev = _cells()
        r = RC.ContextBlindRecipe()
        self.assertTrue(RC.assert_predictor_context_only(r, ev))   # invariant (blind) => context-only

    def test_label_peeking_is_caught_by_guard(self) -> None:
        train, ev = _cells()
        r = RC.LabelPeekingRecipe()
        self.assertFalse(RC.assert_predictor_context_only(r, ev))  # CapabilityError on true_order


class RecoveryTests(unittest.TestCase):
    def test_good_recipe_recovers_order_on_positive_cell(self) -> None:
        train, ev = _cells(kappa=0.75)
        r = RC.GoodContextRecipe()
        r.fit(RC.split_views(train), RC.split_views(ev))
        z = r.predict_latent(ev.context_view(), SamplerSpec(), seed=0)
        # predicted per-item scores correlate with the TRUE order on the positive (non-null) sequences
        pos = ~ev.is_null
        corrs = []
        for i in np.nonzero(pos)[0][:300]:
            a, b = z[i], ev.true_order[i]
            corrs.append(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        self.assertGreater(float(np.nanmean(corrs)), 0.3)          # genuine order recovery from context

    def test_context_blind_gives_uniform_pairwise(self) -> None:
        train, ev = _cells()
        r = RC.ContextBlindRecipe()
        z = r.predict_latent(ev.context_view(), SamplerSpec(), seed=0)
        probs = r.decode_order(z, DecoderSamplerSpec(), seed=0).pairwise_probs
        off = ~np.eye(probs.shape[1], dtype=bool)
        self.assertTrue(np.allclose(probs[:, off], 0.5))           # no order info => 0.5 everywhere


class DecodeTests(unittest.TestCase):
    def test_pairwise_probs_are_antisymmetric_and_calibrated(self) -> None:
        train, ev = _cells()
        r = RC.GoodContextRecipe(); r.fit(RC.split_views(train), RC.split_views(ev))
        z = r.predict_latent(ev.context_view(), SamplerSpec(), seed=0)
        p = r.decode_order(z, DecoderSamplerSpec(), seed=0).pairwise_probs
        L = p.shape[1]
        self.assertTrue(np.allclose(p + np.transpose(p, (0, 2, 1)), 1.0))   # P(i<j)+P(j<i)=1
        self.assertTrue(np.allclose(np.einsum("nii->ni", p), 0.5))          # diagonal 0.5

    def test_sampled_permutations_shape(self) -> None:
        train, ev = _cells(n=100)
        r = RC.GoodContextRecipe(); r.fit(RC.split_views(train), RC.split_views(ev))
        z = r.predict_latent(ev.context_view(), SamplerSpec(), seed=0)
        dec = r.decode_order(z, DecoderSamplerSpec(n_decode_samples=5, return_permutations=True), seed=0)
        self.assertEqual(dec.sampled_permutations.shape, (5, z.shape[0], z.shape[1]))


class RegistryIntegrationTests(unittest.TestCase):
    def test_recipe_hash_recomputed_and_registry_roundtrip(self) -> None:
        r = RC.GoodContextRecipe()
        self.assertNotEqual(r.recipe_hash(), RC.ContextBlindRecipe().recipe_hash())
        reg = REG.OracleRegistry()
        rh = reg.register(r.spec())
        self.assertEqual(rh, r.recipe_hash())                     # registry recomputes the same hash
        from clinical_jepa.eval.oracle_contracts import SplitAssignment
        reg.assign(rh, SplitAssignment(("a",), ("b",), ("c",), ("T_latent_factor",), ("k1",)))
        tr, dv = _cells(n=100)
        art = r.fit(RC.split_views(tr), RC.split_views(dv))
        reg.record_outcome(rh, REG.OUTCOME_CERTIFIED, art,
                           evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # fail-closed: readiness needs an APPROVED calibration identity from trusted policy, and that set
        # ships empty at this stage — a recorded CERTIFIED outcome alone never authorizes (Pi).
        self.assertFalse(reg.authorization_ready(rh))


if __name__ == "__main__":
    unittest.main()
