"""Tests for the frozen contracts + registry state machine (Pi 2nd-pass Phase 0/1). Safe-public."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_contracts as OC
from clinical_jepa.eval import oracle_registry as R


def _spec(**over) -> OC.RecipeSpec:
    base = dict(
        architecture="ctx_mlp", target_encoder="vq", codebook_cfg={"K": 64, "bits": 6},
        losses={"order": 1.0, "recon": 0.5}, optimizer="adamw", schedule="cosine",
        bit_accounting={"target_bits": 6}, decode_policy="pairwise_prob",
        sampler_spec=OC.SamplerSpec(), decoder_sampler_spec=OC.DecoderSamplerSpec(),
        split_ids={"train": "s0", "dev": "s1", "sealed": "s2"}, seed_policy="sha256",
        evaluator_identity="evalcommit1", code_identity="codehash1",
    )
    base.update(over)
    return OC.RecipeSpec(**base)


def _assign(seed_ids=("k1", "k2")) -> OC.SplitAssignment:
    return OC.SplitAssignment(train=("a",), dev=("b",), sealed_cert=("c",),
                              family_ids=("T_latent_factor",), seed_ids=tuple(seed_ids))


def _artifact(rh) -> OC.FittedRecipeArtifact:
    return OC.FittedRecipeArtifact(originating_recipe_hash=rh, artifact_hash="art123")


class ContractHashTests(unittest.TestCase):
    def test_recipe_hash_deterministic_and_field_sensitive(self) -> None:
        a, b = _spec(), _spec()
        self.assertEqual(a.recipe_hash(), b.recipe_hash())
        self.assertNotEqual(a.recipe_hash(), _spec(optimizer="sgd").recipe_hash())
        self.assertNotEqual(a.recipe_hash(),
                            _spec(sampler_spec=OC.SamplerSpec(temperature=2.0)).recipe_hash())

    def test_capability_views_deny_out_of_scope_channels(self) -> None:
        ctx = OC.context_view({"context_features": [1, 2], "item_features": [3]})
        self.assertEqual(ctx.get("context_features"), [1, 2])
        with self.assertRaises(OC.CapabilityError):
            ctx.get("true_order")            # a label is not in the context capability
        with self.assertRaises(OC.CapabilityError):
            ctx.get("family_id")             # nor is family identity
        self.assertIn("context_features", ctx.accessed_channels())

    def test_view_rejects_data_with_nonallowed_channels(self) -> None:
        with self.assertRaises(OC.CapabilityError):
            OC.context_view({"context_features": [1], "null_flag": True})  # label leaked into context

    def test_recipe_and_family_meta_channels_are_disjoint(self) -> None:
        # the recipe never sees family ID / null status (anti-tailoring boundary)
        self.assertNotIn("family_id", OC.CONTEXT_CHANNELS)
        self.assertNotIn("null_flag", OC.CONTEXT_CHANNELS)
        self.assertIn("family_id", OC.FAMILY_META_CHANNELS)


class RegistryLifecycleTests(unittest.TestCase):
    def test_happy_path_to_authorization_ready(self) -> None:
        reg = R.OracleRegistry()
        rh = reg.register(_spec())
        self.assertEqual(reg.recipe_state(rh), R.RECIPE_REGISTERED)
        reg.assign(rh, _assign())
        self.assertEqual(reg.recipe_state(rh), R.RECIPE_ASSIGNED)
        self.assertEqual(reg.seed_state("k1"), R.SEED_ASSIGNED)
        reg.record_outcome(rh, R.OUTCOME_CERTIFIED, _artifact(rh),
                           evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="ph")
        self.assertEqual(reg.recipe_state(rh), R.RECIPE_EVALUATED)
        self.assertEqual(reg.recipe_outcome(rh), R.OUTCOME_CERTIFIED)
        self.assertEqual(reg.seed_state("k1"), R.SEED_RETIRED)   # seeds spent
        self.assertTrue(reg.authorization_ready(rh))

    def test_refuted_retires_seeds_but_is_not_authorization_ready(self) -> None:
        reg = R.OracleRegistry()
        rh = reg.register(_spec())
        reg.assign(rh, _assign())
        reg.record_outcome(rh, R.OUTCOME_REFUTED, _artifact(rh),
                           evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="ph")
        self.assertEqual(reg.seed_state("k1"), R.SEED_RETIRED)   # spent pass OR fail
        self.assertFalse(reg.authorization_ready(rh))            # refuted never authorizes

    def test_unlock_payload_hash_is_persisted_and_required(self) -> None:
        # Pi hardening #1: the outcome is only meaningful with the evidence hash it was computed from.
        reg = R.OracleRegistry()
        rh = reg.register(_spec())
        reg.assign(rh, _assign())
        with self.assertRaises(R.RegistryError):                 # empty payload hash refused
            reg.record_outcome(rh, R.OUTCOME_CERTIFIED, _artifact(rh), evaluator_identity="e",
                               mechanism_hash="m", calibration_hash="c", unlock_payload_hash="")
        reg.record_outcome(rh, R.OUTCOME_CERTIFIED, _artifact(rh), evaluator_identity="e",
                           mechanism_hash="m", calibration_hash="c", unlock_payload_hash="payload-abc")
        self.assertEqual(reg.unlock_payload_hash(rh), "payload-abc")   # persisted in the RECORD
        self.assertTrue(reg.authorization_ready(rh))

    def test_sealed_seed_reuse_refused(self) -> None:
        reg = R.OracleRegistry()
        rh1, rh2 = reg.register(_spec()), reg.register(_spec(optimizer="sgd"))
        reg.assign(rh1, _assign(("k1", "k2")))
        with self.assertRaises(R.RegistryError):
            reg.assign(rh2, _assign(("k2", "k3")))              # k2 already ASSIGNED -> reuse refused

    def test_illegal_transitions_refused(self) -> None:
        reg = R.OracleRegistry()
        rh = reg.register(_spec())
        with self.assertRaises(R.RegistryError):                # evaluate before assign
            reg.record_outcome(rh, R.OUTCOME_CERTIFIED, _artifact(rh),
                               evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="ph")
        reg.assign(rh, _assign())
        with self.assertRaises(R.RegistryError):                # double assign
            reg.assign(rh, _assign(("k9",)))
        with self.assertRaises(R.RegistryError):                # illegal outcome value
            reg.record_outcome(rh, "MAYBE", _artifact(rh),
                               evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="ph")

    def test_claimed_hash_mismatch_and_unknown_recipe_refused(self) -> None:
        reg = R.OracleRegistry()
        with self.assertRaises(R.RegistryError):
            reg.register(_spec(), claimed_hash="not-the-real-hash")
        with self.assertRaises(R.RegistryError):
            reg.assign("deadbeef", _assign())

    def test_artifact_must_originate_from_recipe(self) -> None:
        reg = R.OracleRegistry()
        rh = reg.register(_spec())
        reg.assign(rh, _assign())
        with self.assertRaises(R.RegistryError):
            reg.record_outcome(rh, R.OUTCOME_CERTIFIED, _artifact("someone-else"),
                               evaluator_identity="e", mechanism_hash="m", calibration_hash="c", unlock_payload_hash="ph")


if __name__ == "__main__":
    unittest.main()
