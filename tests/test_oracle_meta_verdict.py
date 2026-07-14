"""UnlockEvaluation + registry-integrated verdict — corrected acceptance matrix (Pi keystone #4/#7).

Safe-public / synthetic. Candidate-only: no manifest, no policy population, governed T4 LOCKED.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_meta_verdict as V
from clinical_jepa.eval import oracle_registry as REG
from clinical_jepa.eval import oracle_unlock as U
from clinical_jepa.eval.oracle_meta_ledger import CI_ENDPOINTS, build_ledger
from clinical_jepa.eval.oracle_meta_gen import HELDOUT_FAMILIES, KAPPA_HELDOUT_ENDPOINTS
from clinical_jepa.eval.oracle_meta_recipe import InvariantLearner, MemorizerRecipe


class _Runs:
    inv = None
    mem = None

    @classmethod
    def get(cls):
        if cls.inv is None:
            cls.inv = V.certify_recipe(lambda: InvariantLearner(), seed=0)
            cls.mem = V.certify_recipe(lambda: MemorizerRecipe(), seed=0)
        return cls.inv, cls.mem


class AcceptanceMatrixTests(unittest.TestCase):
    def test_invariant_certifies_memorizer_refuted(self) -> None:
        inv, mem = _Runs.get()
        self.assertEqual(inv.verdict.outcome, U.CERTIFIED_CANDIDATE)   # fit-once transfer succeeds
        self.assertEqual(mem.verdict.outcome, U.REFUTED)               # shortcut memorizer fails held-out

    def test_output_is_candidate_only(self) -> None:
        inv, _ = _Runs.get()
        self.assertFalse(inv.verdict.governed_manifest_issued)         # never issues a manifest
        self.assertFalse(inv.verdict.can_populate_policy)              # never populates policy

    def test_registry_records_outcome_through_legal_transitions(self) -> None:
        inv, mem = _Runs.get()
        self.assertEqual(inv.registry_outcome, REG.OUTCOME_CERTIFIED)
        self.assertEqual(mem.registry_outcome, REG.OUTCOME_REFUTED)
        self.assertTrue(inv.synthetic_registry_complete)   # certified + seeds retired + identities
        self.assertFalse(mem.synthetic_registry_complete)
        # authorization-ready requires a REAL approved calibration hash -> False in this stage (Pi #9).
        self.assertFalse(inv.authorization_ready)
        self.assertFalse(mem.authorization_ready)

    def test_predict_is_capability_restricted(self) -> None:
        # a recipe that reaches for an eval label through the ContextView is DENIED (physical boundary).
        from clinical_jepa.eval.oracle_meta_recipe import LabelPeekingRecipe
        from clinical_jepa.eval.oracle_contracts import CapabilityError
        from clinical_jepa.eval.oracle_meta_gen import generate_meta_cell
        cell = generate_meta_cell("E_no_h_exogenous", 0.60, "orthogonal", 50, seed=1)
        with self.assertRaises(CapabilityError):
            cell.context_view().get("true_order")                        # label not in the context capability
        with self.assertRaises(RuntimeError):                            # denied end-to-end
            V.certify_recipe(lambda: LabelPeekingRecipe(), seed=0)

    def test_recipe_and_artifact_identities_are_bound(self) -> None:
        inv, _ = _Runs.get()
        self.assertTrue(inv.recipe_hash and inv.artifact_hash)
        self.assertNotEqual(inv.recipe_hash, inv.artifact_hash)   # spec identity separate from fitted


class LedgerTests(unittest.TestCase):
    def test_ledger_has_fixed_expected_cardinality(self) -> None:
        led = build_ledger()
        # CI-based hypotheses = families × endpoints × CI endpoints (fixed, not recipe-dependent).
        self.assertEqual(led.n_ci, len(HELDOUT_FAMILIES) * len(KAPPA_HELDOUT_ENDPOINTS) * len(CI_ENDPOINTS))
        self.assertAlmostEqual(led.ci_alpha(), 0.05 / led.n_ci)
        # hidden-null / support exclusions cannot shrink the ledger (denominator is pre-declared).
        self.assertEqual(build_ledger().n_ci, led.n_ci)


def _valid_identities():
    from clinical_jepa.eval.oracle_meta_recipe import InvariantLearner, sampler_fingerprint
    from clinical_jepa.eval.oracle_meta_gen import invariant_hash
    r = InvariantLearner().fit_on_train(seed=0)
    return r, {"recipe_hash": r.recipe_hash(), "artifact_hash": r.artifact().artifact_hash,
               "mechanism_hash": invariant_hash(), "evaluator_identity": r.spec().evaluator_identity,
               "sampler_fingerprint": sampler_fingerprint(r), "bit_accounting": r.spec().bit_accounting,
               "split_assignment_hash": "sa", "seed_ids": ["k1"], "access_trace": {"families": ["T_latent_factor"]}}


class PureFunctionTests(unittest.TestCase):
    def test_certify_from_unlock_is_deterministic_pure_function(self) -> None:
        recipe, ids = _valid_identities()
        ue = U.compute_unlock(recipe, seed=0, identities=ids)
        v1, v2 = U.certify_from_unlock(ue), U.certify_from_unlock(ue)
        self.assertEqual(v1.outcome, v2.outcome)
        self.assertEqual(v1.outcome, U.CERTIFIED_CANDIDATE)

    def test_fabricated_unlock_is_refused(self) -> None:
        import dataclasses
        recipe, ids = _valid_identities()
        ue = U.compute_unlock(recipe, seed=0, identities=ids)
        self.assertEqual(U.certify_from_unlock(dataclasses.replace(ue, ledger_hash="FAKE")).reason,
                         "ledger_identity_mismatch")
        self.assertEqual(U.certify_from_unlock(dataclasses.replace(ue, identities={})).reason,
                         "missing_identity_fields")
        # a ledger-cardinality lie is refused too
        self.assertEqual(U.certify_from_unlock(dataclasses.replace(ue, ledger_cardinality=999)).reason,
                         "ledger_identity_mismatch")


class SamplerAndBitTests(unittest.TestCase):
    def test_sampled_decode_is_reproducible_and_sampler_is_exercised(self) -> None:
        import numpy as np
        from clinical_jepa.eval.oracle_meta_recipe import sampled_pairwise_probs
        from clinical_jepa.eval.oracle_meta_gen import generate_meta_cell
        r = InvariantLearner().fit_on_train(seed=0)
        self.assertGreater(r.spec().sampler_spec.n_latent_samples, 1)   # a stochastic sampler is registered
        cell = generate_meta_cell("E_no_h_exogenous", 0.60, "orthogonal", 200, seed=9)
        p1 = sampled_pairwise_probs(r, cell, seed=5)
        p2 = sampled_pairwise_probs(r, cell, seed=5)
        self.assertTrue(np.allclose(p1, p2))                            # deterministic given the seed

    def test_sampler_fingerprint_mismatch_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            V.certify_recipe(lambda: InvariantLearner(), seed=0, presented_sampler_fingerprint="WRONG")

    def test_bit_accounting_comes_from_the_registered_recipe(self) -> None:
        r = InvariantLearner()
        self.assertIn("control_bits", r.spec().bit_accounting)
        self.assertIn("target_bits", r.spec().bit_accounting)


class HiddenNullAndPrecisionTests(unittest.TestCase):
    def test_kappa0_is_hidden_null_and_precision_holds(self) -> None:
        # the hidden-null anchor is κ=0 (no context-predictable order): R_bayes − R0 does not clear the
        # margin there. The κ=0.15 held-out endpoint DOES carry signal under the corrected exact
        # estimands, so it is a valid low-OC cell, not necessarily hidden-null.
        from clinical_jepa.eval import oracle_meta_refs as RF
        from clinical_jepa.eval.oracle_meta_gen import generate_meta_cell
        null = generate_meta_cell("E_no_h_exogenous", 0.0, "orthogonal", 2000, seed=5)
        b = RF.briers_from_probs(RF.r_bayes_probs(null), null)
        rb = RF.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=1)[1]
        from clinical_jepa.eval.rung2_contract import ORACLE_R_BAYES_MARGIN
        self.assertLess(rb, ORACLE_R_BAYES_MARGIN)                   # κ=0 is a genuine hidden null

        recipe = InvariantLearner().fit_on_train(seed=0)
        ue = U.compute_unlock(recipe, seed=0)
        for f in ue.families:                                       # every low_oc cell has a valid status
            low = [c for c in f.cells if c.role == "low_oc"][0]
            self.assertIn(low.status, ("SUPPORTED", "HIDDEN_NULL", "SUPPORT_STARVED"))
        self.assertTrue(ue.precision["passes"])
        self.assertTrue(ue.train_family_readiness)


class _LyingRecipe(InvariantLearner):
    """Self-reports a clean provenance but the EXTERNAL loader trace is authoritative — a recipe cannot
    lie its way past contamination since the loader is the only data source."""
    def fit(self, loader, *, max_pairs: int = 40000):
        super().fit(loader, max_pairs=max_pairs)
        self.fit_provenance = {"families": ["E_no_h_exogenous"], "kappas": [0.60]}  # a LIE (ignored)
        return self


class ContaminationAndDeterminismTests(unittest.TestCase):
    def test_contamination_uses_external_loader_trace_not_self_report(self) -> None:
        # the registry-owned loader is the ONLY data source; its trace is train-only and authoritative.
        from clinical_jepa.eval.oracle_meta_recipe import RegistryDataLoader
        from clinical_jepa.eval.oracle_meta_gen import KAPPA_TRAIN_GRID, HELDOUT_FAMILIES
        loader = RegistryDataLoader(V._split_assignment("t"), seed=0)
        list(loader.train_iter()); loader.dev_cell()
        tr = loader.access_trace()
        self.assertFalse(set(tr["families"]) & set(HELDOUT_FAMILIES))     # never a held-out family
        self.assertTrue(set(tr["kappas"]) <= set(KAPPA_TRAIN_GRID))       # never a non-train κ
        # a recipe that LIES about its provenance still certifies via the (clean) external trace.
        self.assertEqual(V.certify_recipe(lambda: _LyingRecipe(), seed=0).registry_outcome,
                         REG.OUTCOME_CERTIFIED)

    def test_recipe_and_artifact_hashes_are_deterministic(self) -> None:
        r1 = InvariantLearner().fit_on_train(seed=0)
        r2 = InvariantLearner().fit_on_train(seed=0)
        self.assertEqual(r1.recipe_hash(), r2.recipe_hash())
        self.assertEqual(r1.artifact().artifact_hash, r2.artifact().artifact_hash)

    def test_artifact_hash_stable_across_held_out_evaluation(self) -> None:
        # compute_unlock does NOT refit; the fitted artifact identity is unchanged by evaluation.
        r = InvariantLearner().fit_on_train(seed=0)
        before = r.artifact().artifact_hash
        U.compute_unlock(r, seed=0)
        self.assertEqual(r.artifact().artifact_hash, before)

    def test_verdict_outcome_reproducible(self) -> None:
        # same recipe + seed -> same outcome and identities (determinism; PYTHONHASHSEED checked in shell).
        inv, _ = _Runs.get()
        again = V.certify_recipe(lambda: InvariantLearner(), seed=0)
        self.assertEqual(inv.verdict.outcome, again.verdict.outcome)
        self.assertEqual(inv.recipe_hash, again.recipe_hash)
        self.assertEqual(inv.artifact_hash, again.artifact_hash)


if __name__ == "__main__":
    unittest.main()
