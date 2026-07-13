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
        self.assertTrue(inv.authorization_ready)      # synthetic-recovery readiness (NOT governed T4)
        self.assertFalse(mem.authorization_ready)     # a refuted recipe never authorizes

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


class PureFunctionTests(unittest.TestCase):
    def test_certify_from_unlock_is_deterministic_pure_function(self) -> None:
        inv, _ = _Runs.get()
        # re-deriving the verdict from the SAME unlock gives the same outcome (pure function of it).
        recipe = InvariantLearner().fit_on_train(seed=0)
        ue = U.compute_unlock(recipe, seed=0)
        v1, v2 = U.certify_from_unlock(ue), U.certify_from_unlock(ue)
        self.assertEqual(v1.outcome, v2.outcome)
        self.assertEqual(v1.outcome, U.CERTIFIED_CANDIDATE)


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
    def test_low_endpoint_is_hidden_null_and_precision_holds(self) -> None:
        recipe = InvariantLearner().fit_on_train(seed=0)
        ue = U.compute_unlock(recipe, seed=0)
        self.assertGreaterEqual(ue.n_hidden_null, 1)                 # weak κ=0.15 endpoint(s) excluded
        for f in ue.families:                                       # a low_oc cell is hidden-null, not a pass
            low = [c for c in f.cells if c.role == "low_oc"][0]
            self.assertIn(low.status, ("HIDDEN_NULL", "SUPPORT_STARVED"))
        self.assertTrue(ue.precision["passes"])
        self.assertTrue(ue.train_family_readiness)


class _ContaminatedRecipe(InvariantLearner):
    """Simulates a recipe that (illegally) touched a held-out family during fit."""
    def fit_on_train(self, *, seed: int = 0, n: int = 1200):
        super().fit_on_train(seed=seed, n=n)
        self.fit_provenance["families"] = sorted(set(self.fit_provenance["families"]) | {"E_no_h_exogenous"})
        return self


class ContaminationAndDeterminismTests(unittest.TestCase):
    def test_held_out_contamination_is_refused(self) -> None:
        # a recipe whose fit provenance includes a held-out family is refused before scoring.
        with self.assertRaises(RuntimeError):
            V.certify_recipe(lambda: _ContaminatedRecipe(), seed=0)

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
