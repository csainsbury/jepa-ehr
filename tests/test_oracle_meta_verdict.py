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


if __name__ == "__main__":
    unittest.main()
