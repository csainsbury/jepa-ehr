"""Rung-2 CONTRACT fail-hard tests (Pi v2 re-gate #6) — the mandatory pre-registration checks:
gate independence, direct-vs-recursive path discipline, MIMIC-2d non-gating, missing-support
NOT_EVALUABLE, T4 governed refusal without an oracle, oracle-assisted strata, nomination-only,
and test sealing. All numeric gates are concrete constants (no un-frozen curves)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung2_contract as C
from clinical_jepa.eval import oracle_policy as OP


class FrozenNumbersTests(unittest.TestCase):
    def test_all_gates_are_concrete_numbers(self) -> None:
        # every categorical gate has a concrete numeric threshold (Pi: "will be frozen" is not a gate)
        for v in (C.PRECEDENCE_SKILL_GATE, C.ORDER_SWAP_EXCESS_GATE, C.ORDER_T0_IMPROVEMENT_GATE,
                  C.GATE_4A_MULTIPLICITY_SKILL, C.GATE_4A_SWAP_SKILL, C.GATE_4A_ECE,
                  C.GATE_4B_KS, C.GATE_4B_CRPS_SKILL, C.GATE_4B_RATE_HEAD_IMPROVEMENT, C.GATE_4B_SWAP,
                  C.COUNT_NOMINATE_MARGIN, C.EXPOSURE_GAP_MARGIN, C.PERT_EPS):
            self.assertIsInstance(v, (int, float))
        self.assertEqual(len(C.config_hash({"x": 1})), 64)
        self.assertFalse(C.frozen_contract()["cross_cutting"]["test_access"])

    def test_rho_t_descriptive_only(self) -> None:
        self.assertEqual(C.RHO_T_ROLE, "descriptive_only")   # Pi #1 — never load-bearing

    def test_ct_head_not_marked_tpp(self) -> None:
        self.assertEqual(C.CT_HEAD_NAME, "continuous_time_multiplicity_head")  # marks not gated


class GateIndependenceTests(unittest.TestCase):
    def test_shared_trained_artifact_fails_hard(self) -> None:
        prov = {"sg3_order": {"trained_checkpoint": "ck_A", "run_id": "r3"},
                "sg4_time": {"trained_checkpoint": "ck_A", "run_id": "r4"}}   # SHARED trained ckpt
        with self.assertRaises(AssertionError):
            C.assert_gates_independent(prov)

    def test_common_frozen_encoder_allowed(self) -> None:
        # a common FROZEN context encoder is fine; only trained artifacts / run ids are policed.
        prov = {"sg3_order": {"frozen_encoder": "enc0", "run_id": "r3", "optimizer": "o3"},
                "sg4_time": {"frozen_encoder": "enc0", "run_id": "r4", "optimizer": "o4"}}
        self.assertTrue(C.assert_gates_independent(prov))


class RolloutPathTests(unittest.TestCase):
    def test_direct_path_forbids_exposure_gap(self) -> None:
        with self.assertRaises(AssertionError):
            C.validate_direct_path_row({"horizon": 90.0, "exposure_gap": 0.1})
        self.assertTrue(C.validate_direct_path_row({"horizon": 90.0, "d_horizon_decay": 0.2}))

    def test_recursive_path_needs_transition_trained_checkpoint(self) -> None:
        self.assertFalse(C.recursive_path_evaluable({"fixed_width_transition_trained": False}))
        self.assertFalse(C.recursive_path_evaluable(None))                    # missing => NOT_EVALUABLE
        self.assertTrue(C.recursive_path_evaluable({"fixed_width_transition_trained": True}))


class PrimaryCellTests(unittest.TestCase):
    def test_mimic_2d_never_gates(self) -> None:
        self.assertFalse(C.is_primary_cell("MIMIC", 2.0))
        self.assertTrue(C.is_primary_cell("MIMIC", 0.5))
        self.assertTrue(C.is_primary_cell("SCID", 730.0))


class OrderSupportTests(unittest.TestCase):
    def test_missing_support_is_not_evaluable(self) -> None:
        self.assertEqual(C.order_support_status(499), C.NOT_EVALUABLE)
        self.assertEqual(C.order_support_status(500), "SUPPORTED")


def _full_oracle_manifest() -> dict:
    """A fully-certified, recipe-bound, property-specific v3 order-T4 authorization manifest."""
    return {
        "schema_version": C.ORACLE_SCHEMA_VERSION, "oracle_mechanism_hash": "mech123",
        "evaluator_commit": "abc123", "certified_recipe_hash": "recipeXYZ", "recipe_registry_id": "reg1",
        "held_out_family_ids": ["E1", "E2"], "sealed_cert_run_id": "run77", "gate_event_ref": "evt-oracle-1",
        "blueprint_hash": "bp999", "oracle_frozen": True, "pi_gate": "PASS",
        "verdict": "synthetic_recovery_CERTIFIED", "codebook_postdates_oracle": True,
        "labels_eval_only_verified": True, "governed_t4_real_output_ceiling": "NOMINATE",
        "transfer_caveat": "synthetic recovery only; real claims stay NOMINATE",
        "unlock_checks": {c: "PASS" for c in C.ORDER_UNLOCK_CHECKS},
        "precision_sim": {"adequate": True}, "realism_envelope": {"within_envelope": True},
        "reference_bounds": {"R_bayes_beats_R0": True, "R0_null_pass": True, "R0_positive_fail": True,
                             "nuisance_incremental_margin_ok": True, "evaluator_realized_alpha": 0.04},
    }


def _test_policy() -> dict:
    """A POPULATED trusted policy that matches _full_oracle_manifest (test-only; the committed
    APPROVED_ORACLE_POLICY stays EMPTY until the implementation gate freezes it)."""
    return {
        "blueprint_hash": "bp999", "gate_event_ref": "evt-oracle-1", "oracle_mechanism_hash": "mech123",
        "schema_version": C.ORACLE_SCHEMA_VERSION,
        "evaluator_commits": ["abc123"], "recipe_registry_ids": ["reg1"], "sealed_cert_run_ids": ["run77"],
    }


def _patch_committed(policy=None):
    """Patch the COMMITTED policy (there is NO caller-injectable policy arg on the guard — Pi #1).
    Tests must patch the trust root, not pass it in."""
    return mock.patch.object(OP, "APPROVED_ORACLE_POLICY", policy if policy is not None else _test_policy())


def _ok(m):  # a passing governed call supplies the recipe hash; trust root is the (patched) committed policy
    with _patch_committed():
        return C.t4_governed_allowed(True, m, presented_recipe_hash="recipeXYZ")


class OracleStopLineTests(unittest.TestCase):
    def test_synthetic_always_allowed_governed_needs_full_cert(self) -> None:
        self.assertTrue(C.requires_oracle(C.T4_TARGET))
        self.assertTrue(C.t4_governed_allowed(False, None))                 # synthetic scaffolding OK
        self.assertFalse(C.t4_governed_allowed(True, None))
        self.assertFalse(C.t4_governed_allowed(True, {"oracle_frozen": True, "pi_gate": "PASS"}))
        self.assertTrue(_ok(_full_oracle_manifest()))

    def test_no_caller_injectable_trust_root(self) -> None:
        # Pi #1 (reproduced defect): the guard must expose NO policy= parameter. Supplying a matching
        # ad-hoc policy must be impossible — the only trust root is the committed module global.
        import inspect
        sig = inspect.signature(C.t4_governed_allowed)
        self.assertNotIn("policy", sig.parameters)
        self.assertNotIn("expected_schema_version", sig.parameters)

    def test_empty_committed_policy_refuses_even_a_full_manifest(self) -> None:
        # Pi #1: the trust anchor is the COMMITTED policy, which ships EMPTY -> fail-closed for all.
        self.assertFalse(OP.policy_is_populated())                          # ships empty
        self.assertFalse(C.t4_governed_allowed(True, _full_oracle_manifest(),
                                               presented_recipe_hash="recipeXYZ"))  # default committed policy
        with _patch_committed({}):                                         # explicitly unpopulated
            self.assertFalse(C.t4_governed_allowed(True, _full_oracle_manifest(),
                                                   presented_recipe_hash="recipeXYZ"))

    def test_recipe_hash_is_the_only_run_input_and_MANDATORY(self) -> None:
        m = _full_oracle_manifest()
        with _patch_committed():
            self.assertFalse(C.t4_governed_allowed(True, m))                # recipe omitted
            self.assertFalse(C.t4_governed_allowed(True, m, presented_recipe_hash=""))

    def test_malformed_nested_manifest_refuses_not_raises(self) -> None:
        # Pi #2 (reproduced defect): list/str/None where a dict is expected must REFUSE, never raise.
        m = _full_oracle_manifest()
        for bad in ({**m, "unlock_checks": ["U1_order_recovery"]},
                    {**m, "reference_bounds": ["nope"]},
                    {**m, "precision_sim": "adequate"},
                    {**m, "realism_envelope": 3},
                    {**m, "reference_bounds": {**m["reference_bounds"], "evaluator_realized_alpha": "low"}},
                    {**m, "reference_bounds": {**m["reference_bounds"], "evaluator_realized_alpha": float("nan")}}):
            self.assertFalse(_ok(bad))          # must return False, not raise

    def test_policy_mismatch_refused(self) -> None:
        # Anchors the caller can no longer choose: a manifest that disagrees with the committed policy fails.
        m = _full_oracle_manifest()
        self.assertFalse(_ok({**m, "blueprint_hash": "WRONG"}))            # blueprint not from policy
        self.assertFalse(_ok({**m, "gate_event_ref": "evt-OTHER"}))        # wrong gate event
        self.assertFalse(_ok({**m, "oracle_mechanism_hash": "WRONG"}))     # wrong mechanism
        self.assertFalse(_ok({**m, "evaluator_commit": "stale999"}))       # stale evaluator not in allowlist
        self.assertFalse(_ok({**m, "recipe_registry_id": "unknownReg"}))   # unknown registry id
        self.assertFalse(_ok({**m, "sealed_cert_run_id": "reusedSeed"}))   # sealed-run not in allowlist

    def test_mismatch_and_stale_fields_refused(self) -> None:
        m = _full_oracle_manifest()
        self.assertFalse(C.t4_governed_allowed(True, m, presented_recipe_hash="WRONG"))     # recipe != certified
        self.assertFalse(_ok({**m, "verdict": "CERTIFIED"}))                # legacy verdict
        self.assertFalse(_ok({**m, "schema_version": "v2"}))                # wrong schema
        self.assertFalse(_ok({**m, "held_out_family_ids": ["E1"]}))         # <2 held-out families
        self.assertFalse(_ok({**m, "held_out_family_ids": ["E1", "E1"]}))   # not DISTINCT
        self.assertFalse(_ok({**m, "held_out_family_ids": ["E1", 2]}))      # malformed type -> refuse not raise
        self.assertFalse(_ok({**m, "held_out_family_ids": ["E1", ""]}))     # empty-string family id
        self.assertFalse(_ok({**m, "held_out_family_ids": "E1,E2"}))        # not even a list
        self.assertFalse(_ok({**m, "governed_t4_real_output_ceiling": "ADOPT"}))
        self.assertFalse(_ok({**m, "transfer_caveat": ""}))
        self.assertFalse(_ok({**m, "unlock_checks": {"U2_null": "FAIL"}}))
        self.assertFalse(_ok({**m, "reference_bounds": {"nuisance_incremental_margin_ok": False}}))

    def test_observed_future_strata_are_oracle_assisted(self) -> None:
        self.assertTrue(C.is_oracle_assisted_stratum("future_occupancy"))
        self.assertFalse(C.is_oracle_assisted_stratum("context_rate_quantile"))  # context-observable OK


class NominationOnlyTests(unittest.TestCase):
    def test_no_adopt_labels(self) -> None:
        self.assertTrue(C.NOMINATION_ONLY)
        for label in (C.NOMINATE_FACTORIZED, C.NOMINATE_CONCAT, C.NEITHER_ADEQUATE,
                      C.NOMINATE_DIRECTION, C.NOT_EVALUABLE):
            self.assertTrue(C.is_nomination_only_decision(label))
        self.assertFalse(C.is_nomination_only_decision("ADOPT_CONCAT"))
        self.assertFalse(C.is_nomination_only_decision("ADOPT_TARGET"))


if __name__ == "__main__":
    unittest.main()
