"""Rung-2 CONTRACT fail-hard tests (Pi v2 re-gate #6) — the mandatory pre-registration checks:
gate independence, direct-vs-recursive path discipline, MIMIC-2d non-gating, missing-support
NOT_EVALUABLE, T4 governed refusal without an oracle, oracle-assisted strata, nomination-only,
and test sealing. All numeric gates are concrete constants (no un-frozen curves)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung2_contract as C


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


class OracleStopLineTests(unittest.TestCase):
    def test_t4_governed_refused_without_frozen_oracle(self) -> None:
        self.assertTrue(C.requires_oracle(C.T4_TARGET))
        self.assertFalse(C.requires_oracle("T2_seq_of_latents"))
        # governed T4 refused without a frozen, Pi-gated oracle
        self.assertFalse(C.t4_governed_allowed(True, None))
        self.assertFalse(C.t4_governed_allowed(True, {"oracle_frozen": True, "pi_gate": "PENDING"}))
        self.assertTrue(C.t4_governed_allowed(True, {"oracle_frozen": True, "pi_gate": "PASS"}))
        # synthetic/safe-public T4 scaffolding always allowed
        self.assertTrue(C.t4_governed_allowed(False, None))

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
