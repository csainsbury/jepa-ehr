"""Rung-2 sub-gate 2 (count interface) + sub-gate 4 (timing 4A/4B) + T4-refusal tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung2_count_interface as CI
from clinical_jepa.eval import rung2_timing as T
from clinical_jepa.eval.rung2_t4_stub import guard_t4
from clinical_jepa.eval.rung2_contract import NOT_EVALUABLE


class CountInterfaceTests(unittest.TestCase):
    def test_rps_lower_for_sharper_correct_dist(self) -> None:
        # sharp-correct distribution has lower RPS than a flat one.
        y = np.array([2, 0, 3])
        sharp = np.zeros((3, 5)); sharp[0, 2] = 1; sharp[1, 0] = 1; sharp[2, 3] = 1
        flat = np.full((3, 5), 0.2)
        self.assertLess(CI.ranked_probability_score(sharp, y).mean(),
                        CI.ranked_probability_score(flat, y).mean())

    def test_point_estimate_b_is_structural_factorized(self) -> None:
        d = CI.count_interface_decision(skill_a_lo=0.3, skill_b_lo=0.9, paired_b_minus_a_lo=0.5,
                                        b_is_point_estimate=True)
        self.assertEqual(d["decision"], "NOMINATE_FACTORIZED")   # B can't win a calibrated comparison

    def test_b_nominates_only_if_superior(self) -> None:
        self.assertEqual(CI.count_interface_decision(skill_a_lo=0.2, skill_b_lo=0.5, paired_b_minus_a_lo=0.1,
                                                     b_is_point_estimate=False)["decision"], "NOMINATE_CONCAT")
        self.assertEqual(CI.count_interface_decision(skill_a_lo=0.5, skill_b_lo=0.5, paired_b_minus_a_lo=0.0,
                                                     b_is_point_estimate=False)["decision"], "NOMINATE_FACTORIZED")
        self.assertEqual(CI.count_interface_decision(skill_a_lo=-0.1, skill_b_lo=-0.1, paired_b_minus_a_lo=0.0,
                                                     b_is_point_estimate=False)["decision"], "NEITHER_ADEQUATE")


class TimingGateTests(unittest.TestCase):
    def test_4a_4b_separate_conjunctive(self) -> None:
        a_pass = T.gate_4a(multiplicity_skill_lo=0.08, swap_skill_lo=0.06, ece_hi=0.03, evaluable=True)
        b_pass = T.gate_4b(ks_upper_ci=0.03, crps_skill_lo=0.08, rate_head_improvement_lo=0.06,
                           swap_skill_lo=0.06, evaluable=True)
        b_fail = T.gate_4b(ks_upper_ci=0.09, crps_skill_lo=0.08, rate_head_improvement_lo=0.06,
                           swap_skill_lo=0.06, evaluable=True)   # KS fails
        self.assertEqual(T.timing_verdict(a_pass, b_pass), "PASS")
        self.assertEqual(T.timing_verdict(a_pass, b_fail), "FAIL")   # conjunctive
        self.assertEqual(T.timing_verdict({"gate_4a": NOT_EVALUABLE}, b_pass), NOT_EVALUABLE)

    def test_4a_multiplicity_required_not_just_p0(self) -> None:
        # low multiplicity skill fails 4A even with good calibration.
        self.assertEqual(T.gate_4a(multiplicity_skill_lo=0.01, swap_skill_lo=0.06, ece_hi=0.02,
                                   evaluable=True)["gate_4a"], "FAIL")

    def test_observed_future_strata_refused(self) -> None:
        with self.assertRaises(AssertionError):
            T.assert_context_observable_strata(["occupancy_bin", "future_occupancy"])
        self.assertTrue(T.assert_context_observable_strata(["occupancy_bin", "context_rate_quantile"]))


class T4RefusalTests(unittest.TestCase):
    def test_governed_t4_refused_without_oracle(self) -> None:
        with self.assertRaises(PermissionError):
            guard_t4(inputs_are_governed=True, oracle_authorization=None)
        with self.assertRaises(PermissionError):
            guard_t4(inputs_are_governed=True, oracle_authorization={"oracle_frozen": True, "pi_gate": "PENDING"})

    def test_synthetic_t4_allowed(self) -> None:
        guard_t4(inputs_are_governed=False, oracle_authorization=None)                      # OK
        guard_t4(inputs_are_governed=True, oracle_authorization={"oracle_frozen": True, "pi_gate": "PASS"})


if __name__ == "__main__":
    unittest.main()
