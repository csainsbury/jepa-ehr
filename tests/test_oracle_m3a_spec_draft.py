"""Step-5 M3a verification-spec DRAFT — HISTORICAL structural integrity (SUPERSEDED by the M3a rebuild).

This module is a frozen HISTORICAL record (Pi: preserve the old dev hash 57ecfc93 as provenance). It is
decoupled from the LIVE schemas, which moved to the full-sequence multi-block unit. The test therefore checks
the module's OWN internal consistency (its historical 4-component D set), NOT the evolving live D menu.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval.oracle_realism_v2_spec import (
    CROSS_STATISTICS, MARGINAL_CHECKS, PARAM_TO_STATISTIC, SOURCE_CONJUNCTION, ESCALATION,
    IDENTIFIABILITY_BATTERY, m3a_spec_dev_hash, open_pi_rulings, M3A_SPEC_VERSION,
)

_M3A_DEV_HASH = "57ecfc9357be972752fcf11f9d45b75e743ca2b57afc38744eea28747fbdc194"
# the historical D dependence set this superseded draft attributed (pre-length_class_mix).
_HIST_D_DEPENDENCE = ("burst_count_length", "burst_timing", "mark_burst_tie", "cluster_size_mark_diversity")


class M3aSpecStructure(unittest.TestCase):
    def test_dev_hash_pinned_and_marked_draft(self) -> None:
        self.assertEqual(m3a_spec_dev_hash(), _M3A_DEV_HASH)
        self.assertTrue(M3A_SPEC_VERSION.endswith("_dev"), "M3a spec must remain a DEV draft until Pi rules")

    def test_all_dependence_params_attributed(self) -> None:
        for p in _HIST_D_DEPENDENCE:
            self.assertIn(p, PARAM_TO_STATISTIC, f"unattributed dependence param {p}")
            for stat in PARAM_TO_STATISTIC[p]:
                self.assertIn(stat, CROSS_STATISTICS, f"{p} -> unknown stat {stat}")

    def test_six_marginals_and_S1_to_S7_present(self) -> None:
        self.assertEqual(len(MARGINAL_CHECKS), 6)
        self.assertEqual(set(CROSS_STATISTICS), {"S1", "S2", "S3", "S4", "S5", "S6", "S7"})
        self.assertTrue(CROSS_STATISTICS["S6"].get("mandatory"), "S6 must be mandatory (Pi P-B)")

    def test_every_cross_stat_has_floor_and_refusal(self) -> None:
        for sid, spec in CROSS_STATISTICS.items():
            self.assertIn("denom_floor", spec, sid)
            self.assertIn("refusal", spec, sid)
            self.assertIn("threshold", spec, sid)

    def test_source_conjunction_and_escalation_declared(self) -> None:
        self.assertEqual(set(SOURCE_CONJUNCTION["required_sources"]), {"SCID", "MIMIC"})
        self.assertEqual(ESCALATION["component_to_check"], PARAM_TO_STATISTIC)
        self.assertIn("immutable", ESCALATION["ledger"])
        self.assertTrue(ESCALATION["ledger"]["immutable"])
        self.assertIn("iteration_cap", ESCALATION)

    def test_identifiability_battery_declares_rank_recovery_collision(self) -> None:
        for k in ("local_rank", "global_recovery", "collision_search", "rule"):
            self.assertIn(k, IDENTIFIABILITY_BATTERY, k)

    def test_open_rulings_surface_all_proposed(self) -> None:
        rulings = open_pi_rulings()
        self.assertEqual(set(rulings["cross_statistic_thresholds"]),
                         {"S1", "S2", "S3", "S4", "S5", "S6", "S7"})
        self.assertTrue(rulings["power"] and rulings["escalation"])

    def test_prior_identities_unmoved(self) -> None:
        from clinical_jepa.eval.oracle_realism_v2 import (
            realism_v2_schema_hash, v2_certification_boundary_hash, m0b_support_policy_hash,
        )
        self.assertEqual(realism_v2_schema_hash(),
                         "2a7405ddadcdfdf3261a2b18e149c5523298a73d3e765cde78b6611927377673")
        self.assertEqual(v2_certification_boundary_hash(),
                         "b33c2d9f6324c84763ebb85fde8912dbf0b84e94b7ee366e4adb879ceb14e8e4")
        self.assertEqual(m0b_support_policy_hash(),
                         "c7532ee9cd8629a20e8943c30729263e429f074258c6d7bb069ab104e688cd6d")


if __name__ == "__main__":
    unittest.main()
