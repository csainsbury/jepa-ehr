"""M3a rebuild step-2 — executable-verifier DESIGN FREEZE structural integrity (awaiting Pi confirmation).

Pins the DEV design hash and checks the design is internally complete: 6 marginals + S1-S8 present, every
check maps to menu D components and every D component is reachable (escalation coverage), ablation covers every
component, and the simulation freezes 25 deterministic seeds. Numeric tolerances/profiles are the DESIGN (to be
confirmed by Pi) — this guards structure, not the eventual frozen values.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_verifier_design import (
    S_ALGORITHMS, CHECK_TO_D_COMPONENTS, PROFILES, SIMULATION, IDENTIFIABILITY, FIXTURE_GENERATOR,
    m3a_design_dev_hash, M3A_DESIGN_VERSION,
)

_DESIGN_DEV_HASH = "3ec8577d7b3b7766733fd1ef6e421106459de3b7910c1ff518b0b420e5f838e1"
_ALL_CHECKS = {"length_ks", "class_tv", "count_ks", "occupancy_abs", "delta_t_zero_abs", "positive_gap_ks",
               "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}


class VerifierDesignStructure(unittest.TestCase):
    def test_dev_hash_pinned_and_marked_dev(self) -> None:
        self.assertEqual(m3a_design_dev_hash(), _DESIGN_DEV_HASH)
        self.assertTrue(M3A_DESIGN_VERSION.endswith("_dev"))

    def test_six_marginals_and_S1_to_S8(self) -> None:
        self.assertEqual(set(S_ALGORITHMS), _ALL_CHECKS)
        self.assertTrue(S_ALGORITHMS["S6"].get("mandatory"))
        for sid in _ALL_CHECKS:
            self.assertIn("floor_unit", S_ALGORITHMS[sid], sid)
            self.assertIn("threshold", S_ALGORITHMS[sid], sid)

    def test_escalation_map_covers_menu_both_ways(self) -> None:
        # every mapped component is in the menu, and every menu component is reachable from some check
        mapped = {c for comps in CHECK_TO_D_COMPONENTS.values() for c in comps}
        self.assertTrue(mapped.issubset(set(V2_D_COMPONENT_MENU)))
        self.assertEqual(mapped, set(V2_D_COMPONENT_MENU), "every D component must be reachable via a check")
        # S5 and S6 route to length_class_mix (the coupling Pi required)
        self.assertIn("length_class_mix", CHECK_TO_D_COMPONENTS["S5"])
        self.assertIn("length_class_mix", CHECK_TO_D_COMPONENTS["S6"])

    def test_ablation_covers_every_component(self) -> None:
        self.assertEqual(set(PROFILES["ablation"]), set(V2_D_COMPONENT_MENU))

    def test_simulation_is_rate_based_25_seeds(self) -> None:
        self.assertEqual(SIMULATION["n_seeds"], 25)
        self.assertEqual(len(SIMULATION["seed_list"]), 25)
        self.assertEqual(SIMULATION["power"]["of_seeds"], 25)
        self.assertGreaterEqual(SIMULATION["power"]["self_known_pass_min"], 24)
        self.assertGreaterEqual(SIMULATION["power"]["misspecified_fail_min"], 20)

    def test_fixture_generator_is_independent(self) -> None:
        self.assertIn("independence_rule", FIXTURE_GENERATOR)
        self.assertIn("NO code path", FIXTURE_GENERATOR["independence_rule"])
        self.assertIn("not_a_candidate", FIXTURE_GENERATOR)

    def test_identifiability_uses_standardized_jacobian(self) -> None:
        self.assertIn("sigma_min/sigma_max", IDENTIFIABILITY["rank_criterion"])
        self.assertIn("collision", IDENTIFIABILITY)

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
