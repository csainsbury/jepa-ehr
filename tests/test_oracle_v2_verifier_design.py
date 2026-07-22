"""M3a rebuild step-2 — executable-verifier DESIGN FREEZE (rev-2) structural integrity.

Pins the rev-2 DEV design hash and checks the folded structure: two separate routes (registered marginals with
exact v1 estimands vs synthetic S1-S9), S9 seam guard present, S2 + S8 out of the D map (terminal), every
D-eligible subcheck maps into the menu and every menu component is reachable, ablation matrix covers every
component, reference-only coarsening, and rate-based 25-seed simulation. Numeric tolerances are the DESIGN
(Pi-confirmed) — this guards structure.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_verifier_design import (
    ROUTES, REGISTERED_MARGINALS, S_ALGORITHMS, CHECK_TO_D_COMPONENTS, ESCALATION, ABLATION_MATRIX,
    CONDITIONAL_COARSENING, SIMULATION, IDENTIFIABILITY, FIXTURE_GENERATOR, FIXTURE_LAW,
    m3a_design_dev_hash, M3A_DESIGN_VERSION,
)

_DESIGN_DEV_HASH = "206f299eb4fb1f2131d2ba742f07f99efd284df51a3471ab4ce950c488abe74e"  # re-minted (Pi step-4 result gate)


class VerifierDesignRev2(unittest.TestCase):
    def test_dev_hash_pinned_and_marked_dev(self) -> None:
        self.assertEqual(m3a_design_dev_hash(), _DESIGN_DEV_HASH)
        self.assertIn("dev", M3A_DESIGN_VERSION)
        self.assertNotIn("frozen", M3A_DESIGN_VERSION)

    def test_two_routes_registered_marginals_exact(self) -> None:
        self.assertEqual(set(ROUTES["marginal_route"]["checks"]), set(REGISTERED_MARGINALS))
        # exact registered estimands preserved (Pi defect 1)
        self.assertIn("POOLED", REGISTERED_MARGINALS["class_tv"]["estimand"])
        self.assertIn("POOLED", REGISTERED_MARGINALS["delta_t_zero_abs"]["estimand"])
        self.assertEqual(ROUTES["sequence_route"]["checks"],
                         ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"])
        self.assertIn("EXPLORATORY", ROUTES["marginal_route"]["claim"])

    def test_S9_seam_guard_present_and_terminal(self) -> None:
        self.assertIn("S9", S_ALGORITHMS)
        self.assertIn("TERMINAL", S_ALGORITHMS["S9"]["escalation"])
        self.assertIn("TERMINAL", S_ALGORITHMS["S8"]["escalation"])

    def test_fixture_law_derives_dt0_not_independent(self) -> None:
        self.assertIn("DERIVED", FIXTURE_LAW["dt0_rate"])
        self.assertTrue(FIXTURE_LAW["singletons_in_S2"])

    def test_S2_not_binned_into_cluster_bins(self) -> None:
        self.assertIn("NOT binned", S_ALGORITHMS["S2"]["notes"])

    def test_escalation_map_excludes_S2_S8_marginals(self) -> None:
        keys = set(CHECK_TO_D_COMPONENTS)
        self.assertNotIn("S2_ks", keys)
        self.assertFalse(any(k.startswith("S8") for k in keys), "S8 must be terminal, not a D route")
        self.assertIn("S2_ks", ESCALATION["terminal_no_D"])
        # every mapped component in the menu; every menu component reachable
        mapped = {c for v in CHECK_TO_D_COMPONENTS.values() for c in v["components"]}
        self.assertTrue(mapped.issubset(set(V2_D_COMPONENT_MENU)))
        self.assertEqual(mapped, set(V2_D_COMPONENT_MENU))
        self.assertNotIn("S5_abs", CHECK_TO_D_COMPONENTS)      # S5 terminal (Pi F2)
        self.assertNotIn("S1_density", CHECK_TO_D_COMPONENTS)  # S1 terminal (Pi F1)
        self.assertNotIn("S6_tv", CHECK_TO_D_COMPONENTS)       # S6 terminal after length_class_mix drop (Pi re-gate)
        self.assertEqual(CHECK_TO_D_COMPONENTS["S7_abs"]["components"], ["cluster_size_mark_diversity"])

    def test_ablation_matrix_covers_every_component(self) -> None:
        self.assertEqual(set(ABLATION_MATRIX), set(V2_D_COMPONENT_MENU))
        for comp, row in ABLATION_MATRIX.items():
            self.assertTrue(row["primary_fail"], comp)

    def test_reference_only_coarsening(self) -> None:
        self.assertIn("REFERENCE ONLY", CONDITIONAL_COARSENING["rule"])
        self.assertEqual(CONDITIONAL_COARSENING["min_retained"]["length_bins"], 3)
        self.assertIn("NEVER coarsened", CONDITIONAL_COARSENING["position_quartiles"])

    def test_simulation_provisional_and_rate_based(self) -> None:
        self.assertIn("per_source_sample_size_candidate", SIMULATION)
        self.assertEqual(SIMULATION["n_seeds"], 25)
        self.assertEqual(len(SIMULATION["seed_list"]), 25)
        self.assertGreaterEqual(SIMULATION["power"]["self_known_pass_min"], 24)
        self.assertGreaterEqual(SIMULATION["power"]["misspecified_fail_min"], 20)
        self.assertTrue(SIMULATION["step4_must_demonstrate"])

    def test_identifiability_no_logit_endpoint_and_onesided(self) -> None:
        self.assertIn("AFFINE", IDENTIFIABILITY["standardization"])
        self.assertIn("no logit", IDENTIFIABILITY["standardization"])
        self.assertIn("sigma_min/sigma_max", IDENTIFIABILITY["rank_criterion"])
        self.assertIn("one-sided", IDENTIFIABILITY["finite_difference"])

    def test_fixture_independent(self) -> None:
        self.assertIn("NO code path", FIXTURE_GENERATOR["independence_rule"])
        self.assertIn("no governed read", FIXTURE_GENERATOR["data_rule"].lower())

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
