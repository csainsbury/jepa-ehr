"""Step-3 (rebuild, final) — fail-closed battery machinery + run-contract (Pi F3 + run-contract fold).

Exercises: the conjunctive verdict semantics (synthetic, no verifier), Wilson CI, deterministic (source,
profile, seed, role) RNG derivation, the forecast (event/cluster/pair volume), a small mechanical orientation
smoke for two components (known-profile repeatability, not recovery), and the fail-closed controls with exact
expected-status maps. The full 25-seed source-conjunction run at N=8000 is the step-4 runner (not here).
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_realism_v2_battery as bat

_BATTERY_IMPL_ID = "7c61fe69587bf77ea653a7000369fbe9a1db3ab07a32dbbb1f560c8149ea07c0"


def _synth(kp, ks, kr):
    return {"primary_fail_per_check": {"S3_tau": {"k": kp, "n": 25}, "S3_loggap": {"k": kp, "n": 25}},
            "specificity_per_check": {"x": {"k": ks, "n": 25}}, "repeatability_rate": {"k": kr, "n": 25}}


class BatteryContract(unittest.TestCase):
    def test_impl_identity_and_registered_contract(self) -> None:
        self.assertEqual(bat.battery_impl_identity(), _BATTERY_IMPL_ID)
        self.assertEqual(bat.REGISTERED_N, 8000)
        self.assertEqual(tuple(bat.SOURCE_PROFILES), ("scid_scale_control", "mimic_scale_control"))
        self.assertEqual(bat.PRIMARY_FAIL_MIN, 20)
        self.assertEqual(bat.SPECIFICITY_MIN, 24)

    def test_conjunctive_verdict(self) -> None:
        both = {"scid_scale_control": _synth(25, 25, 25), "mimic_scale_control": _synth(22, 24, 24)}
        self.assertTrue(bat._component_verdict("burst_timing", both, bat.SOURCE_PROFILES)["conjunctive_pass"])
        # primary 19/25 (< 20) on one source => that source fails => conjunction fails
        one = {"scid_scale_control": _synth(25, 25, 25), "mimic_scale_control": _synth(19, 24, 24)}
        v = bat._component_verdict("burst_timing", one, bat.SOURCE_PROFILES)
        self.assertFalse(v["conjunctive_pass"])
        self.assertFalse(v["per_source_ok"]["mimic_scale_control"])
        # specificity 23/25 (< 24) also fails
        spec = {"scid_scale_control": _synth(25, 23, 25), "mimic_scale_control": _synth(25, 25, 25)}
        self.assertFalse(bat._component_verdict("burst_timing", spec, bat.SOURCE_PROFILES)["conjunctive_pass"])

    def test_wilson_and_seed_derivation(self) -> None:
        r = bat._rate([True] * 24 + [False])
        self.assertEqual((r["k"], r["rate"]), (24, 0.96))
        s1 = bat._derive_seed("fixture", "scid_scale_control", 1000, "reference")
        self.assertEqual(s1, bat._derive_seed("fixture", "scid_scale_control", 1000, "reference"))   # deterministic
        self.assertNotEqual(s1, bat._derive_seed("fixture", "scid_scale_control", 1000, "candidate_A"))  # role varies
        self.assertNotEqual(s1, bat._derive_seed("fixture", "mimic_scale_control", 1000, "reference"))   # source varies

    def test_forecast_by_volume(self) -> None:
        smoke = bat.multiscale_smoke_sampler(n_each=400)
        f = bat.forecast(smoke, source_profile="mimic_scale_control", secs_per_million_events=3.0)
        for k in ("n_sequences", "total_events", "total_clusters", "adjacent_pairs", "est_secs_per_verifier_call"):
            self.assertIn(k, f)
        self.assertEqual(f["n_sequences"], 1200)
        self.assertGreater(f["total_events"], f["n_sequences"])   # scaled by events, not sequence count


class OrientationSmoke(unittest.TestCase):
    def test_two_components_repeatability(self) -> None:
        smoke = bat.multiscale_smoke_sampler(n_each=600)
        for comp in ("burst_timing", "cluster_size_mark_diversity"):
            o = bat.component_ablation(comp, 1000, base_sampler=smoke, source_profile="mimic_scale_control")
            self.assertTrue(o.A_fails_primary, f"{comp}: candidate_A must FAIL primary")
            self.assertTrue(o.A_specificity_ok,
                            f"{comp}: specificity fails {[k for k,v in o.A_specificity.items() if not v]}")
            self.assertTrue(o.known_profile_repeatability,
                            f"{comp}: repeatability fails {[k for k,v in o.D_status.items() if v!=bat.PASS]}")


class ControlsFailClosed(unittest.TestCase):
    def test_null_all_pass(self) -> None:
        smoke = bat.multiscale_smoke_sampler(n_each=600)
        nc = bat.null_control(1000, base_sampler=smoke, source_profile="mimic_scale_control")
        self.assertTrue(nc["all_pass"], f"null FAILs={nc['fails']} NE={nc['not_evaluable']}")

    def test_boundary_expected_status_map(self) -> None:
        # at a representative N (S3 needs >=FLOOR eligible multi-cluster sequences); the exact predeclared
        # NE-else-PASS map holds. (At tiny N some non-length checks are NE for lack of support — a small-N
        # artifact, not the registered behaviour.)
        bc = bat.boundary_control(1000, n_each=1500)
        self.assertTrue(bc["ok"], f"boundary unexpected: {bc['unexpected']}")

    def test_structural_zero(self) -> None:
        sz = bat.structural_zero_control(1000, n_each=600)
        self.assertTrue(sz["zeros_absent"])
        self.assertTrue(sz["ok"])

    def test_source_swap_nondegenerate(self) -> None:
        ss = bat.source_swap_control(1000, n_each=600)
        self.assertTrue(ss["fails_nondegenerate"], f"fails: {ss['fails']}")

    def test_all_check_keys_registry_matches_emitted(self) -> None:  # Pi §4: registry must not drift
        smoke = bat.multiscale_smoke_sampler(n_each=400)
        nc = bat.null_control(1000, base_sampler=smoke, source_profile="mimic_scale_control")
        self.assertEqual(set(nc["status"].keys()), set(bat.ALL_CHECK_KEYS))
        self.assertEqual(set(nc["evidence"].keys()), set(bat.ALL_CHECK_KEYS))

    def test_boundary_structurally_bounded_L_le_7(self) -> None:  # Pi re-gate §4 (bounded-control status only)
        from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
        from clinical_jepa.eval.oracle_realism_v2_verifier import NOT_EVALUABLE
        recs = sample_fixture("MIMIC", bat._BOUNDED_SHORT_PROF, 3000, seed=7)
        self.assertLessEqual(max(r.L_total for r in recs), 7)      # HARD structural bound (no 8-item block)
        bc = bat.boundary_control(1000, n_each=1500)
        for k in ("S9_zero", "S9_class", "S9_gap"):                # seam checks NE by construction, not by tail luck
            self.assertEqual(bc["status"][k], NOT_EVALUABLE, k)
        self.assertTrue(bc["ok"], f"boundary unexpected: {bc['unexpected']}")


if __name__ == "__main__":
    unittest.main()
