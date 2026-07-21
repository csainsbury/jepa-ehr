"""Step-3 (rebuild) — registered step-4 runner/manifest: fail-closed binding + dry-run (Pi §5).

Checks the manifest binds the full run contract, the fail-closed identity pre-flight refuses on any mismatch,
the cap is unambiguous (one worker / <=8 wall-clock hours / <=32 GB), and the mechanical dry-run proves the
pipeline end-to-end (and refuses on a tampered manifest). The full registered run is NOT invoked here.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_realism_v2_step4_runner as rn


class ManifestBinding(unittest.TestCase):
    def setUp(self):
        self.m = rn.build_manifest(reviewed_commit="test_commit")

    def test_binds_full_contract(self) -> None:
        for k in ("identities", "source_profiles", "seeds", "registered_n", "cap", "verdict",
                  "rng_derivation", "checkpoint", "completion_hashes", "manifest_hash", "identifiability_vector"):
            self.assertIn(k, self.m, k)
        self.assertEqual(len(self.m["seeds"]), 25)
        self.assertEqual(self.m["seeds"][0], 1000)
        self.assertEqual(self.m["seeds"][-1], 1024)
        self.assertEqual(self.m["registered_n"], 4000)
        self.assertEqual(set(self.m["identities"]),
                         {"fixture", "verifier", "coupling", "battery", "design", "identifiability", "code_closure"})
        self.assertIn("identifiability", self.m)
        self.assertIn("cost_forecast", self.m["identifiability"])
        self.assertEqual(set(self.m["source_profiles"]), {"scid_scale_control", "mimic_scale_control"})
        self.assertEqual(self.m["identifiability_vector"],
                         ["S3_tau", "S3_loggap", "S4_abs", "S6_tv", "S7_abs"])

    def test_cap_unambiguous(self) -> None:
        self.assertEqual(self.m["cap"]["workers"], 1)
        self.assertEqual(self.m["cap"]["wall_clock_hours"], 8)
        self.assertEqual(self.m["cap"]["ram_gb"], 32)

    def test_manifest_hash_stable_within_process(self) -> None:
        m2 = rn.build_manifest(reviewed_commit="test_commit")
        self.assertEqual(self.m["manifest_hash"], m2["manifest_hash"])
        m3 = rn.build_manifest(reviewed_commit="other")
        self.assertNotEqual(self.m["manifest_hash"], m3["manifest_hash"])   # commit is bound


class FailClosedVerify(unittest.TestCase):
    def test_verify_ok_and_tamper_refuses(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        self.assertTrue(rn.verify_identities(m)["ok"])
        m2 = dict(m, identities=dict(m["identities"], verifier="deadbeef"))
        v = rn.verify_identities(m2)
        self.assertFalse(v["ok"])
        self.assertIn("verifier", v["mismatches"])

    def test_code_closure_identity_stable(self) -> None:
        self.assertEqual(rn.code_closure_identity(), rn.code_closure_identity())


class DryRun(unittest.TestCase):
    def test_mechanical_dry_run(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        d = rn.dry_run(m, n=600, seeds=(1000,), components=["burst_timing"])
        self.assertFalse(d["refused"])
        self.assertIn("burst_timing", d["ablation"])
        self.assertTrue(d["ablation"]["burst_timing"]["A_fails_primary"])
        self.assertTrue(d["ablation"]["burst_timing"]["known_profile_repeatability"])
        for c in ("null", "boundary", "structural_zero", "source_swap"):
            self.assertIn(c, d["controls"])
        self.assertEqual(set(d["forecast_registered_n"]), {"scid_scale_control", "mimic_scale_control"})

    def test_dry_run_refuses_on_mismatch(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        m2 = dict(m, identities=dict(m["identities"], fixture="deadbeef"))
        d = rn.dry_run(m2, n=200)
        self.assertTrue(d["refused"])
        self.assertEqual(d["reason"], "identity mismatch")


if __name__ == "__main__":
    unittest.main()
