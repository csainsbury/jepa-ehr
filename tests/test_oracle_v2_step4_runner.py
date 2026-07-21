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

    def test_two_jobs_distinct_and_job_kind_verified(self) -> None:  # Pi §6
        power = rn.build_manifest(reviewed_commit="c", job_kind="m3a-step4-power-v1")
        ident = rn.build_manifest(reviewed_commit="c", job_kind="m3a-step4-ident-v1")
        self.assertEqual(power["job_kind"], "m3a-step4-power-v1")
        self.assertNotEqual(power["manifest_hash"], ident["manifest_hash"])   # distinct job hashes
        with self.assertRaises(ValueError):
            rn.build_manifest(reviewed_commit="c", job_kind="bogus")
        # verify refuses a tampered job_kind
        bad = dict(power, job_kind="bogus")
        self.assertIn("job_kind", rn.verify_manifest(bad, require_git_head=False)["problems"])


class FailClosedVerify(unittest.TestCase):
    def test_verify_ok_against_trust_root(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        self.assertTrue(rn.verify_manifest(m, require_git_head=False)["ok"])   # fields match the trust root

    def test_tampered_fields_refuse(self) -> None:  # the old fail-OPEN reproductions must now refuse (Pi §1)
        m = rn.build_manifest(reviewed_commit="c")
        m2 = dict(m, seeds=[1], registered_n=1)
        v = rn.verify_manifest(m2, require_git_head=False)
        self.assertFalse(v["ok"])
        self.assertIn("seeds", v["problems"]); self.assertIn("registered_n", v["problems"])
        # tampered identity
        m3 = dict(m, identities=dict(m["identities"], verifier="deadbeef"))
        self.assertFalse(rn.verify_manifest(m3, require_git_head=False)["ok"])
        # arbitrary reviewed_commit is NOT its own trust root: git-head check refuses a bogus commit
        self.assertIn("reviewed_commit_vs_git_head",
                      rn.verify_manifest(rn.build_manifest(reviewed_commit="evil"))["problems"])

    def test_runner_in_code_closure(self) -> None:
        self.assertIn("oracle_realism_v2_step4_runner", rn._CLOSURE_MODULES)
        self.assertEqual(rn.code_closure_identity(), rn.code_closure_identity())


class Benchmark(unittest.TestCase):
    def test_benchmark_binds_volume_time_hardware(self) -> None:  # Pi §4
        b = rn.benchmark()
        for k in ("git_head", "hardware", "registered_n", "volume_per_source", "seconds_per_verifier_call"):
            self.assertIn(k, b)
        self.assertEqual(set(b["volume_per_source"]), {"scid_scale_control", "mimic_scale_control"})
        for v in b["volume_per_source"].values():
            self.assertGreater(v["events"], v["n"])              # forecast scales by event volume, not N
        self.assertGreater(b["seconds_per_verifier_call"], 0.0)


class DryRun(unittest.TestCase):
    def test_mechanical_dry_run(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        d = rn.dry_run(m, n=600, seeds=(1000,), components=["burst_timing"], require_git_head=False)
        self.assertFalse(d["refused"], d.get("problems"))
        self.assertTrue(d["ablation"]["burst_timing"]["A_fails_primary"])
        self.assertTrue(d["ablation"]["burst_timing"]["known_profile_repeatability"])
        for c in ("null", "boundary", "structural_zero", "source_swap"):
            self.assertIn(c, d["controls"])
        self.assertEqual(set(d["forecast_registered_n"]), {"scid_scale_control", "mimic_scale_control"})

    def test_dry_run_refuses_on_mismatch(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        m2 = dict(m, identities=dict(m["identities"], fixture="deadbeef"))
        d = rn.dry_run(m2, n=200, require_git_head=False)
        self.assertTrue(d["refused"])


if __name__ == "__main__":
    unittest.main()
