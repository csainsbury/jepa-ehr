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
        self.assertEqual(self.m["registered_n"], 8000)
        self.assertEqual(set(self.m["identities"]),
                         {"fixture", "verifier", "coupling", "battery", "design", "identifiability", "code_closure"})
        self.assertIn("identifiability", self.m)
        self.assertIn("cost_forecast", self.m["identifiability"])
        self.assertEqual(set(self.m["source_profiles"]), {"scid_scale_control", "mimic_scale_control"})
        self.assertEqual(self.m["identifiability_vector"],
                         ["S3_tau", "S3_loggap", "S4_abs", "S7_abs"])

    def test_binds_control_alloc_and_boundary_profile(self) -> None:  # Pi re-gate #1/#3
        from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
        from clinical_jepa.eval.oracle_contracts import canonical_hash
        self.assertEqual(self.m["control_alloc"], [2667, 2667, 2666])
        self.assertEqual(sum(self.m["control_alloc"]), self.m["control_n"])
        self.assertEqual(self.m["boundary_fixture"]["canonical_profile_hash"],
                         canonical_hash(PROFILES["boundary_short"]))     # design == executed control profile

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
        power = rn.build_manifest(reviewed_commit="c", job_kind="m3a-step4-power-v2")
        ident = rn.build_manifest(reviewed_commit="c", job_kind="m3a-step4-ident-v2")
        self.assertEqual(power["job_kind"], "m3a-step4-power-v2")
        self.assertNotEqual(power["manifest_hash"], ident["manifest_hash"])   # distinct job hashes
        with self.assertRaises(ValueError):
            rn.build_manifest(reviewed_commit="c", job_kind="bogus")
        # verify refuses a tampered job_kind
        bad = dict(power, job_kind="bogus")
        self.assertIn("job_kind", rn.verify_manifest(bad, require_git_head=False)["problems"])


class FailClosedVerify(unittest.TestCase):
    def test_verify_ok_against_trust_root(self) -> None:
        m = rn.build_manifest(reviewed_commit="c")
        self.assertTrue(rn.verify_manifest(m, require_git_head=False)["ok"])   # deep-equal to reconstruction

    def test_whole_manifest_deep_equality(self) -> None:  # Pi §1: reject tampered/omitted/EXTRA fields
        m = rn.build_manifest(reviewed_commit="c")
        # tampered registered field
        self.assertFalse(rn.verify_manifest(dict(m, seeds=[1]), require_git_head=False)["ok"])
        # tampered identity
        self.assertFalse(rn.verify_manifest(
            dict(m, identities=dict(m["identities"], verifier="deadbeef")), require_git_head=False)["ok"])
        # tampered OMITTED field (the old allowlist bypass) — recompute hash, still refuse
        m4 = dict(m); m4["completion_hashes"] = ["evil"]
        m4["manifest_hash"] = None
        import clinical_jepa.eval.oracle_contracts as _c
        m4["manifest_hash"] = _c.canonical_hash({k: v for k, v in m4.items() if k != "manifest_hash"})
        self.assertFalse(rn.verify_manifest(m4, require_git_head=False)["ok"])
        # EXTRA field
        self.assertFalse(rn.verify_manifest(dict(m, extra="x"), require_git_head=False)["ok"])
        # bogus reviewed_commit is NOT its own trust root (git-head refuses)
        self.assertIn("reviewed_commit_vs_git_head",
                      rn.verify_manifest(rn.build_manifest(reviewed_commit="evil"))["problems"])

    def test_runner_in_code_closure_but_policy_excluded(self) -> None:
        self.assertIn("oracle_realism_v2_step4_runner", rn._CLOSURE_MODULES)
        self.assertNotIn("oracle_realism_v2_step4_policy", rn._CLOSURE_MODULES)   # data, not logic
        self.assertEqual(rn.code_closure_identity(), rn.code_closure_identity())


class LaunchGate(unittest.TestCase):
    def test_launch_refused_when_policy_empty(self) -> None:  # Pi §1: empty approval map => nothing launches
        m = rn.build_manifest(reviewed_commit=rn._git_head() or "c", job_kind="m3a-step4-power-v2")
        v = rn.verify_launch(m, run_id="m3a-step4-power-v2-run1", job_kind="m3a-step4-power-v2")
        self.assertFalse(v["ok"])
        self.assertIn("run_id_not_approved", v["problems"])
        r = rn.run_full_battery(m, run_id="m3a-step4-power-v2-run1")
        self.assertEqual(r["status"], "REFUSED")

    def test_cross_job_manifest_refused(self) -> None:  # power entrypoint must reject an ident manifest
        ident_m = rn.build_manifest(reviewed_commit="c", job_kind="m3a-step4-ident-v2")
        v = rn.verify_launch(ident_m, run_id="m3a-step4-power-v2-run1", job_kind="m3a-step4-power-v2")
        self.assertIn("job_kind_mismatch", v["problems"])

    def test_run_id_pattern_and_containment(self) -> None:  # Pi §2
        with self.assertRaises(ValueError):
            rn._validate_run_id("../../etc/passwd", "m3a-step4-power-v2")
        with self.assertRaises(ValueError):
            rn._validate_run_id("m3a-step4-ident-v2-run1", "m3a-step4-power-v2")   # wrong job kind
        rn._validate_run_id("m3a-step4-power-v2-run1", "m3a-step4-power-v2")       # ok

    def test_gate_event_required(self) -> None:  # Pi §3: an approval without a canonical gate_event cannot launch
        import clinical_jepa.eval.oracle_realism_v2_step4_policy as pol
        head = rn._git_head()
        if not head:
            self.skipTest("no git head")
        m = rn.build_manifest(reviewed_commit=head, job_kind="m3a-step4-power-v2")
        rid = "m3a-step4-power-v2-run1"
        saved = dict(pol.APPROVED_STEP4_JOBS)
        try:
            pol.APPROVED_STEP4_JOBS[rid] = {"job_kind": "m3a-step4-power-v2", "reviewed_commit": head,
                                            "manifest_hash": m["manifest_hash"]}   # no gate_event
            v = rn.verify_launch(m, run_id=rid, job_kind="m3a-step4-power-v2")
            self.assertIn("gate_event_missing", v["problems"])
            self.assertIsNone(v["gate_event"])
            pol.APPROVED_STEP4_JOBS[rid]["gate_event"] = "evt-20260722T000000Z-deadbeef"
            v2 = rn.verify_launch(m, run_id=rid, job_kind="m3a-step4-power-v2")
            self.assertTrue(v2["ok"], v2["problems"])
            self.assertEqual(v2["gate_event"], "evt-20260722T000000Z-deadbeef")
            # a non-canonical gate id is refused
            pol.APPROVED_STEP4_JOBS[rid]["gate_event"] = "not-an-event"
            self.assertIn("gate_event_missing", rn.verify_launch(m, run_id=rid, job_kind="m3a-step4-power-v2")["problems"])
        finally:
            pol.APPROVED_STEP4_JOBS.clear(); pol.APPROVED_STEP4_JOBS.update(saved)


class Benchmark(unittest.TestCase):
    def test_benchmark_binds_volume_time_env(self) -> None:  # Pi §4/§7
        b = rn.benchmark()
        for k in ("git_head", "environment_hash", "platform", "registered_n", "volume_per_source",
                  "seconds_per_verifier_call", "workers"):
            self.assertIn(k, b)
        self.assertNotIn("hardware", b)                        # renamed (Pi §7)
        self.assertEqual(set(b["volume_per_source"]), {"scid_scale_control", "mimic_scale_control"})
        for v in b["volume_per_source"].values():
            self.assertGreater(v["events"], v["n"])
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
