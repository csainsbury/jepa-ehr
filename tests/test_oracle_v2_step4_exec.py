"""Step-4 execution engine (Pi §2/§3) — fail-closed, checkpoint/resume, cap, atomic, aggregate.

Machinery correctness at a tiny mechanical scale (the full 25-seed/N=4000 run is the reviewed job): cap-exceed
=> PARTIAL, atomic result + per-replicate serialisation + denominator-map hash, checkpoint/resume, resume
mismatch => PARTIAL, tampered manifest => REFUSED. Plus a pure-function aggregate/verdict test over 25 synthetic
seeds. Verifier-backed parts kept small.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from clinical_jepa.eval import oracle_realism_v2_step4_exec as ex
from clinical_jepa.eval import oracle_realism_v2_step4_runner as rn
from clinical_jepa.eval import oracle_realism_v2_battery as bat
from clinical_jepa.eval.oracle_realism_v2_verifier import PASS, FAIL, NOT_EVALUABLE


def _verify(m):
    return rn.verify_manifest(m, require_git_head=False)


def _inc_clock(dt=10.0):
    """Deterministic monotonic clock: returns 0, dt, 2*dt, ... on successive calls (for cap-boundary tests)."""
    t = {"v": 0.0}
    def clk():
        v = t["v"]; t["v"] += dt; return v
    return clk


class ExecutionMachinery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp(prefix="v2step4exec_")
        cls.m = rn.build_manifest(reviewed_commit="c")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def _run(self, run_id, **kw):
        base = bat.multiscale_smoke_sampler(n_each=300)     # local (a class attr would bind self)
        return ex.execute(self.m, run_id, self.out, base_sampler=base, seeds=[1000],
                          sources=["mimic_scale_control"], components=["burst_timing"], controls=["null"],
                          verify=_verify, **kw)

    def test_cap_exceed_partial(self) -> None:
        r = self._run("runCap", cap_hours=0.0, cap_gb=1e9)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["reason"], "cap exceeded")

    def test_run_persists_and_resumes(self) -> None:
        r1 = self._run("runA", cap_hours=1.0, cap_gb=64)
        self.assertIn(r1["status"], ("PASS", "FAIL"))          # tiny scale => FAIL on thresholds; machinery ok
        self.assertEqual(r1["n_replicates"], 2)                 # 1 ablation + 1 null control
        # §3: real completion hashes — all four, each a 64-hex sha
        ch1 = r1["completion_hashes"]
        for k in ("result_sha256", "evidence_sha256", "runtime_json_sha256", "env_sha256"):
            self.assertEqual(len(ch1[k]), 64, k)
        rd = ex.run_dir(self.out, "runA")
        for f in ("result.json", "checkpoint.json", "runtime.json", "environment.json"):
            self.assertTrue(os.path.exists(os.path.join(rd, f)), f)
        self.assertEqual(len(os.listdir(os.path.join(rd, "replicates"))), 2)
        # §3: per-replicate record persists FULL CheckResult evidence (value + threshold + detail), not just status
        abl = json.load(open(os.path.join(rd, "replicates", "ablation|burst_timing|mimic_scale_control|1000.json")))
        s3 = abl["A_evidence"]["S3_tau"]
        self.assertIn("value", s3); self.assertIn("threshold", s3); self.assertIn("detail", s3)
        r2 = self._run("runA", cap_hours=1.0, cap_gb=64)       # resume: all done
        self.assertEqual(r2["n_replicates"], 2)
        # evidence + decision are deterministic across resume; runtime is observed (may differ) — not asserted equal
        self.assertEqual(r2["completion_hashes"]["evidence_sha256"], ch1["evidence_sha256"])
        self.assertEqual(r2["completion_hashes"]["result_sha256"], ch1["result_sha256"])

    def test_resume_mismatch_partial(self) -> None:
        self._run("runB", cap_hours=1.0, cap_gb=64)
        base = bat.multiscale_smoke_sampler(n_each=300)
        m2 = dict(self.m, manifest_hash="different")
        r = ex.execute(m2, "runB", self.out, base_sampler=base, seeds=[1000],
                       sources=["mimic_scale_control"], components=["burst_timing"], controls=["null"],
                       cap_hours=1.0, cap_gb=64, verify=None)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["reason"], "resume manifest mismatch")

    def test_refused_on_tampered_manifest(self) -> None:
        base = bat.multiscale_smoke_sampler(n_each=300)
        r = ex.execute(dict(self.m, seeds=[1]), "runR", self.out, base_sampler=base, seeds=[1000],
                       sources=["mimic_scale_control"], components=["burst_timing"], controls=["null"],
                       cap_hours=1.0, cap_gb=64, verify=_verify)
        self.assertEqual(r["status"], "REFUSED")   # deep-equality refuses tampered seeds

    def test_resume_cap_accumulation(self) -> None:  # Pi §8: prior elapsed counts against the cap on resume
        # run 1: complete >=1 replicate (banking cum_elapsed>0), then trip => PARTIAL
        r1 = self._run("runAcc", cap_hours=45 / 3600, cap_gb=1e12, clock=_inc_clock(10))
        self.assertEqual(r1["status"], "PARTIAL")
        ck = json.load(open(os.path.join(ex.run_dir(self.out, "runAcc"), "checkpoint.json")))
        self.assertGreater(ck["cum_elapsed"], 0.0)                  # completed-unit time banked
        # run 2: resume; a cap BELOW the already-accumulated time trips before any new work
        r2 = self._run("runAcc", cap_hours=20 / 3600, cap_gb=1e12)
        self.assertEqual(r2["status"], "PARTIAL")
        self.assertEqual(r2["reason"], "cap exceeded")
        self.assertGreaterEqual(r2["cum_elapsed"], ck["cum_elapsed"])   # prior elapsed was carried in

    def test_final_unit_overrun(self) -> None:  # Pi §8/§5: cap re-checked BEFORE the verdict
        # in-loop checks (cum 10, 50) stay under cap=70; total elapsed (~90) exceeds it at the pre-verdict gate
        r = self._run("runFin", cap_hours=70 / 3600, cap_gb=1e12, clock=_inc_clock(10))
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["reason"], "cap exceeded before verdict")


class AggregateVerdict(unittest.TestCase):
    def _records(self, seeds, *, primary_fail=True, spec_pass=True, rep=True, ctrl_ok=True):
        checks = {"S3_tau": FAIL if primary_fail else PASS, "S3_loggap": FAIL if primary_fail else PASS,
                  "S4_abs": PASS if spec_pass else FAIL, "S6_tv": PASS, "S7_abs": PASS, "class_tv": PASS,
                  "S8_class": PASS}
        recs = {}
        for s in seeds:
            recs[f"ablation|burst_timing|mimic_scale_control|{s}"] = {
                "kind": "ablation", "A_status": dict(checks),
                "known_profile_repeatability": rep, "A_specificity_ok": spec_pass}
            recs[f"control|null|mimic_scale_control|{s}"] = {   # source-scoped
                "kind": "control", "name": "null", "all_pass": ctrl_ok, "status": dict(checks)}
            # global controls (Pi §4): keyed by GLOBAL, once per seed
            recs[f"control|boundary|GLOBAL|{s}"] = {
                "kind": "control", "name": "boundary",
                "status": ({"S1_density": NOT_EVALUABLE, "S5_abs": NOT_EVALUABLE, "S6_tv": NOT_EVALUABLE,
                            "S9_zero": NOT_EVALUABLE, "S9_class": NOT_EVALUABLE, "S9_gap": NOT_EVALUABLE,
                            "S4_abs": PASS} if ctrl_ok else {"S4_abs": FAIL})}
            recs[f"control|structural_zero|GLOBAL|{s}"] = {"kind": "control", "name": "structural_zero", "ok": ctrl_ok}
            recs[f"control|source_swap|GLOBAL|{s}"] = {"kind": "control", "name": "source_swap", "fails_nondegenerate": ctrl_ok}
        return recs

    def test_conjunctive_pass_at_25(self) -> None:
        seeds = list(range(1000, 1025))
        v = ex.aggregate(self._records(seeds), ["burst_timing"], ["mimic_scale_control"], seeds)
        self.assertTrue(v["conjunctive_pass"])

    def test_primary_below_20_fails(self) -> None:
        seeds = list(range(1000, 1025))
        recs = self._records(seeds)
        # flip 6 seeds' primary to PASS => only 19/25 FAIL < 20
        for s in seeds[:6]:
            recs[f"ablation|burst_timing|mimic_scale_control|{s}"]["A_status"]["S3_tau"] = PASS
        v = ex.aggregate(recs, ["burst_timing"], ["mimic_scale_control"], seeds)
        self.assertFalse(v["conjunctive_pass"])

    def test_not_evaluable_is_non_passing(self) -> None:
        seeds = list(range(1000, 1025))
        recs = self._records(seeds)
        # a non-attributed check NOT_EVALUABLE on 2 seeds => specificity 23/25 < 24
        for s in seeds[:2]:
            recs[f"ablation|burst_timing|mimic_scale_control|{s}"]["A_status"]["S4_abs"] = NOT_EVALUABLE
        v = ex.aggregate(recs, ["burst_timing"], ["mimic_scale_control"], seeds)
        self.assertFalse(v["conjunctive_pass"])

    def test_controls_routed_source_vs_global(self) -> None:  # Pi §4
        seeds = list(range(1000, 1025))
        v = ex.aggregate(self._records(seeds), ["burst_timing"], ["mimic_scale_control"], seeds)
        ctrl = v["controls"]
        # global controls gated once (not per source); null stays per source
        self.assertIn("global", ctrl)
        self.assertEqual(set(ctrl["global"]), {"boundary_exact", "structural_zero_ok", "source_swap_nondegenerate", "n"})
        self.assertIn("null_pass", ctrl["per_source"]["mimic_scale_control"])
        self.assertNotIn("boundary_exact", ctrl["per_source"]["mimic_scale_control"])
        # per-check known-profile PASS/FAIL/NE diagnostic present
        rates = v["per_component"]["burst_timing"]["per_source"]["mimic_scale_control"]["known_profile_rates"]
        self.assertEqual(rates["S3_tau"]["FAIL"], 25)

    def test_global_control_failure_blocks_pass(self) -> None:  # global gate is conjunctive
        seeds = list(range(1000, 1025))
        recs = self._records(seeds)
        for s in seeds[:2]:                                    # 2 seeds' structural-zero fail => 23/25 < 24
            recs[f"control|structural_zero|GLOBAL|{s}"]["ok"] = False
        v = ex.aggregate(recs, ["burst_timing"], ["mimic_scale_control"], seeds)
        self.assertFalse(v["conjunctive_pass"])


if __name__ == "__main__":
    unittest.main()
