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
        self.assertEqual(len(r1["denominator_map_sha256"]), 64)
        rd = ex.run_dir(self.out, "runA")
        self.assertTrue(os.path.exists(os.path.join(rd, "result.json")))
        self.assertTrue(os.path.exists(os.path.join(rd, "checkpoint.json")))
        self.assertEqual(len(os.listdir(os.path.join(rd, "replicates"))), 2)
        r2 = self._run("runA", cap_hours=1.0, cap_gb=64)       # resume: all done
        self.assertEqual(r2["n_replicates"], 2)
        self.assertEqual(r2["denominator_map_sha256"], r1["denominator_map_sha256"])

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
        self.assertEqual(r["status"], "REFUSED")
        self.assertIn("seeds", r["problems"])


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
            recs[f"control|null|mimic_scale_control|{s}"] = {"kind": "control", "name": "null", "all_pass": ctrl_ok}
            recs[f"control|boundary|mimic_scale_control|{s}"] = {
                "kind": "control", "name": "boundary",
                "status": ({"S1_density": NOT_EVALUABLE, "S5_abs": NOT_EVALUABLE, "S6_tv": NOT_EVALUABLE,
                            "S9_zero": NOT_EVALUABLE, "S9_class": NOT_EVALUABLE, "S9_gap": NOT_EVALUABLE,
                            "S4_abs": PASS} if ctrl_ok else {"S4_abs": FAIL})}
            recs[f"control|structural_zero|mimic_scale_control|{s}"] = {"kind": "control", "name": "structural_zero", "ok": ctrl_ok}
            recs[f"control|source_swap|mimic_scale_control|{s}"] = {"kind": "control", "name": "source_swap", "fails_nondegenerate": ctrl_ok}
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


if __name__ == "__main__":
    unittest.main()
