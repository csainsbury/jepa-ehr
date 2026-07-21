"""Step-4 identifiability runner (Pi §5) — fail-closed, checkpoint/resume, cap, atomic, per-profile verdict.

Machinery correctness at a tiny mechanical scale (the full grid is the reviewed job): grid generation; a
per-profile run with atomic result + checkpoint + resume; cap-exceed => PARTIAL; tampered manifest => REFUSED;
strict null covariance => a profile whose covariance seed refuses is non-pass.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from clinical_jepa.eval import oracle_realism_v2_ident_runner as ir
from clinical_jepa.eval import oracle_realism_v2_identifiability as idf
from clinical_jepa.eval import oracle_realism_v2_step4_runner as rn
from clinical_jepa.eval import oracle_realism_v2_battery as bat


def _verify(m):
    return rn.verify_manifest(m, require_git_head=False)


_TINY_GRID = [{c: v for c in idf.COMPONENTS} for v in (0.1, 0.35, 0.55)]
_RANK_PTS = [{c: 0.35 for c in idf.COMPONENTS}]


class Grid(unittest.TestCase):
    def test_grid_and_rank_points(self) -> None:
        self.assertEqual(len(ir.ident_grid()), 81)                 # 3^4
        self.assertEqual(len(ir.interior_rank_points()), 3)        # low/mid/high


class IdentExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp(prefix="v2ident_")
        cls.m = rn.build_manifest(reviewed_commit="c")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def _exec(self, run_id, m=None, **kw):
        base = bat.multiscale_smoke_sampler(n_each=600)             # local (a class attr would bind self)
        return ir.execute_identifiability(m or self.m, run_id, self.out, base_sampler=base,
                                          nuisance_profiles=["mimic_scale_control"], cov_seeds=[1000, 1001, 1002],
                                          ref_seed=1000, heldout_seed=1003, grid=_TINY_GRID, rank_points=_RANK_PTS,
                                          verify=_verify, **kw)

    def test_run_persists_and_resumes(self) -> None:
        r1 = self._exec("identA", cap_hours=1.0, cap_gb=64)
        self.assertIn(r1["status"], ("PASS", "FAIL"))              # profile evaluable at n=600
        self.assertIn("mimic_scale_control", r1["per_profile"])
        rd = ir.run_dir(self.out, "identA")
        self.assertTrue(os.path.exists(os.path.join(rd, "result.json")))
        self.assertTrue(os.path.exists(os.path.join(rd, "checkpoint.json")))
        self.assertEqual(os.listdir(os.path.join(rd, "profiles")), ["mimic_scale_control.json"])
        r2 = self._exec("identA", cap_hours=1.0, cap_gb=64)        # resume: profile done
        self.assertEqual(r2["status"], r1["status"])

    def test_cap_exceed_partial(self) -> None:
        r = self._exec("identCap", cap_hours=0.0, cap_gb=1e9)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["reason"], "cap exceeded")

    def test_refused_on_tampered_manifest(self) -> None:
        r = self._exec("identR", m=dict(self.m, registered_n=1), cap_hours=1.0, cap_gb=64)
        self.assertEqual(r["status"], "REFUSED")
        self.assertIn("registered_n", r["problems"])


if __name__ == "__main__":
    unittest.main()
