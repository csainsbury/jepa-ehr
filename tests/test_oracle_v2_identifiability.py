"""Step-3 (rebuild) — identifiability battery machinery (Pi §6).

Pure-function checks (FD rule, rank criterion, nearest-grid recovery, collision search, cost forecast, frozen
params) plus a small verifier-backed pipeline (f_theta at theta=0 / interior, Jacobian shape + standardized
rank). The full 3^4 x nuisance x seeds grid runs only under the reviewed step-4 job.
"""
from __future__ import annotations

import numpy as np
import unittest

from clinical_jepa.eval import oracle_realism_v2_identifiability as idf
from clinical_jepa.eval import oracle_realism_v2_battery as bat

_IMPL_ID = "b81baac98da76e6810243ac7aad98d76bdc8232042b4191470f262556e730c0a"


class FrozenParams(unittest.TestCase):
    def test_impl_identity_and_params(self) -> None:
        self.assertEqual(idf.identifiability_impl_identity(), _IMPL_ID)
        self.assertEqual(idf.D_VECTOR, ("S3_tau", "S3_loggap", "S4_abs", "S6_tv", "S7_abs"))
        self.assertEqual(idf.GRID, (0.10, 0.35, 0.55))
        self.assertEqual(idf.PARAM_RANGE, (0.0, 0.6))
        self.assertAlmostEqual(idf.RECOVER_TOL, 0.03)
        self.assertEqual(idf.RANK_MIN, 1e-3)

    def test_fd_rule(self) -> None:
        base = {c: 0.35 for c in idf.COMPONENTS}
        hi, lo, den = idf._fd_pair({**base, "burst_timing": 0.0}, "burst_timing")   # forward at 0
        self.assertEqual(den, idf.FD_STEP); self.assertAlmostEqual(hi["burst_timing"], idf.FD_STEP)
        hi, lo, den = idf._fd_pair({**base, "burst_timing": 0.6}, "burst_timing")   # backward at 0.6
        self.assertEqual(den, idf.FD_STEP); self.assertAlmostEqual(lo["burst_timing"], 0.6 - idf.FD_STEP)
        hi, lo, den = idf._fd_pair(base, "burst_timing")                            # central interior
        self.assertAlmostEqual(den, 2 * idf.FD_STEP)


class PureNumerics(unittest.TestCase):
    def test_standardized_rank(self) -> None:
        Sig = np.eye(5)
        J_good = np.zeros((5, 4)); J_good[np.arange(4), np.arange(4)] = 10.0    # full column rank
        self.assertTrue(idf.standardized_rank(J_good, Sig)["rank_ok"])
        J_bad = J_good.copy(); J_bad[:, 1] = J_bad[:, 0]                          # rank-deficient
        self.assertFalse(idf.standardized_rank(J_bad, Sig)["rank_ok"])

    def test_nearest_grid_recovery(self) -> None:
        thetas = [{c: v for c in idf.COMPONENTS} for v in (0.1, 0.35, 0.55)]
        vecs = [np.full(5, v) for v in (0.1, 0.35, 0.55)]
        rec = idf.nearest_grid_recovery(np.full(5, 0.34), thetas, vecs)
        self.assertEqual(rec, thetas[1])

    def test_collision_search(self) -> None:
        far = [{**{c: 0.1 for c in idf.COMPONENTS}}, {**{c: 0.55 for c in idf.COMPONENTS}}]
        same = [np.zeros(5), np.zeros(5)]                                        # identical vectors, far apart
        self.assertTrue(idf.collision_search(far, same, accept_tol=0.01))
        distinct = [np.zeros(5), np.full(5, 1.0)]
        self.assertFalse(idf.collision_search(far, distinct, accept_tol=0.01))

    def test_cost_forecast(self) -> None:
        f = idf.cost_forecast(seconds_per_eval=8.6)
        self.assertEqual(f["grid_points"], 81)
        self.assertEqual(f["f_evals_full_grid"], 81 * 2 * (1 + 2 * 4))
        self.assertIn("PARTIAL", f["note"])


class Pipeline(unittest.TestCase):
    def test_f_theta_jacobian_and_rank(self) -> None:
        base = bat.multiscale_smoke_sampler(n_each=600)
        sp = "mimic_scale_control"
        z = idf.f_theta({c: 0.0 for c in idf.COMPONENTS}, base_sampler=base, source_profile=sp, seed=1000)
        mid = idf.f_theta({c: 0.35 for c in idf.COMPONENTS}, base_sampler=base, source_profile=sp, seed=1000)
        self.assertEqual(z.shape, (5,))
        self.assertTrue(np.all(np.abs(z) < 0.12))                 # null vs null: small
        self.assertGreater(float(np.sum(np.abs(mid))), float(np.sum(np.abs(z))))   # couplings move the vector
        Sig = idf.null_covariance(base, [1000, 1001, 1002], source_profile=sp)
        J = idf.jacobian({c: 0.35 for c in idf.COMPONENTS}, base_sampler=base, source_profile=sp, seed=1000)
        self.assertEqual(J.shape, (5, 4))
        self.assertIn("rank_ok", idf.standardized_rank(J, Sig))


if __name__ == "__main__":
    unittest.main()
