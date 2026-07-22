"""Step-3 (rebuild) — independent fixture constructor invariants + malformed refusal (Pi binding cond. 2/5).

Tests the claimed invariants DIRECTLY: exact raw-timestamp-equality clustering, canonical run ids, exact ties,
L==1 handling + dt0 exclusion, structural-zero preservation, determinism, and malformed / non-finite / bool
input refusal. Baseline constructor only (no couplings yet).
"""
from __future__ import annotations

import hashlib
from math import log

import numpy as np
import unittest

from clinical_jepa.eval import oracle_realism_v2_fixture as fx

_PROF = {"length": {"family": "discretized_lognormal", "mu": log(99), "sigma": 1.0, "min": 1},
         "class_prior": [0.1, 0.15, 0.2, 0.25, 0.3], "structural_zero_classes": [],
         "cluster_size": {"family": "geometric", "p": 0.45},
         "gap": {"family": "lognormal", "mu": log(1.5), "sigma": 0.9}, "dependence": {}}
_FIXTURE_IMPL_ID = "718788bd21d30479ad7ffc5eac3dced8efc26f3665740c8df86282af15287bf5"


class FixtureInvariants(unittest.TestCase):
    def test_run_and_cluster_invariants(self) -> None:
        for r in fx.sample_fixture("MIMIC", _PROF, 1000, seed=1):
            self.assertEqual(r.L_total, r.class_ids.shape[0])
            self.assertEqual(r.L_total, r.timestamps.shape[0])
            self.assertEqual(r.K, int(r.cluster_ids[-1] + 1))
            self.assertEqual(r.cluster_ids[0], 0)
            self.assertTrue(np.all(np.isin(np.diff(r.cluster_ids), (0, 1))))   # canonical contiguous
            if r.L_total > 1:
                dt = np.diff(r.timestamps)
                self.assertTrue(np.all(dt >= 0.0))                             # nondecreasing
                # within a run all timestamps are EXACTLY equal; across runs strictly positive
                for c in range(r.K):
                    run_ts = r.timestamps[r.cluster_ids == c]
                    self.assertTrue(np.all(run_ts == run_ts[0]), "exact tie within run")
                self.assertTrue(np.all(dt[np.diff(r.cluster_ids) == 1] > 0.0))
                self.assertAlmostEqual(r.positions[0], 0.0)
                self.assertAlmostEqual(r.positions[-1], 1.0)

    def test_L1_handling_and_dt0_exclusion(self) -> None:
        rec = fx.derive_record("SCID", np.array([2]), np.array([0.0]))
        self.assertEqual(rec.L_total, 1)
        self.assertEqual(rec.K, 1)
        self.assertEqual(rec.positions.tolist(), [0.0])
        # dt0 over a sample of only L==1 sequences is NaN (no adjacencies), not 0
        self.assertTrue(np.isnan(fx.reg_dt0_pooled([rec, rec])))

    def test_structural_zero_preserved(self) -> None:
        prof = dict(_PROF, class_prior=[0.4, 0.35, 0.25, 0.0, 0.0], structural_zero_classes=[3, 4])
        allc = np.concatenate([r.class_ids for r in fx.sample_fixture("SCID", prof, 400, seed=2)])
        self.assertTrue(set(np.unique(allc)).issubset({0, 1, 2}))

    def test_determinism(self) -> None:
        def digest(sample):
            h = hashlib.sha256()
            for r in sample:
                h.update(r.class_ids.tobytes()); h.update(np.ascontiguousarray(r.timestamps).tobytes())
            return h.hexdigest()
        a = fx.sample_fixture("MIMIC", _PROF, 200, seed=7)
        b = fx.sample_fixture("MIMIC", _PROF, 200, seed=7)
        c = fx.sample_fixture("MIMIC", _PROF, 200, seed=8)
        self.assertEqual(digest(a), digest(b))
        self.assertNotEqual(digest(a), digest(c))


class MalformedRefusal(unittest.TestCase):
    def test_refusals(self) -> None:
        good_ci, good_ts = np.array([0, 1, 2]), np.array([0.0, 0.0, 1.0])
        with self.assertRaises(fx.MalformedRecord):        # bad source
            fx.derive_record("NOPE", good_ci, good_ts)
        with self.assertRaises(fx.MalformedRecord):        # unequal lengths
            fx.derive_record("SCID", np.array([0, 1]), good_ts)
        with self.assertRaises(fx.MalformedRecord):        # empty
            fx.derive_record("SCID", np.array([], dtype=int), np.array([], dtype=float))
        with self.assertRaises(fx.MalformedRecord):        # bool class_ids
            fx.derive_record("SCID", np.array([True, False, True]), good_ts)
        with self.assertRaises(fx.MalformedRecord):        # integer timestamps (must be float)
            fx.derive_record("SCID", good_ci, np.array([0, 0, 1]))
        with self.assertRaises(fx.MalformedRecord):        # non-finite timestamp
            fx.derive_record("SCID", good_ci, np.array([0.0, np.inf, 1.0]))
        with self.assertRaises(fx.MalformedRecord):        # class out of range
            fx.derive_record("SCID", np.array([0, 1, 9]), good_ts)
        with self.assertRaises(fx.MalformedRecord):        # nonmonotone timestamps
            fx.derive_record("SCID", good_ci, np.array([0.0, 1.0, 0.5]))


class RegisteredMarginals(unittest.TestCase):
    def test_estimands(self) -> None:
        sample = fx.sample_fixture("MIMIC", _PROF, 1500, seed=3)
        props = fx.reg_class_tv_proportions(sample)
        self.assertAlmostEqual(float(props.sum()), 1.0, places=6)
        d = fx.reg_dt0_pooled(sample)
        self.assertTrue(0.0 <= d <= 1.0)
        gaps = fx.reg_positive_gaps(sample)
        self.assertTrue(np.all(gaps > 0.0))
        self.assertTrue(np.allclose(gaps, np.round(gaps, 8)))     # 8dp support


class FixtureIdentity(unittest.TestCase):
    def test_impl_identity_pinned_and_independent(self) -> None:
        self.assertEqual(fx.fixture_impl_identity(), _FIXTURE_IMPL_ID)
        self.assertIn("no import/call of any M2 candidate adapter", fx.FIXTURE_IMPL["independence"])
        self.assertIn("NONE", fx.FIXTURE_IMPL["couplings"])


if __name__ == "__main__":
    unittest.main()
