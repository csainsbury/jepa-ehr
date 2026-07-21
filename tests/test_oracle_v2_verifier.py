"""Step-3 (rebuild) — executable S1–S9 verifier: self-recovery, sensitivity, floors, coarsening.

Validates the statistics compute correctly: self-recovery (candidate ~ reference) yields no FAIL; a targeted
perturbation trips the intended check; floors return NOT_EVALUABLE; and the reference-only coarsening algorithm
behaves. Modest N (event-volume cost is a step-4 concern). Baseline fixture (no couplings yet).
"""
from __future__ import annotations

from math import log

import numpy as np
import unittest

from clinical_jepa.eval import oracle_realism_v2_fixture as fx
from clinical_jepa.eval import oracle_realism_v2_verifier as vf

_VERIFIER_IMPL_ID = "7e97bf3712618fb896da243cc4be583526ff51307ff58a81ecabb69f12305dcb"


def _profile(mu, gap_mu=log(1.2)):
    return {"length": {"family": "discretized_lognormal", "mu": mu, "sigma": 0.35, "min": 1},
            "class_prior": [0.3, 0.25, 0.2, 0.15, 0.1], "structural_zero_classes": [],
            "cluster_size": {"family": "geometric", "p": 0.5},
            "gap": {"family": "lognormal", "mu": gap_mu, "sigma": 0.85}, "dependence": {}}


def _multiscale(seed, n_each=600, gap_mu=log(1.2)):
    """Three length scales (bins 9-32, 33-128, 129-512) so length-conditioned checks can evaluate."""
    recs = []
    for i, mu in enumerate((log(18), log(60), log(250))):
        recs += fx.sample_fixture("MIMIC", _profile(mu, gap_mu), n_each, seed=seed * 10 + i)
    return recs


class SelfRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = _multiscale(1)
        cls.cand = _multiscale(2)
        cls.seq = vf.sequence_route_checks(cls.cand, cls.ref)
        cls.marg = vf.marginal_route_checks(cls.cand, cls.ref)

    def test_no_fail_on_self_recovery(self) -> None:
        for k, v in {**self.marg, **self.seq}.items():
            self.assertNotEqual(v.status, vf.FAIL, f"{k} unexpectedly FAILed at {v.value}")

    def test_length_conditioned_checks_evaluate(self) -> None:
        for k in ("S1_density", "S5_abs", "S6_tv"):
            self.assertEqual(self.seq[k].status, vf.PASS, f"{k} should evaluate+pass under multiscale")

    def test_all_marginals_evaluate(self) -> None:
        for k in ("length_ks", "class_tv", "count_ks", "occupancy_abs", "delta_t_zero_abs", "positive_gap_ks"):
            self.assertEqual(self.marg[k].status, vf.PASS, k)


class Sensitivity(unittest.TestCase):
    def test_gap_perturbation_trips_S3_and_gap_marginal(self) -> None:
        ref = _multiscale(1)
        cand = _multiscale(3, gap_mu=log(1.2 * 1.6))          # 1.6x larger gaps
        seq = vf.sequence_route_checks(cand, ref)
        marg = vf.marginal_route_checks(cand, ref)
        self.assertEqual(seq["S3_loggap"].status, vf.FAIL)     # E[log gap] shifts by log 1.6 > log 1.10
        self.assertEqual(marg["positive_gap_ks"].status, vf.FAIL)
        self.assertEqual(seq["S4_abs"].status, vf.PASS)        # class structure untouched


class Floors(unittest.TestCase):
    def test_tiny_sample_not_evaluable(self) -> None:
        ref = fx.sample_fixture("MIMIC", _profile(log(60)), 40, seed=1)
        cand = fx.sample_fixture("MIMIC", _profile(log(60)), 40, seed=2)
        seq = vf.sequence_route_checks(cand, ref)
        for k in ("S1_density", "S2_ks", "S4_abs", "S7_abs"):
            self.assertEqual(seq[k].status, vf.NOT_EVALUABLE, k)


class Coarsening(unittest.TestCase):
    def test_merges_sparse_into_neighbour(self) -> None:
        # bins: [600, 50, 600, 600, 10] -> sparse (idx1,4) merge into neighbours; 3+ groups remain
        groups = vf.coarsen_reference(np.array([600, 50, 600, 600, 10]))
        self.assertIsNotNone(groups)
        for g in groups:
            self.assertGreaterEqual(sum(np.array([600, 50, 600, 600, 10])[i] for i in g), vf.FLOOR)
        self.assertGreaterEqual(len(groups), vf.MIN_BINS)

    def test_refuses_when_below_min_bins(self) -> None:
        # only one viable bin -> cannot keep >=3 bins each >=500 -> refuse
        self.assertIsNone(vf.coarsen_reference(np.array([50, 50, 3000, 50, 50])))


class VerifierIdentity(unittest.TestCase):
    def test_impl_identity_pinned(self) -> None:
        self.assertEqual(vf.verifier_impl_identity(), _VERIFIER_IMPL_ID)
        self.assertIn("S9_gap", vf.VERIFIER_IMPL["terminal_no_D"])
        self.assertIn("S1_density", vf.VERIFIER_IMPL["terminal_no_D"])   # S1 terminal (Pi F1)
        self.assertIn("S5_abs", vf.VERIFIER_IMPL["terminal_no_D"])       # S5 terminal (Pi F2)


if __name__ == "__main__":
    unittest.main()
