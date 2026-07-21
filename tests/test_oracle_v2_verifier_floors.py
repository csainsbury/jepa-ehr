"""Step-3 (rebuild) — hand-calculated floor fixtures + denominator reporting (Pi denominator hardening).

Deterministic tiny/controlled fixtures exercise each floor path (S1_tau source, S2 cluster, S3 adjacent-pair,
S4 same/adjacent-pair, S7 cluster) and the source-partition guard, and confirm candidate/reference denominators
+ the four S9 KS values are reported.
"""
from __future__ import annotations

from math import log

import numpy as np
import unittest

from clinical_jepa.eval import oracle_realism_v2_fixture as fx
from clinical_jepa.eval import oracle_realism_v2_verifier as vf


def _rec(source, classes, times):
    return fx.derive_record(source, np.asarray(classes), np.asarray(times, dtype=float))


def _multiscale(seed, n=600):
    r = []
    for i, mu in enumerate((log(18), log(60), log(250))):
        p = {"length": {"family": "discretized_lognormal", "mu": mu, "sigma": 0.35, "min": 1},
             "class_prior": [0.3, 0.25, 0.2, 0.15, 0.1], "structural_zero_classes": [],
             "cluster_size": {"family": "geometric", "p": 0.5},
             "gap": {"family": "lognormal", "mu": log(1.2), "sigma": 0.85}, "dependence": {}}
        r += fx.sample_fixture("MIMIC", p, n, seed=seed * 10 + i)
    return r


class SourcePartition(unittest.TestCase):
    def test_mismatched_and_mixed_sources_rejected(self) -> None:
        a = [_rec("MIMIC", [0, 1], [0.0, 1.0])]
        b = [_rec("SCID", [0, 1], [0.0, 1.0])]
        with self.assertRaises(vf.MixedSourceError):
            vf.sequence_route_checks(a, b)                       # candidate MIMIC vs reference SCID
        with self.assertRaises(vf.MixedSourceError):
            vf.sequence_route_checks(a + b, a)                   # mixed-source candidate sample
        with self.assertRaises(vf.MixedSourceError):
            vf.marginal_route_checks(a, b)


class HandFloorFixtures(unittest.TestCase):
    def test_tiny_sample_hits_floors_with_reported_denominators(self) -> None:
        # 3 hand-built 2-cluster sequences per side: every floor is far below 500 => NOT_EVALUABLE with report.
        s = [_rec("MIMIC", [0, 1, 2], [0.0, 1.0, 2.0]) for _ in range(3)]
        seq = vf.sequence_route_checks(s, s)
        self.assertEqual(seq["S1_tau"].status, vf.NOT_EVALUABLE)
        self.assertIn("source-level sequence floor", seq["S1_tau"].detail["reason"])
        self.assertEqual(seq["S1_tau"].detail["seq_cand"], 3)
        self.assertEqual(seq["S2_ks"].status, vf.NOT_EVALUABLE)
        self.assertIn("clusters_cand", seq["S2_ks"].detail)      # cluster denominator reported
        self.assertEqual(seq["S4_abs"].status, vf.NOT_EVALUABLE)
        self.assertIn("same_pairs_cand", seq["S4_abs"].detail)   # pair denominators reported
        self.assertIn("adj_pairs_cand", seq["S4_abs"].detail)

    def test_s2_cluster_denominator_matches_hand_count(self) -> None:
        # 4 sequences, each two singletons + one 2-run => 3 clusters each => 12 clusters total.
        s = [_rec("MIMIC", [0, 1, 2, 2], [0.0, 1.0, 2.0, 2.0]) for _ in range(4)]
        d = vf.sequence_route_checks(s, s)["S2_ks"].detail
        self.assertEqual(d["clusters_cand"], 12)
        self.assertEqual(d["clusters_ref"], 12)


class DenominatorReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = _multiscale(1); cls.cand = _multiscale(2)
        cls.seq = vf.sequence_route_checks(cls.cand, cls.ref)

    def test_S9_emits_all_four_ks_values(self) -> None:
        d = self.seq["S9_gap"].detail
        for k in ("ks_within_cand", "ks_within_ref", "ks_cross_seam", "ks_cross_nonseam"):
            self.assertIn(k, d, k)
        for k in ("seam_adj_cand", "nonseam_adj_cand"):
            self.assertIn(k, d)

    def test_conditional_checks_report_per_bin_denominators(self) -> None:
        # S3 reports adjacent-pair denominators; S7 reports cluster denominators (when evaluable).
        for name, key in (("S3_loggap", "adj_pairs_cand"), ("S7_abs", "clusters_cand")):
            r = self.seq[name]
            if r.status != vf.NOT_EVALUABLE or "under map" in r.detail.get("reason", ""):
                self.assertIn(key, r.detail, f"{name} must report {key}")


if __name__ == "__main__":
    unittest.main()
