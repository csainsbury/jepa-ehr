"""Step-3 (rebuild) — coupling directional movement + direct-testing findings (Pi binding condition 2).

Confirms each WORKING coupling moves its intended verifier check, records the S4<->S7 cross-loading, and
ENCODES the two findings (F1: burst_count_length is a weak/ineffective S1 mover under exact S2 preservation;
F2: length_class_mix moves S6 not S5). Slower (uses the verifier); modest N.
"""
from __future__ import annotations

from math import log

import unittest

from clinical_jepa.eval import oracle_realism_v2_fixture as fx
from clinical_jepa.eval import oracle_realism_v2_verifier as vf
from clinical_jepa.eval import oracle_realism_v2_coupling as cp


def _prof(mu):
    return {"length": {"family": "discretized_lognormal", "mu": mu, "sigma": 0.35, "min": 1},
            "class_prior": [0.3, 0.25, 0.2, 0.15, 0.1], "structural_zero_classes": [],
            "cluster_size": {"family": "geometric", "p": 0.5},
            "gap": {"family": "lognormal", "mu": log(1.2), "sigma": 0.85}, "dependence": {}}


def _multiscale(seed, n=600):
    r = []
    for i, mu in enumerate((log(18), log(60), log(250))):
        r += fx.sample_fixture("MIMIC", _prof(mu), n, seed=seed * 10 + i)
    return r


class CouplingDirection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = _multiscale(1)

    def _checks(self, component, strength, cand_seed):
        cand = cp.apply_coupling(_multiscale(cand_seed), component, strength, seed=1)
        out = vf.sequence_route_checks(cand, self.ref)
        out.update(vf.marginal_route_checks(cand, self.ref))
        return out

    def test_burst_timing_moves_S3(self) -> None:
        c = self._checks("burst_timing", 0.6, 2)
        self.assertEqual(c["S3_tau"].status, vf.FAIL)
        self.assertEqual(c["S3_loggap"].status, vf.FAIL)
        self.assertIn(c["positive_gap_ks"].status, (vf.PASS, vf.NOT_EVALUABLE))    # gap multiset preserved

    def test_mark_burst_tie_moves_S4_and_crossloads_S7(self) -> None:
        c = self._checks("mark_burst_tie", 0.6, 3)
        self.assertEqual(c["S4_abs"].status, vf.FAIL)
        self.assertEqual(c["S7_abs"].status, vf.FAIL)                 # recorded S4<->S7 cross-loading
        self.assertEqual(c["class_tv"].status, vf.PASS)              # exact class counts

    def test_cluster_size_mark_diversity_moves_S7(self) -> None:
        c = self._checks("cluster_size_mark_diversity", 0.6, 4)
        self.assertEqual(c["S7_abs"].status, vf.FAIL)
        self.assertEqual(c["class_tv"].status, vf.PASS)
        self.assertIn(c["S2_ks"].status, (vf.PASS, vf.NOT_EVALUABLE))  # cluster sizes preserved

    def test_length_class_mix_dropped(self) -> None:  # Pi step-4 result gate: it VIOLATES terminal S5 -> dropped
        # the earlier F2 belief ("moves S6 not S5") was refuted at the result gate: candidate-A vs
        # length_class_mix@0.5 gives S5_abs ~19 FAIL/25 on MIMIC (real cross-loading, worse at higher N). Off the
        # active menu; coupling code retained as explored-space evidence but no longer dispatchable.
        self.assertNotIn("length_class_mix", cp.V2_D_COMPONENT_MENU)
        with self.assertRaises(KeyError):
            cp.apply_coupling(self.ref, "length_class_mix", 0.6, seed=1)

    def test_burst_count_length_dropped(self) -> None:             # finding F1 -> Pi dropped it (fast, no verifier)
        self.assertNotIn("burst_count_length", cp.V2_D_COMPONENT_MENU)
        with self.assertRaises(KeyError):
            cp.apply_coupling(self.ref, "burst_count_length", 0.6, seed=1)


if __name__ == "__main__":
    unittest.main()
