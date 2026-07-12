"""Calibration + realism-envelope tests (Pi 2nd-pass Phase 5). SYNTHETIC aggregate fixtures only."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_calibration as CAL
from clinical_jepa.eval.oracle_calibration import AggregateStats
from clinical_jepa.eval.oracle_spec import oracle_mechanism_hash


def _ecdf(points):
    return tuple((float(s), float(c)) for s, c in points)


def _agg(source="SCID", dt0=0.35, counts=(100, 100, 100, 100, 100, 100), occ=0.9,
         n_seq=2000, n_clu=3000):
    return AggregateStats(source=source, n_sequences=n_seq, n_events=6000, n_clusters=n_clu,
                          n_positive_gaps=2500, class_counts=counts, delta_t_zero_fraction=dt0,
                          length_ecdf=_ecdf([(1, 0.1), (5, 0.5), (10, 0.9), (20, 1.0)]),
                          positive_gap_ecdf=_ecdf([(0.1, 0.2), (1.0, 0.6), (5.0, 1.0)]),
                          count_ecdf=_ecdf([(1, 0.2), (5, 0.7), (15, 1.0)]),
                          mean_occupancy_fraction=occ)


class ValidationTests(unittest.TestCase):
    def test_wrong_type_and_class_length_refused(self) -> None:
        self.assertFalse(CAL.validate_aggregate_input({"not": "agg"})[0])
        self.assertFalse(CAL.validate_aggregate_input(_agg(counts=(1, 2, 3)))[0])

    def test_under_supported_is_not_evaluable(self) -> None:
        ok, reason = CAL.validate_aggregate_input(_agg(n_seq=100))    # below ORACLE_ENV_MIN_DENOM
        self.assertFalse(ok)
        self.assertEqual(reason, CAL.NOT_EVALUABLE)

    def test_fraction_out_of_range_refused(self) -> None:
        self.assertFalse(CAL.validate_aggregate_input(_agg(dt0=1.5))[0])


class EnvelopeTests(unittest.TestCase):
    def test_identical_aggregates_are_within_envelope(self) -> None:
        a = _agg()
        res = CAL.realism_envelope(a, a)
        self.assertTrue(res.within_envelope)
        self.assertTrue(all(passed for _, passed in res.checks.values()))

    def test_conjunctive_single_violation_fails(self) -> None:
        target = _agg(dt0=0.35)
        synth = _agg(dt0=0.40)                       # Δt=0 off by 0.05 > 0.02 => fails one check
        res = CAL.realism_envelope(synth, target)
        self.assertFalse(res.within_envelope)        # conjunction: one failure fails the whole envelope
        self.assertFalse(res.checks["delta_t_zero_abs"][1])
        self.assertTrue(res.checks["length_ks"][1])  # others still individually pass

    def test_occupancy_violation_fails(self) -> None:
        res = CAL.realism_envelope(_agg(occ=0.80), _agg(occ=0.90))   # 0.10 > 0.03
        self.assertFalse(res.within_envelope)


class FitTests(unittest.TestCase):
    def test_fit_matches_target_within_envelope_and_is_deterministic(self) -> None:
        import numpy as np
        # a temperature-reachable target: base tempered by ~1.4 (calibration adjusts peakedness + Δt=0);
        # occupancy + ECDFs are structural and already match (same _agg defaults).
        base_counts = np.array([150, 120, 100, 90, 80, 60], float)
        p = (base_counts / base_counts.sum()) ** (1.0 / 1.4)
        tgt_counts = tuple(int(round(x)) for x in p / p.sum() * 600)
        target = _agg(dt0=0.42, counts=tgt_counts)
        base = _agg(dt0=0.20, counts=tuple(int(x) for x in base_counts))
        r1 = CAL.fit_calibration(target, base)
        r2 = CAL.fit_calibration(target, base)
        self.assertTrue(r1.within_envelope, r1.diagnostics)
        self.assertEqual(r1.fitted_param_hash, r2.fitted_param_hash)   # deterministic
        self.assertAlmostEqual(r1.fitted_knobs["zero_gap_bias"], 0.42, places=6)

    def test_calibration_cannot_mutate_the_mechanism(self) -> None:
        before = oracle_mechanism_hash()
        r = CAL.fit_calibration(_agg(), _agg())
        self.assertEqual(r.mechanism_hash, before)      # mechanism hash unchanged by calibration
        self.assertEqual(oracle_mechanism_hash(), before)
        self.assertNotEqual(r.fitted_param_hash, r.spec_hash)   # separate hashes

    def test_fit_refuses_invalid_target(self) -> None:
        r = CAL.fit_calibration(_agg(n_seq=10), _agg())    # under-supported target
        self.assertFalse(r.within_envelope)
        self.assertEqual(r.diagnostics.get("refused"), CAL.NOT_EVALUABLE)
        self.assertEqual(r.fitted_knobs, {})


if __name__ == "__main__":
    unittest.main()
