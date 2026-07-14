"""Calibration + realism-envelope tests (Pi 2nd-pass Phase 5). SYNTHETIC aggregate fixtures only."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_calibration as CAL
from clinical_jepa.eval.oracle_calibration import AggregateStats
from clinical_jepa.eval.oracle_meta_gen import invariant_hash


def _ecdf(points):
    return tuple((float(s), float(c)) for s, c in points)


def _agg(source="SCID", dt0=0.35, counts=(1000, 1000, 1000, 1000, 1000, 1000), occ=0.9,
         n_seq=2000, n_clu=3000):
    return AggregateStats(source=source, n_sequences=n_seq, n_events=sum(counts), n_clusters=n_clu,
                          n_positive_gaps=2500, class_counts=tuple(counts), delta_t_zero_fraction=dt0,
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

    def test_calibration_binds_the_canonical_mechanism_identity(self) -> None:
        # Pi #10: calibration's mechanism hash is the CANONICAL invariant_hash (not the legacy spec).
        before = invariant_hash()
        r = CAL.fit_calibration(_agg(), _agg())
        self.assertEqual(r.mechanism_hash, before)      # bound to the whole-pass mechanism
        self.assertEqual(invariant_hash(), before)      # unchanged by calibration
        self.assertNotEqual(r.fitted_param_hash, r.spec_hash)   # separate hashes
        self.assertEqual(r.spec_hash, CAL.calibration_schema_hash())   # spec_hash = frozen schema

    def test_fit_refuses_invalid_target_or_base(self) -> None:
        r = CAL.fit_calibration(_agg(n_seq=10), _agg())    # under-supported target
        self.assertFalse(r.within_envelope)
        self.assertEqual(r.diagnostics.get("refused"), "target_" + CAL.NOT_EVALUABLE)
        rb = CAL.fit_calibration(_agg(), _agg(n_seq=10))   # under-supported BASE (validated too, Pi #10)
        self.assertEqual(rb.diagnostics.get("refused"), "base_" + CAL.NOT_EVALUABLE)
        self.assertEqual(r.fitted_knobs, {})


class StrengthenedValidationTests(unittest.TestCase):
    def test_class_counts_must_reconcile_with_n_events(self) -> None:
        bad = AggregateStats(source="SCID", n_sequences=2000, n_events=9999, n_clusters=3000,
                             n_positive_gaps=2500, class_counts=(1000,) * 6, delta_t_zero_fraction=0.3,
                             length_ecdf=_ecdf([(1, 0.5), (2, 1.0)]),
                             positive_gap_ecdf=_ecdf([(0.1, 0.5), (1.0, 1.0)]),
                             count_ecdf=_ecdf([(1, 0.5), (2, 1.0)]), mean_occupancy_fraction=0.9)
        self.assertFalse(CAL.validate_aggregate_input(bad)[0])       # sum(counts) 6000 != n_events 9999

    def test_malformed_ecdf_refused(self) -> None:
        non_monotone = _ecdf([(1, 0.6), (2, 0.3), (3, 1.0)])          # cdf decreases
        self.assertFalse(CAL._ecdf_valid(non_monotone))
        not_final_1 = _ecdf([(1, 0.2), (2, 0.5)])                     # final mass != 1
        self.assertFalse(CAL._ecdf_valid(not_final_1))
        non_increasing_supp = _ecdf([(1, 0.2), (1, 0.6), (2, 1.0)])   # duplicate support
        self.assertFalse(CAL._ecdf_valid(non_increasing_supp))

    def test_event_and_gap_denominator_floors(self) -> None:
        low = _agg(counts=(80, 80, 80, 80, 80, 80))                   # n_events 480 < ORACLE_ENV_MIN_DENOM
        self.assertEqual(CAL.validate_aggregate_input(low)[1], CAL.NOT_EVALUABLE)

    def test_source_mismatch_refused(self) -> None:
        self.assertFalse(CAL.realism_envelope(_agg(source="SCID"), _agg(source="MIMIC")).within_envelope)
        r = CAL.fit_calibration(_agg(source="MIMIC"), _agg(source="SCID"))
        self.assertEqual(r.diagnostics.get("refused"), "source_mismatch")


class TimingKnobTests(unittest.TestCase):
    def test_timing_knobs_actually_move_the_gap_ecdf(self) -> None:
        base = _agg()
        slow = CAL._forward_aggregate(base, {"timing_rate_scale": 0.5, "gap_dispersion": 1.0})
        fast = CAL._forward_aggregate(base, {"timing_rate_scale": 2.0, "gap_dispersion": 1.0})
        # a smaller rate scale => larger gaps => the gap ECDF support shifts right (not inert).
        self.assertGreater(slow.positive_gap_ecdf[-1][0], fast.positive_gap_ecdf[-1][0])
        self.assertNotEqual(slow.positive_gap_ecdf, base.positive_gap_ecdf)

    def test_full_input_hash_covers_ecdfs(self) -> None:
        base, target = _agg(), _agg(dt0=0.4)
        h1 = CAL.fit_calibration(target, base).input_hash
        target2 = _agg(dt0=0.4)
        # perturb only an ECDF point -> the input hash must change (full canonical input hashed).
        object.__setattr__(target2, "count_ecdf", _ecdf([(1, 0.3), (5, 0.7), (15, 1.0)]))
        self.assertNotEqual(h1, CAL.fit_calibration(target2, base).input_hash)


class MultiSourceTests(unittest.TestCase):
    def test_collection_requires_every_source_within_envelope(self) -> None:
        # SCID matches itself; MIMIC target differs from its base beyond the envelope (occupancy).
        res = CAL.calibrate_sources(
            targets={"SCID": _agg(source="SCID"), "MIMIC": _agg(source="MIMIC", occ=0.5)},
            bases={"SCID": _agg(source="SCID"), "MIMIC": _agg(source="MIMIC", occ=0.9)})
        self.assertTrue(res.source_coverage_ok)                       # covers exactly {SCID, MIMIC}
        self.assertTrue(res.per_source["SCID"].within_envelope)
        self.assertFalse(res.per_source["MIMIC"].within_envelope)
        self.assertFalse(res.all_sources_within_envelope)             # conjunction fails
        self.assertEqual(res.mechanism_hash, invariant_hash())        # bound to the canonical mechanism

    def test_incomplete_or_extra_source_set_fails_coverage(self) -> None:
        # missing MIMIC -> coverage fails even though SCID passes its envelope (Pi #10 required source set).
        res = CAL.calibrate_sources(targets={"SCID": _agg(source="SCID")},
                                    bases={"SCID": _agg(source="SCID")})
        self.assertFalse(res.source_coverage_ok)
        self.assertFalse(res.all_sources_within_envelope)
        # an UNEXPECTED extra source also fails coverage.
        res2 = CAL.calibrate_sources(
            targets={"SCID": _agg(source="SCID"), "MIMIC": _agg(source="MIMIC"), "EXTRA": _agg(source="EXTRA")},
            bases={"SCID": _agg(source="SCID"), "MIMIC": _agg(source="MIMIC"), "EXTRA": _agg(source="EXTRA")})
        self.assertFalse(res2.source_coverage_ok)


if __name__ == "__main__":
    unittest.main()
