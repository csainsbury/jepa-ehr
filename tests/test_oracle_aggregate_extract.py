"""Aggregate-real extraction — SYNTHETIC fixtures only, NO governed read (Pi C=5 micro-gate).

Exercises the pure aggregation, the C=5 structural-token exclusion, and the fail-closed TRAIN-only /
approval-token / forbidden-key guards, without ever opening a governed HDF5.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_aggregate_extract as X
from clinical_jepa.eval.oracle_calibration import AggregateStats, NOT_EVALUABLE
from clinical_jepa.eval.rung2_contract import ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES


# representative content token ids per class (inside each ORACLE_ENV_CLASS_FAMILIES range)
_CLS_TOK = (10, 60, 100, 1000, 1040)          # demographic, diagnosis, lab, medication, state
_BOS, _DATASET_SCID = 0, 1048                 # structural prefix tokens (must be excluded)


def _seq(classes, days):
    """A synthetic content sequence with a leading [BOS]+DATASET structural prefix that MUST be dropped."""
    toks = [_BOS, _DATASET_SCID] + [_CLS_TOK[c] for c in classes]
    t = [days[0] if days else 0.0, days[0] if days else 0.0] + list(days)
    return np.array(toks), np.array(t, dtype=float)


class PureAggregationTests(unittest.TestCase):
    def test_structural_tokens_excluded_and_counts_correct(self) -> None:
        # one sequence: classes [0,0,1,2] at days [0,0,1,3] -> length 4, clusters {0,1,3}=3, one zero-gap
        toks, days = _seq([0, 0, 1, 2], [0.0, 0.0, 1.0, 3.0])
        f = X._sequence_features(toks, days)
        self.assertEqual(f["length"], 4)                       # 4 content tokens, prefix dropped
        self.assertEqual(f["n_clusters"], 3)                   # days 0,0,1,3 -> clusters at 0,1,3
        self.assertEqual(f["n_zero_adj"], 1)                   # the 0->0 adjacency
        self.assertEqual(f["n_adj"], 3)
        self.assertEqual(f["n_pos_gaps"], 2)                   # 0->1, 1->3
        self.assertEqual(list(f["per_class"]), [2, 1, 1, 0, 0])
        self.assertAlmostEqual(f["occupancy"], 3 / ORACLE_ENV_N_CLASSES)

    def test_aggregate_is_not_evaluable_below_floor(self) -> None:
        seqs = [_seq([0, 1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0, 4.0]) for _ in range(3)]
        self.assertEqual(X.aggregate_from_sequences("SCID", seqs), NOT_EVALUABLE)   # tiny -> not evaluable

    def test_aggregate_well_formed_above_floor(self) -> None:
        rng = np.random.default_rng(0)
        n = ORACLE_ENV_MIN_DENOM + 200
        seqs = []
        for _ in range(n):
            L = int(rng.integers(6, 14))
            classes = list(rng.integers(0, ORACLE_ENV_N_CLASSES, size=L))
            # strictly increasing days with occasional repeats (zero-gap clusters)
            days = np.cumsum(rng.integers(0, 3, size=L)).astype(float)
            seqs.append(_seq(classes, list(days)))
        agg = X.aggregate_from_sequences("MIMIC", seqs)
        self.assertIsInstance(agg, AggregateStats)
        self.assertEqual(agg.source, "MIMIC")
        self.assertEqual(len(agg.class_counts), ORACLE_ENV_N_CLASSES)
        self.assertEqual(agg.n_events, sum(agg.class_counts))
        self.assertGreaterEqual(agg.n_sequences, ORACLE_ENV_MIN_DENOM)
        self.assertTrue(0.0 <= agg.delta_t_zero_fraction <= 1.0)
        self.assertTrue(0.0 <= agg.mean_occupancy_fraction <= 1.0)
        # ECDF support is ascending-unique with final mass 1
        for ecdf in (agg.length_ecdf, agg.count_ecdf):
            supp = [s for s, _ in ecdf]
            self.assertEqual(supp, sorted(set(supp)))
            self.assertAlmostEqual(ecdf[-1][1], 1.0)


class FailClosedGuardTests(unittest.TestCase):
    def test_read_refused_without_approval_token(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.extract_source("SCID", "/approved/joint_flat_corrected_v1/scid_train.h5", "train",
                             approval_token="nope")

    def test_read_refused_on_non_train_split(self) -> None:
        with self.assertRaises(X.ExtractionRefused):
            X.extract_source("SCID", "/approved/joint_flat_corrected_v1/scid_val.h5", "dev",
                             approval_token=X.MICRO_GATE_APPROVAL_TOKEN)

    def test_read_refused_on_test_or_sealed_path(self) -> None:
        for p in ("/approved/joint_flat_corrected_v1/scid_test.h5",
                  "/approved/sealed/scid_train.h5"):
            with self.assertRaises(X.ExtractionRefused):
                X.extract_source("SCID", p, "train", approval_token=X.MICRO_GATE_APPROVAL_TOKEN)

    def test_sanitized_output_is_clean_and_scanner_is_wired(self) -> None:
        # normal aggregate output passes the forbidden-key scan (AggregateStats has only safe field names)
        good = X.sanitized_output({"SCID": NOT_EVALUABLE})
        self.assertEqual(good["sources"]["SCID"], NOT_EVALUABLE)
        self.assertEqual(good["governance_class"],
                         "explicitly_cleared_safe_aggregate_only_no_patient_rows")
        from dataclasses import fields
        from clinical_jepa.validation import FORBIDDEN_AGGREGATE_KEYS
        self.assertFalse({f.name for f in fields(AggregateStats)} & FORBIDDEN_AGGREGATE_KEYS)
        # and the scanner the sanitizer relies on actually flags a row-level key
        from clinical_jepa.validation import _scan_forbidden_aggregate_keys
        self.assertTrue(_scan_forbidden_aggregate_keys({"sources": {"SCID": {"token_ids": [1, 2, 3]}}}))


if __name__ == "__main__":
    unittest.main()
