"""Retrieval occupancy-split tests (Pi R4 Q1): empty and populated targets are
never each other's distractors, and empty/non-empty R@k is reported separately so
strong empty-class retrieval cannot dominate the overall / horizon-decay claim."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval.retrieval import compute_retrieval_metrics, group_key, occupancy_class


def _row(i: int) -> dict:
    empty = (i % 2 == 0)
    return {
        "block_id": f"b{i}",
        "split": "dev",
        "target_type": "T0",
        "source_dataset": "SCID" if i % 4 < 2 else "MIMIC",
        "empty_target": empty,
        "n_target_events": 0 if empty else 5,
        "target_len": 0 if empty else 5,
        "context_len": 10,
    }


class OccupancyClassTests(unittest.TestCase):
    def test_occupancy_class(self) -> None:
        self.assertEqual(occupancy_class({"empty_target": True}), "empty")
        self.assertEqual(occupancy_class({"n_target_events": 0}), "empty")
        self.assertEqual(occupancy_class({"target_len": 0}), "empty")
        self.assertEqual(occupancy_class({"n_target_events": 3}), "populated")
        self.assertEqual(occupancy_class({"target_len": 5}), "populated")
        self.assertEqual(occupancy_class({}), "populated")  # event-index default

    def test_occ_policy_separates_empty_from_populated(self) -> None:
        empty = {"source_dataset": "SCID", "split": "dev", "target_type": "T0",
                 "empty_target": True, "target_len": 0, "context_len": 10}
        pop = {**empty, "empty_target": False, "target_len": 5}
        # Same source/split/type but different occupancy -> different candidate group.
        for policy in ("same_source_split_target_type_occ", "same_source_split_target_type_len_occ_bin"):
            self.assertNotEqual(group_key(empty, policy), group_key(pop, policy), policy)


class OccupancySplitMetricsTests(unittest.TestCase):
    def test_by_occupancy_reported_separately(self) -> None:
        n = 24
        rng = np.random.default_rng(0)
        # Distinct target per block; query = target + small noise -> retrievable.
        target = rng.normal(size=(n, 8)).astype(np.float32)
        query = target + 0.01 * rng.normal(size=(n, 8)).astype(np.float32)
        index = [_row(i) for i in range(n)]

        report = compute_retrieval_metrics(
            query, index, target, index,
            distractor_policy="same_source_split_target_type_occ",
            min_candidates_per_group=2,
        )
        self.assertIn("by_occupancy", report)
        emp = report["by_occupancy"]["empty"]
        pop = report["by_occupancy"]["populated"]
        self.assertGreater(emp["n"], 0)
        self.assertGreater(pop["n"], 0)
        # Separated counts partition the overall evaluated queries.
        self.assertEqual(emp["n"] + pop["n"], report["overall"]["n"])
        # Retrieval works within each class (query ~= its own target).
        self.assertGreaterEqual(emp["recall_at_1"], 0.9)
        self.assertGreaterEqual(pop["recall_at_1"], 0.9)


if __name__ == "__main__":
    unittest.main()
