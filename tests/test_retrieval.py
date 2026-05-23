from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval.retrieval import compute_retrieval_metrics
from clinical_jepa.targets.extract_blocks import _t0_block


class RetrievalMetricTests(unittest.TestCase):
    def test_perfect_same_split_target_retrieval(self) -> None:
        target = np.eye(4, dtype=np.float32)
        query = target.copy()
        index = [
            {"block_id": f"b{i}", "split": "dev", "target_type": "T0"}
            for i in range(4)
        ]
        report = compute_retrieval_metrics(query, index, target, index)
        self.assertEqual(report["overall"]["n"], 4)
        self.assertEqual(report["overall"]["recall_at_1"], 1.0)
        self.assertEqual(report["overall"]["mrr"], 1.0)

    def test_groups_do_not_cross_split(self) -> None:
        target = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        query = target.copy()
        index = [
            {"block_id": "train-a", "split": "train", "target_type": "T0"},
            {"block_id": "train-b", "split": "train", "target_type": "T0"},
            {"block_id": "dev-a", "split": "dev", "target_type": "T0"},
            {"block_id": "dev-b", "split": "dev", "target_type": "T0"},
        ]
        report = compute_retrieval_metrics(query, index, target, index)
        self.assertEqual(set(report["groups"]), {"train|T0", "dev|T0"})
        self.assertEqual(report["overall"]["n"], 4)



class TargetBlockGapTests(unittest.TestCase):
    def test_t0_gap_events_separates_context_and_target(self) -> None:
        block = _t0_block("seq", "dev", seq_len=100, source_dataset="synthetic", ordinal=0, target_window=16, gap_events=7)
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["target_start_ref"] - block["context_end_ref"] - 1, 7)
        self.assertEqual(block["gap_events"], 7)


if __name__ == "__main__":
    unittest.main()
