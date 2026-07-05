from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval.retrieval import compute_retrieval_metrics, group_key


class SourceAwareGroupingTests(unittest.TestCase):
    def test_group_key_includes_source(self) -> None:
        row = {"source_dataset": "MIMIC", "split": "dev", "target_type": "T0"}
        self.assertEqual(group_key(row, "same_source_split_target_type"), ("MIMIC", "dev", "T0"))

    def test_scid_and_mimic_never_share_a_group(self) -> None:
        # 2 SCID + 2 MIMIC, all identical embeddings so only grouping prevents
        # cross-source distractors. Under the source-aware policy each source is
        # its own candidate pool.
        target = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        query = target.copy()
        index = [
            {"block_id": "s0", "source_dataset": "SCID", "split": "dev", "target_type": "T0"},
            {"block_id": "s1", "source_dataset": "SCID", "split": "dev", "target_type": "T0"},
            {"block_id": "m0", "source_dataset": "MIMIC", "split": "dev", "target_type": "T0"},
            {"block_id": "m1", "source_dataset": "MIMIC", "split": "dev", "target_type": "T0"},
        ]
        report = compute_retrieval_metrics(query, index, target, index, distractor_policy="same_source_split_target_type")
        self.assertEqual(set(report["groups"]), {"SCID|dev|T0", "MIMIC|dev|T0"})

    def test_source_aware_policy_registered(self) -> None:
        from clinical_jepa.eval.retrieval import POLICIES

        for policy in [
            "same_source_split",
            "same_source_split_target_type",
            "same_source_split_target_type_len_bin",
            "same_source_split_target_type_len_seq_util_bin",
        ]:
            self.assertIn(policy, POLICIES)

    def test_legacy_policies_unchanged(self) -> None:
        # Legacy (source-agnostic) policy must still ignore source_dataset.
        row = {"source_dataset": "MIMIC", "split": "dev", "target_type": "T0"}
        self.assertEqual(group_key(row, "same_split_target_type"), ("dev", "T0"))


if __name__ == "__main__":
    unittest.main()
