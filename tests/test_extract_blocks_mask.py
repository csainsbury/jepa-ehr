from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import (  # noqa: E402
    build_source_h5s,
    joint_arms_config,
    joint_dataset_config,
    write_tiny_vocab,
    write_yaml,
)

from clinical_jepa.splits.build_index import main as build_index_main  # noqa: E402
from clinical_jepa.targets.extract_blocks import _real_blocks, _t0_block, t0_feasible  # noqa: E402


class MaskAndFeasibilityTests(unittest.TestCase):
    def test_t0_block_starts_after_source_prefix(self) -> None:
        block = _t0_block("seq", "dev", seq_len=220, source_dataset="SCID", ordinal=0, target_window=32, context_start=2, min_context=8)
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["context_start_ref"], 2)
        # No span endpoint touches the masked prefix positions 0/1.
        self.assertGreaterEqual(block["context_start_ref"], 2)

    def test_per_source_window_sizing_saves_short_mimic(self) -> None:
        # Short per-admission MIMIC (len 20) vanishes under the SCID window (32)
        # but survives under the MIMIC window (8).
        self.assertFalse(t0_feasible(20, 32, context_start=2, min_context=8))
        self.assertTrue(t0_feasible(20, 8, context_start=2, min_context=4))


class RealExtractionTests(unittest.TestCase):
    def _run(self, td: Path):
        paths = build_source_h5s(
            td, scid_lens=[220, 220, 220], mimic_lens=[20, 20, 20],
            scid_outcome_at={0: [10]},  # sequence 0 has an outcome inside context
        )
        vocab = write_tiny_vocab(td / "vocab.json")
        idx = td / "idx"
        cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx)
        dataset_cfg_path = write_yaml(td / "dataset.yaml", cfg)
        arms_cfg_path = write_yaml(td / "arms.yaml", joint_arms_config())
        build_index_main(["--dataset-config", dataset_cfg_path, "--output-dir", str(idx)])

        split_manifest = {
            "dataset": "joint-test",
            "source_index_paths": {
                "train": str(idx / "train.index.jsonl"),
                "dev": str(idx / "dev.index.jsonl"),
                "test": str(idx / "test.index.jsonl"),
            },
            "source_h5_paths": {},
        }
        args = argparse.Namespace(
            arms_config=arms_cfg_path,
            targets=["T0"],
            t0_gap_events=0,
            endpoint_proximal_margin=0,
            max_real_train=0,
            max_real_dev=0,
            max_real_test=0,
            t1_anchors_per_sequence=1,
            source_role="primary",
        )
        return _real_blocks(args, cfg, split_manifest)

    def test_mask_source_refusal_and_per_source_yield(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            blocks, details = self._run(Path(td))
            report = details["report"]

            # Every emitted block masks the DATASET/BOS prefix.
            self.assertTrue(blocks)
            for b in blocks:
                self.assertEqual(b["context_start_ref"], 2)
                self.assertTrue(b["endpoint_safe"])
                self.assertEqual(b["context_outcome_positions"], 0)

            # Both sources yield blocks (MIMIC survives via per-source window).
            self.assertIn("SCID", report["blocks_by_source"])
            self.assertIn("MIMIC", report["blocks_by_source"])
            self.assertGreater(report["blocks_by_source"]["MIMIC"], 0)

            # The one SCID sequence with an outcome in context is refused per split.
            self.assertGreaterEqual(sum(report["refused_outcome_in_context"].values()), 3)
            self.assertEqual(report["source_prefix_len"], 2)


if __name__ == "__main__":
    unittest.main()
