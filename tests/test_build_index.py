from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import (  # noqa: E402
    MIMIC_TOKEN,
    SCID_TOKEN,
    build_source_h5s,
    joint_dataset_config,
    make_sequence,
    write_h5,
    write_tiny_vocab,
)

from clinical_jepa.splits.build_index import build_index_for_split  # noqa: E402


class BuildIndexTests(unittest.TestCase):
    def test_index_carries_per_sequence_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = build_source_h5s(root, scid_lens=[220, 200], mimic_lens=[20, 18], splits=("train",))
            vocab = write_tiny_vocab(root / "vocab.json")
            cfg = joint_dataset_config(root, paths, vocab_path=vocab, index_dir=root / "idx")
            rows, report = build_index_for_split(cfg, "train")

            self.assertEqual(report["per_source_counts"], {"SCID": 2, "MIMIC": 2})
            by_source = {r["source_dataset"] for r in rows}
            self.assertEqual(by_source, {"SCID", "MIMIC"})
            for r in rows:
                self.assertIn("seq_len", r)
                self.assertIn("source_h5_path", r)
                self.assertIn("group", r)
                self.assertIn("wall_clock_span_days", r)  # cumulative_days channel present
            scid_lens = sorted(r["seq_len"] for r in rows if r["source_dataset"] == "SCID")
            self.assertEqual(scid_lens, [200, 220])

    def test_token_mismatch_counted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # A "SCID" file whose one sequence actually starts with the MIMIC token.
            bad = make_sequence(MIMIC_TOKEN, 50)
            scid_path = write_h5(root / "scid_train.h5", {"scid_train_0": bad})
            mimic_path = write_h5(root / "mimic_train.h5", {"mimic_train_0": make_sequence(MIMIC_TOKEN, 20)})
            paths = {"scid": {"train": scid_path}, "mimic": {"train": mimic_path}}
            vocab = write_tiny_vocab(root / "vocab.json")
            cfg = joint_dataset_config(root, paths, vocab_path=vocab, index_dir=root / "idx")
            _rows, report = build_index_for_split(cfg, "train")
            self.assertEqual(report["per_source_token_mismatch"].get("SCID"), 1)

    def test_merged_file_source_from_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            merged = write_h5(
                root / "merged_train.h5",
                {
                    "scid_1": make_sequence(SCID_TOKEN, 100),
                    "mimic_1": make_sequence(MIMIC_TOKEN, 20),
                    "mimic_2": make_sequence(MIMIC_TOKEN, 22),
                },
            )
            cfg = {
                "time_channel": "cumulative_days",
                "sources": {
                    "primary": {
                        "source_datasets": [
                            {"kind": "merged", "group_prefixes": {"SCID": "scid_", "MIMIC": "mimic_"}, "h5_paths": {"train": merged}},
                        ]
                    }
                },
            }
            rows, report = build_index_for_split(cfg, "train")
            self.assertEqual(report["per_source_counts"], {"SCID": 1, "MIMIC": 2})


if __name__ == "__main__":
    unittest.main()
