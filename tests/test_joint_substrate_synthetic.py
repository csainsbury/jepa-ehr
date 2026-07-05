from __future__ import annotations

import json
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

from clinical_jepa.audit.run_leakage_audit import main as audit_main  # noqa: E402
from clinical_jepa.splits.build_index import main as build_index_main  # noqa: E402
from clinical_jepa.splits.make_manifest import main as make_manifest_main  # noqa: E402
from clinical_jepa.splits.readiness_manifest import main as readiness_main  # noqa: E402
from clinical_jepa.targets.extract_blocks import main as extract_main  # noqa: E402


class JointSubstrateEndToEndTests(unittest.TestCase):
    def test_index_to_audit_to_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = build_source_h5s(
                td,
                scid_lens=[220, 220, 220, 220],
                mimic_lens=[20, 20, 20, 20],
                scid_outcome_at={0: [10]},  # one outcome-in-context sequence per split
                med_at=[100],               # SCID T1 anchors (out of MIMIC range)
            )
            vocab = write_tiny_vocab(td / "vocab.json")
            idx = td / "idx"
            cfg = joint_dataset_config(td, paths, vocab_path=vocab, index_dir=idx, min_valid_windows=2)
            dataset_cfg = write_yaml(td / "dataset.yaml", cfg)
            arms_cfg = write_yaml(td / "arms.yaml", joint_arms_config())

            # 1. Build the per-split JSONL index (carries source_dataset).
            build_index_main(["--dataset-config", dataset_cfg, "--output-dir", str(idx)])

            # 2. Real split manifest (multi-source: h5 paths optional).
            splits_dir = td / "splits"
            rc = make_manifest_main(["--dataset-config", dataset_cfg, "--output-dir", str(splits_dir)])
            self.assertEqual(rc, 0)
            manifest = json.loads((splits_dir / "split-manifest.json").read_text())
            self.assertEqual(manifest["source_datasets"], ["SCID", "MIMIC"])

            # 3. Extract real target blocks (mask + per-source windows + refusal).
            tb_dir = td / "target-blocks"
            extract_main([
                "--dataset-config", dataset_cfg,
                "--split-manifest", str(splits_dir / "split-manifest.json"),
                "--arms-config", arms_cfg,
                "--output-dir", str(tb_dir),
                "--targets", "T0", "T1",
            ])
            tmanifest = json.loads((tb_dir / "target-block-manifest.json").read_text())
            self.assertTrue(tmanifest["blocks"])
            report = tmanifest["extraction_report"]
            self.assertIn("SCID", report["blocks_by_source"])
            self.assertIn("MIMIC", report["blocks_by_source"])
            self.assertGreaterEqual(sum(report["refused_outcome_in_context"].values()), 1)
            for b in tmanifest["blocks"]:
                self.assertEqual(b["context_start_ref"], 2)

            # 4. Real leakage audit passes on the emitted (clean) blocks.
            audit_out = td / "leakage.json"
            rc = audit_main([
                "--dataset-config", dataset_cfg,
                "--split-manifest", str(splits_dir / "split-manifest.json"),
                "--target-blocks", str(tb_dir / "target-block-manifest.json"),
                "--output", str(audit_out),
            ])
            self.assertEqual(rc, 0)
            audit = json.loads(audit_out.read_text())
            self.assertEqual(audit["overall_status"], "pass")
            self.assertEqual(audit["audits"]["label_feature_separation"]["status"], "pass")
            self.assertEqual(audit["audits"]["forbidden_tokens"]["status"], "pass")

            # 5. Rung -1 readiness gate passes (both sources adequate).
            readiness_out = td / "readiness"
            rc = readiness_main([
                "--dataset-config", dataset_cfg,
                "--arms-config", arms_cfg,
                "--index-dir", str(idx),
                "--output-dir", str(readiness_out),
            ])
            self.assertEqual(rc, 0)
            readiness = json.loads((readiness_out / "rung-minus1-readiness-manifest.json").read_text())
            self.assertEqual(readiness["gate_status"], "pass")


if __name__ == "__main__":
    unittest.main()
