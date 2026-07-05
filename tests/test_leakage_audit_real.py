from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import SCID_TOKEN, make_sequence, write_h5, write_yaml  # noqa: E402

from clinical_jepa.audit.run_leakage_audit import main as audit_main  # noqa: E402


def _block(block_id: str, c0: int, c1: int, t0: int, t1: int, h5_path: str, group: str) -> dict:
    return {
        "block_id": block_id,
        "patient_hash": "h" + block_id,
        "sequence_id": group,
        "sequence_group": group,
        "sequence_file": h5_path,
        "split": "train",
        "target_type": "T0",
        "context_start_ref": c0,
        "context_end_ref": c1,
        "target_start_ref": t0,
        "target_end_ref": t1,
        "horizon_descriptor": "event_gap_0_window_8",
        "source_dataset": "SCID",
    }


def _write_manifest(path: Path, blocks: list[dict]) -> None:
    path.write_text(json.dumps({
        "schema_version": "clinical-jepa-target-block-manifest-v0",
        "created_utc": "2026-07-05T00:00:00Z",
        "targets": ["T0"],
        "counts": {"train": {"T0": len(blocks)}},
        "blocks": blocks,
    }))


class LeakageAuditRealTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # dataset config wiring the mask + outcome channel.
        self.cfg = write_yaml(self.root / "dataset.yaml", {
            "mask": {"source_prefix_len": 2},
            "leakage": {"outcome_label_dataset": "is_outcome_label"},
        })
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps({"dataset": "joint-test"}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_audit(self, blocks: list[dict]) -> tuple[int, dict]:
        manifest = self.root / "blocks.json"
        _write_manifest(manifest, blocks)
        out = self.root / "audit.json"
        rc = audit_main([
            "--dataset-config", self.cfg,
            "--split-manifest", str(self.split),
            "--target-blocks", str(manifest),
            "--output", str(out),
        ])
        return rc, json.loads(out.read_text())

    def test_clean_block_passes(self) -> None:
        h5 = write_h5(self.root / "seq.h5", {"g": make_sequence(SCID_TOKEN, 40)})  # no outcomes
        rc, report = self._run_audit([_block("clean", 2, 10, 11, 18, h5, "g")])
        self.assertEqual(rc, 0)
        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(report["audits"]["label_feature_separation"]["status"], "pass")
        self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "pass")

    def test_planted_is_outcome_leak_is_caught(self) -> None:
        # is_outcome==1 at position 5, inside the context span [2, 10].
        h5 = write_h5(self.root / "seq.h5", {"g": make_sequence(SCID_TOKEN, 40, outcome_at=[5])})
        rc, report = self._run_audit([_block("leaky", 2, 10, 11, 18, h5, "g")])
        self.assertEqual(rc, 2)
        self.assertEqual(report["overall_status"], "fail")
        self.assertEqual(report["audits"]["label_feature_separation"]["status"], "fail")
        self.assertGreaterEqual(report["audits"]["label_feature_separation"]["violations"], 1)
        self.assertGreaterEqual(report["outcome_label_separation"]["leaked_positions"], 1)

    def test_source_prefix_shortcut_is_caught(self) -> None:
        # Context begins at index 0 (includes the DATASET source token) while the
        # config masks source_prefix_len=2 -> source-shortcut violation.
        h5 = write_h5(self.root / "seq.h5", {"g": make_sequence(SCID_TOKEN, 40)})
        rc, report = self._run_audit([_block("shortcut", 0, 10, 11, 18, h5, "g")])
        self.assertEqual(rc, 2)
        self.assertEqual(report["overall_status"], "fail")
        self.assertEqual(report["audits"]["forbidden_tokens"]["status"], "fail")
        self.assertGreaterEqual(report["source_shortcut"]["violating_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
