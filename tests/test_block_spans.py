from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_fixtures import write_yaml  # noqa: E402

from clinical_jepa.targets.block_spans import (  # noqa: E402
    EMPTY_TARGET_REF,
    empty_target_len,
    is_empty_target,
    read_target_span,
    target_occupancy,
)
from clinical_jepa.targets.extract_blocks import EMPTY_TARGET_REF as EXTRACT_SENTINEL  # noqa: E402
from clinical_jepa.audit.run_leakage_audit import main as audit_main  # noqa: E402

POP = {"context_start_ref": 2, "context_end_ref": 10, "target_start_ref": 11, "target_end_ref": 20}
EMPTY = {"context_start_ref": 2, "context_end_ref": 10, "target_start_ref": -1, "target_end_ref": -1,
         "empty_target": True, "n_target_events": 0}


class BlockSpanHelperTests(unittest.TestCase):
    def test_sentinel_in_sync_with_extractor(self) -> None:
        self.assertEqual(EMPTY_TARGET_REF, EXTRACT_SENTINEL)
        self.assertEqual(EMPTY_TARGET_REF, -1)

    def test_is_empty_target(self) -> None:
        self.assertTrue(is_empty_target(EMPTY))
        self.assertTrue(is_empty_target({"target_start_ref": -1, "target_end_ref": -1}))  # refs only
        self.assertFalse(is_empty_target(POP))
        self.assertFalse(is_empty_target({"target_start_ref": 0, "target_end_ref": 5}))

    def test_read_target_span_never_reads_zero_for_empty(self) -> None:
        arr = np.arange(30, dtype=np.int64)
        ids, empty = read_target_span(EMPTY, arr)
        self.assertTrue(empty)
        self.assertEqual(len(ids), 0)  # NOT arr[0:...]
        ids2, empty2 = read_target_span(POP, arr)
        self.assertFalse(empty2)
        self.assertEqual(ids2.tolist(), list(range(11, 21)))

    def test_read_target_span_out_of_range_is_not_empty(self) -> None:
        arr = np.arange(5, dtype=np.int64)  # shorter than POP refs
        ids, empty = read_target_span(POP, arr)
        self.assertFalse(empty)  # populated-but-unreadable, not silence
        self.assertEqual(len(ids), 0)

    def test_empty_target_len(self) -> None:
        self.assertEqual(empty_target_len(EMPTY), 0)  # not the buggy -1-(-1)+1 = 1
        self.assertEqual(empty_target_len(POP), 10)

    def test_target_occupancy(self) -> None:
        self.assertEqual(target_occupancy(EMPTY), (0, 0))
        self.assertEqual(target_occupancy({**POP, "n_target_events": 7}), (1, 7))
        self.assertEqual(target_occupancy(POP), (1, 10))  # falls back to span length


class LeakageSite1RegressionTests(unittest.TestCase):
    """Empty wall-clock targets (target_start_ref=-1) must NOT trip the
    horizon_boundary check (`context_end_ref >= -1` is always True)."""

    def _run(self, blocks):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ds = write_yaml(td / "dataset.yaml", {
                "schema_version": "clinical-jepa-dataset-config-v0",
                "mask": {"source_prefix_len": 0},
                "leakage": {},
            })
            sm = td / "split.json"
            sm.write_text(json.dumps({"dataset": "synthetic"}))
            tb = td / "target-blocks.json"
            tb.write_text(json.dumps({"targets": ["T0"], "blocks": blocks}))
            out = td / "audit.json"
            audit_main(["--dataset-config", ds, "--split-manifest", str(sm),
                        "--target-blocks", str(tb), "--output", str(out)])
            return json.loads(out.read_text())

    def test_mixed_empty_and_populated_passes(self) -> None:
        blocks = []
        for i in range(4):
            blocks.append({"block_id": f"p{i}", "split": "train", "target_type": "T0",
                           "source_dataset": "SCID", **POP})
        for i in range(3):
            blocks.append({"block_id": f"e{i}", "split": "dev", "target_type": "T0",
                           "source_dataset": "SCID", **EMPTY})
        rep = self._run(blocks)
        self.assertEqual(rep["audits"]["horizon_boundary"]["status"], "pass")  # no false violation
        self.assertEqual(rep["overall_status"], "pass")
        self.assertEqual(rep["empty_target_audited"], 3)

    def test_all_empty_still_passes(self) -> None:
        blocks = [{"block_id": f"e{i}", "split": "test", "target_type": "T0",
                   "source_dataset": "MIMIC", **EMPTY} for i in range(5)]
        rep = self._run(blocks)
        self.assertEqual(rep["audits"]["horizon_boundary"]["status"], "pass")
        self.assertEqual(rep["empty_target_audited"], 5)


if __name__ == "__main__":
    unittest.main()
