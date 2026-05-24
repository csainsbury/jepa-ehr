from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval.pseudo_rendering import compute_pseudo_rendering_report

ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    rows = []
    for i in range(4):
        rows.append({
            "block_id": f"block-secret-{i}",
            "patient_hash": f"patient-secret-{i}",
            "split": "dev",
            "target_type": "T0",
            "context_len": 16,
            "target_len": 16,
            "sequence_len": 64,
            "context_med_count": 1,
            "context_lab_count": 2,
            "context_state_count": 0,
            "target_med_count": 1 if i % 2 == 0 else 0,
            "target_lab_count": 1,
            "target_state_count": 0,
            "scenario_consistent": i != 3,
            "negative_control_present": i == 3,
        })
    return rows


class PseudoRenderingTests(unittest.TestCase):
    def test_report_is_aggregate_only(self) -> None:
        target = np.eye(4, dtype=np.float32)
        query = target.copy()
        rows = _rows()
        report = compute_pseudo_rendering_report(query, rows, target, rows, top_k=2, distractor_policy="same_split_target_type")
        self.assertEqual(report["aggregate_only"], True)
        self.assertEqual(report["queries_evaluated"], 4)
        self.assertEqual(report["retrieval"]["recall_at_1"], 1.0)
        self.assertIsNotNone(report["spec_consistency_rate"])
        dumped = json.dumps(report)
        self.assertNotIn("block-secret", dumped)
        self.assertNotIn("patient-secret", dumped)

    def test_warns_when_same_patient_key_missing(self) -> None:
        target = np.eye(4, dtype=np.float32)
        rows = [{k: v for k, v in r.items() if k != "patient_hash"} for r in _rows()]
        report = compute_pseudo_rendering_report(target, rows, target, rows, top_k=2, distractor_policy="same_split_target_type")
        self.assertFalse(report["same_patient_exclusion_applied"])
        self.assertIn("same_patient_exclusion_key_missing", report["warnings"])

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            q = root / "q.npy"
            t = root / "t.npy"
            qi = root / "q.jsonl"
            ti = root / "t.jsonl"
            out = root / "out"
            arr = np.eye(4, dtype=np.float32)
            np.save(q, arr)
            np.save(t, arr)
            rows = _rows()
            qi.write_text("".join(json.dumps(r) + "\n" for r in rows))
            ti.write_text("".join(json.dumps(r) + "\n" for r in rows))
            subprocess.run([
                sys.executable,
                "-m",
                "clinical_jepa.eval.pseudo_rendering",
                "--query-embeddings",
                str(q),
                "--query-index",
                str(qi),
                "--target-embeddings",
                str(t),
                "--target-index",
                str(ti),
                "--output-dir",
                str(out),
                "--top-k",
                "2",
                "--distractor-policy",
                "same_split_target_type",
            ], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report = json.loads((out / "pseudo-rendering-readiness.json").read_text())
            self.assertEqual(report["aggregate_only"], True)
            self.assertTrue((out / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
