from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval.autoregression_readiness import compute_autoregression_readiness_report

ROOT = Path(__file__).resolve().parents[1]


def _rows(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "block_id": f"block-secret-{i}",
            "patient_hash": f"patient-secret-{i}",
            "split": "dev",
            "target_type": "T1",
            "context_len": 16,
            "target_len": 8,
            "sequence_len": 64,
            "context_med_count": 1,
            "context_lab_count": 2,
            "context_state_count": 0,
        })
    return rows


class AutoregressionReadinessTests(unittest.TestCase):
    def test_multihorizon_report_is_aggregate_only(self) -> None:
        n, h, d = 6, 3, 6
        target = np.zeros((n, h, d), dtype=np.float32)
        for step in range(h):
            target[:, step, :] = np.roll(np.eye(d, dtype=np.float32), shift=step, axis=1)[:n]
        pred = target + 0.001
        report = compute_autoregression_readiness_report(pred, target, _rows(n), distractor_policy="same_split_target_type")
        self.assertEqual(report["aggregate_only"], True)
        self.assertEqual(report["n_horizons"], 3)
        self.assertEqual(report["per_horizon"][0]["retrieval"]["recall_at_1"], 1.0)
        self.assertTrue(report["transition_dynamics"])
        dumped = json.dumps(report)
        self.assertNotIn("block-secret", dumped)
        self.assertNotIn("patient-secret", dumped)

    def test_single_horizon_warns_no_transition_check(self) -> None:
        arr = np.eye(4, dtype=np.float32)
        report = compute_autoregression_readiness_report(arr, arr, _rows(4), distractor_policy="same_split_target_type")
        self.assertIn("single_horizon_no_autoregressive_transition_check", report["warnings"])

    def test_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred = root / "pred.npy"
            target = root / "target.npy"
            index = root / "index.jsonl"
            out = root / "out"
            arr = np.stack([np.eye(4, dtype=np.float32), np.roll(np.eye(4, dtype=np.float32), shift=1, axis=1)], axis=1)
            np.save(pred, arr)
            np.save(target, arr)
            index.write_text("".join(json.dumps(r) + "\n" for r in _rows(4)))
            subprocess.run([
                sys.executable,
                "-m",
                "clinical_jepa.eval.autoregression_readiness",
                "--predicted-rollout",
                str(pred),
                "--target-rollout",
                str(target),
                "--index",
                str(index),
                "--output-dir",
                str(out),
                "--distractor-policy",
                "same_split_target_type",
            ], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report = json.loads((out / "autoregression-readiness.json").read_text())
            self.assertEqual(report["aggregate_only"], True)
            self.assertTrue((out / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
