from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "configs/v0/dataset.example.yaml"
ARMS = ROOT / "configs/v0/arms.example.yaml"
METRICS = ROOT / "configs/v0/metrics.example.yaml"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)


class SyntheticPipelineTests(unittest.TestCase):
    def test_split_target_audit_metrics_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            splits = root / "splits"
            targets = root / "target-blocks"
            leakage = root / "leakage.json"
            results = root / "results"
            run([sys.executable, "-m", "clinical_jepa.splits.make_manifest", "--dataset-config", str(DATASET), "--output-dir", str(splits), "--dry-run", "--synthetic-patients", "24"])
            manifest = json.loads((splits / "split-manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "clinical-jepa-split-manifest-v0")
            self.assertEqual(sum(manifest["counts"].values()), 24)
            run([sys.executable, "-m", "clinical_jepa.targets.extract_blocks", "--dataset-config", str(DATASET), "--split-manifest", str(splits / "split-manifest.json"), "--arms-config", str(ARMS), "--output-dir", str(targets), "--targets", "T0", "T1", "--dry-run"])
            tmanifest = json.loads((targets / "target-block-manifest.json").read_text())
            self.assertTrue(tmanifest["blocks"])
            for block in tmanifest["blocks"]:
                self.assertLess(block["context_end_ref"], block["target_start_ref"])
            run([sys.executable, "-m", "clinical_jepa.audit.run_leakage_audit", "--dataset-config", str(DATASET), "--split-manifest", str(splits / "split-manifest.json"), "--target-blocks", str(targets / "target-block-manifest.json"), "--output", str(leakage)])
            audit = json.loads(leakage.read_text())
            self.assertEqual(audit["overall_status"], "pass")
            run([sys.executable, "-m", "clinical_jepa.eval.run_metrics", "--metrics-config", str(METRICS), "--split-manifest", str(splits / "split-manifest.json"), "--target-blocks", str(targets / "target-block-manifest.json"), "--leakage-report", str(leakage), "--output-dir", str(results), "--dry-run"])
            self.assertTrue((results / "baseline-results.json").exists())

    def test_audit_fails_on_boundary_violation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split = root / "split.json"
            target = root / "target.json"
            out = root / "audit.json"
            split.write_text(json.dumps({"dataset": "synthetic", "counts": {"patients_train": 1, "patients_dev": 0, "patients_test": 0}}))
            target.write_text(json.dumps({"targets": ["T0"], "blocks": [{"block_id": "b1", "split": "train", "context_end_ref": 10, "target_start_ref": 10}]}))
            proc = subprocess.run([sys.executable, "-m", "clinical_jepa.audit.run_leakage_audit", "--dataset-config", str(DATASET), "--split-manifest", str(split), "--target-blocks", str(target), "--output", str(out)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(json.loads(out.read_text())["overall_status"], "fail")


if __name__ == "__main__":
    unittest.main()
