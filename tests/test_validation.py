from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.validation import validate_artifact

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def test_split_manifest_validation_passes_and_fails(self) -> None:
        valid = {
            "schema_version": "clinical-jepa-split-manifest-v0",
            "created_utc": "2026-05-23T00:00:00Z",
            "dataset": "synthetic",
            "split_policy": "patient-level-before-windowing",
            "seed": 20260523,
            "hash_method": "salted-hmac-sha256-local-salt-not-exported",
            "salt_exported": False,
            "counts": {"patients_train": 1, "patients_dev": 1, "patients_test": 1},
            "locked_stress_tests": ["eICU-CRD"],
        }
        self.assertEqual(validate_artifact("split-manifest", valid, raise_on_error=False), [])
        invalid = dict(valid)
        invalid.pop("counts")
        self.assertTrue(validate_artifact("split-manifest", invalid, raise_on_error=False))

    def test_aggregate_only_artifacts_reject_identifier_keys(self) -> None:
        valid = {
            "schema_version": "clinical-jepa-scenario-feasibility-v0",
            "created_utc": "2026-05-24T00:00:00Z",
            "scenario_id": "s",
            "aggregate_only": True,
            "overall_decision": "redesign",
            "results": [],
        }
        self.assertEqual(validate_artifact("scenario-feasibility", valid, raise_on_error=False), [])
        invalid = dict(valid)
        invalid["patient_hashes"] = ["not-allowed-even-if-hashed"]
        errors = validate_artifact("scenario-feasibility", invalid, raise_on_error=False)
        self.assertTrue(any("forbidden aggregate-only key" in e for e in errors))

    def test_cli_auto_validation(self) -> None:
        data = {
            "schema_version": "clinical-jepa-query-descriptors-v0",
            "created_utc": "2026-05-23T00:00:00Z",
            "queries": [{"query_id": "q", "target_type": "T0", "horizon": "short", "label_source": "target", "task_type": "binary"}],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "queries.json"
            p.write_text(json.dumps(data))
            proc = subprocess.run([sys.executable, "-m", "clinical_jepa.validation", "--file", str(p)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(proc.returncode, 0, proc.stdout)


if __name__ == "__main__":
    unittest.main()
