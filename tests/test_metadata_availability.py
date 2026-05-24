from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.tte.audit_metadata_availability import audit_metadata_availability, main

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "configs/v0/metadata_availability.example.yaml"
SCENARIO = ROOT / "configs/v0/scenario.example.yaml"


def _blocks() -> list[dict]:
    rows = []
    for i in range(4):
        rows.append({
            "block_id": f"join-{i}",
            "patient_hash": f"not-emitted-{i}",
            "split": "dev",
            "target_type": "T1" if i < 2 else "T0",
            "context_start_ref": 0,
            "context_end_ref": 15,
            "target_start_ref": 16,
            "target_end_ref": 31,
            "source_dataset": "synthetic-mimic",
        })
    return rows


def _requirements() -> dict:
    return {
        "scenario_id": "synthetic",
        "required_fields": [
            "split",
            "target_type",
            "source_dataset",
            "context_len",
            "target_len",
            "context_med_count",
            "context_lab_count",
            "context_state_count",
            "target_med_count",
            "target_lab_count",
            "target_state_count",
            "equivalent_contact_present",
            "negative_control_present",
        ],
        "strongly_preferred_fields": ["source_role", "scenario_consistent"],
        "derivable_fields": {
            "context_len": {"any_of": [["context_len"], ["context_start_ref", "context_end_ref"]]},
            "target_len": {"any_of": [["target_len"], ["target_start_ref", "target_end_ref"]]},
            "source_role": {"any_of": [["source_role"], ["source_dataset"]]},
        },
        "coverage_thresholds": {"required_min_pct": 100.0, "strongly_preferred_min_pct": 80.0},
    }


def _metadata(include_all: bool = True) -> list[dict]:
    rows = []
    for i in range(4):
        row = {
            "block_id": f"join-{i}",
            "context_med_count": 1,
            "context_lab_count": 2,
            "context_state_count": 0,
            "target_med_count": int(i < 2),
            "target_lab_count": 1,
            "target_state_count": 0,
            "equivalent_contact_present": int(i >= 2),
            "negative_control_present": 0,
            "scenario_consistent": True,
        }
        if not include_all:
            row.pop("equivalent_contact_present")
        rows.append(row)
    return rows


class MetadataAvailabilityTests(unittest.TestCase):
    def test_target_blocks_only_parks_missing_metadata(self) -> None:
        report = audit_metadata_availability(_requirements(), _blocks())
        self.assertEqual(report["aggregate_only"], True)
        self.assertEqual(report["overall_decision"], "park")
        self.assertIn("metadata_index_not_provided", report["warnings"])
        self.assertIn("metadata_field_missing:equivalent_contact_present", report["warnings"])
        dumped = json.dumps(report)
        self.assertNotIn("join-", dumped)
        self.assertNotIn("not-emitted", dumped)

    def test_all_required_metadata_passes(self) -> None:
        report = audit_metadata_availability(_requirements(), _blocks(), metadata_rows=_metadata())
        self.assertEqual(report["overall_decision"], "pass")
        required = {r["field"]: r for r in report["field_results"] if r["tier"] == "required"}
        self.assertEqual(required["context_len"]["status"], "derivable")
        self.assertEqual(required["equivalent_contact_present"]["status"], "present")

    def test_missing_equivalent_contact_parks(self) -> None:
        report = audit_metadata_availability(_requirements(), _blocks(), metadata_rows=_metadata(include_all=False))
        self.assertEqual(report["overall_decision"], "park")
        self.assertIn("metadata_field_missing:equivalent_contact_present", report["warnings"])

    def test_cli_rejects_unsafe_metadata_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.json"
            unsafe = root / "metadata.npy"
            target.write_text(json.dumps({"blocks": _blocks()}))
            unsafe.write_bytes(b"not used")
            with self.assertRaises(SystemExit):
                main([
                    "--requirements",
                    str(REQ),
                    "--target-blocks",
                    str(target),
                    "--metadata-index",
                    str(unsafe),
                    "--output-dir",
                    str(root / "out"),
                ])

    def test_cli_writes_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.json"
            meta = root / "metadata.json"
            out = root / "out"
            target.write_text(json.dumps({"blocks": _blocks()}))
            meta.write_text(json.dumps({"rows": _metadata()}))
            proc = subprocess.run([
                sys.executable,
                "-m",
                "clinical_jepa.tte.audit_metadata_availability",
                "--requirements",
                str(REQ),
                "--target-blocks",
                str(target),
                "--scenario-card",
                str(SCENARIO),
                "--metadata-index",
                str(meta),
                "--output-dir",
                str(out),
            ], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertIn("metadata-availability.json", proc.stdout)
            report = json.loads((out / "metadata-availability.json").read_text())
            self.assertEqual(report["aggregate_only"], True)


if __name__ == "__main__":
    unittest.main()
