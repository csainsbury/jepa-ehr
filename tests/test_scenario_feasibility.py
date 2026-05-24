from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.tte.scan_feasibility import scan_scenario_feasibility

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/v0/scenario.example.yaml"


def _manifest() -> dict:
    blocks = []
    for i in range(6):
        blocks.append({
            "block_id": f"secret-t1-{i}",
            "patient_hash": f"patient-secret-{i}",
            "split": "dev",
            "target_type": "T1",
            "context_start_ref": 0,
            "context_end_ref": 20,
            "target_start_ref": 21,
            "target_end_ref": 35,
            "horizon_descriptor": "medication_anchor_event_window",
            "source_dataset": "synthetic-mimic",
            "target_med_count": 1,
            "target_lab_count": 1,
            "target_state_count": 0,
            "negative_control_present": 0,
            "equivalent_contact_present": 0,
        })
    for i in range(6):
        blocks.append({
            "block_id": f"secret-t0-{i}",
            "patient_hash": f"patient-secret-c-{i}",
            "split": "dev",
            "target_type": "T0",
            "context_start_ref": 0,
            "context_end_ref": 22,
            "target_start_ref": 23,
            "target_end_ref": 39,
            "horizon_descriptor": "short",
            "source_dataset": "synthetic-mimic",
            "target_med_count": 0,
            "target_lab_count": 1,
            "target_state_count": 1,
            "negative_control_present": 0,
            "equivalent_contact_present": 1,
        })
    return {
        "schema_version": "clinical-jepa-target-block-manifest-v0",
        "created_utc": "2026-05-24T00:00:00Z",
        "targets": ["T0", "T1"],
        "counts": {"dev": {"T0": 6, "T1": 6}},
        "blocks": blocks,
    }


class ScenarioFeasibilityTests(unittest.TestCase):
    def test_aggregate_scan_does_not_emit_ids(self) -> None:
        scenario = {
            "scenario_id": "synthetic_diuretic_placeholder",
            "aggregate_rules": {
                "eligible_target_types": ["T0", "T1"],
                "incident_target_types": ["T1"],
                "comparator_target_types": ["T0"],
                "min_context_events": 8,
                "min_target_events": 8,
                "proxy_positive_fields": ["target_lab_count"],
                "negative_control_positive_fields": ["negative_control_present"],
                "incident_positive_fields": ["target_med_count"],
                "comparator_positive_fields": ["equivalent_contact_present"],
            },
            "minimums": {"eligible": 10, "incident_initiators": 5, "comparator_candidates": 5},
            "guardrail_status": {
                "equivalent_contact_comparator": "pass",
                "incident_lookback": "pass",
                "endpoint_leakage_embargo": "pass",
                "negative_controls_defined": "pass",
                "contact_intensity_controls_defined": "pass",
                "source_measurability": "pass",
            },
        }
        report = scan_scenario_feasibility(_manifest(), scenario, dry_run=False)
        self.assertEqual(report["aggregate_only"], True)
        self.assertEqual(report["overall_decision"], "promote")
        dumped = json.dumps(report)
        self.assertNotIn("secret-t1", dumped)
        self.assertNotIn("patient-secret", dumped)
        result = report["results"][0]
        self.assertEqual(result["n_incident_initiators"], 6)
        self.assertEqual(result["n_comparator_candidates"], 6)

    def test_missing_configured_metadata_fields_do_not_promote(self) -> None:
        scenario = {
            "scenario_id": "synthetic_missing_metadata",
            "aggregate_rules": {
                "eligible_target_types": ["T0", "T1"],
                "incident_target_types": ["T1"],
                "comparator_target_types": ["T0"],
                "min_context_events": 8,
                "min_target_events": 8,
                "incident_positive_fields": ["missing_incident_marker"],
                "comparator_positive_fields": ["missing_equivalent_contact"],
                "negative_control_positive_fields": ["negative_control_present"],
            },
            "minimums": {"eligible": 10, "incident_initiators": 5, "comparator_candidates": 5},
            "guardrail_status": {
                "equivalent_contact_comparator": "pass",
                "incident_lookback": "pass",
                "endpoint_leakage_embargo": "pass",
                "negative_controls_defined": "pass",
                "contact_intensity_controls_defined": "pass",
                "source_measurability": "pass",
            },
        }
        report = scan_scenario_feasibility(_manifest(), scenario, dry_run=False)
        self.assertNotEqual(report["overall_decision"], "promote")
        warnings = report["warnings"]
        self.assertIn("metadata_field_missing:missing_incident_marker", warnings)
        self.assertIn("metadata_field_missing:missing_equivalent_contact", warnings)

    def test_cli_dry_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.json"
            out = root / "out"
            target.write_text(json.dumps(_manifest()))
            subprocess.run([
                sys.executable,
                "-m",
                "clinical_jepa.tte.scan_feasibility",
                "--target-blocks",
                str(target),
                "--scenario-card",
                str(SCENARIO),
                "--output-dir",
                str(out),
                "--dry-run",
            ], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report = json.loads((out / "scenario-feasibility.json").read_text())
            self.assertTrue((out / "summary.md").exists())
            self.assertNotEqual(report["overall_decision"], "promote")
            self.assertIn("leakage_not_checked", report["warnings"])


if __name__ == "__main__":
    unittest.main()
