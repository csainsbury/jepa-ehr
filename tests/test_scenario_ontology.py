from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.speaker.scenario_ontology import (
    ScenarioOntologySpec,
    TargetFamilySpec,
    build_scenario_ontology_report,
    count_family,
    default_diuretic_synthetic_spec,
    family_features,
    load_scenario_ontology_spec,
    matches_family,
)

ROOT = Path(__file__).resolve().parents[1]


def _signal_rows() -> list[dict]:
    rows: list[dict] = []
    # Context-specific future summaries: context-copy should beat global prior.
    for _ in range(4):
        rows.append(
            {
                "context_events": ["MED:SYN_DIURETIC:LOOP", "LAB:SYN_RENAL:HIGH", "STATE:SYN_CONTACT:HIGH"],
                "target_events": ["MED:SYN_DIURETIC:LOOP", "LAB:SYN_RENAL:HIGH"],
                "matched_random_events": ["LAB:SYN_NEG_GLUCOSE:HIGH"],
            }
        )
    for _ in range(4):
        rows.append(
            {
                "context_events": ["LAB:SYN_ELECTROLYTE:LOW", "STATE:SYN_CONTACT:LOW"],
                "target_events": ["LAB:SYN_ELECTROLYTE:LOW"],
                "time_shift_target_events": ["MED:SYN_DIURETIC:LOOP"],
            }
        )
    for _ in range(4):
        rows.append(
            {
                "context_events": ["LAB:SYN_NEG_GLUCOSE:HIGH", "STATE:SYN_CONTACT:LOW"],
                "target_events": ["LAB:SYN_NEG_GLUCOSE:HIGH"],
                "negative_control_events": ["STATE:SYN_CONTACT:HIGH"],
            }
        )
    return rows


def _base_rate_rows() -> list[dict]:
    rows: list[dict] = []
    # Dominated by a common future family regardless of context.
    for idx in range(20):
        context = ["STATE:SYN_CONTACT:HIGH"] if idx % 2 else ["LAB:SYN_ELECTROLYTE:LOW"]
        rows.append(
            {
                "context_events": context,
                "target_events": ["MED:SYN_DIURETIC:LOOP", "LAB:SYN_RENAL:HIGH"],
            }
        )
    return rows


class ScenarioOntologyTests(unittest.TestCase):
    def test_family_predicates_are_prefix_based_and_exclusion_aware(self) -> None:
        family = TargetFamilySpec(
            name="synthetic_family",
            include_prefixes=("MED:SYN_DIURETIC",),
            exclude_prefixes=("MED:SYN_DIURETIC:EXCLUDE",),
        )
        self.assertTrue(matches_family("med:syn_diuretic:loop", family))
        self.assertFalse(matches_family("MED:SYN_DIURETIC:EXCLUDE_ME", family))
        self.assertFalse(matches_family("LAB:SYN_RENAL:HIGH", family))
        features = family_features(["MED:SYN_DIURETIC:LOOP"], ["MED:SYN_DIURETIC:LOOP"], family)
        self.assertEqual(features["presence"], 1)
        self.assertEqual(features["continuation"], 1)
        self.assertEqual(features["start"], 0)
        self.assertEqual(count_family(["MED:SYN_DIURETIC:LOOP", "MED:SYN_DIURETIC:THIAZIDE"], family), 2)

    def test_scenario_report_scores_signal_controls_and_suppresses_predicates(self) -> None:
        report = build_scenario_ontology_report(_signal_rows(), default_diuretic_synthetic_spec(), scenario_id="synthetic_signal")
        baseline = report["baseline_metrics"]
        self.assertEqual(report["aggregate_only"], True)
        self.assertTrue(report["not_generation"])
        self.assertTrue(report["target_names_suppressed"])
        self.assertGreater(report["n_targets"], 4)
        self.assertGreater(
            baseline["context_summary"]["macro_average_precision"],
            baseline["empirical_prior"]["macro_average_precision"],
        )
        self.assertIn("matched_random_events", report["negative_control_hooks"]["observed_control_event_sets"])
        self.assertGreaterEqual(report["base_rate_domination"]["n_viable_candidate_targets"], 1)
        dumped = json.dumps(report)
        self.assertNotIn("SYN_DIURETIC:LOOP", dumped)
        self.assertNotIn("MED:SYN_DIURETIC", dumped)
        self.assertNotIn("synthetic_diuretic_med_family", dumped)

    def test_base_rate_domination_diagnostic_flags_common_targets(self) -> None:
        report = build_scenario_ontology_report(_base_rate_rows(), default_diuretic_synthetic_spec(), scenario_id="synthetic_base_rate")
        domination = report["base_rate_domination"]
        self.assertTrue(domination["base_rate_domination_flag"])
        self.assertGreaterEqual(domination["n_prior_dominant"], 1)
        self.assertEqual(domination["recommendation"], "refine_or_park_if_local_scan_repeats_base_rate_domination")

    def test_config_loader_and_cli_write_placeholder_plan(self) -> None:
        spec = load_scenario_ontology_spec(ROOT / "configs/v0/scenario_ontology.example.yaml")
        self.assertIsInstance(spec, ScenarioOntologySpec)
        self.assertGreaterEqual(len(spec.target_families), 4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows_path = root / "rows.json"
            out = root / "out"
            rows_path.write_text(json.dumps({"rows": _signal_rows()}))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clinical_jepa.speaker.scenario_ontology",
                    "--spec-config",
                    str(ROOT / "configs/v0/scenario_ontology.example.yaml"),
                    "--input-json",
                    str(rows_path),
                    "--output-dir",
                    str(out),
                    "--scenario-id",
                    "synthetic_cli",
                    "--emit-command-plan",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            report = json.loads((out / "scenario-coded-summary-readout.json").read_text())
            self.assertEqual(report["scenario_id"], "synthetic_cli")
            self.assertTrue((out / "summary.md").exists())
            plan = (out / "command-plan.md").read_text()
            self.assertIn("<LOCAL_PREEXTRACTED_SCENARIO_CODED_SUMMARY_ROWS.json>", plan)
            self.assertIn("does not render/generate event sequences", plan)
            self.assertIn("no HDF5/checkpoint/sidecar paths", json.dumps(report["bridge_contract"]))


if __name__ == "__main__":
    unittest.main()
