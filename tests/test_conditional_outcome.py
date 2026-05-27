from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.speaker.conditional_outcome import (
    ConditionalOutcomeSpec,
    ConditionalTargetFamilySpec,
    build_conditional_outcome_report,
    count_family,
    default_conditional_synthetic_spec,
    load_conditional_outcome_spec,
    matches_family,
    stratum_key,
    target_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def _conditional_signal_rows() -> list[dict]:
    rows: list[dict] = []
    # Same context/utilisation stratum for the medication target, but explicit
    # synthetic readout scores separate positives from negatives. The future
    # target label itself is target_events-only.
    for _ in range(4):
        rows.append(
            {
                "context_events": ["STATE:SYN_CONTACT:LOW", "STATE:SYN_CONTACT:VISIT"],
                "prior_context_events": [],
                "target_events": ["MED:SYN_DIURETIC:LOOP", "LAB:SYN_RENAL:HIGH"],
                "matched_random_events": ["LAB:SYN_NEG_GLUCOSE:HIGH"],
                "time_shift_target_events": [],
                "readout_scores": [0.95, 0.70, 0.10, 0.05],
            }
        )
    for _ in range(4):
        rows.append(
            {
                "context_events": ["STATE:SYN_CONTACT:LOW", "STATE:SYN_CONTACT:VISIT"],
                "prior_context_events": [],
                "target_events": ["LAB:SYN_ELECTROLYTE:LOW"],
                "matched_random_events": ["MED:SYN_DIURETIC:LOOP"],
                "time_shift_target_events": ["LAB:SYN_RENAL:HIGH"],
                "readout_scores": [0.10, 0.10, 0.80, 0.05],
            }
        )
    for _ in range(4):
        rows.append(
            {
                "context_events": ["STATE:SYN_CONTACT:LOW", "STATE:SYN_CONTACT:VISIT"],
                "prior_context_events": [],
                "target_events": ["LAB:SYN_NEG_GLUCOSE:HIGH"],
                "negative_control_events": ["STATE:SYN_CONTACT:VISIT"],
                "readout_scores": [0.05, 0.10, 0.10, 0.20],
            }
        )
    # Interleave rows so constant stratum-prior ties cannot pass just because
    # positives happen to appear first in stable sorting.
    interleaved: list[dict] = []
    for i in range(4):
        interleaved.extend([rows[i], rows[4 + i], rows[8 + i]])
    return interleaved


class ConditionalOutcomeTests(unittest.TestCase):
    def test_target_matching_is_future_only_and_prefix_based(self) -> None:
        family = ConditionalTargetFamilySpec(name="synthetic_family", include_prefixes=("MED:SYN_TARGET",))
        self.assertTrue(matches_family("med:syn_target:item", family))
        self.assertFalse(matches_family("MED:SYN_OTHER:item", family))
        self.assertEqual(count_family(["MED:SYN_TARGET:A", "MED:SYN_TARGET:B"], family), 2)

        spec = ConditionalOutcomeSpec(target_families=(family,))
        rows = [
            {"context_events": [], "target_events": ["MED:SYN_TARGET:A"]},
            {"context_events": ["MED:SYN_TARGET:A", "STATE:SYN_CONTACT:HIGH"], "target_events": ["MED:SYN_TARGET:A"]},
            {"context_events": ["MED:SYN_TARGET:A"], "target_events": []},
        ]
        y = target_matrix(rows, spec)
        self.assertEqual(y.tolist(), [[1.0], [1.0], [0.0]])
        self.assertEqual(stratum_key(rows[0], family, spec), (0, 0, 0, 0))
        self.assertNotEqual(stratum_key(rows[0], family, spec), stratum_key(rows[1], family, spec))

    def test_conditional_report_separates_targets_from_strata_and_suppresses_predicates(self) -> None:
        spec = default_conditional_synthetic_spec()
        report = build_conditional_outcome_report(_conditional_signal_rows(), spec, scenario_id="synthetic_conditional_signal")
        self.assertTrue(report["aggregate_only"])
        self.assertTrue(report["future_only_targets"])
        self.assertTrue(report["context_used_for_strata_not_target_definition"])
        self.assertEqual(report["n_targets"], len(spec.target_families))
        self.assertIn("stratum_prior", report["baseline_metrics"])
        self.assertIn("matched_random_events", report["control_event_diagnostics"])
        self.assertGreaterEqual(report["conditional_outcome_summary"]["n_adequate_support_targets"], 1)
        self.assertGreaterEqual(report["conditional_outcome_summary"]["n_viable_candidate_targets"], 1)
        medication_diag = report["target_diagnostics"][0]
        self.assertGreater(medication_diag["context_average_precision"], medication_diag["stratum_prior_average_precision"])
        self.assertGreater(medication_diag["context_minus_stratum_prior_ap"], 0.0)
        self.assertTrue(all(row["future_only_target"] for row in report["target_diagnostics"]))
        dumped = json.dumps(report)
        self.assertNotIn("SYN_DIURETIC:LOOP", dumped)
        self.assertNotIn("MED:SYN_DIURETIC", dumped)
        self.assertNotIn("synthetic_diuretic_future_presence", dumped)

    def test_config_loader_and_cli_write_placeholder_plan(self) -> None:
        spec = load_conditional_outcome_spec(ROOT / "configs/v0/conditional_outcome.example.yaml")
        self.assertIsInstance(spec, ConditionalOutcomeSpec)
        self.assertGreaterEqual(len(spec.target_families), 4)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows_path = root / "rows.json"
            out = root / "out"
            rows_path.write_text(json.dumps({"rows": _conditional_signal_rows()}))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clinical_jepa.speaker.conditional_outcome",
                    "--spec-config",
                    str(ROOT / "configs/v0/conditional_outcome.example.yaml"),
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
            report = json.loads((out / "conditional-future-outcome-readout.json").read_text())
            self.assertEqual(report["scenario_id"], "synthetic_cli")
            self.assertTrue((out / "summary.md").exists())
            plan = (out / "command-plan.md").read_text()
            self.assertIn("<LOCAL_PREEXTRACTED_CONDITIONAL_OUTCOME_ROWS.json>", plan)
            self.assertIn("does not render/generate event sequences", plan)
            self.assertIn("future-only", plan)
            self.assertIn("no HDF5/checkpoint/sidecar paths", json.dumps(report["bridge_contract"]))


if __name__ == "__main__":
    unittest.main()
