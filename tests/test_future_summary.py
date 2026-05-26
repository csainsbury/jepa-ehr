from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clinical_jepa.speaker.future_summary import (
    FutureSummarySpec,
    build_future_summary_report,
    event_label,
    summarize_events,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows() -> list[dict]:
    # Context and future labels are deliberately correlated for the context-copy
    # baseline, while empirical prior alone must average across both patterns.
    return [
        {
            "context_events": ["MED:DIURETIC:LOOP", "LAB:CREATININE:HIGH", "STATE:ICU"],
            "target_events": ["MED:DIURETIC:LOOP", "LAB:CREATININE:HIGH", "STATE:ICU"],
        },
        {
            "context_events": ["MED:DIURETIC:LOOP", "LAB:POTASSIUM:LOW", "STATE:ICU"],
            "target_events": ["MED:DIURETIC:LOOP", "LAB:POTASSIUM:LOW", "STATE:ICU"],
        },
        {
            "context_events": ["MED:INSULIN:BASAL", "LAB:GLUCOSE:HIGH", "STATE:WARD"],
            "target_events": ["MED:INSULIN:BASAL", "LAB:GLUCOSE:HIGH", "STATE:WARD"],
        },
        {
            "context_events": ["MED:INSULIN:BASAL", "LAB:GLUCOSE:HIGH", "STATE:WARD"],
            "target_events": ["MED:INSULIN:BASAL", "LAB:GLUCOSE:HIGH", "STATE:WARD"],
        },
    ]


class FutureSummaryTests(unittest.TestCase):
    def test_event_summary_uses_coded_families_without_generation(self) -> None:
        spec = FutureSummarySpec(top_k=2)
        summary = summarize_events(["MED:DIURETIC:LOOP", "LAB:CREATININE:HIGH", "NOTE:IGNORED"], spec)
        self.assertEqual(event_label("med:diuretic:loop", label_depth=2), "MED:DIURETIC")
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["type_counts"]["MED"], 1)
        self.assertEqual(summary["type_counts"]["LAB"], 1)
        self.assertIn("MED:DIURETIC", summary["label_presence"])
        self.assertAlmostEqual(sum(summary["type_distribution"].values()), 1.0)

    def test_context_baseline_beats_empirical_prior_on_correlated_synthetic_rows(self) -> None:
        report = build_future_summary_report(_synthetic_rows(), FutureSummarySpec(top_k=3), scenario_id="synthetic")
        prior = report["baselines"]["empirical_prior_presence"]
        context = report["baselines"]["context_presence_copy"]
        self.assertEqual(report["aggregate_only"], True)
        self.assertGreater(report["n_labels"], 3)
        self.assertGreater(context["macro_average_precision"], prior["macro_average_precision"])
        self.assertGreater(context["top_k_recall_mean"], prior["top_k_recall_mean"])
        self.assertTrue(report["bridge_contract"]["not_generation"])
        dumped = json.dumps(report)
        self.assertNotIn("DIURETIC:LOOP", dumped)  # no raw token/example strings
        self.assertNotIn("MED:DIURETIC", dumped)  # label names suppressed for safe aggregate reports
        self.assertTrue(report["label_names_suppressed"])

    def test_cli_writes_aggregate_report_and_placeholder_command_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows_path = root / "rows.json"
            out = root / "out"
            rows_path.write_text(json.dumps({"rows": _synthetic_rows()}))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clinical_jepa.speaker.future_summary",
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
            report = json.loads((out / "future-summary-readout.json").read_text())
            self.assertEqual(report["scenario_id"], "synthetic_cli")
            self.assertEqual(report["aggregate_only"], True)
            self.assertTrue((out / "summary.md").exists())
            plan = (out / "command-plan.md").read_text()
            self.assertIn("<LOCAL_PREEXTRACTED_CODED_EVENT_SUMMARY_ROWS.json>", plan)
            self.assertIn("not render/generate event sequences", plan)


if __name__ == "__main__":
    unittest.main()
