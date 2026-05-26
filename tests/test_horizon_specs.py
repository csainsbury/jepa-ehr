from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clinical_jepa.eval.horizon_specs import (
    build_horizon_spec_report,
    default_event_horizon_specs,
    load_horizon_specs,
    summarize_candidate_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rollout(*, n: int, horizons: int, dim: int = 4, angle_step: float = 0.1) -> np.ndarray:
    arr = np.zeros((n, horizons, dim), dtype=np.float32)
    for h in range(horizons):
        angle = h * angle_step
        arr[:, h, 0] = np.cos(angle)
        arr[:, h, 1] = np.sin(angle)
        # Add a tiny stable row-specific component so the diagnostic handles
        # batches without relying on patient/example identifiers.
        arr[:, h, 2] = np.linspace(0.0, 0.01, n, dtype=np.float32)
    return arr


class HorizonSpecTests(unittest.TestCase):
    def test_default_specs_emit_placeholder_command_plan(self) -> None:
        specs = default_event_horizon_specs()
        self.assertIn("event32_stride64", {spec.spec_id for spec in specs})
        report = build_horizon_spec_report(specs)
        dumped = json.dumps(report)
        self.assertIn("<LOCAL_CHECKPOINT>", dumped)
        self.assertIn("<LOCAL_TARGET_BLOCK_MANIFEST>", dumped)
        self.assertNotIn("/Users/", dumped)
        self.assertEqual(report["aggregate_only"], True)
        self.assertEqual(report["n_specs_evaluated"], 0)

    def test_synthetic_farther_stride_is_more_separable(self) -> None:
        spec_near, spec_far = load_horizon_specs(ROOT / "configs/v0/horizon_specs.example.yaml")[:2]
        near = _synthetic_rollout(n=12, horizons=spec_near.horizon_count, angle_step=0.05)
        far = _synthetic_rollout(n=12, horizons=spec_far.horizon_count, angle_step=0.35)
        near_summary = summarize_candidate_spec(spec_near, near)
        far_summary = summarize_candidate_spec(spec_far, far)
        near_d1 = near_summary["target_horizon_similarity"]["per_distance"][0]["cosine_mean_over_pairs"]
        far_d1 = far_summary["target_horizon_similarity"]["per_distance"][0]["cosine_mean_over_pairs"]
        self.assertLess(far_d1, near_d1)
        self.assertTrue(far_summary["meets_min_target_cosine_drop"])

    def test_cli_writes_summary_and_command_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.npy"
            out = root / "out"
            np.save(target, _synthetic_rollout(n=8, horizons=9, angle_step=0.2))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clinical_jepa.eval.horizon_specs",
                    "--spec-config",
                    str(ROOT / "configs/v0/horizon_specs.example.yaml"),
                    "--target-rollout",
                    f"event32_stride32={target}",
                    "--output-dir",
                    str(out),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            report = json.loads((out / "horizon-spec-diagnostic.json").read_text())
            self.assertEqual(report["aggregate_only"], True)
            self.assertEqual(report["n_specs_evaluated"], 1)
            self.assertTrue((out / "summary.md").exists())
            command_plan = (out / "command-plan.md").read_text()
            self.assertIn("export_mean_token_rollouts", command_plan)
            self.assertIn("autoregression_readiness", command_plan)
            self.assertNotIn("/Users/", command_plan)


if __name__ == "__main__":
    unittest.main()
