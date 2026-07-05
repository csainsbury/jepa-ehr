from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval.source_probe import source_prediction_probe


class SourceProbeTests(unittest.TestCase):
    def test_informative_latents_beat_base_rate(self) -> None:
        # Latents that encode source: SCID centred at +3, MIMIC at -3.
        rng = np.random.default_rng(0)
        n = 120
        sources = ["SCID"] * (n // 2) + ["MIMIC"] * (n // 2)
        base = rng.normal(size=(n, 6)) * 0.3
        shift = np.array([[3.0 if s == "SCID" else -3.0] for s in sources])
        latents = base + shift
        report = source_prediction_probe(latents, sources, seed=1)
        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["source_prediction_accuracy"], 0.9)
        self.assertFalse(report["near_base_rate"])

    def test_masked_latents_sit_near_base_rate(self) -> None:
        # Source-independent latents: a probe cannot recover source (mask works).
        rng = np.random.default_rng(2)
        n = 160
        sources = ["SCID"] * (n // 2) + ["MIMIC"] * (n // 2)
        latents = rng.normal(size=(n, 6))
        report = source_prediction_probe(latents, sources, seed=3, tolerance=0.12)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["near_base_rate"], msg=f"delta={report['above_base_rate_delta']}")

    def test_single_source_is_not_applicable(self) -> None:
        latents = np.random.default_rng(4).normal(size=(20, 4))
        report = source_prediction_probe(latents, ["SCID"] * 20)
        self.assertEqual(report["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
