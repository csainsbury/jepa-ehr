"""Rung-0 verdict CLI tests: load exported sidecars -> per-source three-way verdict;
a run without the sufficiency/raw-count co-gates can never falsely BUILD."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval.rung0_verdict import load_bundle, run_verdict


def _write_cell(d: Path, granularity: str, *, source, W, n=6, hit=True, subwindow_k=None):
    d.mkdir(parents=True, exist_ok=True)
    emb = np.eye(max(n, 4), dtype=np.float32)[:n]
    queries = emb.copy() if hit else np.roll(emb, 1, axis=0)
    np.save(d / f"{granularity}_queries.npy", queries.astype(np.float16))
    np.save(d / f"{granularity}_targets.npy", emb.astype(np.float16))
    pats = [f"p{i % 3}" for i in range(n)]                  # 3 patients -> patient-disjoint works
    with (d / f"{granularity}_index.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"block_id": f"{granularity}{i}", "patient_hash": pats[i],
                                "source_dataset": source, "split": "dev", "window_days": W,
                                "target_type": "T0", "granularity": granularity, "subwindow_k": subwindow_k,
                                "n_target_events": 5, "context_len": 20}) + "\n")


class VerdictTests(unittest.TestCase):
    def test_load_bundle_and_verdict_inconclusive_without_cogates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for W in (30.0, 90.0):
                base = root / "SCID" / f"W{W:g}"
                for g in ("coarse", "coarse_B"):
                    _write_cell(base, g, source="SCID", W=W, hit=True)
                for g in ("fine", "fine_B"):
                    _write_cell(base, g, source="SCID", W=W, hit=True, subwindow_k=0)
            bundle = load_bundle(root, "SCID", [30.0, 90.0])
            self.assertIn(30.0, bundle)
            self.assertIn("coarse", bundle[30.0])

            m = run_verdict(root, {"SCID": {"horizons": [30.0, 90.0], "level_horizons": [30.0, 90.0],
                                            "raw_count_ok": False, "sufficiency_ok": False}},
                            n_boot=200, adequacy_floor=2)
            # coarse == fine (gap ~0) AND no sufficiency co-gate -> never BUILD.
            self.assertNotEqual(m["decisions"]["SCID"], "BUILD")
            self.assertIn(m["decisions"]["SCID"], ("NO-BUILD_INCONCLUSIVE", "NO-BUILD_EFFECT-RULED-OUT"))
            self.assertTrue(m["per_source"]["SCID"]["aggregate_only"])

    def test_missing_sidecars_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = run_verdict(Path(td), {"MIMIC": {"horizons": [1.0], "level_horizons": [1.0]}})
            self.assertEqual(m["decisions"]["MIMIC"], "NO-BUILD_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
