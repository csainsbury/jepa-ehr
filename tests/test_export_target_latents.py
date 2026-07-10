"""Rung-1 export bridge tests: per-arm z+ bundles from real-shaped blocks, dimensional
empties, and TEST-split exclusion (Pi R8 #8 — test never loaded in this run)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.eval.export_target_latents import block_props, build_bundles
from clinical_jepa.targets.target_reps import target_dim

D = 12
V = 40


def _blk(seq_id, split, ts, te, *, empty=False, tq=5.0, W=90.0, ce=4, n=None):
    return {"sequence_id": seq_id, "patient_hash": f"pat-{seq_id}", "split": split,
            "source_dataset": "SCID", "window_days": float(W), "t_query": float(tq),
            "context_end_ref": int(ce), "target_start_ref": -1 if empty else int(ts),
            "target_end_ref": -1 if empty else int(te), "empty_target": bool(empty),
            "n_target_events": int(0 if empty else (n if n is not None else te - ts + 1))}


class BlockPropsTests(unittest.TestCase):
    def test_props_extract(self) -> None:
        ids = np.arange(1, 21, dtype=np.int64)
        days = np.cumsum(np.ones(20, dtype=np.float64))
        p = block_props(_blk("s0", "dev", 5, 9, n=5), ids, days)
        self.assertEqual(p["count"], 5)
        self.assertEqual(len(p["ordered_ids"]), 5)
        self.assertEqual(len(p["dt"]), 4)                         # 5 events -> 4 intervals
        self.assertFalse(p["is_empty"])


@unittest.skipUnless(HAS_TORCH, "torch required")
class BundleTests(unittest.TestCase):
    def _model(self):
        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA
        import torch
        torch.manual_seed(0)
        return MeanTokenJEPA(vocab_size=V, embedding_dim=D, encode_empty=True)

    def test_bundles_exclude_test_and_shape_ok(self) -> None:
        m = self._model()
        ids = np.arange(1, 30, dtype=np.int64)
        days = np.cumsum(np.ones(29, dtype=np.float64))
        seqs = {"s0": {"token_ids": ids, "cumulative_days": days},
                "s1": {"token_ids": ids, "cumulative_days": days}}
        blocks = [
            _blk("s0", "train", 6, 12, ce=5, tq=float(days[5]), W=200.0),
            _blk("s1", "dev", 6, 12, ce=5, tq=float(days[5]), W=200.0),
            _blk("s0", "test", 6, 12, ce=5, tq=float(days[5]), W=200.0),   # must be excluded
        ]
        bundles = build_bundles(blocks, seqs, model=m, arms=["mean_embed", "count_concat"])
        for (arm, source, W), cell in bundles.items():
            self.assertNotIn("test", cell)                        # test sealed
            for split, b in cell.items():
                self.assertEqual(b["z"].shape[1], target_dim(arm, D))
        self.assertIn(("mean_embed", "SCID", 200.0), bundles)


if __name__ == "__main__":
    unittest.main()
