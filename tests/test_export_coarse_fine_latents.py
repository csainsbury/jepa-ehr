"""Export tests (Pi R5 C1): the real export path preserves context-only queries
(mutating targets leaves every query byte-identical) + K× fine expansion + k1 null."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.eval.export_coarse_fine_latents import _read_block, export_blocks  # noqa: E402

VOCAB, DIM = 64, 16


def _item(pid="pA", block_id="b0", window=6.0, delta=2.0, mode="fixed_width_delta"):
    token_ids = np.array([1] + list(range(10, 40)), dtype=np.int64)   # 31 tokens < vocab
    cdays = np.arange(31, dtype=np.float32)
    block = {"block_id": block_id, "patient_hash": pid, "source_dataset": "SCID", "split": "dev",
             "context_start_ref": 2, "context_end_ref": 12, "window_days": window, "t_query": 12.0,
             "target_start_ref": 13, "target_end_ref": 17, "empty_target": False, "censored": False,
             "n_target_events": 5}
    return _read_block(block, token_ids, cdays, delta_days=delta, mode=mode, max_context=32)


@unittest.skipUnless(HAS_TORCH, "torch required")
class ExportTests(unittest.TestCase):
    def _model(self):
        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA
        import torch
        torch.manual_seed(0)
        return MeanTokenJEPA(VOCAB, DIM, autoregression_mode="recursive", encode_empty=True)

    def test_read_block_partitions(self) -> None:
        it = _item()
        self.assertEqual(it["K"], 3)                         # round(6/2)
        self.assertEqual(len(it["subwindows"]), 3)
        self.assertEqual([s["n"] for s in it["subwindows"]], [1, 2, 2])

    def test_export_shapes_and_fine_expansion(self) -> None:
        m = self._model()
        items = [_item(pid=f"p{i}", block_id=f"b{i}") for i in range(5)]
        out = export_blocks(m, items, W=6.0, delta_days=2.0, mode="fixed_width_delta", budget_B=1, seed=0)
        self.assertEqual(len(out["coarse"]["index"]), 5)
        self.assertEqual(len(out["fine"]["index"]), 5 * 3)   # K=3 sub-windows per block
        self.assertEqual(out["coarse"]["queries"].shape, (5, DIM))
        self.assertEqual(out["fine"]["queries"].shape, (15, DIM))
        self.assertIn("coarse_B", out)                       # budget_B=1 -> all populated targets qualify

    def test_queries_are_context_only_invariant_to_targets(self) -> None:
        import copy
        m = self._model()
        items = [_item(pid=f"p{i}", block_id=f"b{i}") for i in range(4)]
        out1 = export_blocks(m, items, W=6.0, delta_days=2.0, mode="fixed_width_delta", budget_B=1, seed=0)
        # Corrupt every TARGET (full + sub-window ids) but keep context identical.
        mutated = copy.deepcopy(items)
        for it in mutated:
            it["full_ids"] = np.array([39, 39, 39], dtype=np.int64)
            for s in it["subwindows"]:
                s["ids"] = np.array([38], dtype=np.int64)
                s["is_empty"] = False
        out2 = export_blocks(m, mutated, W=6.0, delta_days=2.0, mode="fixed_width_delta", budget_B=1, seed=0)
        # C1: queries depend on CONTEXT ONLY -> byte-identical despite target corruption.
        self.assertTrue(np.array_equal(out1["coarse"]["queries"], out2["coarse"]["queries"]))
        self.assertTrue(np.array_equal(out1["fine"]["queries"], out2["fine"]["queries"]))
        # ... while the TARGETS DID change (sanity: the corruption took effect).
        self.assertFalse(np.array_equal(out1["fine"]["targets"], out2["fine"]["targets"]))

    def test_k1_null_fine_equals_coarse(self) -> None:
        m = self._model()
        items = [_item(pid=f"p{i}", block_id=f"b{i}", mode="k1_null") for i in range(3)]
        out = export_blocks(m, items, W=6.0, delta_days=2.0, mode="k1_null", budget_B=1, seed=0)
        self.assertEqual(items[0]["K"], 1)
        self.assertIn("k1", out)
        # K=1: the single fine sub-window IS the full-W target; its query is the coarse query.
        self.assertTrue(np.allclose(out["k1"]["queries"], out["coarse"]["queries"]))


if __name__ == "__main__":
    unittest.main()
