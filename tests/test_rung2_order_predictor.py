"""Rung-2 sub-gate 3 order-predictor test (synthetic): when context carries the instance order, the
context-only predicted ẑ decodes precedence better than chance (prediction-achieved)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from clinical_jepa.arms.rung2.order_predictor import predict_precedence, train_order_predictor
    from clinical_jepa.targets.order_targets import build_order_target


@unittest.skipUnless(HAS_TORCH, "torch required")
class OrderPredictorTests(unittest.TestCase):
    def test_context_carrying_order_is_recovered(self) -> None:
        rng = np.random.default_rng(0)
        V, D, n_win = 30, 12, 120
        E = rng.standard_normal((V, D)).astype(np.float32)
        z_empty = np.zeros(D, dtype=np.float32)
        ctx, targ, id_lists = [], [], []
        for _ in range(n_win):
            k = 4
            ids = rng.integers(2, V, size=k)
            t2, _ = build_order_target("T2_seq_of_latents", ids, E=E, z_empty=z_empty)
            targ.append(t2)
            # context carries the true order: flattened ordered embeddings (so ẑ can recover it)
            c = np.concatenate([E[i] for i in ids]).astype(np.float32)
            ctx.append(np.concatenate([c, rng.normal(size=4).astype(np.float32)]))
            id_lists.append(ids)
        ctx = np.asarray(ctx, dtype=np.float32); targ = np.asarray(targ, dtype=np.float32)
        model = train_order_predictor(ctx, targ, id_lists, E, D, steps=200)
        P = predict_precedence(model, ctx, id_lists)
        # precedence accuracy: predicted P(a<b) > 0.5 for a<b (the true order is index order)
        correct = tot = 0
        for mat, ids in zip(P, id_lists):
            n = len(ids)
            for a in range(n):
                for b in range(a + 1, n):
                    correct += int(mat[a, b] > 0.5); tot += 1
        self.assertGreater(correct / tot, 0.6)               # recovers order above chance


if __name__ == "__main__":
    unittest.main()
