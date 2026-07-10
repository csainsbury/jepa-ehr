"""Rung-0 leak-free latent tests (Pi R5 C1/C2/C3): predicted queries are CONTEXT-ONLY
(the future-cardinality leak Pi caught), the pooling identity holds numerically on
populated windows, and the event budget is matched bilaterally."""
from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.eval.coarse_fine_latents import (  # noqa: E402
    budget_subsample_ids,
    coarse_query,
    fine_queries,
    pooling_identity_residual,
    predicted_counts,
)

DIM = 16
VOCAB = 64


class C1SignatureTests(unittest.TestCase):
    def test_queries_take_context_only_no_target_arg(self) -> None:
        # The future-cardinality leak is structurally impossible: no query function
        # accepts a target / count / n_k argument (Pi R5 C1).
        self.assertEqual(set(inspect.signature(coarse_query).parameters), {"model", "ctx_ids"})
        self.assertEqual(set(inspect.signature(fine_queries).parameters), {"model", "ctx_ids", "K"})
        self.assertEqual(set(inspect.signature(predicted_counts).parameters), {"model", "ctx_ids", "K"})
        for fn in (coarse_query, fine_queries, predicted_counts):
            src = inspect.getsource(fn)
            self.assertNotIn("n_target", src)
            self.assertNotIn("target_latent", src)  # queries never touch targets


@unittest.skipUnless(HAS_TORCH, "torch required")
class C1C2Tests(unittest.TestCase):
    def _model(self):
        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA
        import torch
        torch.manual_seed(0)
        return MeanTokenJEPA(VOCAB, DIM, autoregression_mode="horizon_conditioned", max_horizons=4, encode_empty=True)

    def test_coarse_query_is_context_only_single_step(self) -> None:
        import torch
        m = self._model()
        ctx = torch.tensor([[1, 10, 11, 12], [1, 20, 21, 0]], dtype=torch.long)
        q = coarse_query(m, ctx)
        ref = m.predict_rollout_from_context_ids(ctx, 1)[:, 0, :]
        self.assertTrue(torch.allclose(q, ref))          # exactly the context-only prediction
        # Mutating a hypothetical target cannot change q (q is a pure fn of ctx).
        self.assertTrue(torch.allclose(coarse_query(m, ctx), q))
        self.assertEqual(fine_queries(m, ctx, 3).shape, (2, 3, DIM))

    def test_pooling_identity_holds_on_populated(self) -> None:
        m = self._model()
        # full-W [10,11,12,13,14] split into [10,11] + [12,13,14]; weighted mean == full mean.
        resid = pooling_identity_residual(m, [[10, 11], [12, 13, 14]])
        self.assertLess(resid, 1e-4)


class C3BudgetTests(unittest.TestCase):
    def test_budget_subsample_matched_and_stratified(self) -> None:
        ids = [10, 11, 12, 13, 14]
        s = budget_subsample_ids(ids, budget_B=3, seed=0)
        self.assertEqual(len(s), 3)
        self.assertTrue(sorted(s.tolist()) == s.tolist())        # sorted (stable)
        # deterministic (fixed seed)
        self.assertEqual(s.tolist(), budget_subsample_ids(ids, 3, 0).tolist())
        # < B events -> separate stratum, never padded
        self.assertIsNone(budget_subsample_ids([10, 11], budget_B=3, seed=0))
        # pads dropped before counting
        self.assertIsNone(budget_subsample_ids([10, 0, 0], budget_B=2, seed=0))


if __name__ == "__main__":
    unittest.main()
