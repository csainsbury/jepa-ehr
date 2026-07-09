"""Rung-1 empty-decode tests (Pi R4): the empty/count-0 head decodes silence from
frozen z+, the decode rule skips token/timing on empties, and the empty-recall>=0.95
falsifier fires when z_empty is NOT separable (simulated collapse)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.eval.rung1_decode import (  # noqa: E402
    build_empty_count_decoder,
    decode_rule,
    empty_decode_metrics,
    nonempty_cell_mask,
    train_empty_count_decoder,
)

DIM = 16


@unittest.skipUnless(HAS_TORCH, "torch required")
class Rung1EmptyDecodeTests(unittest.TestCase):
    def _separable(self, n: int = 40):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(0)
        z_empty = F.normalize(torch.randn(DIM), dim=0)
        emp = z_empty.unsqueeze(0).expand(n, -1) + 0.01 * torch.randn(n, DIM)   # tight silence cluster
        pop = torch.randn(n, DIM) + 3.0 * F.normalize(torch.randn(DIM), dim=0)   # elsewhere
        z_plus = torch.cat([emp, pop], 0)
        is_empty = torch.tensor([True] * n + [False] * n)
        count = torch.tensor([0] * n + [1 + (i % 4) for i in range(n)])
        return z_plus, is_empty, count

    def test_separable_passes_falsifier(self) -> None:
        z_plus, is_empty, count = self._separable()
        dec = build_empty_count_decoder(DIM)
        train_empty_count_decoder(dec, z_plus, is_empty, count, steps=400)
        m = empty_decode_metrics(dec, z_plus, is_empty, count)
        self.assertGreaterEqual(m["empty_recall"], 0.95)
        self.assertTrue(m["passes_empty_falsifier"])
        self.assertLess(m["empty_false_positive_rate"], 0.1)     # precision-side reported too
        self.assertGreater(m["count0_accuracy_on_true_empty"], 0.95)
        self.assertGreater(m["exact_count_incl_zero"], 0.5)      # count-0 is a real class

    def test_decode_rule_skips_tokens_for_empties(self) -> None:
        import torch

        z_plus, is_empty, count = self._separable()
        dec = build_empty_count_decoder(DIM)
        train_empty_count_decoder(dec, z_plus, is_empty, count, steps=400)
        out = decode_rule(dec, z_plus)
        # Every decoded-empty row has count 0 and is excluded from token decoding.
        self.assertTrue(bool((out["decoded_count"][out["decoded_empty"]] == 0).all()))
        self.assertTrue(torch.equal(out["decode_tokens_mask"], ~out["decoded_empty"]))
        self.assertTrue(torch.equal(nonempty_cell_mask(is_empty), ~is_empty))

    def test_nonseparable_fails_falsifier_collapse_alarm(self) -> None:
        # z_empty collapsed into the populated manifold: empties and populated share
        # IDENTICAL latents with contradictory labels -> unlearnable -> recall < floor.
        import torch

        torch.manual_seed(1)
        z = torch.randn(40, DIM)
        z_plus = torch.cat([z, z], 0)
        is_empty = torch.tensor([True] * 40 + [False] * 40)
        count = torch.tensor([0] * 40 + [2] * 40)
        dec = build_empty_count_decoder(DIM)
        train_empty_count_decoder(dec, z_plus, is_empty, count, steps=300)
        m = empty_decode_metrics(dec, z_plus, is_empty, count)
        self.assertLess(m["empty_recall"], 0.95)
        self.assertFalse(m["passes_empty_falsifier"])   # collapse alarm fires


if __name__ == "__main__":
    unittest.main()
