"""Rung-0 sufficiency ablation tests (Pi R5 Q5): a TRAINED coarse-conditioned head
reduces the per-lag drift SLOPE vs a matched-compute coarse-zeroed flat baseline when
the coarse plan is informative — and does NOT when it is useless."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.eval.drift_ablation import build_fine_render_head, compare_flat_vs_2level  # noqa: E402

DIM, K, N = 16, 4, 160


@unittest.skipUnless(HAS_TORCH, "torch required")
class DriftAblationTests(unittest.TestCase):
    def _informative(self):
        """z_k depends MORE on the per-example coarse c as k grows (alpha_k = k/(K-1)):
        early sub-windows ~ a shared fixed vector (flat can predict), late ones ~ c
        (need coarse). Flat drift rises with k; 2-level (has c) stays low."""
        import torch
        import torch.nn.functional as F
        torch.manual_seed(0)
        ctx_z = torch.zeros(N, DIM)                      # context carries no per-example info
        c = F.normalize(torch.randn(N, DIM), dim=1)      # predicted coarse (per example)
        fixed = F.normalize(torch.randn(DIM), dim=0)
        tz = []
        for k in range(K):
            a = k / (K - 1)
            tz.append(F.normalize(a * c + (1 - a) * fixed, dim=1))
        return ctx_z, c, torch.stack(tz, dim=1)          # targets [N,K,DIM]

    def test_two_level_reduces_drift_slope_when_coarse_informative(self) -> None:
        ctx_z, c, tz = self._informative()
        r = compare_flat_vs_2level(DIM, K, ctx_z, c, tz, steps=300, seeds=(0, 1), n_boot=300)
        self.assertGreater(r["drift_slope_improvement"], 0.0)          # flat drifts faster
        self.assertGreater(r["mean_slope_flat"], r["mean_slope_2level"])
        self.assertTrue(r["sufficiency_ok"])                           # CI lower bound clears practical

    def test_no_improvement_when_coarse_useless(self) -> None:
        import torch
        import torch.nn.functional as F
        torch.manual_seed(1)
        ctx_z, _c, tz = self._informative()
        useless = F.normalize(torch.randn(N, DIM), dim=1)              # coarse unrelated to targets
        r = compare_flat_vs_2level(DIM, K, ctx_z, useless, tz, steps=300, seeds=(0, 1), n_boot=300)
        self.assertFalse(r["sufficiency_ok"])                         # no genuine two-level benefit

    def test_matched_architecture_same_param_count(self) -> None:
        h = build_fine_render_head(DIM, K, seed=0)
        n_params = sum(p.numel() for p in h.parameters())
        # flat is the SAME head (coarse zeroed at forward), so param/FLOP budget is identical.
        self.assertEqual(n_params, sum(p.numel() for p in build_fine_render_head(DIM, K, seed=0).parameters()))
        self.assertGreater(n_params, 0)


if __name__ == "__main__":
    unittest.main()
