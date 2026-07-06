"""Encode-empty v0B model tests (Pi R4 Q2): the frozen silence prototype +
occupancy/count heads. z_empty must be an immovable buffer, not a trainable
Parameter; target_latent must route empties to it; checkpoints must round-trip;
and old (encode_empty=False) checkpoints must still load strict."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.arms.v0b.mean_token_model import (  # noqa: E402
    MeanTokenJEPA,
    build_mean_token_jepa_from_checkpoint,
)

DIM = 16
VOCAB = 1051  # 1050 substrate + room; encode-empty needs no vocab bump but this is fine


@unittest.skipUnless(HAS_TORCH, "torch required")
class EmptyPrototypeTests(unittest.TestCase):
    def test_prototype_is_frozen_buffer_not_parameter(self) -> None:
        m = MeanTokenJEPA(VOCAB, DIM, encode_empty=True)
        buffers = dict(m.named_buffers())
        params = dict(m.named_parameters())
        self.assertIn("empty_prototype", buffers)
        self.assertNotIn("empty_prototype", params)           # not a Parameter
        self.assertFalse(buffers["empty_prototype"].requires_grad)

    def test_prototype_unit_norm_and_seeded_deterministic(self) -> None:
        import torch

        m1 = MeanTokenJEPA(VOCAB, DIM, encode_empty=True, empty_prototype_seed=123)
        m2 = MeanTokenJEPA(VOCAB, DIM, encode_empty=True, empty_prototype_seed=123)
        self.assertAlmostEqual(float(m1.empty_prototype.norm()), 1.0, places=5)
        self.assertTrue(torch.allclose(m1.empty_prototype, m2.empty_prototype))  # deterministic
        m1.reseed_empty_prototype(999)
        self.assertFalse(torch.allclose(m1.empty_prototype, m2.empty_prototype))
        self.assertAlmostEqual(float(m1.empty_prototype.norm()), 1.0, places=5)

    def test_prototype_excluded_from_optimizer(self) -> None:
        import torch

        m = MeanTokenJEPA(VOCAB, DIM, encode_empty=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        tracked = {id(p) for group in opt.param_groups for p in group["params"]}
        self.assertNotIn(id(m.empty_prototype), tracked)

    def test_target_latent_routes_empties_to_prototype(self) -> None:
        import torch

        m = MeanTokenJEPA(VOCAB, DIM, encode_empty=True)
        ids = torch.tensor([[1, 60, 60, 0], [1, 60, 0, 0]], dtype=torch.long)  # 2 populated rows
        is_empty = torch.tensor([True, False])
        z = m.target_latent(ids, is_empty)
        self.assertEqual(z.shape, (2, DIM))
        self.assertTrue(torch.allclose(z[0], m.empty_prototype.to(z.dtype)))    # empty -> prototype
        self.assertTrue(torch.allclose(z[1], m.mean_embed(ids[1:2])[0]))         # populated -> mean_embed
        # is_empty=None => plain mean_embed (no zero-vector NaN on all-pad handled elsewhere).
        self.assertTrue(torch.allclose(m.target_latent(ids, None), m.mean_embed(ids)))

    def test_occupancy_heads_shape(self) -> None:
        import torch

        m = MeanTokenJEPA(VOCAB, DIM, encode_empty=True)
        ctx = torch.randn(5, DIM)
        occ, cnt = m.predict_occupancy_from_latent(ctx, horizon_count=1)
        self.assertEqual(occ.shape, (5, 1, 1))
        self.assertEqual(cnt.shape, (5, 1, 1))

    def test_horizon_conditioned_has_per_horizon_heads(self) -> None:
        import torch

        m = MeanTokenJEPA(VOCAB, DIM, autoregression_mode="horizon_conditioned", max_horizons=3, encode_empty=True)
        self.assertEqual(len(m.occupancy_heads), 3)
        occ, cnt = m.predict_occupancy_from_latent(torch.randn(4, DIM), horizon_count=2)
        self.assertEqual(occ.shape, (4, 2, 1))

    def test_checkpoint_round_trip_preserves_prototype_and_heads(self) -> None:
        import torch

        m = MeanTokenJEPA(VOCAB, DIM, encode_empty=True, empty_prototype_seed=7)
        ckpt = {
            "model_state_dict": m.state_dict(),
            "vocab_size": VOCAB,
            "embedding_dim": DIM,
            "architecture": m.architecture_metadata(),
        }
        m2 = build_mean_token_jepa_from_checkpoint(ckpt)  # strict load
        self.assertTrue(m2.encode_empty)
        self.assertTrue(torch.allclose(m.empty_prototype, m2.empty_prototype))  # buffer round-trips

    def test_old_checkpoint_without_encode_empty_loads_strict(self) -> None:
        m_old = MeanTokenJEPA(VOCAB, DIM, encode_empty=False)  # pre-encode-empty architecture
        self.assertNotIn("empty_prototype", dict(m_old.named_buffers()))
        ckpt = {
            "model_state_dict": m_old.state_dict(),
            "vocab_size": VOCAB,
            "embedding_dim": DIM,
            "architecture": m_old.architecture_metadata(),
        }
        m2 = build_mean_token_jepa_from_checkpoint(ckpt)  # must not raise
        self.assertFalse(m2.encode_empty)


if __name__ == "__main__":
    unittest.main()
