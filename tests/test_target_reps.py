"""Rung-1 parameter-free target representation tests: dimensional empties, the frozen
time featurizer, temporal-slot structure, and the load-bearing arm-A permutation-pair
invariance (same multiset, altered order => bit-identical z+; Pi R8 #5)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

from clinical_jepa.targets.target_reps import ARM_NAMES, build_target_rep, target_dim, time_features
from clinical_jepa.eval.rung1_contract import D_TIME, M_PRIMARY

DIM = 16
VOCAB = 64


def _block(target_start, target_end, *, empty=False, t_query=10.0, W=90.0, context_end=4, n=None):
    return {
        "target_start_ref": -1 if empty else int(target_start),
        "target_end_ref": -1 if empty else int(target_end),
        "empty_target": bool(empty),
        "t_query": float(t_query),
        "window_days": float(W),
        "context_end_ref": int(context_end),
        "n_target_events": int(0 if empty else (n if n is not None else target_end - target_start + 1)),
    }


class TimeFeatureTests(unittest.TestCase):
    def test_shape_and_determinism(self) -> None:
        tau = np.array([0.0, 0.3, 0.9]); log_dt = np.array([0.0, 1.0, 2.0])
        f1 = time_features(tau, log_dt, D_TIME)
        f2 = time_features(tau, log_dt, D_TIME)
        self.assertEqual(f1.shape, (3, D_TIME))
        self.assertTrue(np.array_equal(f1, f2))


@unittest.skipUnless(HAS_TORCH, "torch required")
class RepTests(unittest.TestCase):
    def _model(self):
        from clinical_jepa.arms.v0b.mean_token_model import MeanTokenJEPA
        import torch
        torch.manual_seed(0)
        return MeanTokenJEPA(vocab_size=VOCAB, embedding_dim=DIM, encode_empty=True)

    def _seq(self, n=40):
        # token ids 1..VOCAB-1, cumulative_days monotone increasing
        rng = np.random.default_rng(0)
        ids = rng.integers(2, VOCAB, size=n).astype(np.int64)
        days = np.cumsum(rng.uniform(0.5, 5.0, size=n)).astype(np.float64)
        return ids, days

    def test_dims_per_arm(self) -> None:
        m = self._model(); ids, days = self._seq()
        blk = _block(5, 14, context_end=4, t_query=float(days[4]), W=200.0)
        for arm in ARM_NAMES:
            z = build_target_rep(arm, blk, ids, days, model=m, d_time=D_TIME, slots=M_PRIMARY)
            self.assertEqual(z.shape[0], target_dim(arm, DIM, d_time=D_TIME, slots=M_PRIMARY), arm)

    def test_dimensional_empties(self) -> None:
        m = self._model(); ids, days = self._seq()
        blk = _block(0, 0, empty=True)
        for arm in ARM_NAMES:
            z = build_target_rep(arm, blk, ids, days, model=m, slots=M_PRIMARY)
            self.assertEqual(z.shape[0], target_dim(arm, DIM, slots=M_PRIMARY), arm)
            self.assertTrue(np.all(np.isfinite(z)))

    def test_arm_a_permutation_pair_invariance(self) -> None:
        # THE contract invariant (Pi R8 #5): same multiset, altered order => bit-identical z+.
        m = self._model(); ids, days = self._seq()
        blk = _block(5, 14, context_end=4, t_query=float(days[4]), W=500.0)
        z1 = build_target_rep("mean_embed", blk, ids, days, model=m)
        ids_perm = ids.copy()
        seg = ids_perm[5:15]
        ids_perm[5:15] = seg[::-1]                                    # reverse target-span order
        z2 = build_target_rep("mean_embed", blk, ids_perm, days, model=m)
        self.assertTrue(np.array_equal(z1, z2))                      # bit-identical

    def test_count_concat_encodes_count(self) -> None:
        m = self._model(); ids, days = self._seq()
        b_small = _block(5, 7, context_end=4, n=3)
        b_big = _block(5, 24, context_end=4, n=20)
        z_small = build_target_rep("count_concat", b_small, ids, days, model=m)
        z_big = build_target_rep("count_concat", b_big, ids, days, model=m)
        self.assertAlmostEqual(z_small[-1], float(np.log1p(3)), places=5)
        self.assertAlmostEqual(z_big[-1], float(np.log1p(20)), places=5)

    def test_temporal_slot_width_is_M_times_D(self) -> None:
        m = self._model(); ids, days = self._seq()
        blk = _block(5, 20, context_end=4, t_query=float(days[4]), W=float(days[20] - days[4]) + 1)
        z4 = build_target_rep("temporal_slot", blk, ids, days, model=m, slots=4)
        z8 = build_target_rep("temporal_slot", blk, ids, days, model=m, slots=8)
        self.assertEqual(z4.shape[0], 4 * DIM)
        self.assertEqual(z8.shape[0], 8 * DIM)


if __name__ == "__main__":
    unittest.main()
