"""Rung-2 frozen order-target tests (T1-T3): dims, no-silent-truncation flag + bit accounting,
dimensional empties, and that T2 seq-of-latents preserves order (unlike the pooled T1)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.targets.order_targets import (
    ORDER_TARGET_NAMES, build_order_target, order_target_dim, rank_code,
)

D = 12
V = 40


class RankCodeTests(unittest.TestCase):
    def test_deterministic_shape(self) -> None:
        a = rank_code(np.arange(5)); b = rank_code(np.arange(5))
        self.assertEqual(a.shape, (5, 8))
        self.assertTrue(np.array_equal(a, b))


class OrderTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.E = np.random.default_rng(0).standard_normal((V, D)).astype(np.float32)
        self.z_empty = np.zeros(D, dtype=np.float32)

    def test_dims_and_empty(self) -> None:
        ids = np.array([3, 7, 9, 2], dtype=np.int64)
        for name in ORDER_TARGET_NAMES:
            z, meta = build_order_target(name, ids, E=self.E, z_empty=self.z_empty)
            self.assertEqual(z.shape[0], order_target_dim(name, D), name)
            ze, m0 = build_order_target(name, np.array([], dtype=np.int64), E=self.E, z_empty=self.z_empty)
            self.assertEqual(ze.shape[0], order_target_dim(name, D), name + " empty")
            self.assertTrue(m0["empty"])

    def test_no_silent_truncation_flag_and_bits(self) -> None:
        long = np.arange(1, 25, dtype=np.int64)              # 24 events > L_max 16
        z, meta = build_order_target("T2_seq_of_latents", long, E=self.E, z_empty=self.z_empty)
        self.assertTrue(meta["truncated"])                   # flag set, not silent
        self.assertEqual(meta["L_used"], 16)
        self.assertEqual(meta["bits"], z.shape[0] * 32)      # explicit bit accounting

    def test_T2_preserves_order_T1_does_not(self) -> None:
        ids = np.array([3, 7, 9, 2], dtype=np.int64)
        rev = ids[::-1].copy()
        z2, _ = build_order_target("T2_seq_of_latents", ids, E=self.E, z_empty=self.z_empty)
        z2r, _ = build_order_target("T2_seq_of_latents", rev, E=self.E, z_empty=self.z_empty)
        self.assertFalse(np.allclose(z2, z2r))               # seq-of-latents encodes order
        z1, _ = build_order_target("T1_pooled_ordinal", ids, E=self.E, z_empty=self.z_empty)
        z1r, _ = build_order_target("T1_pooled_ordinal", rev, E=self.E, z_empty=self.z_empty)
        # T1's mean-E block is order-blind; the ordinal tag/moment carry (weak) order -> differ,
        # but the first D dims (pooled mean) are identical.
        self.assertTrue(np.allclose(z1[:D], z1r[:D]))


if __name__ == "__main__":
    unittest.main()
