"""Sub-window partition tests (rung 0 C6): near-fixed-width δ carving is exact and
the sub-window event counts sum to the full-W count (a TARGET-side identity)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.targets.subwindow_blocks import (
    annotate_block_subwindows,
    carve_subwindows,
    resolve_partition,
)


class ResolvePartitionTests(unittest.TestCase):
    def test_fixed_width_delta_exact_no_remainder(self) -> None:
        # 730d / 30d -> K=round(24.33)=24, w=730/24 (no 10-day remainder cell).
        K, w = resolve_partition(730.0, 30.0, "fixed_width_delta")
        self.assertEqual(K, 24)
        self.assertAlmostEqual(w * K, 730.0, places=6)          # tiles W exactly
        self.assertAlmostEqual(w, 730.0 / 24, places=6)
        # 365/30 -> round(12.17)=12
        self.assertEqual(resolve_partition(365.0, 30.0, "fixed_width_delta")[0], 12)

    def test_modes(self) -> None:
        self.assertEqual(resolve_partition(90.0, 30.0, "k1_null"), (1, 90.0))
        self.assertEqual(resolve_partition(90.0, 30.0, "k2_proportional"), (2, 45.0))
        self.assertEqual(resolve_partition(90.0, 30.0, "k4"), (4, 22.5))


class CarveTests(unittest.TestCase):
    def test_subwindows_partition_and_counts_sum(self) -> None:
        cdays = np.arange(30, dtype=np.float32)     # days 0..29
        # context_end=12 (day 12), t_query=12, W=6 -> [12,18): days 13..17 = 5 events.
        K, w = resolve_partition(6.0, 2.0, "fixed_width_delta")  # K=3, w=2
        subs = carve_subwindows(cdays, 30, 12, 12.0, K, w)
        self.assertEqual(K, 3)
        counts = [s["n_target_events"] for s in subs]
        self.assertEqual(sum(counts), 5)            # sums to the full-W count
        self.assertEqual(counts, [1, 2, 2])         # [13] / [14,15] / [16,17]
        self.assertEqual([s["subwindow_k"] for s in subs], [0, 1, 2])
        self.assertFalse(any(s["empty_target"] for s in subs))

    def test_all_empty_window(self) -> None:
        # Sparse days: after context (day 12) the next event is far beyond W.
        cdays = np.array(list(range(13)) + [500.0] * 17, dtype=np.float32)
        K, w = resolve_partition(4.0, 2.0, "fixed_width_delta")  # K=2, w=2
        subs = carve_subwindows(cdays, 30, 12, 12.0, K, w)
        self.assertTrue(all(s["empty_target"] for s in subs))
        self.assertTrue(all(s["n_target_events"] == 0 for s in subs))

    def test_annotate_records_delta_deviation(self) -> None:
        cdays = np.arange(40, dtype=np.float32)
        block = {"window_days": 730.0, "context_end_ref": 12, "t_query": 12.0, "n_target_events": 27}
        ann = annotate_block_subwindows(block, cdays, delta_days=30.0, seq_len=40)
        self.assertEqual(ann["K"], 24)
        self.assertAlmostEqual(ann["w_minus_delta"], 730.0 / 24 - 30.0, places=6)
        self.assertEqual(ann["partition_mode"], "fixed_width_delta")


if __name__ == "__main__":
    unittest.main()
