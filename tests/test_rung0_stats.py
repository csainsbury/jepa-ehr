"""Rung-0 paired-inference + decision tests (Pi R5 C5/C7)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval.rung0_stats import (
    assert_k1_null,
    decision,
    paired_bootstrap_gap,
    paired_bootstrap_slope,
    target_geometry,
)


def _recs(coarse_ranks, fine_ranks, n_pat=10, per=5):
    recs = []
    i = 0
    for p in range(n_pat):
        for _ in range(per):
            recs.append({"patient": f"p{p}", "coarse_rank": coarse_ranks[i % len(coarse_ranks)],
                         "fine_rank": fine_ranks[i % len(fine_ranks)]})
            i += 1
    return recs


class PairedGapTests(unittest.TestCase):
    def test_coarse_beats_fine_ci_excludes_zero(self) -> None:
        # coarse always rank 1 (hit@10), fine alternates 1 / 20 -> gap ~0.5.
        g = paired_bootstrap_gap(_recs([1], [1, 20]), n_boot=500, seed=0)
        self.assertGreater(g["gap"], 0.10)
        self.assertGreater(g["ci_lo"], 0.0)          # paired CI excludes 0

    def test_null_gap_zero(self) -> None:
        recs = [{"patient": f"p{p}", "coarse_rank": 3, "fine_rank": 3} for p in range(20) for _ in range(3)]
        g = paired_bootstrap_gap(recs, n_boot=200, seed=0)
        self.assertAlmostEqual(g["gap"], 0.0, places=9)
        assert_k1_null(g)                             # harness assertion passes


class PairedSlopeTests(unittest.TestCase):
    def test_coarse_decays_slower_positive_diff(self) -> None:
        # coarse R@10 flat across W; fine R@10 drops -> fine decays faster -> diff>0.
        rbw = {}
        for W, fr in ((30.0, 1), (90.0, 1), (365.0, 12), (730.0, 30)):
            # coarse rank 1 always; fine rank grows with W (worse retrieval)
            rbw[W] = [{"patient": f"p{p}", "coarse_rank": 1, "fine_rank": fr} for p in range(12) for _ in range(4)]
        s = paired_bootstrap_slope(rbw, n_boot=300, seed=0)
        self.assertGreater(s["slope_diff_fine_minus_coarse"], 0.0)   # coarse decays slower
        self.assertGreater(s["implied_range_widening"], 0.0)


class DecisionTests(unittest.TestCase):
    def _clear(self):  # a clearly-passing gap dict
        return {"gap": 0.3, "ci_lo": 0.2, "ci_hi": 0.4}

    def test_build_when_all_pass(self) -> None:
        d = decision(level_gap=self._clear(), coarse_b_gap=self._clear(),
                     slope={"ci_lo": 0.1, "ci_hi": 0.3}, raw_count_ok=True, veto=False,
                     sufficiency_ok=True, adequate=True)
        self.assertEqual(d["decision"], "BUILD")

    def test_inconclusive_when_underpowered(self) -> None:
        d = decision(level_gap=self._clear(), coarse_b_gap=self._clear(),
                     slope={"ci_lo": 0.1, "ci_hi": 0.3}, raw_count_ok=True, veto=False,
                     sufficiency_ok=True, adequate=False)
        self.assertEqual(d["decision"], "NO-BUILD_INCONCLUSIVE")

    def test_inconclusive_when_ci_spans_threshold(self) -> None:
        # coarse_b CI includes the practical level -> not ruled out, inconclusive.
        d = decision(level_gap=self._clear(), coarse_b_gap={"gap": 0.08, "ci_lo": 0.02, "ci_hi": 0.15},
                     slope={"ci_lo": -0.01, "ci_hi": 0.2}, raw_count_ok=True, veto=False,
                     sufficiency_ok=True, adequate=True)
        self.assertEqual(d["decision"], "NO-BUILD_INCONCLUSIVE")

    def test_effect_ruled_out(self) -> None:
        d = decision(level_gap={"gap": 0.01, "ci_lo": -0.02, "ci_hi": 0.03},
                     coarse_b_gap={"gap": 0.01, "ci_lo": -0.02, "ci_hi": 0.04},  # ci_hi < 0.10
                     slope={"ci_lo": -0.1, "ci_hi": 0.02},                        # ci_hi < 0.05
                     raw_count_ok=True, veto=False, sufficiency_ok=True, adequate=True)
        self.assertEqual(d["decision"], "NO-BUILD_EFFECT-RULED-OUT")

    def test_ruled_out_requires_all_coprimary_cells(self) -> None:
        # Worst co-primary cell rules out (ci_hi<0.10) + slope ruled out, BUT another
        # co-primary cell still shows a real effect -> coarse_b_ruled_out=False ->
        # INCONCLUSIVE, not EFFECT-RULED-OUT (verification-found gate-logic fix).
        base = dict(level_gap={"gap": 0.01, "ci_lo": -0.02, "ci_hi": 0.03},
                    coarse_b_gap={"gap": 0.01, "ci_lo": -0.02, "ci_hi": 0.04},  # worst cell
                    slope={"ci_lo": -0.1, "ci_hi": 0.02}, raw_count_ok=True, veto=False,
                    sufficiency_ok=True, adequate=True)
        self.assertEqual(decision(**base, coarse_b_ruled_out=False)["decision"], "NO-BUILD_INCONCLUSIVE")
        self.assertEqual(decision(**base, coarse_b_ruled_out=True)["decision"], "NO-BUILD_EFFECT-RULED-OUT")

    def test_veto_blocks_build(self) -> None:
        d = decision(level_gap=self._clear(), coarse_b_gap=self._clear(),
                     slope={"ci_lo": 0.1, "ci_hi": 0.3}, raw_count_ok=True, veto=True,
                     sufficiency_ok=True, adequate=True)
        self.assertNotEqual(d["decision"], "BUILD")

    def test_k1_null_raises_on_nonzero(self) -> None:
        with self.assertRaises(AssertionError):
            assert_k1_null({"gap": 0.2})


class TargetGeometryTests(unittest.TestCase):
    def test_duplicate_and_effective_rank(self) -> None:
        import numpy as np
        x = np.eye(4)
        g = target_geometry(np.vstack([x, x[0]]))    # one duplicate row
        self.assertGreater(g["duplicate_rate"], 0.0)
        self.assertGreater(g["effective_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
