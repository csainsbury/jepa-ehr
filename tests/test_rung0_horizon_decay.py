"""Rung-0 retrieval-orchestration tests (Pi R5 C4/C5): patient-disjoint ranking,
2-channel paired streams, and the driver plumbing → 3-way verdict."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval.rung0_horizon_decay import build_manifest, evaluate_source
from clinical_jepa.eval.rung0_retrieval import frozen_group_key, rung0_rank
from clinical_jepa.eval.rung0_stats import paired_gap_streams, paired_slope_streams


def _index(n, patients, *, W=30.0, granularity="coarse", subwindow_k=None):
    return [{"block_id": f"b{i}", "patient_hash": patients[i], "source_dataset": "SCID",
             "split": "dev", "window_days": W, "target_type": "T0", "granularity": granularity,
             "subwindow_k": subwindow_k, "context_len": 20, "n_target_events": 5} for i in range(n)]


class RankerTests(unittest.TestCase):
    def test_patient_disjoint_excludes_same_patient(self) -> None:
        # 4 orthonormal targets; queries retrieve their own (rank 1). Patients A,A,B,B.
        emb = np.eye(4, dtype=np.float32)
        idx = _index(4, ["A", "A", "B", "B"])
        res = rung0_rank(emb, emb, idx, max_candidates=100, min_candidates=2, seed=0)
        recs = {r["patient"]: r for r in res["records"]}  # last per patient; check counts below
        self.assertEqual(res["n_ranked"], 4)
        for r in res["records"]:
            # a patient-A query sees only the 2 patient-B targets + its own true target.
            self.assertEqual(r["n_candidates"], 3)      # patient-disjoint (excludes the sibling)
            self.assertEqual(r["rank"], 1)              # query == true target

    def test_frozen_key_separates_granularity_and_W(self) -> None:
        a = {"source_dataset": "SCID", "split": "dev", "window_days": 30.0, "granularity": "coarse", "subwindow_k": None, "context_len": 20, "n_target_events": 5}
        self.assertNotEqual(frozen_group_key(a), frozen_group_key({**a, "window_days": 90.0}))
        self.assertNotEqual(frozen_group_key(a), frozen_group_key({**a, "granularity": "fine", "subwindow_k": 0}))


class StreamStatsTests(unittest.TestCase):
    def test_gap_streams_coarse_dominant(self) -> None:
        c = [{"patient": f"p{p}", "rank": 1} for p in range(10) for _ in range(3)]
        f = [{"patient": f"p{p}", "rank": 20} for p in range(10) for _ in range(6)]  # K× more, all miss
        g = paired_gap_streams(c, f, n_boot=300, seed=0)
        self.assertGreater(g["gap"], 0.5)
        self.assertGreater(g["ci_lo"], 0.0)

    def test_slope_streams_coarse_decays_slower(self) -> None:
        cby = {30.0: [{"patient": f"p{p}", "rank": 1} for p in range(12)],
               90.0: [{"patient": f"p{p}", "rank": 1} for p in range(12)]}     # coarse flat
        fby = {30.0: [{"patient": f"p{p}", "rank": 1} for p in range(12)],     # fine hits at 30
               90.0: [{"patient": f"p{p}", "rank": 30} for p in range(12)]}    # fine misses at 90
        s = paired_slope_streams(cby, fby, n_boot=200, seed=0)
        self.assertGreater(s["slope_diff_fine_minus_coarse"], 0.0)


class DriverTests(unittest.TestCase):
    def _cell(self, n, patients, *, W, granularity, hit=True, subwindow_k=None):
        emb = np.eye(max(n, 4), dtype=np.float32)[:n]
        queries = emb.copy() if hit else np.roll(emb, 1, axis=0)  # miss => query != own target
        return {"queries": queries, "targets": emb, "index": _index(n, patients, W=W, granularity=granularity, subwindow_k=subwindow_k)}

    def test_evaluate_source_runs_and_null_is_not_build(self) -> None:
        pats = ["A", "A", "B", "B"]
        per_W = {}
        for W in (30.0, 90.0):
            per_W[W] = {
                "coarse": self._cell(4, pats, W=W, granularity="coarse", hit=True),
                "fine": self._cell(4, pats, W=W, granularity="fine", hit=True, subwindow_k=0),   # fine also hits => gap ~0
                "coarse_B": self._cell(4, pats, W=W, granularity="coarse_B", hit=True),
                "fine_B": self._cell(4, pats, W=W, granularity="fine_B", hit=True, subwindow_k=0),
            }
        v = evaluate_source("SCID", per_W, level_horizons=[30.0, 90.0], n_boot=200,
                            adequacy_floor=2, raw_count_ok=True, sufficiency_ok=True)
        self.assertIn(v["decision"], ("NO-BUILD_INCONCLUSIVE", "NO-BUILD_EFFECT-RULED-OUT"))  # no coarse edge
        self.assertTrue(v["aggregate_only"])
        m = build_manifest([v])
        self.assertEqual(m["decisions"]["SCID"], v["decision"])
        self.assertIn("SCID", m["per_source"])


if __name__ == "__main__":
    unittest.main()
