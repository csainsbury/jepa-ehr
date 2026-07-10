"""Rung-1 driver end-to-end (synthetic): the pipeline runs, count is decodable when the
latent carries it (count_concat), the incumbent order verdict is forced to content-prior,
and the manifest is well-formed + test-sealed (Pi R7/R8)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from clinical_jepa.eval.rung1_ceiling import count_row, run_ceiling

D = 12
V = 40


def _bundle(arm, *, n_pat=60, per=4, count_in_z, seed=0):
    rng = np.random.default_rng(seed)
    z, counts, patients, dt_lists, ordered_ids = [], [], [], [], []
    for p in range(n_pat):
        for _ in range(per):
            n = int(rng.integers(1, 6))
            base = rng.normal(size=D).astype(np.float32)
            if count_in_z:
                base[-1] = np.float32(np.log1p(n))          # count linearly present
            z.append(base)
            counts.append(n)
            patients.append(f"p{p}")
            dt_lists.append(rng.exponential(1.0, size=max(n - 1, 0)))
            ordered_ids.append(rng.integers(1, V, size=n))
    return {"z": np.asarray(z), "counts": np.asarray(counts), "patients": np.asarray(patients),
            "dt_lists": dt_lists, "ordered_ids": [np.asarray(o) for o in ordered_ids]}


@unittest.skipUnless(HAS_TORCH, "torch required")
class DriverTests(unittest.TestCase):
    def test_count_decodable_when_latent_carries_it(self) -> None:
        tr = _bundle("count_concat", count_in_z=True, seed=1)
        dev = _bundle("count_concat", count_in_z=True, seed=2)
        row = count_row("count_concat", "SCID", 90.0, tr, dev, embedding_dim=D, floor=30, n_boot=200)
        self.assertTrue(row["evaluable"])
        self.assertTrue(row["m2_gate_ok"])                  # exact-count clears 0.80
        self.assertGreater(row["m2_excess_lo"], 0.10)       # own swap-excess clears the margin
        self.assertTrue(row["m2_copy_ok"])

    def test_count_not_decodable_when_latent_lacks_it(self) -> None:
        tr = _bundle("mean_embed", count_in_z=False, seed=3)
        dev = _bundle("mean_embed", count_in_z=False, seed=4)
        row = count_row("mean_embed", "SCID", 90.0, tr, dev, embedding_dim=D, floor=30, n_boot=200)
        self.assertTrue(row["evaluable"])
        self.assertFalse(row["m2_gate_ok"])                 # count not recoverable from noise

    def test_pipeline_manifest_sealed_and_scoped(self) -> None:
        E = np.random.default_rng(0).normal(size=(V, D)).astype(np.float32)
        bundles = {}
        for arm in ("mean_embed", "count_concat"):
            bundles[(arm, "SCID", 90.0)] = {"train": _bundle(arm, count_in_z=(arm == "count_concat"), seed=10),
                                            "dev": _bundle(arm, count_in_z=(arm == "count_concat"), seed=11)}
        m = run_ceiling(bundles, embedding_dim=D, E=E, arms=["mean_embed", "count_concat"],
                        properties=("count", "order"), run_config={"SCID": {"horizons": [90]}},
                        count_floor=30, order_floor=30, timing_cluster_floor=30, timing_interval_floor=30)
        self.assertFalse(m["test_access"])                  # test sealed
        self.assertEqual(len(m["config_hash"]), 64)
        self.assertEqual(m["rung1a"]["arm"], "mean_embed")
        # incumbent order verdict is forced to content-prior regardless of raw score
        self.assertEqual(m["rung1a"]["properties"]["order"]["verdict"],
                         "STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY")
        self.assertFalse(m["rung1a"]["properties"]["order"]["can_nominate"])
        # count_concat should nominate count (decodable, 1b, direct scope)
        noms = [(n["arm"], n["property"]) for n in m["nominations"]]
        self.assertIn(("count_concat", "count"), noms)


if __name__ == "__main__":
    unittest.main()
