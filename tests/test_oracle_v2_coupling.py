"""Step-3 (rebuild) — exact coupling-construction invariants (Pi binding condition 2).

Tests the EXACT invariants each coupling preserves BY CONSTRUCTION directly, plus s==0 no-op and s>0 change.
Directional movement of the intended verifier check is covered in test_oracle_v2_coupling_direction.py (slower).
"""
from __future__ import annotations

from math import log

import numpy as np
import unittest

from clinical_jepa.eval import oracle_realism_v2_fixture as fx
from clinical_jepa.eval import oracle_realism_v2_coupling as cp

_COUPLING_IMPL_ID = "839204245178abe069b6d4f959600054d4474d2cf71df809fb1e9fae695ee85c"
_PROF = {"length": {"family": "discretized_lognormal", "mu": log(120), "sigma": 0.9, "min": 1},
         "class_prior": [0.3, 0.25, 0.2, 0.15, 0.1], "structural_zero_classes": [],
         "cluster_size": {"family": "geometric", "p": 0.5},
         "gap": {"family": "lognormal", "mu": log(1.2), "sigma": 0.85}, "dependence": {}}


def _pooled_L(s): return np.sort([r.L_total for r in s])
def _pooled_K(s): return np.sort([r.K for r in s])
def _pooled_class(s):
    c = np.zeros(5, int)
    for r in s:
        c += np.bincount(r.class_ids, minlength=5)
    return c
def _seq_counts(s): return [tuple(np.bincount(r.class_ids, minlength=5)) for r in s]
def _seq_runs(s): return [tuple(np.sort(np.bincount(r.cluster_ids))) for r in s]
def _seq_gaps(s):
    out = []
    for r in s:
        fi = cp._first_indices(r)
        out.append(tuple(np.round(np.sort(np.diff(r.timestamps[fi])), 8)) if r.K > 1 else ())
    return out
def _changed(a, b):
    return any((not np.array_equal(x.class_ids, y.class_ids)) or (not np.allclose(x.timestamps, y.timestamps))
               for x, y in zip(a, b))


class CouplingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = fx.sample_fixture("MIMIC", _PROF, 800, seed=5)

    def test_noop_at_zero(self) -> None:
        for comp in cp.V2_D_COMPONENT_MENU:
            out = cp.apply_coupling(self.s, comp, 0.0, seed=1)
            self.assertEqual(_seq_counts(out), _seq_counts(self.s), comp)
            self.assertFalse(_changed(out, self.s), f"{comp} must be a no-op at s=0")

    def test_changes_at_half(self) -> None:
        for comp in cp.V2_D_COMPONENT_MENU:
            self.assertTrue(_changed(cp.apply_coupling(self.s, comp, 0.5, seed=1), self.s), comp)

    def test_burst_count_length_rejected_as_D_component(self) -> None:  # Pi F1: dropped from active D
        from clinical_jepa.eval.oracle_realism_v2 import REJECTED_D_COMPONENTS
        self.assertIn("burst_count_length", REJECTED_D_COMPONENTS)
        self.assertNotIn("burst_count_length", cp.V2_D_COMPONENT_MENU)
        with self.assertRaises(KeyError):
            cp.apply_coupling(self.s, "burst_count_length", 0.5, seed=1)

    def test_burst_timing_preserves_per_sequence_gap_multiset(self) -> None:
        out = cp.apply_coupling(self.s, "burst_timing", 0.5, seed=1)
        self.assertEqual(_seq_gaps(out), _seq_gaps(self.s))

    def test_mark_burst_tie_preserves_per_sequence_class_counts(self) -> None:
        out = cp.apply_coupling(self.s, "mark_burst_tie", 0.5, seed=1)
        self.assertEqual(_seq_counts(out), _seq_counts(self.s))

    def test_diversity_preserves_counts_and_cluster_sizes(self) -> None:
        out = cp.apply_coupling(self.s, "cluster_size_mark_diversity", 0.5, seed=1)
        self.assertEqual(_seq_counts(out), _seq_counts(self.s))
        self.assertEqual(_seq_runs(out), _seq_runs(self.s))

    def test_length_class_mix_preserves_pooled_class_counts(self) -> None:
        out = cp.apply_coupling(self.s, "length_class_mix", 0.5, seed=1)
        self.assertTrue(np.array_equal(_pooled_class(out), _pooled_class(self.s)))

    def test_rejects_bad_component_and_strength(self) -> None:
        with self.assertRaises(KeyError):
            cp.apply_coupling(self.s, "nope", 0.3, seed=1)
        with self.assertRaises(ValueError):
            cp.apply_coupling(self.s, "burst_timing", 0.9, seed=1)      # out of [0,0.6]

    def test_impl_identity_pinned(self) -> None:
        self.assertEqual(cp.coupling_impl_identity(), _COUPLING_IMPL_ID)


if __name__ == "__main__":
    unittest.main()
