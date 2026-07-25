"""The rank benchmark must compare the production function to a GENUINELY quadratic reference.

Pi rev-25 caught this: the benchmark timed `cell_upper_p` as its "reference" and labelled the result
quadratic / 400 MB. That was true until ruling 4 made `cell_upper_p` the sorted form — after which the
benchmark compared the production function to ITSELF and still published a 400 MB attribution. The run
behind config identity 2e68817d reported `speedup 1.0x` alongside `400.0 MB transient`, which is
self-evidently contradictory, and it shipped because nobody read the rank ladder.

These tests pin the invariant that would have caught it: the two sides must be DIFFERENT algorithms with
different asymptotics, and the reported speedup must actually grow with A.
"""
from __future__ import annotations

import unittest

import numpy as np

from scripts.oracle_realism_v3_randomization import _cell_upper_p_quadratic, cell_upper_p


class TestRankReferenceIsGenuinelyQuadratic(unittest.TestCase):
    def test_production_and_reference_are_distinct_functions(self):
        """If these are ever the same object, the benchmark is timing one algorithm twice."""
        self.assertIsNot(cell_upper_p, _cell_upper_p_quadratic)

    def test_they_agree_exactly(self):
        """Distinct implementations, identical output — that is what makes the oracle usable."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            e = np.abs(rng.normal(size=int(rng.integers(2, 80))))
            np.testing.assert_array_equal(cell_upper_p(e), _cell_upper_p_quadratic(e))

    def test_reference_allocates_quadratically_and_production_does_not(self):
        """The 400 MB attribution belongs to the reference; the production path must not pay it."""
        peaks = {}
        for name, fn in (("reference", _cell_upper_p_quadratic), ("production", cell_upper_p)):
            e = np.abs(np.random.default_rng(5).normal(size=4000))
            import tracemalloc
            tracemalloc.start()
            fn(e)
            peaks[name] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        # A x A bytes for A=4000 is ~16 MB; the sorted path is a few tens of KB.
        self.assertGreater(peaks["reference"], 4_000_000)
        self.assertLess(peaks["production"], 1_000_000)
        self.assertGreater(peaks["reference"], 10 * peaks["production"])

    def test_speedup_grows_with_A(self):
        """A quadratic-vs-linearithmic comparison must widen with A. A flat ~1.0x ratio is the exact
        signature of the bug this guards: the benchmark timing the same function on both sides."""
        import time

        def secs(fn, e, reps):
            fn(e)
            t = time.perf_counter()
            for _ in range(reps):
                fn(e)
            return (time.perf_counter() - t) / reps

        ratios = []
        for A, reps in ((1000, 20), (4000, 5)):
            e = np.abs(np.random.default_rng(7).normal(size=A))
            ratios.append(secs(_cell_upper_p_quadratic, e, reps) / max(secs(cell_upper_p, e, reps), 1e-12))
        self.assertGreater(ratios[0], 2.0, f"no speedup at A=1000 ({ratios[0]:.2f}x) — same algorithm?")
        self.assertGreater(ratios[1], ratios[0], f"speedup did not grow with A: {ratios}")


if __name__ == "__main__":
    unittest.main()
