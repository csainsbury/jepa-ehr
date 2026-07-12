"""Power / MDE + precision-simulation tests (Pi 2nd-pass Phase 4b). Safe-public / synthetic."""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_power as P
from clinical_jepa.eval.oracle_metrics import clopper_pearson_lower, clopper_pearson_upper
from clinical_jepa.eval.rung2_contract import ORACLE_POWER_FLOOR


class BinomialBoundTests(unittest.TestCase):
    def test_lower_upper_symmetry_and_ordering(self) -> None:
        lo, up = clopper_pearson_lower(18, 20), clopper_pearson_upper(18, 20)
        self.assertLess(lo, 18 / 20)
        self.assertGreater(up, 18 / 20)
        self.assertAlmostEqual(clopper_pearson_lower(20, 20), 1.0 - clopper_pearson_upper(0, 20), places=9)
        self.assertEqual(clopper_pearson_lower(0, 20), 0.0)


class CompletePassTests(unittest.TestCase):
    def test_pass_on_strong_positive_fail_on_null(self) -> None:
        self.assertTrue(P.complete_pass_decision(P._good_factory, "T_latent_factor", 0.75, seed=1))
        self.assertFalse(P.complete_pass_decision(P._good_factory, "T_latent_factor", 0.0, seed=1))


class PowerStudyTests(unittest.TestCase):
    def test_independent_seed_power_high_at_effect_low_at_null(self) -> None:
        hi = P.power_study(P._good_factory, "T_latent_factor", 0.75, n_seeds=25, n=400, ref_n=1500)
        self.assertEqual(hi.n_pass, hi.n_seeds)                 # every independent seed passes
        self.assertGreaterEqual(hi.lower_ci, ORACLE_POWER_FLOOR)  # exact one-sided lower bound clears floor
        self.assertTrue(hi.meets_floor)
        null = P.power_study(P._good_factory, "T_latent_factor", 0.0, n_seeds=25, n=400, ref_n=1500)
        self.assertFalse(null.meets_floor)                     # no power on a true null

    def test_mde_is_a_grid_point_no_interpolation(self) -> None:
        m = P.mde(P._good_factory, "T_latent_factor", (0.15, 0.35, 0.60), n_seeds=20, n=400)
        self.assertIn(m, (0.15, 0.35, 0.60))                   # a grid point, never interpolated


class PrecisionSimulationTests(unittest.TestCase):
    def test_evaluator_holds_type_I_and_power(self) -> None:
        pr = P.precision_simulation(n=300)                     # frozen study count
        self.assertLessEqual(pr.type_I_upper, 0.05)            # type-I bounded under alpha
        self.assertGreaterEqual(pr.power_lower, ORACLE_POWER_FLOOR)
        self.assertTrue(pr.mc_adequate)                        # enough studies to resolve the tolerance
        self.assertTrue(pr.passes)

    def test_too_few_studies_refuses_as_mc_inadequate(self) -> None:
        pr = P.precision_simulation(n_studies=15, n=300)       # too few to resolve -> fail-closed
        self.assertFalse(pr.mc_adequate)
        self.assertFalse(pr.passes)


if __name__ == "__main__":
    unittest.main()
