"""Load-bearing end-to-end oracle tests (Pi consolidated #7).

These run the FULL synthetic pipeline (generate -> reference bracket -> context-only evaluator) on
small fully-synthetic cells and assert the certification INVARIANTS that make the oracle trustworthy.
Everything is safe-public; no governed data, no sealed-cert splits, no manifest issuance.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_evaluator as E
from clinical_jepa.eval.oracle_generator import generate_cell
from clinical_jepa.eval.oracle_metrics import sequence_skill
from clinical_jepa.eval.oracle_spec import get_family

N = 600
KAPPA = 0.75          # strong-but-not-perfect coupling for the positive cells


def _cell(family_id: str, kappa: float = KAPPA, cell: str = "orthogonal", seed: int = 1):
    return generate_cell(get_family(family_id), kappa, cell, N, seed=seed)


class ContextPredictorRecoversOrderTests(unittest.TestCase):
    def test_context_predictor_beats_gate_on_positive_hfamily(self) -> None:
        cell = _cell("T_latent_factor")
        rb = E.reference_bracket(cell)
        # E-O1: fair context ceiling clears the PRACTICAL gate, not merely > 0 (Pi #3).
        self.assertGreaterEqual(rb.R_bayes_pos.lower_ci, E.ORACLE_EO1_SKILL_GATE)
        self.assertTrue(rb.R_bayes_beats_R0)


class R0FloorTests(unittest.TestCase):
    def test_R0_passes_null_and_fails_positive(self) -> None:
        cell = _cell("T_hmm_markov")
        rb = E.reference_bracket(cell)
        self.assertTrue(rb.R0_null_pass)         # content-prior does not fire on nulls
        self.assertTrue(rb.R0_positive_fail)     # content-prior alone cannot explain positives
        # the certifiable signal is CONTEXT-conditional: R_bayes >> R0 on positives.
        self.assertGreater(rb.R_bayes_pos.mean_skill, rb.R0_pos.upper_ci)


class NuisanceBracketTests(unittest.TestCase):
    def test_nuis_loses_incremental_in_orthogonal_captures_leak(self) -> None:
        orth = _cell("T_latent_factor", cell="orthogonal", seed=2)
        leak = _cell("T_latent_factor", cell="correlated_leak", seed=3)
        self.assertTrue(E.nuisance_incremental_ok(orth, leak))
        # and the leak never beats the fair context ceiling
        rb_leak = E.reference_bracket(leak)
        nu_leak = E._skill(E.predict_nuisance(leak), leak, want_null=False, seed=99)
        self.assertLessEqual(nu_leak.mean_skill, rb_leak.R_bayes_pos.upper_ci + 1e-9)


class ContextBlindMustFailTests(unittest.TestCase):
    def test_perfect_coupling_but_context_blind_gets_no_skill(self) -> None:
        cell = _cell("T_latent_factor")
        blind = E.predict_context_blind(cell)
        r = E._skill(blind, cell, want_null=False, seed=7)
        self.assertLess(r.upper_ci, E.ORACLE_EO1_SKILL_GATE)   # blinded context => below the gate
        self.assertFalse(r.fires)


class LabelPerturbationInvarianceTests(unittest.TestCase):
    def test_context_predictor_ignores_the_label(self) -> None:
        cell = _cell("T_hmm_markov")
        before = E.predict_context(cell)
        # perturb the FUTURE/label: shuffle s_true and flip nuisance — evaluator inputs are x_ctx/f only.
        from dataclasses import replace
        rng = np.random.default_rng(0)
        perturbed = replace(cell, s_true=rng.standard_normal(cell.s_true.shape), u=-cell.u)
        after = E.predict_context(perturbed)
        self.assertTrue(np.allclose(before, after))            # predictions invariant to label change


class ShortcutFailsNoHTests(unittest.TestCase):
    def test_h_projection_shortcut_fails_no_h_family(self) -> None:
        no_h = _cell("E_no_h_exogenous")
        self.assertTrue(E.shortcut_fails_no_h(no_h))           # shortcut skill <= ORACLE_SHORTCUT_MAX_SKILL
        # but a PROPER context predictor still recovers order on the no-h family (order IS in context).
        rb = E.reference_bracket(no_h)
        self.assertGreater(rb.R_bayes_pos.mean_skill, E.ORACLE_SHORTCUT_MAX_SKILL)


class NullFiringRateTests(unittest.TestCase):
    def test_context_predictor_does_not_fire_on_nulls(self) -> None:
        cell = _cell("T_latent_factor")
        rb = E.reference_bracket(cell)
        self.assertFalse(rb.R_bayes_null.fires)                # sequence-level null does not fire
        self.assertLess(rb.R_bayes_null.upper_ci, E.ORACLE_EO1_SKILL_GATE)


class SequenceNullStatisticTests(unittest.TestCase):
    def test_min_pairs_and_single_decision_per_sequence(self) -> None:
        # a sequence with too few eligible pairs does not contribute.
        pred = [np.array([1.0, 2.0])]        # 1 pair < ORACLE_NULL_MIN_PAIRS
        true = [np.array([1.0, 2.0])]
        r = sequence_skill(pred, true)
        self.assertEqual(r.n_contributing, 0)
        self.assertFalse(r.fires)


if __name__ == "__main__":
    unittest.main()
