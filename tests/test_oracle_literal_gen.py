"""Falsify the literal families against the frozen discriminated schema (Pi 2nd-pass Phase 2)."""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval import oracle_contracts as OC
from clinical_jepa.eval.oracle_literal_gen import (
    N_CLASSES, generate_literal_cell, _MECHANISMS,
)
from clinical_jepa.eval.oracle_spec import no_h_families
from clinical_jepa.eval.rung2_contract import ORDER_SUPPORT_FLOOR

FAMILIES = tuple(_MECHANISMS)


class SchemaTests(unittest.TestCase):
    def test_every_family_emits_the_common_core(self) -> None:
        for fam in FAMILIES:
            c = generate_literal_cell(fam, 0.5, "orthogonal", 40, seed=1)
            self.assertEqual(c.context_features.shape[0], 40)
            self.assertEqual(c.item_features.shape[1], c.true_order.shape[1])
            self.assertEqual(c.future_multiset.shape, c.true_order.shape)
            self.assertTrue(((c.future_multiset >= 0) & (c.future_multiset < N_CLASSES)).all())
            self.assertEqual(c.nuisance_u.shape, c.true_order.shape)
            self.assertTrue(c.mechanism_params_hash)

    def test_context_view_exposes_only_allowlisted_channels(self) -> None:
        c = generate_literal_cell("T_latent_factor", 0.5, "orthogonal", 20, seed=2)
        view = c.context_view()
        for ch in c.observable_allowlist:
            view.get(ch)                                  # allowed
        with self.assertRaises(OC.CapabilityError):
            view.get("true_order")                        # label never in context
        with self.assertRaises(OC.CapabilityError):
            view.get("family_id")                         # family identity never in context


class StructuralDifferenceTests(unittest.TestCase):
    # families whose order is driven by a genuine HIDDEN common cause h vs an OBSERVABLE state:
    HIDDEN_H = {"T_hmm_markov", "T_latent_factor", "E_offgrid_nonlinear"}
    OBSERVABLE_STATE = {"E_no_h_exogenous", "T_realized_history"}   # exogenous clock / realized prefix

    def test_hidden_vs_observable_state_by_family(self) -> None:
        no_h_ids = {f.family_id for f in no_h_families()}
        self.assertIn("E_no_h_exogenous", no_h_ids)
        for fam in FAMILIES:
            c = generate_literal_cell(fam, 0.5, "orthogonal", 30, seed=3)
            if fam in self.HIDDEN_H:
                self.assertIsNotNone(c.hidden_state)                 # a genuine hidden common cause
            else:
                self.assertIn(fam, self.OBSERVABLE_STATE)
                self.assertIsNone(c.hidden_state)                    # state is observable, not hidden
                self.assertIsNotNone(c.observed_covariates)          # exposed as a context covariate
                self.assertIn("observed_covariates", c.observable_allowlist)

    def test_hmm_terminal_state_is_one_hot(self) -> None:
        c = generate_literal_cell("T_hmm_markov", 0.5, "orthogonal", 50, seed=4)
        rowsums = c.hidden_state.sum(1)
        self.assertTrue(np.allclose(rowsums, 1.0))                    # one-hot terminal state
        self.assertTrue(set(np.unique(c.hidden_state)).issubset({0.0, 1.0}))

    def test_offgrid_driver_is_heavier_tailed_than_gaussian_factor(self) -> None:
        off = generate_literal_cell("E_offgrid_nonlinear", 0.5, "orthogonal", 4000, seed=5)
        lin = generate_literal_cell("T_latent_factor", 0.5, "orthogonal", 4000, seed=5)

        def excess_kurt(x):
            z = (x - x.mean()) / (x.std() + 1e-9)
            return float((z ** 4).mean() - 3.0)
        self.assertGreater(excess_kurt(off.hidden_state), excess_kurt(lin.hidden_state) + 1.0)


class TimingTests(unittest.TestCase):
    def test_zero_gap_multiplicity_and_positive_inter_cluster_gaps(self) -> None:
        c = generate_literal_cell("T_realized_history", 0.5, "orthogonal", 300, seed=6)
        # some clusters have multiplicity > 1 (Δt=0 simultaneity)
        self.assertTrue((c.multiplicity > 1).any())
        # timestamps nondecreasing; every strictly-positive step corresponds to a new cluster
        dt = np.diff(c.future_timestamps, axis=1)
        self.assertTrue((dt >= -1e-9).all())
        new_cluster = np.diff(c.cluster_ids, axis=1) > 0
        self.assertTrue((dt[new_cluster] > 0).all())                 # inter-cluster gaps strictly positive
        self.assertTrue((np.abs(dt[~new_cluster]) < 1e-9).all())     # within-cluster gaps are Δt=0


class SupportTests(unittest.TestCase):
    def test_support_starved_cell_is_tagged(self) -> None:
        c = generate_literal_cell("T_latent_factor", 0.5, "orthogonal", 5000, seed=7, support_starved=True)
        self.assertEqual(c.support_status, "SUPPORT_STARVED")
        self.assertLess(c.context_features.shape[0], ORDER_SUPPORT_FLOOR)
        ok = generate_literal_cell("T_latent_factor", 0.5, "orthogonal", ORDER_SUPPORT_FLOOR + 10, seed=7)
        self.assertEqual(ok.support_status, "SUPPORTED")


class ProvenanceTests(unittest.TestCase):
    def test_mechanism_hash_is_family_and_kappa_sensitive(self) -> None:
        a = generate_literal_cell("T_hmm_markov", 0.5, "orthogonal", 20, seed=8).mechanism_params_hash
        b = generate_literal_cell("T_hmm_markov", 0.5, "orthogonal", 20, seed=9).mechanism_params_hash
        c = generate_literal_cell("T_hmm_markov", 0.75, "orthogonal", 20, seed=8).mechanism_params_hash
        d = generate_literal_cell("T_latent_factor", 0.5, "orthogonal", 20, seed=8).mechanism_params_hash
        self.assertEqual(a, b)               # params frozen per family, independent of the draw seed
        self.assertNotEqual(a, c)            # kappa-sensitive
        self.assertNotEqual(a, d)            # family-sensitive


if __name__ == "__main__":
    unittest.main()
