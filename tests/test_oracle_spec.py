"""Tests for the FROZEN oracle spec (Pi #5/#6). Pure-spec invariants; no generation, no data."""
from __future__ import annotations

import unittest

import clinical_jepa.eval.oracle_spec as S
from clinical_jepa.eval import rung2_contract as C


class FamilyInventoryTests(unittest.TestCase):
    def test_at_least_three_train_two_heldout(self) -> None:
        self.assertGreaterEqual(len(S.train_families()), 3)
        self.assertGreaterEqual(len(S.heldout_families()), C.ORACLE_N_HELDOUT_FAMILIES)

    def test_exactly_the_expected_no_h_family_exists(self) -> None:
        no_h = S.no_h_families()
        self.assertTrue(any(f.family_id == "E_no_h_exogenous" for f in no_h))
        for f in no_h:
            self.assertFalse(f.has_h)
            self.assertEqual(f.split, "held_out")   # the shortcut probe is a held-out generalization test

    def test_family_ids_unique(self) -> None:
        ids = [f.family_id for f in S.STRUCTURAL_FAMILIES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_offgrid_family_is_off_the_train_grid(self) -> None:
        off = S.get_family("E_offgrid_nonlinear")
        for k in off.kappa_cells:
            self.assertNotIn(k, S.KAPPA_TRAIN_GRID)

    def test_offgrid_endpoints_match_contract_and_are_disjoint(self) -> None:
        # Pi 2nd-pass: the off-grid cells ARE the two frozen endpoints (0.15, 0.60), NOT a band; they
        # must equal ORACLE_OFFGRID_KAPPA, be disjoint from the train grid, and be separate from kappa_mid.
        self.assertEqual(tuple(S.KAPPA_OFFGRID), tuple(C.ORACLE_OFFGRID_KAPPA))
        for k in S.KAPPA_OFFGRID:
            self.assertNotIn(k, S.KAPPA_TRAIN_GRID)
        self.assertNotIn(C.ORACLE_POWER_KAPPA_MID, S.KAPPA_TRAIN_GRID)
        self.assertNotIn(C.ORACLE_POWER_KAPPA_MID, S.KAPPA_OFFGRID)   # power point is its own probe

    def test_every_family_has_null_and_both_nuisance_cells(self) -> None:
        for f in S.STRUCTURAL_FAMILIES:
            self.assertGreater(f.null_mixture_weight, 0.0)   # camouflaged nulls present
            self.assertIn("orthogonal", f.nuisance_cells)    # R_nuis-must-lose cell
            self.assertIn("correlated_leak", f.nuisance_cells)  # R_nuis-captures-leak cell
            self.assertTrue(f.camouflaged_null)
            self.assertGreater(f.n_sequences, 0)

    def test_null_kappa_is_on_the_train_grid(self) -> None:
        # kappa=0 (the mechanistic null) must be reachable so R0 can be exercised on true nulls.
        self.assertIn(0.0, S.KAPPA_TRAIN_GRID)

    def test_get_family_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            S.get_family("nope")


class HashTests(unittest.TestCase):
    def test_mechanism_hash_deterministic_and_hex(self) -> None:
        h1, h2 = S.oracle_mechanism_hash(), S.oracle_mechanism_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        int(h1, 16)  # valid hex

    def test_mechanism_and_calibration_hashes_differ(self) -> None:
        # Pi #6: calibration output has its OWN hash, separate from the mechanism hash.
        self.assertNotEqual(S.oracle_mechanism_hash(), S.calibration_hash())

    def test_mechanism_hash_binds_executable_generator_config(self) -> None:
        # Pi #5/#8 (reproduced defect): a change to the generator's numeric mechanism MUST move the
        # hash. GENERATOR_CONFIG is what the generator actually executes.
        from unittest import mock
        base = S.oracle_mechanism_hash()
        bumped = {**S.GENERATOR_CONFIG, "d_h": S.GENERATOR_CONFIG["d_h"] + 1}
        with mock.patch.object(S, "GENERATOR_CONFIG", bumped):
            self.assertNotEqual(S.oracle_mechanism_hash(), base)
        # the generator reads the SAME constants the hash binds
        from clinical_jepa.eval import oracle_generator as G
        self.assertEqual(G.D_H, S.GENERATOR_CONFIG["d_h"])
        self.assertEqual(G.L_ITEMS, S.GENERATOR_CONFIG["l_items"])

    def test_calibration_cannot_list_mechanism_as_tunable(self) -> None:
        tunable = set(S.CALIBRATION_SPEC["tunable_params"])
        frozen = set(S.CALIBRATION_SPEC["frozen_against_calibration"])
        # order mechanism / seeds / grid / registry are frozen, never tunable.
        for must_be_frozen in ("order_mechanism", "seeds", "kappa_grid", "recipe_registry",
                               "evaluator_metrics", "structural_family_definitions"):
            self.assertIn(must_be_frozen, frozen)
            self.assertNotIn(must_be_frozen, tunable)

    def test_calibration_failure_is_fail_closed(self) -> None:
        self.assertIn("REFUSE", S.CALIBRATION_SPEC["failure_behavior"])
        self.assertEqual(S.CALIBRATION_SPEC["governance_class"],
                         "explicitly_cleared_safe_aggregate_only_no_patient_rows")
        # the two governance strings must be ONE reconciled statement (Pi: they disagreed)
        from clinical_jepa.eval import oracle_calibration as CAL
        self.assertEqual(S.CALIBRATION_SPEC["governance_class"], CAL._GOVERNANCE_CLASS)

    def test_spec_summary_is_safe_public(self) -> None:
        s = S.spec_summary()
        self.assertEqual(s["n_no_h_families"], len(S.no_h_families()))
        self.assertEqual(s["oracle_mechanism_hash"], S.oracle_mechanism_hash())
        self.assertGreaterEqual(s["n_heldout_families"], C.ORACLE_N_HELDOUT_FAMILIES)


if __name__ == "__main__":
    unittest.main()
