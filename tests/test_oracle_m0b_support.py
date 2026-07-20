"""Step-4 M0b — support-floor / min-length accounting for the v2 order core (Pi P-A owns L=1).

The v2 realism layer emits variable realized lengths; every restricted order core must be classified
EXPLICITLY and never silently: vacuous L<=1, support-starved cell / pair, and the structural occupancy cap
at L<5 (occupancy = distinct/C, C=5). This is a v2 realism-side accounting layer only — restricted cores
never reach fixed-L certification (the guard rejects them), which this test also reconfirms.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.oracle_realism_v2 import (
    account_order_support, restricted_core_support, restrict_order_core, m0b_support_policy_hash,
    assert_canonical_certification_cell, CertificationBoundaryError,
    SUPPORT_OK, SUPPORT_STARVED, VACUOUS_ORDER, M0B_CELL_SUPPORT_FLOOR, M0B_PAIR_DENOM_FLOOR,
    realism_v2_schema_hash, v2_certification_boundary_hash,
)

_M0B_POLICY_HASH = "876bffb6b79f7a9127616fea1fb5a9231ef48561cc0808ab8512a0c0099317d8"


class M0bAccounting(unittest.TestCase):
    def test_vacuous_order_for_L_le_1(self) -> None:
        for L in (0, 1):
            acc = account_order_support(L, 10_000, 10_000)
            self.assertEqual(acc.status, VACUOUS_ORDER, f"L={L}")
            self.assertTrue(acc.reasons, "vacuous order must carry a reason (never silent)")

    def test_per_cell_starvation(self) -> None:
        acc = account_order_support(6, M0B_CELL_SUPPORT_FLOOR - 1, 10_000)
        self.assertEqual(acc.status, SUPPORT_STARVED)
        self.assertTrue(any("per-cell" in r for r in acc.reasons))

    def test_per_pair_starvation(self) -> None:
        acc = account_order_support(6, 10_000, M0B_PAIR_DENOM_FLOOR - 1)
        self.assertEqual(acc.status, SUPPORT_STARVED)
        self.assertTrue(any("per-pair" in r for r in acc.reasons))

    def test_occupancy_cap_is_a_flag_not_a_failure(self) -> None:
        # L<5 with floors met stays SUPPORTED, but the structural occupancy cap L/5 is flagged explicitly.
        acc = account_order_support(4, 10_000, 10_000)
        self.assertEqual(acc.status, SUPPORT_OK)
        self.assertTrue(acc.occupancy_capped)
        self.assertAlmostEqual(acc.occupancy_cap, 4 / 5)
        self.assertTrue(any("occupancy structurally capped" in r for r in acc.reasons))

    def test_clean_supported_is_reasonless(self) -> None:
        acc = account_order_support(6, 10_000, 10_000)
        self.assertEqual(acc.status, SUPPORT_OK)
        self.assertFalse(acc.occupancy_capped)
        self.assertAlmostEqual(acc.occupancy_cap, 1.0)
        self.assertEqual(acc.reasons, ())

    def test_never_silent_off_nominal(self) -> None:
        # any off-nominal classification must carry at least one reason
        for acc in (account_order_support(1, 10_000, 10_000),
                    account_order_support(6, 10, 10_000),
                    account_order_support(6, 10_000, 10),
                    account_order_support(3, 10_000, 10_000)):
            if acc.status != SUPPORT_OK or acc.occupancy_capped:
                self.assertTrue(acc.reasons)


class M0bFromRestrictedCore(unittest.TestCase):
    def test_min_denom_from_real_core(self) -> None:
        cell = generate_literal_cell("T_latent_factor", 0.35, "orthogonal", 800, seed=17)
        core = restrict_order_core(cell, (0, 2, 3, 6))
        acc = restricted_core_support(core)
        self.assertEqual(acc.realized_length, 4)
        self.assertEqual(acc.n_sequences, 800)
        # continuous scores => all 800 sequences eligible for every surviving pair
        self.assertEqual(acc.per_pair_min_denom, 800)
        # 800 >= 500 cell floor and 800 >= 500 pair floor => SUPPORTED, occupancy capped (L=4)
        self.assertEqual(acc.status, SUPPORT_OK)
        self.assertTrue(acc.occupancy_capped)

    def test_rejects_non_core(self) -> None:
        cell = generate_literal_cell("T_hmm_markov", 0.35, "orthogonal", 20, seed=1)
        with self.assertRaises(TypeError):
            restricted_core_support(cell)

    def test_core_is_classifiable_but_never_certifiable(self) -> None:
        # M0b classifies a restricted core; the certification guard still rejects it (complementary roles).
        cell = generate_literal_cell("T_hmm_markov", 0.6, "orthogonal", 700, seed=5)
        core = restrict_order_core(cell, (1, 4, 7))
        self.assertIn(restricted_core_support(core).status, (SUPPORT_OK, SUPPORT_STARVED))
        with self.assertRaises(CertificationBoundaryError):
            assert_canonical_certification_cell(core, entrypoint="test")


class M0bIdentity(unittest.TestCase):
    def test_policy_hash_pinned_and_others_unmoved(self) -> None:
        self.assertEqual(m0b_support_policy_hash(), _M0B_POLICY_HASH)
        self.assertEqual(realism_v2_schema_hash(),
                         "704b079a6137a9dbadbf9938b031b4af5cedfbaadb81fbfa9d872c6519cd6f85")
        self.assertEqual(v2_certification_boundary_hash(),
                         "b33c2d9f6324c84763ebb85fde8912dbf0b84e94b7ee366e4adb879ceb14e8e4")
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_aggregate_extract import extraction_code_identity
        self.assertEqual(invariant_hash(),
                         "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0")
        self.assertEqual(extraction_code_identity(),
                         "fd4b30be8c63a072cdcf35523443abcff45be6f7c1d12f138baa1e695829e6a0")


if __name__ == "__main__":
    unittest.main()
