"""Step-1 of the M3a rebuild — full-sequence multi-block re-architecture + re-gated DEV identities.

Pi M3a gate (thread thr-20260720T143304Z): the realism UNIT is the full content-token sequence
(L_total = 8*B + R), while certification stays a fixed 8-item block. This test pins the re-architected DEV
schemas and identities and the certification narrowing (only complete 8-item blocks certifiable; the final
restricted block + cross-block/tail pairs are emission-only). No sampling / fitting — schema + identity only.
"""
from __future__ import annotations

import unittest

import numpy as np

from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval import oracle_realism_v2 as v2

# re-gated DEV identities (all moved by the re-architecture)
_REALISM_V2 = "2a7405ddadcdfdf3261a2b18e149c5523298a73d3e765cde78b6611927377673"
_MARGINAL = "adfdf975fa827eae12cd580d662ad2b09630c1141d29165a71a8e69df1a7de50"
_ADAPTER_IFACE = "4ff9a5dabbb78fdaa41642e04bc3c9559ca5c9a8af9ad8c46bfb8131382dd19d"
_M0B = "c7532ee9cd8629a20e8943c30729263e429f074258c6d7bb069ab104e688cd6d"


class BlockCompositionSchema(unittest.TestCase):
    def test_canonical_decomposition_and_certification_narrowing(self) -> None:
        bc = v2.V2_BLOCK_COMPOSITION
        self.assertEqual(bc["canonical"], "L_total = 8*B + R")
        self.assertEqual(bc["block_len"], 8)
        self.assertTrue(bc["empty_forbidden"])
        # ONLY complete 8-item blocks are certifiable; tail + cross-block pairs are not
        self.assertIn("complete 8-item blocks", bc["certifiable_unit"])
        for nc in ("final_restricted_block", "cross_block_pairs", "restricted_tail_pairs"):
            self.assertIn(nc, bc["not_certifiable"])

    def test_bins_are_overflow_not_capped_at_8(self) -> None:
        length = v2.V2_FROZEN_BINS["length"]
        self.assertEqual(length[-1], (2049, None), "length bins must have an open overflow bin")
        self.assertTrue(any(hi is None or hi > 8 for _, hi in length), "length bins must extend past 8")
        self.assertEqual(v2.V2_FROZEN_BINS["cluster_size"][-1], (17, None))
        self.assertEqual(v2.V2_FROZEN_BINS["position_quartiles"], 4)

    def test_marginal_schema_is_full_sequence_multiblock(self) -> None:
        ms = v2.V2_MARGINAL_SCHEMA
        self.assertIn("multi-block", ms["unit"])
        self.assertEqual(ms["length_law"], "source_conditioned_full_sequence_length")
        self.assertIs(ms["block_composition"], v2.V2_BLOCK_COMPOSITION)
        self.assertIs(ms["bins"], v2.V2_FROZEN_BINS)


class ReGatedIdentities(unittest.TestCase):
    def test_moved_dev_identities_pinned(self) -> None:
        self.assertEqual(v2.realism_v2_schema_hash(), _REALISM_V2)
        self.assertEqual(v2.v2_marginal_schema_hash(), _MARGINAL)
        self.assertEqual(v2.v2_adapter_interface_hash(), _ADAPTER_IFACE)
        self.assertEqual(v2.m0b_support_policy_hash(), _M0B)

    def test_m0b_policy_has_multiblock_levels(self) -> None:
        levels = v2.M0B_SUPPORT_POLICY["levels"]
        self.assertEqual(set(levels),
                         {"sequence", "complete_block", "restricted_tail", "within_block_pair"})
        self.assertEqual(set(v2.M0B_SUPPORT_POLICY["level_floors"]), set(levels))

    def test_adapter_emits_multiblock_sequence(self) -> None:
        emits = " ".join(v2.V2_ADAPTER_INTERFACE["emits"])
        self.assertIn("block_sequence", emits)
        self.assertIn("final_restricted_block", emits)
        self.assertIn("ONLY complete 8-item blocks", v2.V2_ADAPTER_INTERFACE["certification_boundary"])


class ActiveSetDIdentity(unittest.TestCase):
    def test_menu_includes_length_class_mix(self) -> None:
        self.assertIn("length_class_mix", v2.V2_D_COMPONENT_MENU)

    def test_active_sets_are_distinct_and_ordered(self) -> None:
        a = v2.v2_active_d_identity(["burst_count_length"])
        b = v2.v2_active_d_identity(["burst_count_length", "length_class_mix"])
        self.assertNotEqual(a, b)
        # order-independent (set semantics)
        self.assertEqual(v2.v2_active_d_identity(["length_class_mix", "burst_count_length"]), b)
        # dev != final
        self.assertNotEqual(v2.v2_active_d_identity(["burst_count_length"]),
                            v2.v2_active_d_identity(["burst_count_length"], final=True))

    def test_empty_and_unknown_rejected(self) -> None:
        with self.assertRaises(ValueError):
            v2.v2_active_d_identity([])
        with self.assertRaises(KeyError):
            v2.v2_active_d_identity(["not_a_component"])


class CertificationNarrowing(unittest.TestCase):
    def test_complete_block_certifiable_tail_not(self) -> None:
        # a canonical 8-item block passes the guard; the restricted tail (a RestrictedOrderCore) is rejected
        cell = generate_literal_cell("T_latent_factor", 0.6, "orthogonal", 40, seed=5)
        self.assertIsNone(v2.assert_canonical_certification_cell(cell, entrypoint="test"))
        tail = v2.restrict_order_core(cell, (0, 2, 3))     # a length-3 emission-only tail
        with self.assertRaises(v2.CertificationBoundaryError):
            v2.assert_canonical_certification_cell(tail, entrypoint="test")

    def test_frozen_v1_and_boundary_unmoved(self) -> None:
        self.assertEqual(v2.v2_certification_boundary_hash(),
                         "b33c2d9f6324c84763ebb85fde8912dbf0b84e94b7ee366e4adb879ceb14e8e4")
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_aggregate_extract import extraction_code_identity
        self.assertEqual(invariant_hash(),
                         "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0")
        self.assertEqual(extraction_code_identity(),
                         "fd4b30be8c63a072cdcf35523443abcff45be6f7c1d12f138baa1e695829e6a0")


if __name__ == "__main__":
    unittest.main()
