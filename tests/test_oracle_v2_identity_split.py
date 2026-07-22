"""Step-3 Option-A / Option-D identity split (Pi P-D-1; thread thr-20260719T155635Z).

The scaffold previously declared a single copula join, conflating the falsifiable baseline (A) with the
escalation target (D). This step freezes ONE shared marginal schema plus DISTINCT `A_independent` and
`D_copula` identities (dev + final), and an adapter INTERFACE STUB carrying no behaviour. M2 will bind A
first; D is a controls-driven escalation with its own identity + ledger entry.

Everything here is synthetic-only identity scaffolding — no sampling law, parameter fit, or target comparison.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval.oracle_realism_v2 import (
    V2_VARIANT_A_INDEPENDENT, V2_VARIANT_D_COPULA, V2_ADAPTER_INTERFACE,
    v2_marginal_schema_hash, v2_variant_identity, v2_adapter_interface_hash,
    v2_certification_boundary_hash, realism_v2_schema_hash,
)

# pinned step-3 identities
_MARGINAL = "1ac8c61c7a2a541355253831de87f785db35b0f5bff16b1522d75fabedccb947"
_A_DEV = "531c938e15a73c6263335f089166766cee21193775e235cc6d5b4eb2f5de4cb1"
_D_DEV = "8f7e53676b144a6c017c186d9219ba64ac117a0a69b8c0f09f97ea8fccf76089"
_A_FINAL = "7151f4e56f93ae778c2a1ff55e16c32066a801400cd231b2aa2de8d921010b86"
_D_FINAL = "42e7ab4f7488a5b2a37d3c16c268aac94682465ef27b465e76162360a42b7819"
_ADAPTER_IFACE = "4ff9a5dabbb78fdaa41642e04bc3c9559ca5c9a8af9ad8c46bfb8131382dd19d"
_DEV_SCAFFOLD = "2a7405ddadcdfdf3261a2b18e149c5523298a73d3e765cde78b6611927377673"


class VariantIdentitySplit(unittest.TestCase):
    def test_identities_pinned(self) -> None:
        self.assertEqual(v2_marginal_schema_hash(), _MARGINAL)
        self.assertEqual(v2_variant_identity("A_independent"), _A_DEV)
        self.assertEqual(v2_variant_identity("D_copula"), _D_DEV)
        self.assertEqual(v2_variant_identity("A_independent", final=True), _A_FINAL)
        self.assertEqual(v2_variant_identity("D_copula", final=True), _D_FINAL)
        self.assertEqual(v2_adapter_interface_hash(), _ADAPTER_IFACE)

    def test_A_and_D_and_dev_final_all_distinct(self) -> None:
        ids = {
            v2_variant_identity("A_independent"), v2_variant_identity("D_copula"),
            v2_variant_identity("A_independent", final=True), v2_variant_identity("D_copula", final=True),
        }
        self.assertEqual(len(ids), 4, "A/D x dev/final must be four distinct identities")

    def test_unknown_variant_rejected(self) -> None:
        with self.assertRaises(KeyError):
            v2_variant_identity("B_something")

    def test_A_is_independent_baseline_D_is_copula_escalation(self) -> None:
        self.assertEqual(V2_VARIANT_A_INDEPENDENT["join"], "independent_source_conditioned_marginals")
        self.assertEqual(V2_VARIANT_A_INDEPENDENT["dependence_params"], [])          # baseline: no coupling
        self.assertEqual(V2_VARIANT_D_COPULA["join"], "sparse_compound_burst_copula")
        self.assertTrue(V2_VARIANT_D_COPULA["dependence_params"])                     # D has coupling params
        self.assertIn("controls_driven", V2_VARIANT_D_COPULA["role"])

    def test_adapter_is_interface_stub_only(self) -> None:
        # the stub forbids any behaviour before the M3a freeze
        for forbidden in ("sampling_law", "parameter_fit", "target_comparison"):
            self.assertIn(forbidden, V2_ADAPTER_INTERFACE["forbidden_pre_m3a"])
        self.assertTrue(V2_ADAPTER_INTERFACE["adapter_stub"].endswith("_dev"))


class Step3IdentityInvariants(unittest.TestCase):
    def test_dev_scaffold_bumped_boundary_and_v1_unmoved(self) -> None:
        # the umbrella dev-scaffold hash bumped INTENTIONALLY for the split...
        self.assertEqual(realism_v2_schema_hash(), _DEV_SCAFFOLD)
        # ...but the certification boundary identity and frozen v1 identities did NOT move.
        self.assertEqual(v2_certification_boundary_hash(),
                         "b33c2d9f6324c84763ebb85fde8912dbf0b84e94b7ee366e4adb879ceb14e8e4")
        from clinical_jepa.eval.oracle_meta_gen import invariant_hash
        from clinical_jepa.eval.oracle_aggregate_extract import (
            base_schema_hash, generator_fit_schema_hash, extraction_code_identity,
        )
        self.assertEqual(invariant_hash(),
                         "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0")
        self.assertEqual(base_schema_hash(),
                         "13a0b4dedfd1ec773f29f680b5e752b2d6c7111cbc57d396704e6e75a619c8be")
        self.assertEqual(generator_fit_schema_hash(),
                         "b6aa74e1fd3ddc0565957328c8c7f489c87dd372eef1321af9344aea147b180f")
        self.assertEqual(extraction_code_identity(),
                         "fd4b30be8c63a072cdcf35523443abcff45be6f7c1d12f138baa1e695829e6a0")


if __name__ == "__main__":
    unittest.main()
