"""Rung-1 CONTRACT fail-hard tests (Pi R8 #5). These encode the load-bearing invariants
that must hold before any governed run — attribution, information-scope, nomination,
adequacy, primary horizons, and test sealing."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_jepa.eval import rung1_contract as C


class AttributionTests(unittest.TestCase):
    def test_m2_nonlinear_decodable_when_m1_fails(self) -> None:
        # A weak M1 must NOT veto a genuinely nonlinear M2 (the R7 #1 correction).
        cls = C.classify_readout(m1_gate_ok=False, m1_excess_lo=-0.5,
                                 m2_gate_ok=True, m2_excess_lo=0.30, m2_copy_ok=True,
                                 evaluable=True, precise=True)
        self.assertEqual(cls, C.DECODABLE_NONLINEAR)

    def test_m2_raw_pass_swap_fail_is_prior_masked(self) -> None:
        cls = C.classify_readout(m1_gate_ok=False, m1_excess_lo=0.0,
                                 m2_gate_ok=True, m2_excess_lo=0.01, m2_copy_ok=True,
                                 evaluable=True, precise=True)
        self.assertEqual(cls, C.PRIOR_MASKED)
        cls2 = C.classify_readout(m1_gate_ok=False, m1_excess_lo=0.0,
                                  m2_gate_ok=True, m2_excess_lo=0.5, m2_copy_ok=False,  # copies
                                  evaluable=True, precise=True)
        self.assertEqual(cls2, C.PRIOR_MASKED)

    def test_simple_decodable_when_m1_clears(self) -> None:
        cls = C.classify_readout(m1_gate_ok=True, m1_excess_lo=0.20,
                                 m2_gate_ok=False, m2_excess_lo=0.0, m2_copy_ok=True,
                                 evaluable=True, precise=True)
        self.assertEqual(cls, C.DECODABLE_SIMPLE)

    def test_under_floor_vs_wide_ci_differ(self) -> None:
        under = C.classify_readout(m1_gate_ok=False, m1_excess_lo=0.0, m2_gate_ok=False,
                                   m2_excess_lo=0.0, m2_copy_ok=True, evaluable=False, precise=False)
        wide = C.classify_readout(m1_gate_ok=False, m1_excess_lo=0.0, m2_gate_ok=False,
                                  m2_excess_lo=0.0, m2_copy_ok=True, evaluable=True, precise=False)
        self.assertEqual(under, C.NOT_EVALUABLE)
        self.assertEqual(wide, C.INCONCLUSIVE)
        self.assertNotEqual(under, wide)


class InformationScopeTests(unittest.TestCase):
    def test_arm_a_order_never_direct(self) -> None:
        # Even if the base class says nonlinear-decodable, arm-A order is forced to
        # content-prior-only and can never nominate (Pi R8 #1).
        for base in (C.DECODABLE_NONLINEAR, C.DECODABLE_SIMPLE, C.PRIOR_MASKED, C.NOT_DECODABLE):
            v = C.scoped_verdict("mean_embed", "order", base)
            self.assertEqual(v["verdict"], C.STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY)
            self.assertEqual(v["information_scope"], C.SCOPE_CONTENT_PROXY)
            self.assertFalse(v["can_nominate"])

    def test_temporal_slot_cannot_emit_exact_order(self) -> None:
        v = C.scoped_verdict("temporal_slot", "order", C.DECODABLE_NONLINEAR)
        self.assertEqual(v["verdict"], C.COARSE_SLOT_DECODABLE)
        self.assertEqual(v["information_scope"], C.SCOPE_COARSE_SLOT)
        self.assertNotEqual(v["verdict"], "DIRECT_ORDER")
        self.assertTrue(C.direct_order_forbidden("temporal_slot"))
        self.assertTrue(C.direct_order_forbidden("mean_embed"))

    def test_tap_timing_direct_ceiling(self) -> None:
        v = C.scoped_verdict("tap_concat", "timing", C.DECODABLE_NONLINEAR)
        self.assertEqual(v["verdict"], C.DIRECT_TIMING_CEILING_DECODABLE)
        self.assertEqual(v["information_scope"], C.SCOPE_DIRECT)
        self.assertTrue(v["can_nominate"])                       # 1b + direct scope

    def test_arm_a_timing_only_content_proxy(self) -> None:
        v = C.scoped_verdict("mean_embed", "timing", C.DECODABLE_NONLINEAR)
        self.assertEqual(v["verdict"], C.CONTENT_PROXY_DECODABLE)
        self.assertFalse(v["can_nominate"])                      # 1a incumbent never nominates

    def test_content_proxy_and_oracle_never_nominate(self) -> None:
        self.assertFalse(C.can_nominate(C.SCOPE_CONTENT_PROXY))
        self.assertFalse(C.can_nominate(C.SCOPE_ORACLE_ASSISTED))
        self.assertTrue(C.can_nominate(C.SCOPE_DIRECT))
        self.assertTrue(C.can_nominate(C.SCOPE_COARSE_SLOT))

    def test_not_evaluated_order_maps_to_structural_for_order_blind(self) -> None:
        # order-blind arms: unconditional order NOT_EVALUATED -> the structural finding stands.
        v = C.scoped_verdict("mean_embed", "order", C.NOT_EVALUATED)
        self.assertEqual(v["verdict"], C.STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY)
        self.assertFalse(v["can_nominate"])

    def test_marginal_only_timing_passes_through_and_never_nominates(self) -> None:
        for arm in ("mean_embed", "tap_concat"):
            v = C.scoped_verdict(arm, "timing", C.MARGINAL_ONLY)
            self.assertEqual(v["verdict"], C.MARGINAL_ONLY)
            self.assertFalse(v["can_nominate"])

    def test_incumbent_1a_never_nominates(self) -> None:
        for prop in C.PROPERTIES:
            self.assertFalse(C.scoped_verdict("mean_embed", prop, C.DECODABLE_NONLINEAR)["can_nominate"])

    def test_1b_count_can_nominate(self) -> None:
        v = C.scoped_verdict("count_concat", "count", C.DECODABLE_NONLINEAR)
        self.assertTrue(v["can_nominate"])
        self.assertEqual(v["information_scope"], C.SCOPE_DIRECT)


class PrimaryHorizonTests(unittest.TestCase):
    def test_mimic_2d_not_primary(self) -> None:
        self.assertFalse(C.is_primary_cell("MIMIC", 2.0))
        self.assertTrue(C.is_primary_cell("MIMIC", 0.5))
        self.assertTrue(C.is_primary_cell("SCID", 730.0))       # SCID has no sensitivity horizon


class MatchedHeadTests(unittest.TestCase):
    def test_hidden_shrinks_as_input_grows(self) -> None:
        D = 256
        h_base = C.matched_head_hidden(C.input_dim("mean_embed", D), 1, D)
        h_slot = C.matched_head_hidden(C.input_dim("temporal_slot", D), 1, D)  # M*D input
        self.assertGreater(C.input_dim("temporal_slot", D), C.input_dim("mean_embed", D))
        self.assertLess(h_slot, h_base)                          # capacity equalised, not gifted

    def test_input_dims(self) -> None:
        D = 128
        self.assertEqual(C.input_dim("mean_embed", D), D)
        self.assertEqual(C.input_dim("tap_concat", D), D + C.D_TIME)
        self.assertEqual(C.input_dim("count_concat", D), D + 1)
        self.assertEqual(C.input_dim("temporal_slot", D), C.M_PRIMARY * D)


class SwapTests(unittest.TestCase):
    def test_derangement_no_self_patient_disjoint_deterministic(self) -> None:
        pats = ["A", "A", "B", "B", "C", "C"]
        p1 = C.deterministic_derangement(pats, seed=1)
        p2 = C.deterministic_derangement(pats, seed=1)
        self.assertTrue((p1 == p2).all())                        # deterministic
        for i, j in enumerate(p1):
            self.assertNotEqual(i, int(j))                       # no self-pair
            self.assertNotEqual(pats[i], pats[int(j)])           # patient-disjoint

    def test_no_partner_when_single_patient(self) -> None:
        p = C.deterministic_derangement(["A", "A", "A"], seed=1)
        self.assertTrue((p == -1).all())                         # no patient-disjoint partner


class ConfigHashTests(unittest.TestCase):
    def test_hash_stable_and_sensitive(self) -> None:
        h1 = C.config_hash({"SCID": {"horizons": [30, 90]}})
        h2 = C.config_hash({"SCID": {"horizons": [30, 90]}})
        h3 = C.config_hash({"SCID": {"horizons": [30, 90, 365]}})
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_test_access_sealed(self) -> None:
        self.assertFalse(C.frozen_contract()["test_access"])     # no test in this run


if __name__ == "__main__":
    unittest.main()
