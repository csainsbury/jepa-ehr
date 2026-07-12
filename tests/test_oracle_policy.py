"""Tests for the trusted committed approved-oracle policy (Pi #1/#2).

The committed policy ships EMPTY -> every governed T4 is refused (fail-closed). These tests assert
that empty behavior AND the fail-closed membership logic against a POPULATED test policy.
"""
from __future__ import annotations

import unittest

from clinical_jepa.eval import oracle_policy as P


def _populated() -> dict:
    return {
        "blueprint_hash": "bp999", "gate_event_ref": "evt-oracle-1", "oracle_mechanism_hash": "mech123",
        "schema_version": "clinical-jepa-oracle-order-authorization-v3",
        "evaluator_commits": ["abc123"], "recipe_registry_ids": ["reg1"], "sealed_cert_run_ids": ["run77"],
    }


def _matching_manifest() -> dict:
    return {
        "blueprint_hash": "bp999", "gate_event_ref": "evt-oracle-1", "oracle_mechanism_hash": "mech123",
        "schema_version": "clinical-jepa-oracle-order-authorization-v3",
        "evaluator_commit": "abc123", "recipe_registry_id": "reg1", "sealed_cert_run_id": "run77",
    }


class CommittedPolicyShipsEmptyTests(unittest.TestCase):
    def test_committed_policy_is_empty_and_unpopulated(self) -> None:
        pol = P.load_approved_oracle_policy()
        self.assertIsNone(pol["blueprint_hash"])
        self.assertIsNone(pol["gate_event_ref"])
        self.assertIsNone(pol["oracle_mechanism_hash"])
        self.assertEqual(pol["evaluator_commits"], [])
        self.assertFalse(P.policy_is_populated())            # default committed policy is NOT usable

    def test_empty_policy_rejects_any_manifest(self) -> None:
        self.assertFalse(P.manifest_matches_policy(_matching_manifest()))          # default (empty)
        self.assertFalse(P.manifest_matches_policy(_matching_manifest(), {}))      # explicit empty


class PopulatedPolicyMembershipTests(unittest.TestCase):
    def test_full_match_passes(self) -> None:
        self.assertTrue(P.policy_is_populated(_populated()))
        self.assertTrue(P.manifest_matches_policy(_matching_manifest(), _populated()))

    def test_each_anchor_mismatch_fails(self) -> None:
        for k in ("blueprint_hash", "gate_event_ref", "oracle_mechanism_hash", "schema_version"):
            m = _matching_manifest()
            m[k] = "WRONG"
            self.assertFalse(P.manifest_matches_policy(m, _populated()))

    def test_membership_anchors_must_be_in_allowlist(self) -> None:
        for k, mkey in (("evaluator_commits", "evaluator_commit"),
                        ("recipe_registry_ids", "recipe_registry_id"),
                        ("sealed_cert_run_ids", "sealed_cert_run_id")):
            m = _matching_manifest()
            m[mkey] = "not-in-allowlist"
            self.assertFalse(P.manifest_matches_policy(m, _populated()))

    def test_partial_policy_is_not_populated(self) -> None:
        for missing in ("blueprint_hash", "gate_event_ref", "oracle_mechanism_hash"):
            pol = _populated()
            pol[missing] = None
            self.assertFalse(P.policy_is_populated(pol))
            self.assertFalse(P.manifest_matches_policy(_matching_manifest(), pol))


if __name__ == "__main__":
    unittest.main()
