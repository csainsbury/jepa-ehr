"""TRUSTED approved-aggregate-read policy LOGIC (Pi micro-gate REVISE#1..#3).

The policy DATA lives in ``oracle_aggregate_policy_data`` so this logic module can be hashed into the code
identity without circularity. Authorization is NOT non-emptiness: the populated policy must bind the
approving gate event, the reviewed commit, the CURRENT frozen identity hashes (mechanism / ledger /
calibration schema / evaluator / vocab / extraction schema / BASE schema / code / state-root), the exact
config hash, immutable TRAIN artifact identities, the exact sources + split, and the one-time run id.
``aggregate_read_authorized`` re-derives the live identities and refuses on any mismatch, so a stale policy
(pinned to a superseded mechanism/code) also refuses.
"""
from __future__ import annotations

from typing import Any

from clinical_jepa.eval.oracle_aggregate_policy_data import APPROVED_AGGREGATE_READ_POLICY

_SCALAR_ANCHORS = ("gate_event_ref", "reviewed_commit", "invariant_hash", "ledger_hash",
                   "calibration_schema_hash", "evaluator_identity", "vocab_hash", "vocab_name",
                   "extraction_schema_hash", "base_schema_hash", "generator_fit_schema_hash",
                   "calibration_adapter_schema_hash", "code_identity", "state_root_identity",
                   "provenance_procedure_hash", "config_hash", "split", "run_id")
# anchors re-derived live and compared exactly (gate_event_ref/reviewed_commit are bound at policy-
# population review, not derivable here).
_LIVE_ANCHORS = ("invariant_hash", "ledger_hash", "calibration_schema_hash", "evaluator_identity",
                 "vocab_hash", "vocab_name", "extraction_schema_hash", "base_schema_hash",
                 "generator_fit_schema_hash", "calibration_adapter_schema_hash", "code_identity",
                 "state_root_identity", "provenance_procedure_hash", "config_hash", "run_id")


def load_policy() -> dict[str, Any]:
    p = APPROVED_AGGREGATE_READ_POLICY
    return {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}


def policy_is_populated(policy: dict[str, Any] | None = None) -> bool:
    p = policy if policy is not None else APPROVED_AGGREGATE_READ_POLICY
    return all(isinstance(p.get(k), str) and p[k] for k in _SCALAR_ANCHORS) and bool(p.get("sources"))


def aggregate_read_authorized(policy: dict[str, Any], live: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed. ``live`` carries the currently-derived identities. The policy must be populated AND the
    source LIST must equal the required list exactly (duplicates refused) AND every live anchor must equal
    its live value."""
    if not policy_is_populated(policy):
        return False, "aggregate_read_policy_empty"
    from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
    if list(policy.get("sources", [])) != list(REQUIRED_SOURCES):       # ONE canonical order = run order (Pi #7)
        return False, "policy_sources_not_canonical_list"
    if policy.get("split") != "train":
        return False, "policy_split_not_train"
    for k in _LIVE_ANCHORS:
        if policy.get(k) != live.get(k):
            return False, f"policy_{k}_mismatch"
    return True, "authorized"
