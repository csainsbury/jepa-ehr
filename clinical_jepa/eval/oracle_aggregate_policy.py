"""TRUSTED, COMMITTED approved-aggregate-read policy (Pi micro-gate defect #1).

The first draft let the run operator self-authorize with a public committed token — string equality, not
approval. This committed, read-only allowlist is populated ONLY after the calibration micro-gate PASSes.
While EMPTY (as it is now) every aggregate-real read is refused — fail-closed, exactly like
``oracle_policy.APPROVED_ORACLE_POLICY``.

Authorization is NOT non-emptiness: the populated policy must bind the approving gate event, the reviewed
commit, the CURRENT frozen identity hashes (mechanism / ledger / calibration schema / evaluator / vocab),
the extraction schema, the exact sources + split, the approved local-config hash, and a single one-time
run id. ``aggregate_read_authorized`` re-derives the live identities and refuses on any mismatch, so a
stale policy (pinned to a superseded mechanism) also refuses.
"""
from __future__ import annotations

from typing import Any

# EMPTY until the calibration micro-gate PASSes. Do NOT populate without a Pi micro-gate ruling that
# supplies every field below. Populating this is the ONLY thing that authorizes a governed aggregate read.
APPROVED_AGGREGATE_READ_POLICY: dict[str, Any] = {
    "gate_event_ref": None,          # approving Pi micro-gate event / thread reference
    "reviewed_commit": None,         # the exact reviewed extraction commit
    "invariant_hash": None,          # frozen mechanism hash at approval
    "ledger_hash": None,
    "calibration_schema_hash": None,
    "evaluator_identity": None,
    "vocab_hash": None,              # flatascend_joint_corrected_v1 vocab hash
    "vocab_name": None,
    "extraction_schema_hash": None,  # the frozen extraction field/range/convention schema
    "config_hash": None,             # approved LOCAL config (real paths) content hash
    "sources": [],                   # must equal REQUIRED_SOURCES exactly
    "split": None,                   # must be "train"
    "run_id": None,                  # single approved one-time run id
}

_SCALAR_ANCHORS = ("gate_event_ref", "reviewed_commit", "invariant_hash", "ledger_hash",
                   "calibration_schema_hash", "evaluator_identity", "vocab_hash", "vocab_name",
                   "extraction_schema_hash", "config_hash", "split", "run_id")


def load_policy() -> dict[str, Any]:
    p = APPROVED_AGGREGATE_READ_POLICY
    return {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}


def policy_is_populated(policy: dict[str, Any] | None = None) -> bool:
    p = policy if policy is not None else APPROVED_AGGREGATE_READ_POLICY
    return (all(isinstance(p.get(k), str) and p[k] for k in _SCALAR_ANCHORS)
            and bool(p.get("sources")))


def aggregate_read_authorized(policy: dict[str, Any], live: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed. ``live`` carries the CURRENTLY-derived identities (mechanism/ledger/calibration/
    evaluator/vocab/extraction-schema/config hashes, sources, split, run_id). The policy must be populated
    AND every anchor must equal its live value — so a policy pinned to a superseded identity refuses."""
    if not policy_is_populated(policy):
        return False, "aggregate_read_policy_empty"
    from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
    if set(policy.get("sources", [])) != set(REQUIRED_SOURCES):
        return False, "policy_sources_not_required_set"
    if policy.get("split") != "train":
        return False, "policy_split_not_train"
    for k in ("invariant_hash", "ledger_hash", "calibration_schema_hash", "evaluator_identity",
              "vocab_hash", "vocab_name", "extraction_schema_hash", "config_hash", "run_id"):
        if policy.get(k) != live.get(k):
            return False, f"policy_{k}_mismatch"
    return True, "authorized"
