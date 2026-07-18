"""Approved-aggregate-read policy DATA — the populated allowlist only (Pi micro-gate REVISE#3 #5).

Split out from ``oracle_aggregate_policy`` (the logic) so the executable logic closure can be content-hashed
into ``extraction_code_identity`` WITHOUT this data module — a policy-population commit edits ONLY this file
and therefore does not change the code identity (no circular hash), while any change to executable logic
does invalidate authorization.

Ships EMPTY and fail-closed: while empty, every aggregate-real read is refused. Populating it is the ONLY
thing that authorizes a governed read, and only via the separate reviewed policy-population delta.
"""
from __future__ import annotations

from typing import Any

APPROVED_AGGREGATE_READ_POLICY: dict[str, Any] = {
    "gate_event_ref": None,          # approving Pi micro-gate event / thread reference
    "reviewed_commit": None,         # the exact reviewed extraction commit
    "invariant_hash": None,
    "ledger_hash": None,
    "calibration_schema_hash": None,
    "evaluator_identity": None,
    "vocab_hash": None,
    "vocab_name": None,
    "extraction_schema_hash": None,
    "base_schema_hash": None,        # declared synthetic BASE population identity
    "code_identity": None,           # complete executable logic-closure hash
    "state_root_identity": None,     # absolute canonical state-root identity
    "config_hash": None,             # approved LOCAL config (real paths) content hash
    "train_artifact_identities": None,   # per-source immutable TRAIN artifact identity (or provenance rule)
    "sources": [],                   # must equal REQUIRED_SOURCES exactly
    "split": None,                   # must be "train"
    "run_id": None,                  # single approved one-time run id
}
