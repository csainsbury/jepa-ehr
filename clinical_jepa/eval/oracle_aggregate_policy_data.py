"""Approved-aggregate-read policy DATA — the populated allowlist only (Pi micro-gate REVISE#3 #5).

Split out from ``oracle_aggregate_policy`` (the logic) so the executable logic closure can be content-hashed
into ``extraction_code_identity`` WITHOUT this data module — a policy-population/retirement commit edits ONLY
this file and therefore does not change the code identity (no circular hash), while any change to executable
logic does invalidate authorization.

RETIRED (fail-closed) after the one-time run was SPENT and the result-gate ruled (Pi
`jepa-pi-oracle-calibration-result-gate.md`, evt-20260719T012528Z). The active policy is EMPTY again so a
clone / deleted local state / recreated checkout cannot reuse the spent one-time authorization. The spent
run is recorded in ``SPENT_RUNS`` (sanitized: identities + hashes only — no governed paths/rows). Do NOT
repopulate without a new gate and a new run id; the local COMPLETE state/result artifacts are preserved.
"""
from __future__ import annotations

from typing import Any

# EMPTY / fail-closed. Every aggregate-real read is refused while this is empty.
APPROVED_AGGREGATE_READ_POLICY: dict[str, Any] = {
    "gate_event_ref": None,
    "reviewed_commit": None,
    "invariant_hash": None,
    "ledger_hash": None,
    "calibration_schema_hash": None,
    "evaluator_identity": None,
    "vocab_hash": None,
    "vocab_name": None,
    "extraction_schema_hash": None,
    "base_schema_hash": None,
    "generator_fit_schema_hash": None,
    "calibration_adapter_schema_hash": None,
    "code_identity": None,
    "state_root_identity": None,
    "provenance_procedure_hash": None,
    "config_hash": None,
    "sources": [],
    "split": None,
    "run_id": None,
}

# Durable sanitized record of spent one-time runs — a run id here has already been consumed and MUST NOT be
# reused (a new governed read requires a new question/schema/run id and a fresh gate).
SPENT_RUNS: list[dict[str, Any]] = [
    {
        "run_id": "aggcalib-microgate-run-1",
        "reviewed_commit": "6d565b3f29128000defeda9ba623f5a8eb6b468e",
        "policy_population_commit": "3e6e33f0615deebdf3edaafe06080f385926db92",
        "gate_event_ref": "evt-20260718T170631Z-6c149959|thr-20260711T091727Z-9e9e31f6",
        "result_gate_event_ref": "evt-20260719T012528Z-8e5f96c9",
        "result_sha256": "bd64a29e2d517df347941ed4575574ada64d6667e0e17a92dcb5e17dba02b163",
        "state_sha256": "b3f18d1911aa2c953dbc703c24a343fd7559d43d23314ca6564b640dd03713ae",
        "extracted_content_digests": {
            "SCID": "9a8ffb40ff56abc63deeace9b6c0e3e7d4d18806d2d347e3e6f0a4409e197f1d",
            "MIMIC": "d3c23a06a2207d30e1c36d05d6209f749942b5e99ee6c9db6aa6027ca0f7943d",
        },
        "verdict": "execution_integrity=PASS; aggregate_realism_eligibility=FAIL; authorization_consequence=NONE",
    },
]
