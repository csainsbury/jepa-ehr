"""Approved-aggregate-read policy DATA — the populated allowlist only (Pi micro-gate REVISE#3 #5).

Split out from ``oracle_aggregate_policy`` (the logic) so the executable logic closure can be content-hashed
into ``extraction_code_identity`` WITHOUT this data module — a policy-population commit edits ONLY this file
and therefore does not change the code identity (no circular hash), while any change to executable logic
does invalidate authorization.

POPULATED at the calibration-micro-gate PASS (Pi `jepa-pi-oracle-microgate-revise5-pass.md`,
evt-20260718T170631Z-6c149959) for reviewed commit 6d565b3. Populating this authorizes a governed read
ONLY through the fail-closed runner AND only after the separate policy-population delta is PASSed and the
one-time run is explicitly executed. Extracted-content provenance is pre-read UNVERIFIED and requires a
separate result gate; it is NOT a whole-HDF5 digest.
"""
from __future__ import annotations

from typing import Any

APPROVED_AGGREGATE_READ_POLICY: dict[str, Any] = {
    "gate_event_ref": "evt-20260718T170631Z-6c149959|thr-20260711T091727Z-9e9e31f6|jepa-pi-oracle-microgate-revise5-pass.md",
    "reviewed_commit": "6d565b3f29128000defeda9ba623f5a8eb6b468e",
    "invariant_hash": "e2371fade71dad81eea692e3848691c6debd2c919eee9b0aefbc35de6af986b0",
    "ledger_hash": "fb83cbd85bd0676d0276dd5ad9bf73d06ef08fad0d97cb780233f08b9d00b9f1",
    "calibration_schema_hash": "f4f86336fff104d89ec5b589e5bfc7368b2f97efac48d6b00e5b686b250d7aab",
    "evaluator_identity": "oracle_meta_eval_v5",
    "vocab_hash": "4b57b210ab4b3ec6",
    "vocab_name": "flatascend_joint_corrected_v1",
    "extraction_schema_hash": "980a63554541e8edf18a272ee09f4f46ead34293d17138c459314cca572f8ed1",
    "base_schema_hash": "13a0b4dedfd1ec773f29f680b5e752b2d6c7111cbc57d396704e6e75a619c8be",
    "generator_fit_schema_hash": "b6aa74e1fd3ddc0565957328c8c7f489c87dd372eef1321af9344aea147b180f",
    "calibration_adapter_schema_hash": "79577295e636ad3d7a2b44d3e13f542867077d989b386544611b9e713a34d70d",
    "code_identity": "fd4b30be8c63a072cdcf35523443abcff45be6f7c1d12f138baa1e695829e6a0",
    "state_root_identity": "c4a12e46f630f4bff78a4c5a9d68dfd109a4c18d14b955ba938708d29201cc9f",
    "provenance_procedure_hash": "4c7662029c451ea7185d49edfab96ac538b8870a14e81defe4cad2006092467c",
    "config_hash": "db06ec7beeee03056f06b1fd9788fb6b4cd895d07d85eaaac56ed9230c7c33a3",
    "sources": ["SCID", "MIMIC"],
    "split": "train",
    "run_id": "aggcalib-microgate-run-1",
}
