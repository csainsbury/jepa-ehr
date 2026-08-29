#!/usr/bin/env python3
"""Guarded fail-closed entrypoint for the BP011 one-model residual beta."""
from __future__ import annotations

from pathlib import Path
import sys

from clinical_jepa.eval.j04c_v3_r0resid import canonical_json_bytes, parse_canonical_json, sha256_hex
from clinical_jepa.eval.j04c_v3_r0resid_1m import (
    FAILURE_CODES, OneModelApprovedSeedEnvelope, PrototypeInvariantError,
    approved_envelope_from_dict, build_provenance_from_dict, failure_artifact,
    run_one_model_beta, seed_manifest_from_dict, validate_seed_audit, validate_success_schema,
)


def _arguments(argv: list[str]) -> dict[str, str]:
    allowed = {
        "--build-provenance", "--build-provenance-sha256", "--seed-manifest",
        "--approved-seed-envelope", "--approved-seed-envelope-sha256",
        "--historical-path-inventory",
    }
    if len(argv) != 12 or any(argv[i] not in allowed for i in range(0, 12, 2)):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    result = dict(zip(argv[0::2], argv[1::2]))
    if set(result) != allowed or any(not item for item in result.values()):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return result


def _normalize_failure(error: Exception, phase: str) -> tuple[str, str]:
    text = str(error)
    phase_defaults = {
        "INPUT": "INPUT_SCHEMA", "PROVENANCE": "PROVENANCE_CONTENT",
        "SEED_AUDIT": "SEED_AUDIT_DIGEST", "GENERATION": "GENERATION_INVARIANT",
        "TRAINING": "TRAINING_INVARIANT", "READOUT": "READOUT_INVALID",
        "BOOTSTRAP": "BOOTSTRAP_INVALID", "SERIALIZATION": "SERIALIZATION_INVALID",
    }
    code = text if text in FAILURE_CODES else phase_defaults.get(phase, "SERIALIZATION_INVALID")
    if code.startswith("TRAINING_"):
        phase = "TRAINING"
    elif code == "READOUT_INVALID":
        phase = "READOUT"
    elif code == "BOOTSTRAP_INVALID":
        phase = "BOOTSTRAP"
    elif code == "SERIALIZATION_INVALID":
        phase = "SERIALIZATION"
    return phase, code


def main(argv: list[str] | None = None) -> int:
    phase_state = ["INPUT"]
    def set_phase(value: str) -> None:
        phase_state[0] = value
    try:
        args = _arguments(list(sys.argv[1:] if argv is None else argv))
        provenance_raw = Path(args["--build-provenance"]).read_bytes()
        manifest_raw = Path(args["--seed-manifest"]).read_bytes()
        envelope_raw = Path(args["--approved-seed-envelope"]).read_bytes()
        inventory_raw = Path(args["--historical-path-inventory"]).read_bytes()
        provenance_value = parse_canonical_json(provenance_raw)
        manifest_value = parse_canonical_json(manifest_raw)
        envelope_value = parse_canonical_json(envelope_raw)
        inventory_value = parse_canonical_json(inventory_raw)

        set_phase("PROVENANCE")
        if sha256_hex(provenance_raw) != args["--build-provenance-sha256"]:
            raise PrototypeInvariantError("PROVENANCE_DIGEST")
        provenance = build_provenance_from_dict(provenance_value)

        set_phase("SEED_AUDIT")
        if sha256_hex(envelope_raw) != args["--approved-seed-envelope-sha256"]:
            raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
        envelope: OneModelApprovedSeedEnvelope = approved_envelope_from_dict(envelope_value)
        manifest = seed_manifest_from_dict(manifest_value)
        _, seed_audit = validate_seed_audit(
            manifest, manifest_raw, envelope, inventory_value, inventory_raw,
        )

        set_phase("GENERATION")
        result = run_one_model_beta(
            manifest, provenance, seed_audit,
            build_provenance_sha256=args["--build-provenance-sha256"],
            approved_envelope_sha256=args["--approved-seed-envelope-sha256"],
            phase_callback=set_phase,
        )
        set_phase("SERIALIZATION")
        validate_success_schema(result)
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    except Exception as error:
        phase, code = _normalize_failure(error, phase_state[0])
        sys.stdout.buffer.write(canonical_json_bytes(failure_artifact(phase, code)) + b"\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
