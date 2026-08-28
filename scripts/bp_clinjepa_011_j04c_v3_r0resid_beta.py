#!/usr/bin/env python3
"""Guarded, fail-closed entrypoint for the BP011 J04c-v3 residual beta."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from clinical_jepa.eval.j04c_v3_r0resid import (
    FAILURE_CODES, ApprovedSeedEnvelope, PrototypeInvariantError,
    approved_envelope_from_dict, build_provenance_from_dict, canonical_json_bytes,
    failure_artifact, parse_canonical_json, run_production_beta,
    seed_manifest_from_dict, sha256_hex, validate_seed_audit,
    validate_success_schema,
)


def _arguments(argv: list[str]) -> dict[str, str]:
    allowed = {
        "--build-provenance", "--build-provenance-sha256", "--seed-manifest",
        "--approved-seed-envelope", "--approved-seed-envelope-sha256",
        "--historical-path-inventory",
    }
    if len(argv) != 12 or any(argv[index] not in allowed for index in range(0, 12, 2)):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    result = dict(zip(argv[0::2], argv[1::2]))
    if set(result) != allowed or any(not value for value in result.values()):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return result


def _read_named(filename: str) -> bytes:
    return Path(filename).read_bytes()


def main(argv: list[str] | None = None) -> int:
    phase = "INPUT"
    try:
        args = _arguments(list(sys.argv[1:] if argv is None else argv))
        provenance_raw = _read_named(args["--build-provenance"])
        manifest_raw = _read_named(args["--seed-manifest"])
        envelope_raw = _read_named(args["--approved-seed-envelope"])
        inventory_raw = _read_named(args["--historical-path-inventory"])
        provenance_value = parse_canonical_json(provenance_raw)
        manifest_value = parse_canonical_json(manifest_raw)
        envelope_value = parse_canonical_json(envelope_raw)
        inventory_value = parse_canonical_json(inventory_raw)

        phase = "PROVENANCE"
        if sha256_hex(provenance_raw) != args["--build-provenance-sha256"]:
            raise PrototypeInvariantError("PROVENANCE_DIGEST")
        provenance = build_provenance_from_dict(provenance_value)

        phase = "SEED_AUDIT"
        if sha256_hex(envelope_raw) != args["--approved-seed-envelope-sha256"]:
            raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
        envelope: ApprovedSeedEnvelope = approved_envelope_from_dict(envelope_value)
        manifest = seed_manifest_from_dict(manifest_value)
        _, seed_audit = validate_seed_audit(manifest, manifest_raw, envelope, inventory_value, inventory_raw)

        phase = "GENERATION"
        result = run_production_beta(
            manifest, provenance, seed_audit,
            build_provenance_sha256=args["--build-provenance-sha256"],
            approved_envelope_sha256=args["--approved-seed-envelope-sha256"],
        )
        phase = "SERIALIZATION"
        validate_success_schema(result)
        encoded = canonical_json_bytes(result) + b"\n"
        sys.stdout.buffer.write(encoded)
        return 0
    except Exception as error:
        text = str(error)
        code = text if text in FAILURE_CODES else {
            "INPUT": "INPUT_SCHEMA", "PROVENANCE": "PROVENANCE_CONTENT",
            "SEED_AUDIT": "SEED_AUDIT_DIGEST", "GENERATION": "GENERATION_INVARIANT",
            "TRAINING": "TRAINING_INVARIANT", "READOUT": "READOUT_INVALID",
            "BOOTSTRAP": "BOOTSTRAP_INVALID", "SERIALIZATION": "SERIALIZATION_INVALID",
        }.get(phase, "SERIALIZATION_INVALID")
        if code.startswith("TRAINING_"):
            phase = "TRAINING"
        elif code == "READOUT_INVALID":
            phase = "READOUT"
        elif code == "BOOTSTRAP_INVALID":
            phase = "BOOTSTRAP"
        elif code == "SERIALIZATION_INVALID":
            phase = "SERIALIZATION"
        sys.stdout.buffer.write(canonical_json_bytes(failure_artifact(phase, code)) + b"\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
