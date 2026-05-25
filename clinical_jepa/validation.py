from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"

KIND_TO_SCHEMA = {
    "split-manifest": "clinical-jepa-split-manifest-v0.schema.json",
    "target-block-manifest": "clinical-jepa-target-block-manifest-v0.schema.json",
    "leakage-audit": "clinical-jepa-leakage-audit-v0.schema.json",
    "metrics-bundle": "clinical-jepa-metrics-bundle-v0.schema.json",
    "query-descriptors": "clinical-jepa-query-descriptors-v0.schema.json",
    "v0a-embedding-cache": "clinical-jepa-v0a-embedding-cache-v0.schema.json",
    "v0b-train-manifest": "clinical-jepa-v0b-train-manifest-v0.schema.json",
    "metadata-availability": "clinical-jepa-metadata-availability-v0.schema.json",
    "scenario-feasibility": "clinical-jepa-scenario-feasibility-v0.schema.json",
    "pseudo-rendering-readiness": "clinical-jepa-pseudo-rendering-readiness-v0.schema.json",
    "autoregression-readiness": "clinical-jepa-autoregression-readiness-v0.schema.json",
}

VERSION_TO_KIND = {
    "clinical-jepa-split-manifest-v0": "split-manifest",
    "clinical-jepa-target-block-manifest-v0": "target-block-manifest",
    "clinical-jepa-leakage-audit-v0": "leakage-audit",
    "clinical-jepa-query-descriptors-v0": "query-descriptors",
    "clinical-jepa-v0a-embedding-cache-v0": "v0a-embedding-cache",
    "clinical-jepa-v0b-train-manifest-v0": "v0b-train-manifest",
    "clinical-jepa-metadata-availability-v0": "metadata-availability",
    "clinical-jepa-scenario-feasibility-v0": "scenario-feasibility",
    "clinical-jepa-pseudo-rendering-readiness-v0": "pseudo-rendering-readiness",
    "clinical-jepa-autoregression-readiness-v0": "autoregression-readiness",
}

TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}

AGGREGATE_ONLY_KINDS = {"metadata-availability", "scenario-feasibility", "pseudo-rendering-readiness", "autoregression-readiness"}
FORBIDDEN_AGGREGATE_KEYS = {
    "block_id",
    "block_ids",
    "patient_hash",
    "patient_hashes",
    "sequence_id",
    "sequence_ids",
    "sequence_file",
    "source_id",
    "source_ids",
    "token",
    "tokens",
    "token_id",
    "token_ids",
    "raw_tokens",
    "h5_path",
    "hdf5_path",
    "embedding",
    "embeddings",
    "checkpoint",
    "checkpoints",
}


def load_schema(kind: str) -> dict[str, Any]:
    if kind not in KIND_TO_SCHEMA:
        raise ValueError(f"Unknown schema kind: {kind}")
    return json.loads((SCHEMA_DIR / KIND_TO_SCHEMA[kind]).read_text())


def infer_kind(data: dict[str, Any], path: Path | None = None) -> str:
    version = data.get("schema_version")
    if version in VERSION_TO_KIND:
        return VERSION_TO_KIND[version]
    if path and path.name in {"baseline-results.json", "retrieval-results.json"}:
        return "metrics-bundle"
    raise ValueError("Cannot infer schema kind; pass --kind")


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_type_ok(value, item) for item in expected)
    typ = TYPE_MAP[expected]
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, typ)


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/definitions/"):
        raise ValueError(f"Unsupported $ref: {ref}")
    key = ref.split("/")[-1]
    return root["definitions"][key]


def validate_schema(data: Any, schema: dict[str, Any], *, root: dict[str, Any] | None = None, path: str = "$") -> list[str]:
    root = root or schema
    schema = _resolve_ref(schema, root)
    errors: list[str] = []
    if "type" in schema and not _type_ok(data, schema["type"]):
        errors.append(f"{path}: expected {schema['type']}, got {type(data).__name__}")
        return errors
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value {data!r} not in enum {schema['enum']!r}")
    if "const" in schema and data != schema["const"]:
        errors.append(f"{path}: value {data!r} != const {schema['const']!r}")
    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required key {req!r}")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in data:
                errors.extend(validate_schema(data[key], subschema, root=root, path=f"{path}.{key}"))
    if isinstance(data, list) and "items" in schema:
        for idx, item in enumerate(data):
            errors.extend(validate_schema(item, schema["items"], root=root, path=f"{path}[{idx}]"))
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data!r} < minimum {schema['minimum']!r}")
    return errors


def _scan_forbidden_aggregate_keys(data: Any, *, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in FORBIDDEN_AGGREGATE_KEYS:
                errors.append(f"{path}: forbidden aggregate-only key {key!r}")
            errors.extend(_scan_forbidden_aggregate_keys(value, path=f"{path}.{key}"))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            errors.extend(_scan_forbidden_aggregate_keys(item, path=f"{path}[{idx}]"))
    return errors


def validate_artifact(kind: str, data: dict[str, Any], *, raise_on_error: bool = True) -> list[str]:
    schema = load_schema(kind)
    errors = validate_schema(data, schema)
    if kind in AGGREGATE_ONLY_KINDS:
        errors.extend(_scan_forbidden_aggregate_keys(data))
    if errors and raise_on_error:
        raise ValueError("Schema validation failed for " + kind + ":\n" + "\n".join(errors))
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate Clinical-JEPA JSON artifacts against local schemas")
    ap.add_argument("--kind", default="auto", choices=["auto", *sorted(KIND_TO_SCHEMA)])
    ap.add_argument("--file", required=True)
    args = ap.parse_args(argv)
    path = Path(args.file)
    data = json.loads(path.read_text())
    kind = infer_kind(data, path) if args.kind == "auto" else args.kind
    errors = validate_artifact(kind, data, raise_on_error=False)
    if errors:
        print(json.dumps({"file": str(path), "kind": kind, "ok": False, "errors": errors}, indent=2))
        return 2
    print(json.dumps({"file": str(path), "kind": kind, "ok": True}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
