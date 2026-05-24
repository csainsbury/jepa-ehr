from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, write_json
from clinical_jepa.validation import validate_artifact

ALLOWED_SOURCE_ROLES = {"primary", "inspect_external", "external_validation", "other_aggregate", "unknown"}
ALLOWED_SPLITS = {"train", "dev", "test", "stress", "unknown"}

FORBIDDEN_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".parquet",
    ".feather",
    ".arrow",
    ".csv",
    ".tsv",
}

SAFE_METADATA_FIELDS = {
    "split",
    "target_type",
    "source_dataset",
    "source_role",
    "context_len",
    "target_len",
    "sequence_len",
    "contact_count",
    "context_start_ref",
    "context_end_ref",
    "target_start_ref",
    "target_end_ref",
    "context_med_count",
    "context_lab_count",
    "context_state_count",
    "target_med_count",
    "target_lab_count",
    "target_state_count",
    "equivalent_contact_present",
    "negative_control_present",
    "diuretic_initiator_present",
    "active_med_comparator_present",
    "prior_exposure_lookback_complete",
    "prior_diuretic_or_fluid_med_exposure",
    "scenario_specific_lab_family_present",
    "surveillance_control_present",
    "biologic_negative_control_present",
    "scenario_consistent",
}


def _scenario_id(requirements: dict[str, Any], scenario: dict[str, Any] | None) -> str:
    if scenario:
        return str(scenario.get("scenario_id") or scenario.get("id") or requirements.get("scenario_id") or "unnamed_scenario")
    return str(requirements.get("scenario_id") or "unnamed_scenario")


def _reject_unsafe_input_path(path: str | Path, *, role: str) -> Path:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        raise SystemExit(f"Refusing {role} with unsafe/raw-like suffix: {p}")
    if suffix not in {".json", ".jsonl", ".yaml", ".yml"}:
        raise SystemExit(f"Refusing {role}; only JSON/JSONL/YAML metadata inputs are allowed: {p}")
    parts = {part.lower() for part in p.parts}
    # Allow public schemas/configs, but reject obvious raw/generated artifact paths.
    if role == "metadata-index" and parts.intersection({"raw", "checkpoints", "embeddings", "bundles", "sequences", "token_ids", "sequence_file"}):
        raise SystemExit(f"Refusing {role} from raw/checkpoint/embedding-like path: {p}")
    return p


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = _reject_unsafe_input_path(path, role="metadata-index")
    if p.suffix == ".jsonl":
        return _iter_jsonl(p)
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return [dict(r) for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [dict(r) for r in data["rows"] if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("blocks"), list):
        return [dict(r) for r in data["blocks"] if isinstance(r, dict)]
    raise ValueError(f"Unsupported metadata rows format: {p}")


def _load_target_blocks(path: str | Path) -> list[dict[str, Any]]:
    p = _reject_unsafe_input_path(path, role="target-blocks")
    data = read_json(p)
    blocks = data.get("blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list):
        raise ValueError("target-blocks must be a JSON manifest with a blocks list")
    return [dict(b) for b in blocks if isinstance(b, dict)]


def _merge_rows(blocks: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block = {str(r.get("block_id")): r for r in metadata_rows if r.get("block_id")}
    merged: list[dict[str, Any]] = []
    for block in blocks:
        row = dict(block)
        meta = by_block.get(str(block.get("block_id")))
        if meta:
            for key in SAFE_METADATA_FIELDS:
                if key in meta:
                    row[key] = meta[key]
        merged.append(row)
    return merged


def _normalize_source_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in ALLOWED_SOURCE_ROLES:
        return role
    if "inspect" in role:
        return "inspect_external"
    if role in {"mimic", "mimic_train", "mimic_dev", "mimic_test"}:
        return "primary"
    return "other_aggregate" if role else "unknown"


def _normalize_split(value: Any) -> str:
    split = str(value or "").strip().lower()
    return split if split in ALLOWED_SPLITS else "unknown"


def _source_role(row: dict[str, Any]) -> str:
    if row.get("source_role"):
        return _normalize_source_role(row["source_role"])
    src = str(row.get("source_dataset") or "").lower()
    if "inspect" in src:
        return "inspect_external"
    if src:
        return "primary"
    return "unknown"


def _has_field(row: dict[str, Any], field: str, requirements: dict[str, Any]) -> bool:
    if field in row and row[field] is not None:
        return True
    derivable = requirements.get("derivable_fields") or {}
    spec = derivable.get(field) or {}
    for option in spec.get("any_of") or []:
        if all(k in row and row[k] is not None for k in option):
            return True
    return False


def _field_source(rows: list[dict[str, Any]], field: str, requirements: dict[str, Any]) -> str:
    if any(field in r and r[field] is not None for r in rows):
        return "present"
    derivable = requirements.get("derivable_fields") or {}
    spec = derivable.get(field) or {}
    for option in spec.get("any_of") or []:
        if any(all(k in r and r[k] is not None for k in option) for r in rows):
            return "derivable_from:" + "+".join(option)
    return "missing"


def _field_result(rows: list[dict[str, Any]], field: str, tier: str, requirements: dict[str, Any], threshold: float) -> dict[str, Any]:
    denom = len(rows)
    n_present = sum(1 for r in rows if _has_field(r, field, requirements))
    coverage = (100.0 * n_present / denom) if denom else 0.0
    source = _field_source(rows, field, requirements)
    if coverage >= threshold:
        status = "derivable" if source.startswith("derivable_from:") else "present"
    elif coverage > 0:
        status = "partial"
    else:
        status = "missing"
    warning = None
    if status in {"missing", "partial"}:
        warning = f"metadata_field_{status}:{field}"
    return {
        "field": field,
        "tier": tier,
        "status": status,
        "coverage_pct": float(coverage),
        "source": source,
        "warning": warning,
    }


def _group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = collections.Counter()
    for row in rows:
        grouped[(_source_role(row), _normalize_split(row.get("split")))] += 1
    return [
        {"source_role": role, "split": split, "n_rows": int(n)}
        for (role, split), n in sorted(grouped.items())
    ]


def audit_metadata_availability(
    requirements: dict[str, Any],
    target_blocks: list[dict[str, Any]],
    *,
    scenario: dict[str, Any] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata_rows = metadata_rows or []
    merged = _merge_rows(target_blocks, metadata_rows)
    thresholds = requirements.get("coverage_thresholds") or {}
    required_threshold = float(thresholds.get("required_min_pct", 100.0) or 100.0)
    preferred_threshold = float(thresholds.get("strongly_preferred_min_pct", 80.0) or 80.0)
    required_fields = [str(x) for x in requirements.get("required_fields", [])]
    preferred_fields = [str(x) for x in requirements.get("strongly_preferred_fields", [])]

    field_results = [
        _field_result(merged, field, "required", requirements, required_threshold)
        for field in required_fields
    ]
    field_results.extend(
        _field_result(merged, field, "strongly_preferred", requirements, preferred_threshold)
        for field in preferred_fields
    )

    required_failures = [r for r in field_results if r["tier"] == "required" and r["status"] not in {"present", "derivable"}]
    warnings = [str(r["warning"]) for r in field_results if r.get("warning")]
    if not metadata_rows:
        warnings.append("metadata_index_not_provided")
    if not target_blocks:
        warnings.append("target_blocks_empty")

    overall_decision = "pass" if target_blocks and not required_failures else "park"
    report = {
        "schema_version": "clinical-jepa-metadata-availability-v0",
        "created_utc": now_utc(),
        "scenario_id": _scenario_id(requirements, scenario),
        "aggregate_only": True,
        "overall_decision": overall_decision,
        "n_target_rows": len(target_blocks),
        "n_metadata_rows": len(metadata_rows),
        "n_merged_rows": len(merged),
        "field_results": field_results,
        "group_summaries": _group_summaries(merged),
        "pass_park_gates": {
            "required_fields_total": len(required_fields),
            "required_fields_passing": len(required_fields) - len(required_failures),
            "required_min_pct": required_threshold,
            "strongly_preferred_min_pct": preferred_threshold,
            "metadata_index_provided": bool(metadata_rows),
        },
        "warnings": sorted(set(warnings)),
        "notes": "Aggregate-only metadata availability audit. It reports field coverage only; no join keys, patient hashes, token strings, token IDs, HDF5 paths, embeddings, checkpoints, or row examples are emitted.",
    }
    validate_artifact("metadata-availability", report)
    return report


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA metadata availability audit",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Decision: **{report['overall_decision']}**",
        f"Target rows inspected: {report['n_target_rows']}",
        f"Metadata rows inspected: {report['n_metadata_rows']}",
        "",
        "Aggregate field-coverage audit only; not a scenario scan or causal analysis.",
        "",
        "## Field results",
        "",
        "| Field | Tier | Status | Coverage % | Source |",
        "|---|---|---:|---:|---|",
    ]
    for r in report.get("field_results", []):
        lines.append(f"| `{r['field']}` | {r['tier']} | {r['status']} | {r['coverage_pct']:.1f} | {r['source']} |")
    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        lines.extend(f"- {w}" for w in report["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate-only metadata availability audit for Clinical-JEPA TTE/readout scans")
    ap.add_argument("--requirements", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-card")
    ap.add_argument("--metadata-index")
    args = ap.parse_args(argv)

    requirements_path = _reject_unsafe_input_path(args.requirements, role="requirements")
    scenario_path = _reject_unsafe_input_path(args.scenario_card, role="scenario-card") if args.scenario_card else None
    requirements = load_yaml(requirements_path)
    scenario = load_yaml(scenario_path) if scenario_path else None
    target_blocks = _load_target_blocks(args.target_blocks)
    metadata_rows = _load_rows(args.metadata_index)
    report = audit_metadata_availability(requirements, target_blocks, scenario=scenario, metadata_rows=metadata_rows)
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "metadata-availability.json", report)
    (outdir / "summary.md").write_text(_summary_md(report))
    print(json.dumps({"output": str(outdir / "metadata-availability.json"), "overall_decision": report["overall_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
