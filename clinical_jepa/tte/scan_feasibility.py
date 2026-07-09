from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.targets.block_spans import empty_target_len
from clinical_jepa.tte.audit_metadata_availability import _reject_unsafe_input_path
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json
from clinical_jepa.validation import validate_artifact

SPLITS = ("train", "dev", "test", "stress")
SOURCE_ROLES = {"primary", "inspect_external", "external_validation", "other_aggregate", "unknown"}
COUNT_FIELDS = (
    "context_med_count",
    "context_lab_count",
    "context_state_count",
    "target_med_count",
    "target_lab_count",
    "target_state_count",
)
SAFE_METADATA_FIELDS = {
    *COUNT_FIELDS,
    "context_len",
    "target_len",
    "sequence_len",
    "contact_count",
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


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(_num(row, key, default))


def _pct(numer: float, denom: float) -> float:
    return float(100.0 * numer / denom) if denom else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _target_len(row: dict[str, Any]) -> int:
    if row.get("target_len") is not None:
        return _int(row, "target_len")
    return empty_target_len(row)  # 0 for empty/censored; inclusive span else


def _context_len(row: dict[str, Any]) -> int:
    if row.get("context_len") is not None:
        return _int(row, "context_len")
    if row.get("context_start_ref") is not None and row.get("context_end_ref") is not None:
        return max(0, _int(row, "context_end_ref") - _int(row, "context_start_ref") + 1)
    return 0


def _contact_count(row: dict[str, Any]) -> float:
    explicit = row.get("contact_count")
    if explicit is not None:
        return _num(row, "contact_count")
    # Event count is an intentionally coarse fallback for aggregate-only scans.
    return float(_context_len(row) + _target_len(row))


def _normalize_source_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in SOURCE_ROLES:
        return role
    if "inspect" in role:
        return "inspect_external"
    if role in {"mimic", "mimic_train", "mimic_dev", "mimic_test"}:
        return "primary"
    return "other_aggregate" if role else "unknown"


def _source_role(row: dict[str, Any], default_source_role: str) -> str:
    if row.get("source_role"):
        return _normalize_source_role(row["source_role"])
    src = str(row.get("source_dataset") or "").lower()
    if "inspect" in src:
        return "inspect_external"
    if default_source_role:
        return _normalize_source_role(default_source_role)
    return "primary"


def _load_blocks(target_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = target_manifest.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("target-block manifest must contain a blocks list")
    return [dict(b) for b in blocks if isinstance(b, dict)]


def _iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_metadata_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = _reject_unsafe_input_path(path, role="metadata-index")
    if p.suffix == ".jsonl":
        return _iter_jsonl(p)
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [r for r in data["rows"] if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("blocks"), list):
        return [r for r in data["blocks"] if isinstance(r, dict)]
    raise ValueError(f"Unsupported metadata index format: {p}")


def _merge_metadata(blocks: list[dict[str, Any]], metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not metadata_rows:
        return blocks
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


def _scenario_id(scenario: dict[str, Any]) -> str:
    return str(scenario.get("scenario_id") or scenario.get("id") or "unnamed_scenario")


def _rules(scenario: dict[str, Any]) -> dict[str, Any]:
    return scenario.get("aggregate_rules") or scenario.get("rules") or {}


def _minimums(scenario: dict[str, Any]) -> dict[str, Any]:
    return scenario.get("minimums") or {}


def _field_positive(row: dict[str, Any], fields: list[str]) -> bool:
    if not fields:
        return False
    return any(_num(row, f, 0.0) > 0 for f in fields)


def _scenario_guardrail_warnings(scenario: dict[str, Any], *, leakage_checked: bool) -> list[str]:
    required = scenario.get("required_guardrails") or [
        "equivalent_contact_comparator",
        "incident_lookback",
        "endpoint_leakage_embargo",
        "negative_controls_defined",
        "contact_intensity_controls_defined",
        "source_measurability",
    ]
    status = scenario.get("guardrail_status") or {}
    warnings: list[str] = []
    if not leakage_checked:
        warnings.append("leakage_not_checked")
    for key in required:
        value = status.get(key)
        if value is True:
            continue
        if isinstance(value, str) and value.strip().lower() in {"pass", "passed", "verified", "yes", "true"}:
            continue
        warnings.append(f"guardrail_not_verified:{key}")
    return warnings


def _is_eligible(row: dict[str, Any], rules: dict[str, Any]) -> tuple[bool, str | None]:
    target_types = set(rules.get("eligible_target_types") or ["T0", "T1"])
    if str(row.get("target_type")) not in target_types:
        return False, "target_type_not_eligible"
    min_context = int(rules.get("min_context_events", 0) or 0)
    if _context_len(row) < min_context:
        return False, "inadequate_lookback"
    min_target = int(rules.get("min_target_events", 0) or 0)
    if _target_len(row) < min_target:
        return False, "missing_followup_or_target"
    return True, None


def _is_incident(row: dict[str, Any], rules: dict[str, Any]) -> bool:
    incident_types = set(rules.get("incident_target_types") or ["T1"])
    fields = list(rules.get("incident_positive_fields") or [])
    type_match = str(row.get("target_type")) in incident_types
    if fields:
        return type_match and _field_positive(row, fields)
    return type_match


def _is_comparator(row: dict[str, Any], rules: dict[str, Any]) -> bool:
    comparator_types = set(rules.get("comparator_target_types") or ["T0"])
    fields = list(rules.get("comparator_positive_fields") or [])
    type_match = str(row.get("target_type")) in comparator_types
    if fields:
        return type_match and _field_positive(row, fields)
    return type_match


def _decision(result: dict[str, Any], scenario: dict[str, Any]) -> str:
    minimums = _minimums(scenario)
    min_initiators = int(minimums.get("incident_initiators", 1) or 0)
    min_comparators = int(minimums.get("comparator_candidates", 1) or 0)
    min_eligible = int(minimums.get("eligible", 1) or 0)
    min_followup_pct = float(minimums.get("followup_available_pct", 0.0) or 0.0)
    min_target_pct = float(minimums.get("target_block_available_pct", 0.0) or 0.0)

    if result["n_eligible"] < min_eligible:
        return "park"
    if result["n_incident_initiators"] < min_initiators:
        return "redesign"
    if result["n_comparator_candidates"] < min_comparators:
        return "redesign"
    if result["followup_available_pct"] < min_followup_pct:
        return "redesign"
    if result["target_block_available_pct"] < min_target_pct:
        return "redesign"
    if result["warnings"]:
        return "redesign"
    return "promote"


def _configured_metadata_fields(rules: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for key in [
        "incident_positive_fields",
        "comparator_positive_fields",
        "proxy_positive_fields",
        "negative_control_positive_fields",
    ]:
        fields.update(str(x) for x in (rules.get(key) or []))
    return fields


def _summarize_group(rows: list[dict[str, Any]], scenario: dict[str, Any], *, source_role: str, split: str, leakage_checked: bool) -> dict[str, Any]:
    rules = _rules(scenario)
    exclusion_counts: collections.Counter[str] = collections.Counter()
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        ok, reason = _is_eligible(row, rules)
        if ok:
            eligible_rows.append(row)
        elif reason:
            exclusion_counts[reason] += 1

    n_eligible = len(eligible_rows)
    incident = [r for r in eligible_rows if _is_incident(r, rules)]
    comparators = [r for r in eligible_rows if _is_comparator(r, rules)]
    baseline_min = int(rules.get("baseline_lookback_events", rules.get("min_context_events", 0)) or 0)
    followup_min = int(rules.get("followup_events", rules.get("min_target_events", 0)) or 0)
    proxy_fields = list(rules.get("proxy_positive_fields") or ["target_lab_count", "target_state_count"])
    negative_fields = list(rules.get("negative_control_positive_fields") or [])
    configured_fields = _configured_metadata_fields(rules)
    missing_fields = sorted(f for f in configured_fields if not any(f in r for r in rows))

    context_ok = sum(1 for r in eligible_rows if _context_len(r) >= baseline_min)
    followup_ok = sum(1 for r in eligible_rows if _target_len(r) >= followup_min)
    target_ok = sum(1 for r in eligible_rows if _target_len(r) > 0)
    proxy_positive = sum(1 for r in eligible_rows if _field_positive(r, proxy_fields))
    negative_positive = sum(1 for r in eligible_rows if _field_positive(r, negative_fields))
    contact_counts = [_contact_count(r) for r in eligible_rows]

    patient_hashes = {str(r.get("patient_hash")) for r in rows if r.get("patient_hash")}
    result: dict[str, Any] = {
        "scenario_id": _scenario_id(scenario),
        "source_role": source_role,
        "split": split,
        "n_subjects": len(patient_hashes) if patient_hashes else 0,
        "n_sequences": len(rows),
        "n_eligible": n_eligible,
        "n_incident_initiators": len(incident),
        "n_comparator_candidates": len(comparators),
        "baseline_lookback_complete_pct": _pct(context_ok, n_eligible),
        "followup_available_pct": _pct(followup_ok, n_eligible),
        "target_block_available_pct": _pct(target_ok, n_eligible),
        "proxy_event_rate_pct": _pct(proxy_positive, n_eligible),
        "negative_control_rate_pct": _pct(negative_positive, n_eligible),
        "median_contact_count": _median(contact_counts),
        "median_lab_or_event_density": _median([_num(r, "target_lab_count", 0.0) + _num(r, "context_lab_count", 0.0) for r in eligible_rows]),
        "event_family_summary": {f: int(sum(_num(r, f, 0.0) for r in eligible_rows)) for f in COUNT_FIELDS},
        "missing_metadata_fields": missing_fields,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "warnings": _scenario_guardrail_warnings(scenario, leakage_checked=leakage_checked),
        "decision": "insufficient_information",
        "notes": "Aggregate-only feasibility counts; no IDs, raw tokens, or patient examples are emitted.",
    }

    minimums = _minimums(scenario)
    if result["n_incident_initiators"] < int(minimums.get("incident_initiators", 1) or 0):
        result["warnings"].append("incident_initiator_count_below_minimum")
    if result["n_comparator_candidates"] < int(minimums.get("comparator_candidates", 1) or 0):
        result["warnings"].append("comparator_count_below_minimum")
    if not negative_fields:
        result["warnings"].append("negative_control_fields_not_configured")
    if not proxy_fields:
        result["warnings"].append("proxy_fields_not_configured")
    if not rules.get("comparator_positive_fields"):
        result["warnings"].append("comparator_equivalent_contact_fields_not_configured")
    if not rules.get("incident_positive_fields"):
        result["warnings"].append("incident_positive_fields_not_configured")
    if result["n_subjects"] == 0:
        result["warnings"].append("subject_count_unavailable_in_input")
    for field in missing_fields:
        result["warnings"].append(f"metadata_field_missing:{field}")

    result["decision"] = _decision(result, scenario)
    return result


def _expected_availability_fields(scenario: dict[str, Any]) -> set[str]:
    return {
        "split",
        "target_type",
        "source_dataset",
        "context_len",
        "target_len",
        "context_med_count",
        "context_lab_count",
        "context_state_count",
        "target_med_count",
        "target_lab_count",
        "target_state_count",
        *_configured_metadata_fields(_rules(scenario)),
    }


def _metadata_availability_warnings(
    report: dict[str, Any] | None,
    *,
    dry_run: bool,
    scenario: dict[str, Any],
    n_target_rows: int,
    n_metadata_rows: int,
) -> list[str]:
    if report:
        errors = validate_artifact("metadata-availability", report, raise_on_error=False)
        if errors:
            return ["metadata_availability_invalid"]
        warnings: list[str] = []
        decision = report.get("overall_decision")
        if decision != "pass":
            warnings.append(f"metadata_availability_not_pass:{decision or 'unknown'}")
        if report.get("aggregate_only") is not True:
            warnings.append("metadata_availability_not_aggregate_only")
        if str(report.get("scenario_id")) != str(_scenario_id(scenario)):
            warnings.append("metadata_availability_scenario_mismatch")
        if int(report.get("n_target_rows", -1)) != int(n_target_rows):
            warnings.append("metadata_availability_target_row_count_mismatch")
        if int(report.get("n_metadata_rows", -1)) != int(n_metadata_rows):
            warnings.append("metadata_availability_metadata_row_count_mismatch")
        gates = report.get("pass_park_gates") if isinstance(report.get("pass_park_gates"), dict) else {}
        required_total = int(gates.get("required_fields_total", 0) or 0)
        required_passing = int(gates.get("required_fields_passing", -1) or -1)
        if required_total <= 0 or required_passing != required_total:
            warnings.append("metadata_availability_required_gates_not_satisfied")
        required_results = [r for r in report.get("field_results", []) if isinstance(r, dict) and r.get("tier") == "required"]
        if len(required_results) != required_total:
            warnings.append("metadata_availability_required_field_result_count_mismatch")
        bad_required = [r for r in required_results if r.get("status") not in {"present", "derivable"}]
        if bad_required:
            warnings.append("metadata_availability_required_fields_not_available")
        reported_required_fields = {str(r.get("field")) for r in required_results}
        missing_expected = sorted(_expected_availability_fields(scenario) - reported_required_fields)
        if missing_expected:
            warnings.append("metadata_availability_expected_fields_missing:" + ",".join(missing_expected))
        return warnings
    if dry_run:
        return []
    return ["metadata_availability_not_checked"]


def scan_scenario_feasibility(
    target_manifest: dict[str, Any],
    scenario: dict[str, Any],
    *,
    default_source_role: str = "primary",
    dry_run: bool = False,
    leakage_checked: bool | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    metadata_availability_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if leakage_checked is None:
        leakage_checked = not dry_run
    blocks = _merge_metadata(_load_blocks(target_manifest), metadata_rows or [])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for block in blocks:
        split = str(block.get("split") or "unknown")
        if split not in SPLITS:
            split = "unknown"
        role = _source_role(block, default_source_role)
        grouped[(role, split)].append(block)

    results = [
        _summarize_group(rows, scenario, source_role=role, split=split, leakage_checked=leakage_checked)
        for (role, split), rows in sorted(grouped.items())
    ]
    metadata_warnings = _metadata_availability_warnings(
        metadata_availability_report,
        dry_run=dry_run,
        scenario=scenario,
        n_target_rows=len(blocks),
        n_metadata_rows=len(metadata_rows or []),
    )
    overall_decision = "promote" if results and all(r["decision"] == "promote" for r in results) else "redesign"
    if not results:
        overall_decision = "park"
    if any(r["decision"] == "park" for r in results) or metadata_warnings:
        overall_decision = "park"

    report = {
        "schema_version": "clinical-jepa-scenario-feasibility-v0",
        "created_utc": now_utc(),
        "scenario_id": _scenario_id(scenario),
        "dry_run": bool(dry_run),
        "leakage_checked": bool(leakage_checked),
        "aggregate_only": True,
        "source_target_blocks": str(target_manifest.get("schema_version", "target-block-manifest")),
        "n_groups": len(results),
        "overall_decision": overall_decision,
        "results": results,
        "warnings": sorted({w for r in results for w in r.get("warnings", [])} | set(metadata_warnings)),
        "notes": "Scenario feasibility is aggregate-only. It is a TTE/specification readiness diagnostic, not a causal estimate.",
    }
    validate_artifact("scenario-feasibility", report)
    return report


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA scenario feasibility scan",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Overall decision: **{report['overall_decision']}**",
        f"Dry run: {report['dry_run']}",
        "",
        "This is an aggregate-only TTE/specification readiness diagnostic, not a causal analysis.",
        "",
        "## Groups",
        "",
    ]
    for r in report.get("results", []):
        lines.extend([
            f"### {r['source_role']} / {r['split']}",
            "",
            f"- Eligible: {r['n_eligible']} / sequences: {r['n_sequences']} / subjects counted: {r['n_subjects']}",
            f"- Incident initiators: {r['n_incident_initiators']}",
            f"- Comparator candidates: {r['n_comparator_candidates']}",
            f"- Baseline complete: {r['baseline_lookback_complete_pct']:.1f}%",
            f"- Follow-up available: {r['followup_available_pct']:.1f}%",
            f"- Proxy event rate: {r['proxy_event_rate_pct']:.1f}%",
            f"- Negative-control rate: {r['negative_control_rate_pct']:.1f}%",
            f"- Decision: {r['decision']}",
            f"- Warnings: {', '.join(r['warnings']) if r['warnings'] else 'none'}",
            "",
        ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate-only Clinical-JEPA TTE scenario feasibility scanner")
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--scenario-card", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata-index", help="Optional JSON/JSONL index keyed by block_id containing aggregate-only fields such as counts and guardrail flags")
    ap.add_argument("--metadata-availability-report", help="Optional metadata availability audit; must pass for real-mode promotion")
    ap.add_argument("--leakage-report", help="Required unless --dry-run is set")
    ap.add_argument("--source-role", default="primary", help="Default source role when target blocks do not specify one")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.dry_run:
        if not args.leakage_report:
            raise SystemExit("--leakage-report is required unless --dry-run is set")
        require_pass_leakage(args.leakage_report)

    target_manifest = read_json(args.target_blocks)
    scenario = load_yaml(args.scenario_card)
    metadata_rows = _load_metadata_rows(args.metadata_index)
    metadata_availability_report = read_json(_reject_unsafe_input_path(args.metadata_availability_report, role="metadata-availability-report")) if args.metadata_availability_report else None
    report = scan_scenario_feasibility(
        target_manifest,
        scenario,
        default_source_role=args.source_role,
        dry_run=args.dry_run,
        leakage_checked=not args.dry_run,
        metadata_rows=metadata_rows,
        metadata_availability_report=metadata_availability_report,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "scenario-feasibility.json", report)
    (outdir / "summary.md").write_text(_summary_md(report))
    print(json.dumps({"output": str(outdir / "scenario-feasibility.json"), "overall_decision": report["overall_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
