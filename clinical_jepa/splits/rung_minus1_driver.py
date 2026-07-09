"""Composite rung -1 gate driver (Pi round-3 Q3).

The rung -1 gate has two halves the individual CLIs report separately:
  - substrate floor+presence -> ``readiness_manifest.gate_status``
  - real-data leakage/mask audit -> ``run_leakage_audit.overall_status``
Nothing AND-combined them, so a consumer could read readiness "PASS" as complete
on the substrate half alone (Pi Q3). This driver combines both statuses **plus a
governance scan** into ONE fail-closed composite gate, so downstream automation
checks a single ``composite_status``.

It is a *combiner*: it consumes the two aggregate-only manifests produced by
``clinical_jepa.splits.readiness_manifest`` and ``clinical_jepa.audit.run_leakage_audit``
(optionally recomputing readiness from the index so a stale readiness manifest
cannot slip through), validates each against its schema, scans both for
aggregate-only violations, checks provenance consistency against the dataset
config, and emits a schema-validated composite manifest. Aggregate-only; no
sequence ids / tokens / real paths beyond the input manifest paths it was given.

Fail-closed: any component or governance check failing => ``composite_status=fail``
and a non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, write_json
from clinical_jepa.validation import (
    _scan_forbidden_aggregate_keys,
    validate_artifact,
)


class CompositeGateError(RuntimeError):
    """Raised when the composite rung -1 gate does not pass (fail-closed)."""


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "detail": detail}


def _expected_sources(dataset_cfg: dict[str, Any]) -> set[str]:
    readiness = dataset_cfg.get("readiness", {}) or {}
    if readiness.get("expected_sources"):
        return {str(s) for s in readiness["expected_sources"]}
    primary = (dataset_cfg.get("sources", {}) or {}).get("primary", {}) or {}
    return {str(s.get("name")) for s in (primary.get("source_datasets") or []) if s.get("name")}


def build_composite_gate(
    readiness: dict[str, Any],
    leakage: dict[str, Any],
    *,
    dataset_cfg: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    wallclock_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """AND-combine readiness + leakage + governance into one composite gate.

    Every check must pass for ``composite_status == "pass"``. Schema-invalid or
    aggregate-only-violating inputs FAIL the gate (they are not silently trusted).
    """
    checks: list[dict[str, Any]] = []

    # 1. Both inputs are schema-valid for their declared kind.
    readiness_errors = validate_artifact("rung-minus1-readiness", readiness, raise_on_error=False)
    leakage_errors = validate_artifact("leakage-audit", leakage, raise_on_error=False)
    checks.append(_check("readiness_schema_valid", not readiness_errors,
                         "ok" if not readiness_errors else f"{len(readiness_errors)} schema errors: {readiness_errors[:3]}"))
    checks.append(_check("leakage_schema_valid", not leakage_errors,
                         "ok" if not leakage_errors else f"{len(leakage_errors)} schema errors: {leakage_errors[:3]}"))

    # 2. Component statuses (the two halves of the rung -1 gate). We do NOT trust
    # the self-reported aggregate verbatim: a readiness gate_status="pass" with a
    # non-empty under_floor/missing, or a leakage overall_status="pass" whose
    # critical audits merely "not_configured"/"not_applicable" (the gameable
    # pass-by-absence the fail-hard design closes only when require_source_mask is
    # asserted), must NOT pass this composite.
    readiness_gate = str(readiness.get("gate_status"))
    under_floor = readiness.get("under_floor") or []
    missing = readiness.get("missing") or []
    checks.append(_check("readiness_gate_pass", readiness_gate == "pass" and not under_floor and not missing,
                         f"gate_status={readiness_gate!r} under_floor={under_floor} missing={missing}"))

    leakage_status = str(leakage.get("overall_status"))
    audits = leakage.get("audits", {}) or {}

    def _astat(name: str) -> str:
        return str((audits.get(name) or {}).get("status"))

    any_failed_audit = any(str((v or {}).get("status")) == "fail" for v in audits.values())
    # The two leakage-critical audits for rung -1 must be genuinely verified on
    # real data (== "pass"), NOT "not_configured"/"not_applicable": the source
    # shortcut mask (forbidden_tokens) and the is_outcome separation.
    critical_ok = _astat("forbidden_tokens") == "pass" and _astat("label_feature_separation") == "pass"
    checks.append(_check("leakage_audit_pass", leakage_status == "pass",
                         f"overall_status={leakage_status!r}"))
    checks.append(_check("leakage_no_failed_audits", not any_failed_audit,
                         f"failed audits={[k for k,v in audits.items() if str((v or {}).get('status'))=='fail']}"))
    checks.append(_check("leakage_critical_audits_verified", critical_ok,
                         f"forbidden_tokens={_astat('forbidden_tokens')!r} label_feature_separation={_astat('label_feature_separation')!r}"
                         " (both must be 'pass', not not_configured/not_applicable)"))
    # Defense in depth: `not_configured` means a configured check did not actually
    # run (the gameable pass-by-absence). NO audit may be `not_configured`.
    # `not_applicable` is allowed only for cached_embeddings (nothing to leak when
    # there is no embedding manifest); a `not_applicable` anywhere else is suspect.
    unconfigured = [k for k, v in audits.items() if str((v or {}).get("status")) == "not_configured"]
    stray_na = [k for k, v in audits.items()
                if str((v or {}).get("status")) == "not_applicable" and k != "cached_embeddings"]
    checks.append(_check("leakage_no_unconfigured_audits", not unconfigured and not stray_na,
                         f"not_configured={unconfigured} stray_not_applicable={stray_na}"))

    # 3. Governance scan: both must be aggregate-only in flag AND content.
    gov_errors: list[str] = []
    if readiness.get("aggregate_only") is not True:
        gov_errors.append("readiness.aggregate_only is not True")
    if leakage.get("aggregate_only") is not True:
        gov_errors.append("leakage.aggregate_only is not True")
    gov_errors.extend(f"readiness{e}" for e in _scan_forbidden_aggregate_keys(readiness))
    gov_errors.extend(f"leakage{e}" for e in _scan_forbidden_aggregate_keys(leakage))
    checks.append(_check("governance_aggregate_only", not gov_errors,
                         "ok" if not gov_errors else f"{len(gov_errors)} violations: {gov_errors[:3]}"))

    # 4a. Source-prefix FLOOR (independent of cfg): the rung -1 mask requires
    # source_prefix_len >= 2 (DATASET token + [BOS]); a mask of 0 means no source
    # masking happened at all, so equality alone (0==0) must not pass.
    try:
        r_prefix = int(readiness.get("source_prefix_len"))
    except (TypeError, ValueError):
        r_prefix = -2
    min_prefix = 2
    if dataset_cfg is not None:
        min_prefix = max(2, int((dataset_cfg.get("mask", {}) or {}).get("min_source_prefix_len", 2)))
    checks.append(_check("source_prefix_floor", r_prefix >= min_prefix,
                         f"readiness source_prefix_len={r_prefix} < required min {min_prefix}"))

    # 4b. Provenance consistency against the dataset config (guards stale-manifest mixing).
    # A governance gate must not pass without provenance verification: if no dataset
    # config is supplied, the consistency checks cannot run, so the gate fails closed.
    checks.append(_check("provenance_verified", dataset_cfg is not None,
                         "ok" if dataset_cfg is not None else "no dataset-config supplied; provenance not verified"))
    if dataset_cfg is not None:
        cfg_prefix = int((dataset_cfg.get("mask", {}) or {}).get("source_prefix_len", -1))
        checks.append(_check("source_prefix_consistency", cfg_prefix == r_prefix,
                             f"config source_prefix_len={cfg_prefix} vs readiness={r_prefix}"))

        expected = _expected_sources(dataset_cfg)
        r_sources = set(readiness.get("per_source", {}).keys())
        l_sources = set(leakage.get("blocks_by_source", {}).keys())
        # An empty expected set means we cannot verify source presence -> fail.
        checks.append(_check("readiness_sources_present", bool(expected) and expected.issubset(r_sources),
                             f"expected={sorted(expected)} readiness_per_source={sorted(r_sources)}"))
        # Leakage blocks may legitimately be empty for a source, but must not
        # introduce an unexpected source (would signal a mismatched run).
        checks.append(_check("leakage_sources_subset", l_sources.issubset(expected) if l_sources else True,
                             f"leakage_blocks_by_source={sorted(l_sources)} expected={sorted(expected)}"))

    # 5. Wall-clock readiness (optional; only when gating the WALL-CLOCK rung).
    # Each expected source must have >= 1 adequate (non-degenerate, floor-clearing)
    # horizon; empty/censored-dominated horizons are flagged conditional/incomplete.
    wallclock_status = "not_provided"
    if wallclock_readiness is not None:
        wc_errors = validate_artifact("wallclock-readiness", wallclock_readiness, raise_on_error=False)
        checks.append(_check("wallclock_schema_valid", not wc_errors,
                             "ok" if not wc_errors else f"{len(wc_errors)} schema errors: {wc_errors[:3]}"))
        adq = wallclock_readiness.get("adequate_horizons_by_source", {}) or {}
        per_src = wallclock_readiness.get("per_source", {}) or {}
        srcs = set(per_src.keys()) or set(adq.keys())
        all_adequate = bool(srcs) and all(adq.get(s) for s in srcs)
        checks.append(_check("wallclock_horizons_adequate", all_adequate,
                             f"adequate_horizons_by_source={adq}"))
        gov_wc = (wallclock_readiness.get("aggregate_only") is True) and not _scan_forbidden_aggregate_keys(wallclock_readiness)
        checks.append(_check("wallclock_aggregate_only", gov_wc, "ok" if gov_wc else "governance violation"))
        wallclock_status = "pass" if (not wc_errors and all_adequate and gov_wc) else "fail"

    failed = [c["name"] for c in checks if c["status"] != "pass"]
    composite_status = "pass" if not failed else "fail"

    manifest = {
        "schema_version": "clinical-jepa-rung-minus1-composite-v0",
        "created_utc": now_utc(),
        "composite_status": composite_status,
        "component_status": {
            "readiness_gate": readiness_gate,
            "leakage_audit": leakage_status,
            "governance_scan": "pass" if not gov_errors else "fail",
            "wallclock_readiness": wallclock_status,
        },
        "checks": checks,
        "failed_checks": failed,
        "inputs": dict(inputs or {}),
        "aggregate_only": True,
        "notes": (
            "composite rung -1 gate: AND-combines readiness (substrate floor+presence) "
            "+ leakage audit (mask + is_outcome separation) + governance scan; fail-closed."
        ),
    }
    validate_artifact("rung-minus1-composite", manifest)
    return manifest


def assert_composite_or_raise(manifest: dict[str, Any]) -> None:
    if manifest.get("composite_status") != "pass":
        raise CompositeGateError(
            "rung -1 composite gate FAILED (fail-closed): failed_checks -> "
            + json.dumps(manifest.get("failed_checks", []))
        )


def _resolve_readiness(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (readiness_manifest, dataset_cfg_or_None).

    Either read a provided readiness manifest, or recompute it from the index +
    configs (so a stale readiness manifest cannot slip through).
    """
    dataset_cfg = load_yaml(args.dataset_config) if args.dataset_config else None
    if args.readiness_manifest:
        return read_json(args.readiness_manifest), dataset_cfg
    if not (args.index_dir and args.dataset_config and args.arms_config):
        raise SystemExit(
            "provide --readiness-manifest, or --index-dir + --dataset-config + --arms-config to recompute it"
        )
    from clinical_jepa.splits.readiness_manifest import build_readiness_manifest

    arms_cfg = load_yaml(args.arms_config)
    index_paths: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        p = Path(args.index_dir) / f"{split}.index.jsonl"
        if p.exists():
            index_paths[split] = str(p)
    readiness = build_readiness_manifest(
        dataset_cfg, arms_cfg, index_paths, min_valid_windows=args.min_valid_windows,
    )
    return readiness, dataset_cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Composite rung -1 gate: AND-combine readiness + leakage + governance (fail-closed)")
    ap.add_argument("--readiness-manifest", help="Path to a rung -1 readiness manifest (else recompute from --index-dir)")
    ap.add_argument("--index-dir", help="Directory of {split}.index.jsonl (to recompute readiness)")
    ap.add_argument("--dataset-config", help="Dataset config (for recompute + provenance consistency checks)")
    ap.add_argument("--arms-config", help="Arms config (for recompute)")
    ap.add_argument("--min-valid-windows", type=int, default=None)
    ap.add_argument("--leakage-audit", required=True, help="Path to a run_leakage_audit report")
    ap.add_argument("--wallclock-readiness", help="Optional wall-clock readiness manifest (gates the wall-clock rung)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    readiness, dataset_cfg = _resolve_readiness(args)
    leakage = read_json(args.leakage_audit)
    wallclock = read_json(args.wallclock_readiness) if args.wallclock_readiness else None
    inputs = {
        "readiness_manifest": str(args.readiness_manifest or f"(recomputed from {args.index_dir})"),
        "leakage_audit": str(args.leakage_audit),
    }
    if args.wallclock_readiness:
        inputs["wallclock_readiness"] = str(args.wallclock_readiness)
    manifest = build_composite_gate(readiness, leakage, dataset_cfg=dataset_cfg, inputs=inputs,
                                    wallclock_readiness=wallclock)

    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "rung-minus1-composite-gate.json", manifest)
    print(json.dumps({
        "output": str(outdir / "rung-minus1-composite-gate.json"),
        "composite_status": manifest["composite_status"],
        "component_status": manifest["component_status"],
        "failed_checks": manifest["failed_checks"],
    }, indent=2))
    assert_composite_or_raise(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
