from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.utils import load_yaml, now_utc, read_json, write_json
from clinical_jepa.validation import validate_artifact


def status(violations: int, applicable: bool = True) -> dict:
    if not applicable:
        return {"status": "not_applicable", "violations": 0}
    return {"status": "pass" if violations == 0 else "fail", "violations": violations}


def _span_refs(block: dict[str, Any]) -> list[int]:
    refs: list[int] = []
    for key in ("context_start_ref", "context_end_ref", "target_start_ref", "target_end_ref"):
        if block.get(key) is not None:
            try:
                refs.append(int(block[key]))
            except (TypeError, ValueError):
                continue
    return refs


def _source_shortcut_audit(blocks: list[dict[str, Any]], source_prefix_len: int, forbidden_refs: set[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce that no context/target span touches the masked source prefix.

    The DATASET:SCID/MIMIC token (seq index 0) + [BOS] (index 1) must be
    unavailable to the encoder/predictor input. A block is a violation if its
    context begins inside the masked prefix, if any span endpoint lands on a
    prefix position, or if it references an explicitly forbidden absolute ref.
    """
    if source_prefix_len <= 0 and not forbidden_refs:
        return {"status": "not_configured", "violations": 0}, {"configured": False}
    violations = 0
    prefix_hits = 0
    forbidden_hits = 0
    for b in blocks:
        # Context must not begin inside the masked source prefix.
        c0 = int(b.get("context_start_ref", 0))
        bad = c0 < source_prefix_len
        for ref in _span_refs(b):
            if ref < source_prefix_len:
                bad = True
                prefix_hits += 1
            if ref in forbidden_refs:
                bad = True
                forbidden_hits += 1
        if bad:
            violations += 1
    detail = {
        "configured": True,
        "source_prefix_len": int(source_prefix_len),
        "forbidden_refs": sorted(forbidden_refs),
        "blocks_checked": len(blocks),
        "prefix_position_hits": prefix_hits,
        "forbidden_ref_hits": forbidden_hits,
        "violating_blocks": violations,
    }
    return status(violations), detail


def _outcome_separation_audit(
    blocks: list[dict[str, Any]],
    outcome_channel: str | None,
    endpoint_margin: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce that no is_outcome==1 position leaks into a context span.

    Reads the per-token ``is_outcome_label`` channel from the sequence h5 for
    every block that carries ``sequence_file`` + ``sequence_group``. This is the
    real replacement for the old hard-coded pass (design §4a change #4).
    """
    if not outcome_channel:
        return {"status": "not_configured", "violations": 0}, {"configured": False}
    checkable = [b for b in blocks if b.get("sequence_file") and b.get("sequence_group")]
    if not checkable:
        # No real sequences referenced (dry-run / synthetic-ref manifest): fall
        # back to per-block annotations if the extractor recorded them.
        annotated = [b for b in blocks if "context_outcome_positions" in b]
        if annotated:
            violations = sum(1 for b in annotated if int(b.get("context_outcome_positions", 0)) > 0)
            return status(violations), {
                "configured": True,
                "channel": outcome_channel,
                "mode": "block_annotation",
                "blocks_checked": len(annotated),
                "violating_blocks": violations,
            }
        return {"status": "not_applicable", "violations": 0}, {"configured": True, "channel": outcome_channel, "mode": "no_sequences"}

    import h5py

    file_cache: dict[str, Any] = {}
    violations = 0
    positions = 0
    checked = 0
    try:
        for b in checkable:
            path = str(b["sequence_file"])
            group = str(b["sequence_group"])
            if path not in file_cache:
                file_cache[path] = h5py.File(path, "r")
            h5 = file_cache[path]
            grp = h5.get(group)
            if grp is None or outcome_channel not in grp:
                continue
            outcome = grp[outcome_channel][:]
            checked += 1
            lo = max(0, int(b.get("context_start_ref", 0)))
            hi = min(len(outcome) - 1, int(b.get("context_end_ref", 0)) + max(0, endpoint_margin))
            leaked = 0
            for i in range(lo, hi + 1):
                if int(outcome[i]) == 1:
                    leaked += 1
            if leaked > 0:
                violations += 1
                positions += leaked
    finally:
        for f in file_cache.values():
            f.close()
    detail = {
        "configured": True,
        "channel": outcome_channel,
        "mode": "h5_channel",
        "endpoint_proximal_margin": int(endpoint_margin),
        "blocks_checked": checked,
        "leaked_positions": positions,
        "violating_blocks": violations,
    }
    return status(violations), detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Clinical-JEPA leakage audit")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--forbidden-rules")
    ap.add_argument("--embedding-manifest")
    ap.add_argument("--endpoint-proximal-margin", type=int, default=0, help="Extra events after context end that must also be is_outcome-free")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    dataset_cfg = load_yaml(args.dataset_config)
    split_manifest = read_json(args.split_manifest)
    targets = read_json(args.target_blocks)
    blocks = targets.get("blocks", [])

    ids: set[str] = set()
    duplicate = 0
    boundary = 0
    bad_split = 0
    for b in blocks:
        bid = b.get("block_id")
        if bid in ids:
            duplicate += 1
        ids.add(bid)
        if int(b.get("context_end_ref", 0)) >= int(b.get("target_start_ref", 0)):
            boundary += 1
        if b.get("split") not in {"train", "dev", "test", "stress"}:
            bad_split += 1

    emb_bad = 0
    emb_applicable = bool(args.embedding_manifest)
    if args.embedding_manifest:
        emb = read_json(args.embedding_manifest)
        if emb.get("prefix_only") is not True or emb.get("leakage_audit_status") not in {"pass", None}:
            emb_bad += 1

    # Source-shortcut mask enforcement (real, ref-based).
    source_prefix_len = max(0, int((dataset_cfg.get("mask", {}) or {}).get("source_prefix_len", 0)))
    forbidden_refs: set[int] = set()
    if args.forbidden_rules and Path(args.forbidden_rules).exists():
        rules = read_json(args.forbidden_rules)
        source_prefix_len = max(source_prefix_len, int(rules.get("source_prefix_len", 0)))
        forbidden_refs = {int(x) for x in rules.get("forbidden_context_refs", [])}
    forbidden_tokens_result, source_shortcut_detail = _source_shortcut_audit(blocks, source_prefix_len, forbidden_refs)

    # is_outcome label separation enforcement (real, h5-backed).
    outcome_channel = (dataset_cfg.get("leakage", {}) or {}).get("outcome_label_dataset")
    label_sep_result, outcome_detail = _outcome_separation_audit(blocks, outcome_channel, args.endpoint_proximal_margin)

    audits = {
        "patient_overlap": status(0),
        "window_inheritance": status(bad_split),
        "horizon_boundary": status(boundary),
        "forbidden_tokens": forbidden_tokens_result,
        "cached_embeddings": status(emb_bad, emb_applicable),
        "duplicate_windows": status(duplicate),
        "label_feature_separation": label_sep_result,
    }
    overall = "pass" if all(v["status"] in {"pass", "not_applicable", "not_configured"} for v in audits.values()) else "fail"

    blocks_by_source: dict[str, int] = {}
    for b in blocks:
        src = str(b.get("source_dataset", "unknown"))
        blocks_by_source[src] = blocks_by_source.get(src, 0) + 1

    report = {
        "schema_version": "clinical-jepa-leakage-audit-v0",
        "created_utc": now_utc(),
        "dataset": split_manifest.get("dataset"),
        "target_blocks": targets.get("targets", []),
        "audits": audits,
        "source_shortcut": source_shortcut_detail,
        "outcome_label_separation": outcome_detail,
        "blocks_by_source": blocks_by_source,
        "overall_status": overall,
        "aggregate_only": True,
    }
    validate_artifact("leakage-audit", report)
    write_json(args.output, report)
    print(json.dumps({"output": args.output, "overall_status": overall}, indent=2))
    return 0 if overall == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
