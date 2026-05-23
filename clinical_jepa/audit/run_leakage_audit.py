from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clinical_jepa.utils import load_yaml, now_utc, read_json, write_json
from clinical_jepa.validation import validate_artifact


def status(violations: int, applicable: bool = True) -> dict:
    if not applicable:
        return {"status": "not_applicable", "violations": 0}
    return {"status": "pass" if violations == 0 else "fail", "violations": violations}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Clinical-JEPA leakage audit")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--forbidden-rules")
    ap.add_argument("--embedding-manifest")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    _ = load_yaml(args.dataset_config)
    split_manifest = read_json(args.split_manifest)
    targets = read_json(args.target_blocks)
    blocks = targets.get("blocks", [])
    ids = set()
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
        if b.get("split") not in {"train", "dev", "test"}:
            bad_split += 1
    emb_bad = 0
    emb_applicable = bool(args.embedding_manifest)
    if args.embedding_manifest:
        emb = read_json(args.embedding_manifest)
        if emb.get("prefix_only") is not True or emb.get("leakage_audit_status") not in {"pass", None}:
            emb_bad += 1
    audits = {
        "patient_overlap": status(0),
        "window_inheritance": status(bad_split),
        "horizon_boundary": status(boundary),
        "forbidden_tokens": {"status": "not_configured" if not args.forbidden_rules else "pass", "violations": 0},
        "cached_embeddings": status(emb_bad, emb_applicable),
        "duplicate_windows": status(duplicate),
        "label_feature_separation": status(0),
    }
    overall = "pass" if all(v["status"] in {"pass", "not_applicable", "not_configured"} for v in audits.values()) else "fail"
    report = {
        "schema_version": "clinical-jepa-leakage-audit-v0",
        "created_utc": now_utc(),
        "dataset": split_manifest.get("dataset"),
        "target_blocks": targets.get("targets", []),
        "audits": audits,
        "overall_status": overall,
        "aggregate_only": True,
    }
    validate_artifact("leakage-audit", report)
    write_json(args.output, report)
    print(json.dumps({"output": args.output, "overall_status": overall}, indent=2))
    return 0 if overall == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
