from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, stable_hmac, write_json
from clinical_jepa.validation import validate_artifact

# Default number of leading sequence positions that carry the source-shortcut
# prefix (index 0 = DATASET:SCID/MIMIC, index 1 = [BOS]). These are stripped
# from every context/target span so the encoder/predictor input cannot use the
# source token as a shortcut. `source_dataset` is retained on the block for
# evaluator-side stratification only. See rung0_1_run_specs.md (cross-cutting
# guards) and clinical-jepa-native-generation-design.md §4a change #3.
DEFAULT_SOURCE_PREFIX_LEN = 0
DEFAULT_MIN_CONTEXT = 8


def fake_hash(split: str, idx: int) -> str:
    return stable_hmac(f"{split}-{idx}", "synthetic-target-salt")


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_id_to_token(vocab_json_path: str | None) -> dict[int, str]:
    if not vocab_json_path:
        return {}
    p = Path(vocab_json_path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    if "id_to_token" in raw:
        return {int(k): str(v) for k, v in raw["id_to_token"].items()}
    if "token_to_id" in raw:
        return {int(v): str(k) for k, v in raw["token_to_id"].items()}
    # FlatASCEND vocabulary.json is id-string -> token.
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def _limit_for_split(args: argparse.Namespace, split: str) -> int:
    return int({"train": args.max_real_train, "dev": args.max_real_dev, "test": args.max_real_test}[split])


def _source_spec(split_manifest: dict[str, Any], source_role: str) -> dict[str, Any]:
    if source_role == "primary":
        return split_manifest
    if source_role == "external_validation":
        spec = split_manifest.get("external_validation") or {}
        if not spec:
            raise ValueError("split manifest lacks external_validation source metadata")
        return spec
    raise ValueError(f"unsupported source role: {source_role}")


def _source_paths(source_spec: dict[str, Any], split: str) -> tuple[str, str | None]:
    """Return (index_path, split_level_h5_path_or_None) for a split.

    For a joint / multi-source substrate the split-level h5 path may be absent
    (each index row carries its own ``source_h5_path``). In that case the h5
    path is returned as ``None`` and the extractor uses the per-row path.
    """
    index_paths = source_spec.get("source_index_paths", {})
    h5_paths = source_spec.get("source_h5_paths", {}) or {}
    if split not in index_paths:
        raise ValueError(f"source metadata lacks source index path for split={split}")
    return str(index_paths[split]), (str(h5_paths[split]) if split in h5_paths else None)


def t0_feasible(
    seq_len: int,
    target_window: int,
    gap_events: int = 0,
    *,
    context_start: int = 0,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> bool:
    """Whether a T0 block can be carved from a sequence of ``seq_len`` events.

    Shared by the extractor and the rung -1 readiness manifest so block-yield
    counts and actual emission use identical arithmetic.
    """
    context_start = max(0, int(context_start))
    gap_events = max(0, int(gap_events))
    min_context = max(1, int(min_context))
    target_window = max(1, int(target_window))
    if seq_len < context_start + min_context + gap_events + target_window + 2:
        return False
    max_context_end = seq_len - gap_events - target_window - 2
    return (context_start + min_context - 1) <= max_context_end


def _t0_block(
    seq_id: str,
    split: str,
    seq_len: int,
    source_dataset: str,
    ordinal: int,
    target_window: int,
    gap_events: int = 0,
    *,
    context_start: int = 0,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> dict[str, Any] | None:
    context_start = max(0, int(context_start))
    gap_events = max(0, int(gap_events))
    min_context = max(1, int(min_context))
    if not t0_feasible(seq_len, target_window, gap_events, context_start=context_start, min_context=min_context):
        return None
    max_context_end = seq_len - gap_events - target_window - 2
    lowest_context_end = context_start + min_context - 1
    midpoint = (context_start + seq_len) // 2
    context_end = min(max(lowest_context_end, midpoint), max_context_end)
    target_start = context_end + 1 + gap_events
    target_end = min(seq_len - 1, target_start + target_window - 1)
    if context_end >= target_start or target_start >= target_end or context_end < context_start:
        return None
    return {
        "block_id": stable_hmac(f"T0|{seq_id}|{context_start}|{context_end}|{target_start}|{target_end}|gap{gap_events}|{ordinal}", "clinical-jepa-real-block-v0"),
        "patient_hash": stable_hmac(seq_id, "clinical-jepa-rekeyed-seq"),
        "sequence_id": seq_id,
        "sequence_group": seq_id,
        "split": split,
        "target_type": "T0",
        "context_start_ref": int(context_start),
        "context_end_ref": int(context_end),
        "target_start_ref": int(target_start),
        "target_end_ref": int(target_end),
        "horizon_descriptor": f"event_gap_{gap_events}_window_{target_window}",
        "gap_events": int(gap_events),
        "source_dataset": source_dataset,
        "unit": "event_index",
    }


def _t1_block(
    seq_id: str,
    split: str,
    seq_len: int,
    source_dataset: str,
    anchor_idx: int,
    ordinal: int,
    target_window: int,
    *,
    context_start: int = 0,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> dict[str, Any] | None:
    context_start = max(0, int(context_start))
    min_context = max(1, int(min_context))
    if anchor_idx < context_start + min_context or anchor_idx >= seq_len - 3:
        return None
    context_end = anchor_idx - 1
    target_start = anchor_idx
    target_end = min(seq_len - 1, target_start + target_window - 1)
    if context_end >= target_start or target_start >= target_end or context_end < context_start:
        return None
    return {
        "block_id": stable_hmac(f"T1|{seq_id}|{context_start}|{anchor_idx}|{target_end}|{ordinal}", "clinical-jepa-real-block-v0"),
        "patient_hash": stable_hmac(seq_id, "clinical-jepa-rekeyed-seq"),
        "sequence_id": seq_id,
        "sequence_group": seq_id,
        "split": split,
        "target_type": "T1",
        "context_start_ref": int(context_start),
        "context_end_ref": int(context_end),
        "target_start_ref": int(target_start),
        "target_end_ref": int(target_end),
        "horizon_descriptor": "medication_anchor_event_window",
        "source_dataset": source_dataset,
        "anchor_type": "medication_token",
        "unit": "event_index",
    }


def _resolve_source_prefix_len(dataset_cfg: dict[str, Any]) -> int:
    mask = dataset_cfg.get("mask", {}) or {}
    return max(0, int(mask.get("source_prefix_len", DEFAULT_SOURCE_PREFIX_LEN)))


def _resolve_outcome_channel(dataset_cfg: dict[str, Any]) -> str | None:
    leakage = dataset_cfg.get("leakage", {}) or {}
    channel = leakage.get("outcome_label_dataset")
    return str(channel) if channel else None


def _resolve_windows(common: dict[str, Any], target_type: str, source_dataset: str | None) -> tuple[int, int]:
    """Resolve (event_count_window, min_context) for a target type + source.

    Per-source overrides live under ``target_blocks.<type>.per_source.<SOURCE>``
    so short per-admission MIMIC sequences can use a smaller window instead of
    being silently dropped (rung0_1_run_specs.md: MIMIC windowing).
    """
    tcfg = common.get("target_blocks", {}) or {}
    tspec = tcfg.get(target_type, {})
    if not isinstance(tspec, dict):
        tspec = {}
    default_window = int(tspec.get("event_count_window", 32))
    default_min_context = int(tspec.get("min_context", DEFAULT_MIN_CONTEXT))
    per_source = (tspec.get("per_source", {}) or {}).get(source_dataset or "", {}) or {}
    window = int(per_source.get("event_count_window", default_window))
    min_context = int(per_source.get("min_context", default_min_context))
    return max(1, window), max(1, min_context)


def _outcome_positions_in_span(outcome: Any, start: int, end: int) -> int:
    """Count is_outcome==1 positions in the inclusive span [start, end]."""
    if outcome is None:
        return 0
    lo = max(0, int(start))
    hi = min(len(outcome) - 1, int(end))
    if hi < lo:
        return 0
    count = 0
    for i in range(lo, hi + 1):
        if int(outcome[i]) == 1:
            count += 1
    return count


def _real_blocks(args: argparse.Namespace, dataset_cfg: dict[str, Any], split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py  # imported only for real extraction

    arms = load_yaml(args.arms_config)
    common = arms.get("common", {})
    vocab = _load_id_to_token(dataset_cfg.get("vocabulary", {}).get("vocab_json_path"))
    med_ids = {tid for tid, tok in vocab.items() if tok.startswith("MED:")}

    source_prefix_len = _resolve_source_prefix_len(dataset_cfg)
    outcome_channel = _resolve_outcome_channel(dataset_cfg)
    endpoint_margin = max(0, int(args.endpoint_proximal_margin))

    blocks: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    source_spec = _source_spec(split_manifest, args.source_role)
    fallback_source = source_spec.get("dataset") or source_spec.get("name") or split_manifest.get("dataset", "flatascend-b1a-rekeyed")
    processed = {"train": 0, "dev": 0, "test": 0}
    processed_by_source: dict[str, int] = {}
    skipped_short = {"train": 0, "dev": 0, "test": 0}
    t1_no_anchor = {"train": 0, "dev": 0, "test": 0}
    refused_outcome_in_context = {"train": 0, "dev": 0, "test": 0}
    blocks_by_source: dict[str, int] = {}

    for split in ["train", "dev", "test"]:
        index_path, split_h5_path = _source_paths(source_spec, split)
        limit = _limit_for_split(args, split)
        file_cache: dict[str, Any] = {}
        try:
            for row in _iter_jsonl(index_path):
                if limit and processed[split] >= limit:
                    break
                seq_id = str(row.get("sequence_id") or row.get("group"))
                group_name = str(row.get("group") or seq_id)
                seq_len = int(row.get("seq_len") or 0)
                source_dataset = str(row.get("source_dataset") or fallback_source)
                h5_path = str(row.get("source_h5_path") or split_h5_path or "")
                if not seq_id or seq_len <= 0 or not h5_path:
                    continue
                processed[split] += 1
                processed_by_source[source_dataset] = processed_by_source.get(source_dataset, 0) + 1

                if h5_path not in file_cache:
                    file_cache[h5_path] = h5py.File(h5_path, "r")
                h5 = file_cache[h5_path]
                grp = h5.get(group_name)
                token_ids = grp["token_ids"][:] if (grp is not None and "token_ids" in grp) else []
                outcome = None
                if outcome_channel and grp is not None and outcome_channel in grp:
                    outcome = grp[outcome_channel][:]

                t0_window, t0_min_context = _resolve_windows(common, "T0", source_dataset)
                t1_window, t1_min_context = _resolve_windows(common, "T1", source_dataset)

                made = False
                if "T0" in args.targets:
                    block = _t0_block(
                        seq_id, split, seq_len, source_dataset, 0, t0_window, args.t0_gap_events,
                        context_start=source_prefix_len, min_context=t0_min_context,
                    )
                    if block is not None:
                        # Refuse to emit if is_outcome==1 leaks into context (or the
                        # endpoint-proximal margin just before the target start).
                        leak = _outcome_positions_in_span(outcome, block["context_start_ref"], block["context_end_ref"] + endpoint_margin)
                        if leak > 0:
                            refused_outcome_in_context[split] += 1
                        else:
                            block["sequence_file"] = h5_path
                            block["endpoint_safe"] = True
                            block["context_outcome_positions"] = 0
                            blocks.append(block)
                            counts.setdefault(split, {}).setdefault("T0", 0)
                            counts[split]["T0"] += 1
                            blocks_by_source[source_dataset] = blocks_by_source.get(source_dataset, 0) + 1
                            made = True

                if "T1" in args.targets and med_ids:
                    anchors = [
                        i for i, tid in enumerate(token_ids)
                        if int(tid) in med_ids and i >= source_prefix_len + t1_min_context and i < seq_len - 3
                    ]
                    if not anchors:
                        t1_no_anchor[split] += 1
                    for j, anchor_idx in enumerate(anchors[: max(0, args.t1_anchors_per_sequence)]):
                        block = _t1_block(
                            seq_id, split, seq_len, source_dataset, int(anchor_idx), j, t1_window,
                            context_start=source_prefix_len, min_context=t1_min_context,
                        )
                        if block is None:
                            continue
                        leak = _outcome_positions_in_span(outcome, block["context_start_ref"], block["context_end_ref"] + endpoint_margin)
                        if leak > 0:
                            refused_outcome_in_context[split] += 1
                            continue
                        block["sequence_file"] = h5_path
                        block["endpoint_safe"] = True
                        block["context_outcome_positions"] = 0
                        blocks.append(block)
                        counts.setdefault(split, {}).setdefault("T1", 0)
                        counts[split]["T1"] += 1
                        blocks_by_source[source_dataset] = blocks_by_source.get(source_dataset, 0) + 1
                        made = True

                if not made:
                    skipped_short[split] += 1
        finally:
            for f in file_cache.values():
                f.close()

    report = {
        "processed_sequences": processed,
        "processed_by_source": processed_by_source,
        "skipped_without_blocks": skipped_short,
        "t1_no_medication_anchor": t1_no_anchor,
        "refused_outcome_in_context": refused_outcome_in_context,
        "blocks_by_source": blocks_by_source,
        "targets": args.targets,
        "caps": {"train": args.max_real_train, "dev": args.max_real_dev, "test": args.max_real_test},
        "t0_gap_events": int(args.t0_gap_events),
        "source_prefix_len": int(source_prefix_len),
        "endpoint_proximal_margin": int(endpoint_margin),
        "outcome_label_channel": outcome_channel,
        "source_role": args.source_role,
        "aggregate_only": True,
    }
    return blocks, {"counts": counts, "report": report}


def _synthetic_blocks(args: argparse.Namespace, split_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count_map = {
        "train": int(split_manifest["counts"].get("patients_train", 0)),
        "dev": int(split_manifest["counts"].get("patients_dev", 0)),
        "test": int(split_manifest["counts"].get("patients_test", 0)),
    }
    blocks = []
    for split, n in count_map.items():
        for i in range(min(n, args.max_synthetic_per_split)):
            ph = fake_hash(split, i)
            base = 10 + i * 5
            if "T0" in args.targets:
                blocks.append({"block_id": stable_hmac(f"T0-{split}-{i}", "synthetic-block"), "patient_hash": ph, "split": split, "target_type": "T0", "context_start_ref": 0, "context_end_ref": base, "target_start_ref": base + 1, "target_end_ref": base + 32, "horizon_descriptor": "short", "source_dataset": split_manifest.get("dataset", "synthetic")})
            if "T1" in args.targets:
                blocks.append({"block_id": stable_hmac(f"T1-{split}-{i}", "synthetic-block"), "patient_hash": ph, "split": split, "target_type": "T1", "context_start_ref": 0, "context_end_ref": base + 5, "target_start_ref": base + 6, "target_end_ref": base + 18, "horizon_descriptor": "medium", "source_dataset": split_manifest.get("dataset", "synthetic"), "anchor_type": "new_start"})
    counts: dict[str, dict[str, int]] = {}
    for b in blocks:
        counts.setdefault(b["split"], {}).setdefault(b["target_type"], 0)
        counts[b["split"]][b["target_type"]] += 1
    return blocks, {"counts": counts, "report": {"dry_run": True, "aggregate_only": True}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract Clinical-JEPA target blocks")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--targets", nargs="+", default=["T0", "T1"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-synthetic-per-split", type=int, default=12)
    ap.add_argument("--max-real-train", type=int, default=0, help="0 means no cap")
    ap.add_argument("--max-real-dev", type=int, default=0, help="0 means no cap")
    ap.add_argument("--max-real-test", type=int, default=0, help="0 means no cap")
    ap.add_argument("--t1-anchors-per-sequence", type=int, default=1)
    ap.add_argument("--t0-gap-events", type=int, default=0, help="Number of events to skip between context end and T0 target start")
    ap.add_argument("--endpoint-proximal-margin", type=int, default=0, help="Additional events after context end that must also be free of is_outcome==1 (endpoint-proximal exclusion)")
    ap.add_argument("--source-role", default="primary", choices=["primary", "external_validation"], help="Source to extract from when the split manifest includes external validation metadata")
    args = ap.parse_args(argv)

    dataset_cfg = load_yaml(args.dataset_config)
    _arms = load_yaml(args.arms_config)
    split_manifest = read_json(args.split_manifest)
    outdir = ensure_dir(args.output_dir)

    if args.dry_run:
        blocks, details = _synthetic_blocks(args, split_manifest)
        dry_run = True
        notes = "synthetic block refs only; no raw tokens or timestamps"
    else:
        blocks, details = _real_blocks(args, dataset_cfg, split_manifest)
        dry_run = False
        notes = "Real re-keyed bundle block refs only; no raw tokens, source ids, timestamps, or patient examples."

    manifest = {
        "schema_version": "clinical-jepa-target-block-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": dry_run,
        "source_split_manifest": str(Path(args.split_manifest)),
        "source_role": args.source_role,
        "targets": args.targets,
        "counts": details["counts"],
        "blocks": blocks,
        "extraction_report": details.get("report", {}),
        "notes": notes,
    }
    validate_artifact("target-block-manifest", manifest)
    write_json(outdir / "target-block-manifest.json", manifest)
    write_json(outdir / "target-extraction-report.json", {"created_utc": now_utc(), "n_blocks": len(blocks), **details.get("report", {})})
    print(json.dumps({"target_blocks": str(outdir / "target-block-manifest.json"), "n_blocks": len(blocks), "dry_run": dry_run, "counts": details["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
