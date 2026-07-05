"""Rung -1 substrate / eval-readiness manifest emitter (fail-closed).

Operationalises the rung -1 readiness gate (rung0_1_run_specs.md; design §4a
change #1): emit a per-source manifest — sequence counts, token-count
median/quantiles, wall-clock span quantiles, block yield under each proposed
(source-specific) horizon, and split counts — and **fail closed** if either
source contributes fewer than the configured minimum valid matched windows per
source per split. No silent near-all-SCID result.

Aggregate-only: sequence ids never leave this module; only integer counts and
quantiles are written.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import length_bin
from clinical_jepa.targets.extract_blocks import (
    _iter_jsonl,
    _resolve_source_prefix_len,
    _resolve_windows,
    t0_feasible,
)
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, write_json
from clinical_jepa.validation import validate_artifact

DEFAULT_MIN_VALID_WINDOWS = 500
DEFAULT_MIN_MATCHED_CANDIDATES = 2
DEFAULT_REQUIRED_SPLITS = ("train", "dev", "test")
DEFAULT_EXPECTED_SOURCES = ("MIMIC", "SCID")


class ReadinessGateError(RuntimeError):
    """Raised when a source under-contributes valid matched windows, or an
    expected source / required split is absent (fail-closed)."""


def _context_rate_bin(seq_len: int, span_days: float | None) -> str:
    """Coarse events-per-wall-clock-day bin for candidate matching.

    Consistent with the retrieval candidate-matching intent: windows are only
    "matched" when they share source + horizon + length + event-rate structure,
    so a window's true target has length/rate-comparable distractors.
    """
    if span_days is None or span_days <= 0:
        return "rate_na"
    rate = float(seq_len) / float(span_days)
    for edge, name in ((0.1, "r<0.1"), (0.5, "0.1-0.5"), (1.0, "0.5-1"), (2.0, "1-2"), (5.0, "2-5"), (10.0, "5-10")):
        if rate < edge:
            return name
    return "10+"


def _expected_sources(dataset_cfg: dict[str, Any], readiness_cfg: dict[str, Any]) -> set[str]:
    if readiness_cfg.get("expected_sources"):
        return {str(s) for s in readiness_cfg["expected_sources"]}
    primary = (dataset_cfg.get("sources", {}) or {}).get("primary", {}) or {}
    names = {str(s.get("name")) for s in (primary.get("source_datasets") or []) if s.get("name")}
    return names or set(DEFAULT_EXPECTED_SOURCES)


def _required_splits(readiness_cfg: dict[str, Any]) -> tuple[str, ...]:
    cfg = readiness_cfg.get("required_splits")
    if cfg:
        return tuple(str(s) for s in cfg)
    return DEFAULT_REQUIRED_SPLITS


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _candidate_windows(dataset_cfg: dict[str, Any], common: dict[str, Any], source: str) -> list[tuple[str, int, int]]:
    """(name, event_count_window, min_context) horizons to report yield for."""
    primary_window, min_context = _resolve_windows(common, "T0", source)
    specs: list[tuple[str, int, int]] = [("primary_T0", primary_window, min_context)]
    extra = (dataset_cfg.get("readiness", {}) or {}).get("candidate_windows", []) or []
    for w in extra:
        w = int(w)
        specs.append((f"event_window_{w}", w, min_context))
    return specs


def build_readiness_manifest(
    dataset_cfg: dict[str, Any],
    arms_cfg: dict[str, Any],
    index_paths: dict[str, str],
    *,
    min_valid_windows: int | None = None,
    t0_gap_events: int = 0,
    min_matched_candidates: int | None = None,
) -> dict[str, Any]:
    common = arms_cfg.get("common", {}) or {}
    source_prefix_len = _resolve_source_prefix_len(dataset_cfg)
    readiness_cfg = dataset_cfg.get("readiness", {}) or {}
    if min_valid_windows is None:
        min_valid_windows = int(readiness_cfg.get("min_valid_windows_per_source_per_split", DEFAULT_MIN_VALID_WINDOWS))
    if min_matched_candidates is None:
        min_matched_candidates = int(readiness_cfg.get("min_matched_candidates", DEFAULT_MIN_MATCHED_CANDIDATES))
    gap = int(readiness_cfg.get("t0_gap_events", t0_gap_events))
    expected_sources = _expected_sources(dataset_cfg, readiness_cfg)
    required_splits = _required_splits(readiness_cfg)

    # per_source[source][split] -> stats
    per_source: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, int] = {}

    for split, index_path in index_paths.items():
        if not index_path or not Path(index_path).exists():
            continue
        # Accumulate per-source seq lengths / spans for this split.
        seq_lens: dict[str, list[int]] = {}
        spans: dict[str, list[float]] = {}
        yields: dict[str, dict[str, int]] = {}
        action_counts: dict[str, list[int]] = {}
        action_available: dict[str, bool] = {}
        # Candidate-matching group keys for primary_T0-feasible sequences.
        matched_keys: dict[str, list[tuple[str, str]]] = {}
        for row in _iter_jsonl(index_path):
            source = str(row.get("source_dataset") or "unknown")
            seq_len = int(row.get("seq_len") or 0)
            if seq_len <= 0:
                continue
            seq_lens.setdefault(source, []).append(seq_len)
            span = float(row["wall_clock_span_days"]) if "wall_clock_span_days" in row else None
            if span is not None:
                spans.setdefault(source, []).append(span)
            if "n_candidate_actions" in row:
                action_counts.setdefault(source, []).append(int(row.get("n_candidate_actions") or 0))
                action_available[source] = True
            else:
                action_available.setdefault(source, False)
            source_yields = yields.setdefault(source, {})
            primary_window = None
            primary_min_context = None
            for name, window, min_context in _candidate_windows(dataset_cfg, common, source):
                feasible = t0_feasible(
                    seq_len, window, gap,
                    context_start=source_prefix_len, min_context=min_context,
                )
                source_yields[name] = source_yields.get(name, 0) + (1 if feasible else 0)
                if name == "primary_T0":
                    primary_window, primary_min_context = window, min_context
            # A primary_T0-feasible sequence contributes a candidate-matching key
            # (length bin + event-rate bin) so we can count *matched* windows.
            if primary_window is not None and t0_feasible(
                seq_len, primary_window, gap,
                context_start=source_prefix_len, min_context=primary_min_context,
            ):
                key = (length_bin(seq_len), _context_rate_bin(seq_len, span))
                matched_keys.setdefault(source, []).append(key)

        split_counts[split] = sum(len(v) for v in seq_lens.values())
        for source, lens in seq_lens.items():
            block_yield = yields.get(source, {})
            feasible_windows = int(block_yield.get("primary_T0", 0))
            keys = matched_keys.get(source, [])
            key_counts = Counter(keys)
            # "Matched" = feasible AND lands in a (length, rate) group with enough
            # peers to be a valid retrieval candidate (source-stratified).
            matched = sum(1 for k in keys if key_counts[k] >= min_matched_candidates)
            counts_list = action_counts.get(source, [])
            has_actions = bool(action_available.get(source))
            candidate_action_frequency = (
                float(sum(1 for c in counts_list if c > 0) / len(lens)) if (has_actions and lens) else None
            )
            candidate_action_mean = (
                float(np.mean(counts_list)) if (has_actions and counts_list) else None
            )
            per_source.setdefault(source, {})[split] = {
                "n_sequences": len(lens),
                "token_count_quantiles": _quantiles([float(x) for x in lens]),
                "wall_clock_span_quantiles": _quantiles(spans.get(source, [])),
                "block_yield": block_yield,
                "feasible_windows": feasible_windows,
                "valid_matched_windows": int(matched),
                "matched_candidate_yield": float(matched / feasible_windows) if feasible_windows else 0.0,
                "candidate_action_frequency": candidate_action_frequency,
                "candidate_action_mean_per_sequence": candidate_action_mean,
                "candidate_action_available": has_actions,
                "min_matched_candidates": int(min_matched_candidates),
                "meets_floor": int(matched) >= min_valid_windows,
            }

    # Fail-closed #1: any (source, split) below the matched-window floor.
    under_floor: list[dict[str, Any]] = []
    for source, by_split in per_source.items():
        for split, stats in by_split.items():
            if not stats["meets_floor"]:
                under_floor.append({
                    "source": source,
                    "split": split,
                    "valid_matched_windows": stats["valid_matched_windows"],
                })

    # Fail-closed #2: an expected source or required split is absent / empty.
    # Absence must FAIL, not silently drop from per_source (Pi round 2).
    missing: list[dict[str, Any]] = []
    for source in sorted(expected_sources):
        for split in required_splits:
            stats = per_source.get(source, {}).get(split)
            if not stats or int(stats.get("n_sequences", 0)) <= 0:
                missing.append({"source": source, "split": split, "reason": "absent_or_empty"})

    gate_status = "pass" if (not under_floor and not missing) else "fail"

    manifest = {
        "schema_version": "clinical-jepa-rung-minus1-readiness-v0",
        "created_utc": now_utc(),
        "gate_status": gate_status,
        "min_valid_windows_per_source_per_split": int(min_valid_windows),
        "min_matched_candidates": int(min_matched_candidates),
        "source_prefix_len": int(source_prefix_len),
        "t0_gap_events": int(gap),
        "expected_sources": sorted(expected_sources),
        "required_splits": list(required_splits),
        "split_counts": split_counts,
        "per_source": per_source,
        "under_floor": under_floor,
        "missing": missing,
        "aggregate_only": True,
        "notes": "rung -1 readiness gate; aggregate counts/quantiles only, no sequence ids.",
    }
    validate_artifact("rung-minus1-readiness", manifest)
    return manifest


def assert_gate_or_raise(manifest: dict[str, Any]) -> None:
    if manifest.get("gate_status") != "pass":
        under = json.dumps(manifest.get("under_floor", []))
        missing = json.dumps(manifest.get("missing", []))
        raise ReadinessGateError(
            "rung -1 readiness gate FAILED (fail-closed): "
            f"under_floor (below matched-window floor) -> {under}; "
            f"missing (absent expected source / required split) -> {missing}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit the rung -1 substrate/eval-readiness manifest (fail-closed)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--index-dir", help="Directory of {split}.index.jsonl files (from build_index)")
    ap.add_argument("--train-index")
    ap.add_argument("--dev-index")
    ap.add_argument("--test-index")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-valid-windows", type=int, default=None, help="Override the per-source per-split floor (default from config or 500)")
    args = ap.parse_args(argv)

    dataset_cfg = load_yaml(args.dataset_config)
    arms_cfg = load_yaml(args.arms_config)

    index_paths: dict[str, str] = {}
    if args.index_dir:
        for split in ("train", "dev", "test"):
            p = Path(args.index_dir) / f"{split}.index.jsonl"
            if p.exists():
                index_paths[split] = str(p)
    for split, override in (("train", args.train_index), ("dev", args.dev_index), ("test", args.test_index)):
        if override:
            index_paths[split] = override
    if not index_paths:
        raise SystemExit("No index files found; pass --index-dir or --{train,dev,test}-index")

    outdir = ensure_dir(args.output_dir)
    manifest = build_readiness_manifest(
        dataset_cfg, arms_cfg, index_paths, min_valid_windows=args.min_valid_windows,
    )
    write_json(outdir / "rung-minus1-readiness-manifest.json", manifest)
    print(json.dumps({
        "output": str(outdir / "rung-minus1-readiness-manifest.json"),
        "gate_status": manifest["gate_status"],
        "split_counts": manifest["split_counts"],
        "under_floor": manifest["under_floor"],
        "missing": manifest["missing"],
    }, indent=2))
    # Fail closed: non-zero exit if any source under-contributes.
    assert_gate_or_raise(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
