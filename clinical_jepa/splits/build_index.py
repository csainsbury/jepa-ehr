"""Build the per-split JSONL sequence index for the joint MIMIC+SCI-D substrate.

The joint corrected substrate ships only per-source h5 files (+ vocab metadata):
there is no JSONL index, which the target-block extractor, split manifest, and
every arm depend on. This tool emits one JSONL index per split, carrying the
per-sequence ``source_dataset`` (SCID / MIMIC) so the rest of the pipeline can
window, stratify, and normalise candidates per source.

Governance: this reads governed h5 sequence data and must run under the existing
governance boundary. It writes only sequence ids, group keys, integer lengths,
source labels, and file paths (no raw tokens / values); the JSONL index is a
gitignored local artifact. Synthetic tests exercise it end-to-end. Running it on
the real governed substrate is a governed step, not part of the safe_distilled
first-pass port.

Each emitted row:
    {"sequence_id", "group", "seq_len", "source_dataset", "source_h5_path"[, "wall_clock_span_days"]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from clinical_jepa.utils import ensure_dir, now_utc, write_json


def _source_datasets(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    primary = (dataset_cfg.get("sources", {}) or {}).get("primary", {}) or {}
    sources = primary.get("source_datasets")
    if not sources:
        raise ValueError(
            "dataset config sources.primary.source_datasets missing; the joint "
            "index builder needs an explicit per-source list (name + h5_paths)"
        )
    return list(sources)


def _derive_source_for_group(
    group: str,
    source_entry: dict[str, Any],
) -> str | None:
    """Return the source label for a group key, or None to skip it."""
    if str(source_entry.get("kind", "per_source")) == "merged":
        prefixes: dict[str, str] = source_entry.get("group_prefixes", {}) or {}
        for name, prefix in prefixes.items():
            if group.startswith(str(prefix)):
                return str(name)
        return None
    return str(source_entry["name"])


def _verify_token_source(
    token0: int,
    declared_source: str,
    source_token_ids: dict[str, int],
) -> bool:
    """True if token index 0 matches the declared source (or no map given)."""
    if not source_token_ids:
        return True
    expected = source_token_ids.get(declared_source)
    if expected is None:
        return True
    return int(token0) == int(expected)


def _iter_groups(h5: Any) -> Iterable[str]:
    for key in h5.keys():
        yield str(key)


def build_index_for_split(
    dataset_cfg: dict[str, Any],
    split: str,
    *,
    max_per_source: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py

    time_channel = dataset_cfg.get("time_channel")
    source_token_ids = {str(k): int(v) for k, v in (dataset_cfg.get("source_token_ids", {}) or {}).items()}
    verify_source_token = bool(dataset_cfg.get("verify_source_token", False))

    rows: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    per_source_token_mismatch: dict[str, int] = {}
    for source_entry in _source_datasets(dataset_cfg):
        h5_paths = source_entry.get("h5_paths", {}) or {}
        if split not in h5_paths:
            continue
        h5_path = str(h5_paths[split])
        emitted = 0
        with h5py.File(h5_path, "r") as h5:
            for group in _iter_groups(h5):
                if max_per_source and emitted >= max_per_source:
                    break
                grp = h5.get(group)
                if grp is None or "token_ids" not in grp:
                    continue
                source_dataset = _derive_source_for_group(group, source_entry)
                if source_dataset is None:
                    continue
                token_ids = grp["token_ids"]
                seq_len = int(token_ids.shape[0])
                if seq_len <= 0:
                    continue
                if verify_source_token and source_token_ids:
                    token0 = int(token_ids[0])
                    if not _verify_token_source(token0, source_dataset, source_token_ids):
                        per_source_token_mismatch[source_dataset] = per_source_token_mismatch.get(source_dataset, 0) + 1
                row: dict[str, Any] = {
                    "sequence_id": group,
                    "group": group,
                    "seq_len": seq_len,
                    "source_dataset": source_dataset,
                    "source_h5_path": h5_path,
                }
                if time_channel and time_channel in grp:
                    cdays = grp[time_channel][:]
                    if len(cdays) > 0:
                        row["wall_clock_span_days"] = float(cdays[-1]) - float(cdays[0])
                rows.append(row)
                per_source_counts[source_dataset] = per_source_counts.get(source_dataset, 0) + 1
                emitted += 1

    report = {
        "split": split,
        "n_rows": len(rows),
        "per_source_counts": per_source_counts,
        "per_source_token_mismatch": per_source_token_mismatch,
        "verify_source_token": verify_source_token,
        "time_channel": time_channel,
        "aggregate_only": True,
    }
    return rows, report


def _write_index(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    from clinical_jepa.utils import load_yaml

    ap = argparse.ArgumentParser(description="Build per-split JSONL sequence index for the joint substrate")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--max-per-source", type=int, default=0, help="0 means no cap (per source per split)")
    args = ap.parse_args(argv)

    dataset_cfg = load_yaml(args.dataset_config)
    outdir = ensure_dir(args.output_dir)
    reports: dict[str, Any] = {}
    index_paths: dict[str, str] = {}
    for split in args.splits:
        rows, report = build_index_for_split(dataset_cfg, split, max_per_source=args.max_per_source)
        index_path = outdir / f"{split}.index.jsonl"
        _write_index(rows, index_path)
        index_paths[split] = str(index_path)
        reports[split] = report

    build_report = {
        "created_utc": now_utc(),
        "index_paths": index_paths,
        "splits": reports,
        "aggregate_only": True,
        "notes": "per-split JSONL sequence index; sequence ids + paths only, no raw tokens or values.",
    }
    write_json(outdir / "index-build-report.json", build_report)
    print(json.dumps({"index_paths": index_paths, "per_split_rows": {s: reports[s]["n_rows"] for s in reports}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
