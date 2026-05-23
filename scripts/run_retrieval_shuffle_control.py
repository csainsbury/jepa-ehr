#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import POLICIES, compute_retrieval_metrics, group_key, read_jsonl
from clinical_jepa.utils import ensure_dir, now_utc, write_json


def _summarise_controls(controls: list[dict[str, Any]]) -> dict[str, float | None]:
    r10 = [c["overall"]["recall_at_10"] for c in controls]
    mrr = [c["overall"]["mrr"] for c in controls]
    return {
        "recall_at_10_mean": float(np.mean(r10)) if r10 else None,
        "recall_at_10_min": float(np.min(r10)) if r10 else None,
        "recall_at_10_max": float(np.max(r10)) if r10 else None,
        "mrr_mean": float(np.mean(mrr)) if mrr else None,
        "mrr_min": float(np.min(mrr)) if mrr else None,
        "mrr_max": float(np.max(mrr)) if mrr else None,
    }


def _augment_from_target_blocks(rows: list[dict[str, Any]], target_blocks_path: str | None) -> list[dict[str, Any]]:
    if not target_blocks_path:
        return rows
    manifest = json.loads(Path(target_blocks_path).read_text())
    blocks = {str(b.get("block_id")): b for b in manifest.get("blocks", []) if b.get("block_id")}
    out: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        block = blocks.get(str(row.get("block_id"))) or {}
        for key in ["patient_hash", "context_start_ref", "context_end_ref", "target_start_ref", "target_end_ref", "source_dataset"]:
            if key in block and key not in merged:
                merged[key] = block[key]
        out.append(merged)
    return out


def _time_shift_permutation(rows: list[dict[str, Any]], policy: str) -> np.ndarray:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(group_key(row, policy), []).append(i)
    perm = np.arange(len(rows), dtype=np.int64)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        ordered = sorted(
            idxs,
            key=lambda i: (
                int(rows[i].get("context_end_ref", rows[i].get("context_len", 0)) or 0),
                int(rows[i].get("target_start_ref", 0) or 0),
                str(rows[i].get("patient_hash", rows[i].get("block_id", ""))),
            ),
        )
        shifted = ordered[1:] + ordered[:1]
        for dst, src in zip(ordered, shifted):
            perm[dst] = src
    return perm


def _random_target_controls(q: np.ndarray, qidx: list[dict[str, Any]], t: np.ndarray, tidx: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    controls = []
    for i in range(args.n_shuffles):
        perm = rng.permutation(len(t))
        shuffled = compute_retrieval_metrics(q, qidx, t[perm], tidx, distractor_policy=args.distractor_policy, max_candidates_per_group=args.max_candidates_per_group, seed=args.seed + i + 1)
        controls.append({"shuffle": i, "overall": shuffled["overall"], "skipped_no_target": shuffled["skipped_no_target"], "skipped_no_candidates": shuffled["skipped_no_candidates"]})
    return controls


def _random_query_controls(q: np.ndarray, qidx: list[dict[str, Any]], t: np.ndarray, tidx: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    controls = []
    for i in range(args.n_shuffles):
        perm = rng.permutation(len(q))
        shuffled = compute_retrieval_metrics(q[perm], qidx, t, tidx, distractor_policy=args.distractor_policy, max_candidates_per_group=args.max_candidates_per_group, seed=args.seed + 1000 + i)
        controls.append({"shuffle": i, "overall": shuffled["overall"], "skipped_no_target": shuffled["skipped_no_target"], "skipped_no_candidates": shuffled["skipped_no_candidates"]})
    return controls


def main() -> int:
    ap = argparse.ArgumentParser(description="Run aggregate retrieval shuffle controls")
    ap.add_argument("--query-embeddings", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--target-embeddings", required=True)
    ap.add_argument("--target-index", required=True)
    ap.add_argument("--target-blocks", help="Optional target-block manifest for time-shift metadata")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--distractor-policy", default="same_split_target_type_len_bin", choices=POLICIES)
    ap.add_argument("--max-candidates-per-group", type=int, default=4096)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260523)
    args = ap.parse_args()

    q = np.load(args.query_embeddings)
    t = np.load(args.target_embeddings)
    qidx = _augment_from_target_blocks(read_jsonl(args.query_index), args.target_blocks)
    tidx = _augment_from_target_blocks(read_jsonl(args.target_index), args.target_blocks)
    outdir = ensure_dir(args.output_dir)

    observed = compute_retrieval_metrics(q, qidx, t, tidx, distractor_policy=args.distractor_policy, max_candidates_per_group=args.max_candidates_per_group, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    target_controls = _random_target_controls(q, qidx, t, tidx, args, rng)
    query_controls = _random_query_controls(q, qidx, t, tidx, args, rng)

    time_shift_perm = _time_shift_permutation(tidx, args.distractor_policy)
    time_shift = compute_retrieval_metrics(q, qidx, t[time_shift_perm], tidx, distractor_policy=args.distractor_policy, max_candidates_per_group=args.max_candidates_per_group, seed=args.seed + 2000)

    report = {
        "created_utc": now_utc(),
        "distractor_policy": args.distractor_policy,
        "max_candidates_per_group": args.max_candidates_per_group,
        "n_shuffles": args.n_shuffles,
        "observed": observed,
        "shuffled_target_controls": target_controls,
        "shuffled_query_controls": query_controls,
        "within_group_time_shift_control": {"overall": time_shift["overall"], "skipped_no_target": time_shift["skipped_no_target"], "skipped_no_candidates": time_shift["skipped_no_candidates"]},
        "control_summary": {
            "target_shuffle": _summarise_controls(target_controls),
            "query_shuffle": _summarise_controls(query_controls),
            "time_shift": {"recall_at_10": time_shift["overall"]["recall_at_10"], "mrr": time_shift["overall"]["mrr"]},
        },
        "aggregate_only": True,
        "notes": "Aggregate row-shuffle controls only; target-block metadata is used for grouping/time-shift controls but no token sequences or patient examples are written.",
    }
    write_json(outdir / "retrieval-shuffle-control.json", report)
    lines = [
        "# Retrieval shuffle control",
        "",
        f"Policy: `{args.distractor_policy}`",
        f"Observed Recall@10: {observed['overall']['recall_at_10']:.4f}",
        f"Observed MRR: {observed['overall']['mrr']:.4f}",
        f"Target-shuffle Recall@10 mean: {report['control_summary']['target_shuffle']['recall_at_10_mean']:.4f}",
        f"Query-shuffle Recall@10 mean: {report['control_summary']['query_shuffle']['recall_at_10_mean']:.4f}",
        f"Within-group time-shift Recall@10: {report['control_summary']['time_shift']['recall_at_10']:.4f}",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(outdir / "retrieval-shuffle-control.json"), "observed_recall_at_10": observed["overall"]["recall_at_10"], "target_shuffle_recall_at_10_mean": report["control_summary"]["target_shuffle"]["recall_at_10_mean"], "query_shuffle_recall_at_10_mean": report["control_summary"]["query_shuffle"]["recall_at_10_mean"], "time_shift_recall_at_10": report["control_summary"]["time_shift"]["recall_at_10"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
