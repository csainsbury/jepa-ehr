from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.utils import ensure_dir, now_utc, write_json

POLICIES = [
    "same_split",
    "same_split_target_type",
    "same_split_target_type_len_bin",
    "same_split_target_type_len_seq_util_bin",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom < 1e-8] = 1.0
    return x / denom


def length_bin(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        v = int(value)
    except Exception:
        return "missing"
    if v <= 8:
        return "001-008"
    if v <= 16:
        return "009-016"
    if v <= 32:
        return "017-032"
    if v <= 64:
        return "033-064"
    if v <= 128:
        return "065-128"
    if v <= 256:
        return "129-256"
    if v <= 512:
        return "257-512"
    if v <= 1024:
        return "513-1024"
    return "1025-plus"


def count_bin(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        v = int(value)
    except Exception:
        return "missing"
    if v <= 0:
        return "000"
    if v == 1:
        return "001"
    if v <= 3:
        return "002-003"
    if v <= 7:
        return "004-007"
    if v <= 15:
        return "008-015"
    if v <= 31:
        return "016-031"
    return "032-plus"


def group_key(row: dict[str, Any], policy: str) -> tuple[Any, ...]:
    if policy == "same_split":
        return (row.get("split"),)
    if policy == "same_split_target_type":
        return (row.get("split"), row.get("target_type"))
    if policy == "same_split_target_type_len_bin":
        return (
            row.get("split"),
            row.get("target_type"),
            length_bin(row.get("context_len")),
            length_bin(row.get("target_len")),
        )
    if policy == "same_split_target_type_len_seq_util_bin":
        return (
            row.get("split"),
            row.get("target_type"),
            length_bin(row.get("context_len")),
            length_bin(row.get("target_len")),
            length_bin(row.get("sequence_len")),
            count_bin(row.get("context_med_count")),
            count_bin(row.get("context_lab_count")),
            count_bin(row.get("context_state_count")),
        )
    raise ValueError(f"Unsupported distractor policy: {policy}")


def _summarize(rank_values: list[int], ks: tuple[int, ...]) -> dict[str, Any]:
    n = len(rank_values)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "mean_rank": None, "median_rank": None, **{f"recall_at_{k}": 0.0 for k in ks}}
    arr = np.asarray(rank_values, dtype=np.float32)
    out: dict[str, Any] = {
        "n": n,
        "mrr": float(np.mean(1.0 / arr)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
    }
    for k in ks:
        out[f"recall_at_{k}"] = float(np.mean(arr <= k))
    return out


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    query_index: list[dict[str, Any]],
    target_embeddings: np.ndarray,
    target_index: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    distractor_policy: str = "same_split_target_type",
    max_candidates_per_group: int = 0,
    min_candidates_per_group: int = 0,
    seed: int = 20260523,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Compute true-target retrieval ranks with aggregate-only output.

    `max_candidates_per_group` samples a stable distractor pool per group. Each
    query's true target is always inserted into its batch candidate set, so ranks
    are with respect to sampled distractors plus the true target. Use 0 for all
    candidates in each group. `min_candidates_per_group` skips very small groups
    when chance rates would be high under fine-grained matching policies.
    """
    if len(query_embeddings) != len(query_index):
        raise ValueError("query embedding rows and query index length differ")
    if len(target_embeddings) != len(target_index):
        raise ValueError("target embedding rows and target index length differ")

    q = normalize(query_embeddings)
    t = normalize(target_embeddings)
    target_by_block = {str(row.get("block_id")): i for i, row in enumerate(target_index)}

    target_groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(target_index):
        target_groups.setdefault(group_key(row, distractor_policy), []).append(i)

    query_groups: dict[tuple[Any, ...], list[int]] = {}
    skipped_no_target = 0
    for i, row in enumerate(query_index):
        block_id = str(row.get("block_id"))
        if block_id not in target_by_block:
            skipped_no_target += 1
            continue
        query_groups.setdefault(group_key(row, distractor_policy), []).append(i)

    rng = random.Random(seed)
    all_ranks: list[int] = []
    per_group: dict[str, list[int]] = {}
    skipped_no_candidates = 0
    sampled_candidate_counts: dict[str, int] = {}

    for key, q_indices in query_groups.items():
        group_targets = list(target_groups.get(key, []))
        if len(group_targets) <= 1 or (min_candidates_per_group and len(group_targets) < min_candidates_per_group):
            skipped_no_candidates += len(q_indices)
            continue
        if max_candidates_per_group and len(group_targets) > max_candidates_per_group:
            base = list(group_targets)
            rng.shuffle(base)
            base = base[:max_candidates_per_group]
        else:
            base = group_targets
        group_name = "|".join(str(x) for x in key)
        sampled_candidate_counts[group_name] = len(base)
        group_ranks: list[int] = []

        for start in range(0, len(q_indices), batch_size):
            batch_q_indices = q_indices[start : start + batch_size]
            true_target_indices = [target_by_block[str(query_index[qi].get("block_id"))] for qi in batch_q_indices]
            candidates = list(dict.fromkeys([*base, *true_target_indices]))
            candidate_pos = {idx: pos for pos, idx in enumerate(candidates)}
            cand = np.asarray(candidates, dtype=np.int64)
            sims = q[np.asarray(batch_q_indices, dtype=np.int64)] @ t[cand].T
            true_cols = np.asarray([candidate_pos[idx] for idx in true_target_indices], dtype=np.int64)
            true_sims = sims[np.arange(len(batch_q_indices)), true_cols]
            ranks = (sims > true_sims[:, None]).sum(axis=1).astype(np.int64) + 1
            group_ranks.extend(int(r) for r in ranks)

        all_ranks.extend(group_ranks)
        per_group[group_name] = group_ranks

    group_sizes = [len(v) for v in target_groups.values()]
    return {
        "created_utc": now_utc(),
        "distractor_policy": distractor_policy,
        "max_candidates_per_group": max_candidates_per_group,
        "min_candidates_per_group": min_candidates_per_group,
        "batch_size": batch_size,
        "query_rows": int(len(query_index)),
        "target_rows": int(len(target_index)),
        "n_candidate_groups": len(target_groups),
        "candidate_group_size_summary": {
            "min": int(min(group_sizes)) if group_sizes else 0,
            "median": float(np.median(group_sizes)) if group_sizes else 0,
            "max": int(max(group_sizes)) if group_sizes else 0,
        },
        "skipped_no_target": skipped_no_target,
        "skipped_no_candidates": skipped_no_candidates,
        "sampled_candidate_counts": sampled_candidate_counts,
        "overall": _summarize(all_ranks, ks),
        "groups": {name: _summarize(ranks, ks) for name, ranks in per_group.items()},
        "aggregate_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute aggregate target-retrieval metrics from embedding arrays")
    ap.add_argument("--query-embeddings", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--target-embeddings", required=True)
    ap.add_argument("--target-index", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--distractor-policy", default="same_split_target_type", choices=POLICIES)
    ap.add_argument("--max-candidates-per-group", type=int, default=0)
    ap.add_argument("--min-candidates-per-group", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260523)
    args = ap.parse_args(argv)

    report = compute_retrieval_metrics(
        np.load(args.query_embeddings),
        read_jsonl(args.query_index),
        np.load(args.target_embeddings),
        read_jsonl(args.target_index),
        distractor_policy=args.distractor_policy,
        max_candidates_per_group=args.max_candidates_per_group,
        min_candidates_per_group=args.min_candidates_per_group,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "retrieval-metrics.json", report)
    lines = [
        "# Retrieval metrics",
        "",
        f"Distractor policy: `{report['distractor_policy']}`",
        f"Queries evaluated: {report['overall']['n']}",
        f"Candidate groups: {report['n_candidate_groups']}",
        "",
        "## Overall",
        "",
        f"- Recall@1: {report['overall']['recall_at_1']:.4f}",
        f"- Recall@5: {report['overall']['recall_at_5']:.4f}",
        f"- Recall@10: {report['overall']['recall_at_10']:.4f}",
        f"- MRR: {report['overall']['mrr']:.4f}",
        f"- Median rank: {report['overall']['median_rank']}",
        "",
        "## Groups",
        "",
    ]
    for name, summary in sorted(report["groups"].items()):
        lines.append(f"- {name}: n={summary['n']}, R@10={summary['recall_at_10']:.4f}, MRR={summary['mrr']:.4f}, median_rank={summary['median_rank']}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(outdir / "retrieval-metrics.json"), "queries": report["overall"]["n"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
