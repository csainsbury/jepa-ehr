#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.retrieval import compute_retrieval_metrics, read_jsonl
from clinical_jepa.utils import ensure_dir, now_utc, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Run aggregate retrieval shuffle controls")
    ap.add_argument("--query-embeddings", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--target-embeddings", required=True)
    ap.add_argument("--target-index", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--distractor-policy", default="same_split_target_type_len_bin", choices=["same_split", "same_split_target_type", "same_split_target_type_len_bin"])
    ap.add_argument("--max-candidates-per-group", type=int, default=4096)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260523)
    args = ap.parse_args()

    q = np.load(args.query_embeddings)
    t = np.load(args.target_embeddings)
    qidx = read_jsonl(args.query_index)
    tidx = read_jsonl(args.target_index)
    outdir = ensure_dir(args.output_dir)

    observed = compute_retrieval_metrics(
        q,
        qidx,
        t,
        tidx,
        distractor_policy=args.distractor_policy,
        max_candidates_per_group=args.max_candidates_per_group,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    controls = []
    for i in range(args.n_shuffles):
        perm = rng.permutation(len(t))
        shuffled = compute_retrieval_metrics(
            q,
            qidx,
            t[perm],
            tidx,
            distractor_policy=args.distractor_policy,
            max_candidates_per_group=args.max_candidates_per_group,
            seed=args.seed + i + 1,
        )
        controls.append({
            "shuffle": i,
            "overall": shuffled["overall"],
            "skipped_no_target": shuffled["skipped_no_target"],
            "skipped_no_candidates": shuffled["skipped_no_candidates"],
        })
    control_r10 = [c["overall"]["recall_at_10"] for c in controls]
    control_mrr = [c["overall"]["mrr"] for c in controls]
    report = {
        "created_utc": now_utc(),
        "distractor_policy": args.distractor_policy,
        "max_candidates_per_group": args.max_candidates_per_group,
        "n_shuffles": args.n_shuffles,
        "observed": observed,
        "shuffled_target_controls": controls,
        "control_summary": {
            "recall_at_10_mean": float(np.mean(control_r10)) if control_r10 else None,
            "recall_at_10_min": float(np.min(control_r10)) if control_r10 else None,
            "recall_at_10_max": float(np.max(control_r10)) if control_r10 else None,
            "mrr_mean": float(np.mean(control_mrr)) if control_mrr else None,
            "mrr_min": float(np.min(control_mrr)) if control_mrr else None,
            "mrr_max": float(np.max(control_mrr)) if control_mrr else None,
        },
        "aggregate_only": True,
        "notes": "Target embedding rows are shuffled while target index/block ids are held fixed; no token sequences or patient examples are written.",
    }
    write_json(outdir / "retrieval-shuffle-control.json", report)
    lines = [
        "# Retrieval shuffle control",
        "",
        f"Policy: `{args.distractor_policy}`",
        f"Observed Recall@10: {observed['overall']['recall_at_10']:.4f}",
        f"Observed MRR: {observed['overall']['mrr']:.4f}",
        f"Shuffle Recall@10 mean: {report['control_summary']['recall_at_10_mean']:.4f}",
        f"Shuffle MRR mean: {report['control_summary']['mrr_mean']:.4f}",
        "",
    ]
    for c in controls:
        lines.append(f"- shuffle {c['shuffle']}: R@10={c['overall']['recall_at_10']:.4f}, MRR={c['overall']['mrr']:.4f}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(outdir / "retrieval-shuffle-control.json"), "observed_recall_at_10": observed["overall"]["recall_at_10"], "shuffle_recall_at_10_mean": report["control_summary"]["recall_at_10_mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
