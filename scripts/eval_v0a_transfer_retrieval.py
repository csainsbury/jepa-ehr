#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.arms.v0a.train_predictor import _ridge_regression_fit, _ridge_regression_predict
from clinical_jepa.eval.retrieval import compute_retrieval_metrics, read_jsonl
from clinical_jepa.utils import ensure_dir, now_utc, read_json, write_json


def _files(manifest: dict[str, Any]) -> dict[str, str]:
    files = manifest.get("embedding_files") or {}
    if not files:
        raise SystemExit("embedding manifest lacks embedding_files")
    return {str(k): str(v) for k, v in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train v0A context->target ridge on one source and evaluate retrieval on another")
    ap.add_argument("--train-embedding-manifest", required=True)
    ap.add_argument("--eval-embedding-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--retrieval-policy", default="same_split_target_type_len_bin", choices=["same_split", "same_split_target_type", "same_split_target_type_len_bin", "same_split_target_type_len_seq_util_bin"])
    ap.add_argument("--retrieval-max-candidates", type=int, default=4096)
    ap.add_argument("--ridge-lambda", type=float, default=10.0)
    args = ap.parse_args()

    train_manifest = read_json(args.train_embedding_manifest)
    eval_manifest = read_json(args.eval_embedding_manifest)
    train_files = _files(train_manifest)
    eval_files = _files(eval_manifest)
    train_rows = read_jsonl(train_files["index_jsonl"])
    eval_rows = read_jsonl(eval_files["index_jsonl"])
    train_idx = np.asarray([i for i, row in enumerate(train_rows) if row.get("split") == "train"], dtype=np.int64)
    if len(train_idx) == 0:
        raise SystemExit("no train rows in training embedding manifest")

    pairs = [
        ("mean_to_mean", "final_mean_fp16", "target_final_mean_fp16"),
        ("final_to_mean", "final_token_fp16", "target_final_mean_fp16"),
        ("final_to_final", "final_token_fp16", "target_final_token_fp16"),
    ]
    outdir = ensure_dir(args.output_dir)
    retrievals: dict[str, Any] = {}
    for name, context_key, target_key in pairs:
        if context_key not in train_files or target_key not in train_files or context_key not in eval_files or target_key not in eval_files:
            continue
        train_x = np.load(train_files[context_key]).astype(np.float32)
        train_y = np.load(train_files[target_key]).astype(np.float32)
        eval_x = np.load(eval_files[context_key]).astype(np.float32)
        eval_y = np.load(eval_files[target_key]).astype(np.float32)
        w, stats = _ridge_regression_fit(train_x[train_idx], train_y[train_idx], lam=args.ridge_lambda)
        pred = _ridge_regression_predict(eval_x, w, stats).astype(np.float16)
        pred_path = outdir / f"v0a-transfer-target-pred-{name}.fp16.npy"
        np.save(pred_path, pred)
        report = compute_retrieval_metrics(
            pred,
            eval_rows,
            eval_y.astype(np.float16),
            eval_rows,
            distractor_policy=args.retrieval_policy,
            max_candidates_per_group=args.retrieval_max_candidates,
        )
        report["predictor"] = {
            "name": name,
            "context_embedding": context_key,
            "target_embedding": target_key,
            "ridge_lambda": args.ridge_lambda,
            "trained_on_rows": int(len(train_idx)),
            "trained_on_manifest": str(args.train_embedding_manifest),
            "evaluated_on_manifest": str(args.eval_embedding_manifest),
            "predicted_embedding_file": str(pred_path),
        }
        retrievals[name] = report

    best = None
    for pred, rep in retrievals.items():
        item = {"predictor": pred, **rep["overall"]}
        if best is None or item.get("recall_at_10", 0.0) > best.get("recall_at_10", 0.0):
            best = item
    summary = {
        "created_utc": now_utc(),
        "retrieval_policy": args.retrieval_policy,
        "retrieval_max_candidates": args.retrieval_max_candidates,
        "ridge_lambda": args.ridge_lambda,
        "train_embedding_manifest": str(args.train_embedding_manifest),
        "eval_embedding_manifest": str(args.eval_embedding_manifest),
        "train_rows_used": int(len(train_idx)),
        "eval_rows": int(len(eval_rows)),
        "retrieval": retrievals,
        "best": best,
        "aggregate_only": True,
        "notes": "Ridge map is fitted only on the training source's train split and applied to the evaluation source embeddings; no token sequences or patient examples are written.",
    }
    write_json(outdir / "v0a-transfer-retrieval.json", summary)
    lines = [
        "# v0A transfer retrieval",
        "",
        f"Policy: `{args.retrieval_policy}`",
        f"Train rows used: {len(train_idx)}",
        f"Eval rows: {len(eval_rows)}",
        "",
    ]
    for name, report in retrievals.items():
        r = report["overall"]
        lines.append(f"- {name}: R@10={r['recall_at_10']:.4f}, MRR={r['mrr']:.4f}, median_rank={r['median_rank']}, n={r['n']}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(outdir / "v0a-transfer-retrieval.json"), "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
