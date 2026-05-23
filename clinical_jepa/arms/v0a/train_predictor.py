from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, write_json


def _load_id_to_token(vocab_json_path: str | None) -> dict[int, str]:
    if not vocab_json_path:
        return {}
    raw = json.loads(Path(vocab_json_path).read_text())
    if "id_to_token" in raw:
        return {int(k): str(v) for k, v in raw["id_to_token"].items()}
    if "token_to_id" in raw:
        return {int(v): str(k) for k, v in raw["token_to_id"].items()}
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def _med_family(tok: str) -> str | None:
    if tok.startswith("MED:"):
        parts = tok.split(":")
        return ":".join(parts[:2]) if len(parts) >= 2 else tok
    return None


def _lab_family(tok: str) -> str | None:
    if tok.startswith("LAB:"):
        parts = tok.split(":")
        return ":".join(parts[:2]) if len(parts) >= 2 else tok
    return None


def _state_family(tok: str) -> str | None:
    if tok.startswith("STATE:"):
        parts = tok.split(":")
        return ":".join(parts[:2]) if len(parts) >= 2 else tok
    return None


def _first_label(tokens: list[str], fn) -> str | None:
    for tok in tokens:
        lab = fn(tok)
        if lab:
            return lab
    return None


def _read_embedding_index(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_target_labels(block_ids: set[str], target_manifest: dict[str, Any], id_to_token: dict[int, str]) -> dict[str, dict[str, Any]]:
    import h5py

    blocks = {b.get("block_id"): b for b in target_manifest.get("blocks", []) if b.get("block_id") in block_ids and b.get("target_type") == "T0"}
    labels: dict[str, dict[str, Any]] = {}
    cache: dict[str, Any] = {}
    try:
        for bid, b in blocks.items():
            path = str(b["sequence_file"])
            if path not in cache:
                cache[path] = h5py.File(path, "r")
            h5 = cache[path]
            group = str(b.get("sequence_group") or b.get("sequence_id"))
            arr = h5[group]["token_ids"][:]
            t0 = max(0, int(b["target_start_ref"]))
            t1 = min(len(arr) - 1, int(b["target_end_ref"]))
            toks = [id_to_token.get(int(t), "") for t in arr[t0 : t1 + 1]]
            labels[str(bid)] = {
                "split": b.get("split"),
                "med": _first_label(toks, _med_family),
                "lab": _first_label(toks, _lab_family),
                "state": _first_label(toks, _state_family),
            }
    finally:
        for f in cache.values():
            f.close()
    return labels


def _standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (train_x - mu) / sigma, (x - mu) / sigma, sigma


def _ridge_train(x: np.ndarray, y_idx: np.ndarray, n_classes: int, lam: float = 1.0) -> np.ndarray:
    # Add intercept column; solve multiclass ridge in closed form.
    x1 = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1).astype(np.float64)
    y = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    y[np.arange(x.shape[0]), y_idx] = 1.0
    xtx = x1.T @ x1
    xtx += lam * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x1.T @ y
    return np.linalg.solve(xtx, xty)


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x1 = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1).astype(np.float64)
    return (x1 @ w).argmax(axis=1)


def _centroid_predict(train_x: np.ndarray, train_y: np.ndarray, x: np.ndarray, n_classes: int) -> np.ndarray:
    cents = np.zeros((n_classes, train_x.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        rows = train_x[train_y == c]
        if len(rows):
            cents[c] = rows.mean(axis=0)
    # cosine similarity
    xx = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    cc = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    return (xx @ cc.T).argmax(axis=1)


def _topk_prior(train_y: np.ndarray, k: int) -> set[int]:
    c = collections.Counter(int(v) for v in train_y)
    return {cls for cls, _ in c.most_common(k)}


def _evaluate_task(x: np.ndarray, rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]], task: str, label_key: str, pooling_name: str) -> list[dict[str, Any]]:
    valid = []
    for i, row in enumerate(rows):
        bid = str(row.get("block_id"))
        lab = labels.get(bid, {}).get(label_key)
        split = labels.get(bid, {}).get("split") or row.get("split")
        if lab:
            valid.append((i, str(split), str(lab)))
    train_labs = sorted({lab for _, split, lab in valid if split == "train"})
    if len(train_labs) < 2:
        return []
    lab_to_idx = {lab: i for i, lab in enumerate(train_labs)}
    train_idx = [i for i, split, lab in valid if split == "train" and lab in lab_to_idx]
    train_y = np.array([lab_to_idx[lab] for _, split, lab in valid if split == "train" and lab in lab_to_idx], dtype=np.int64)
    train_x = x[train_idx].astype(np.float32)
    train_x_std, _, _ = _standardize(train_x, train_x)
    w = _ridge_train(train_x_std, train_y, len(train_labs), lam=1.0)
    prior1 = _topk_prior(train_y, 1)
    prior5 = _topk_prior(train_y, min(5, len(train_labs)))
    metrics = []
    for split in ["dev", "test"]:
        eval_pairs = [(i, lab) for i, sp, lab in valid if sp == split and lab in lab_to_idx]
        if not eval_pairs:
            continue
        eval_idx = [i for i, _ in eval_pairs]
        y = np.array([lab_to_idx[lab] for _, lab in eval_pairs], dtype=np.int64)
        _, eval_x_std, _ = _standardize(train_x, x[eval_idx].astype(np.float32))
        pred_ridge = _ridge_predict(eval_x_std, w)
        pred_cent = _centroid_predict(train_x_std, train_y, eval_x_std, len(train_labs))
        for baseline, pred in [("ridge_linear_probe", pred_ridge), ("nearest_centroid_probe", pred_cent)]:
            metrics.append({
                "task": task,
                "pooling": pooling_name,
                "split": split,
                "baseline": baseline,
                "n_train": int(len(train_y)),
                "n_evaluated": int(len(y)),
                "n_classes_train": int(len(train_labs)),
                "top1_accuracy": float((pred == y).mean()),
            })
        metrics.append({
            "task": task,
            "pooling": pooling_name,
            "split": split,
            "baseline": "empirical_train_prior",
            "n_train": int(len(train_y)),
            "n_evaluated": int(len(y)),
            "n_classes_train": int(len(train_labs)),
            "top1_accuracy": float(np.mean([int(v in prior1) for v in y])),
            "top5_accuracy": float(np.mean([int(v in prior5) for v in y])),
        })
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0A FlatASCEND predictor/probe")
    ap.add_argument("--embedding-manifest", required=True)
    ap.add_argument("--variant", default="linear,mlp")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset-config")
    ap.add_argument("--target-blocks")
    args = ap.parse_args(argv)
    emb = read_json(args.embedding_manifest)
    if emb.get("prefix_only") is not True:
        raise SystemExit("Embedding manifest is not prefix_only")
    outdir = ensure_dir(args.output_dir)
    variants = [v.strip() for v in args.variant.split(",") if v.strip()]

    if args.dry_run:
        manifest = {"schema_version": "clinical-jepa-v0a-train-manifest-v0", "created_utc": now_utc(), "dry_run": True, "variants": variants, "embedding_manifest": args.embedding_manifest, "trained": False, "notes": "dry-run predictor grid only"}
        write_json(outdir / "train-manifest.json", manifest)
        write_json(outdir / "prediction-manifest.json", {"created_utc": now_utc(), "dry_run": True, "aggregate_only": True})
        print(json.dumps({"train_manifest": str(outdir / "train-manifest.json")}, indent=2))
        return 0

    if not args.dataset_config or not args.target_blocks:
        raise SystemExit("--dataset-config and --target-blocks are required for real v0A predictor probes")
    dataset = load_yaml(args.dataset_config)
    target_manifest = read_json(args.target_blocks)
    files = emb.get("embedding_files", {})
    rows = _read_embedding_index(files["index_jsonl"])
    block_ids = {str(r.get("block_id")) for r in rows}
    id_to_token = _load_id_to_token(dataset.get("vocabulary", {}).get("vocab_json_path"))
    labels = _read_target_labels(block_ids, target_manifest, id_to_token)
    all_metrics = []
    for pooling_key, pooling_name in [("final_mean_fp16", "final_mean"), ("final_token_fp16", "final_token")]:
        x = np.load(files[pooling_key]).astype(np.float32)
        all_metrics.extend(_evaluate_task(x, rows, labels, "next_med_family", "med", pooling_name))
        all_metrics.extend(_evaluate_task(x, rows, labels, "next_lab_family", "lab", pooling_name))
        all_metrics.extend(_evaluate_task(x, rows, labels, "next_state_family", "state", pooling_name))
    manifest = {"schema_version": "clinical-jepa-v0a-train-manifest-v0", "created_utc": now_utc(), "dry_run": False, "variants": ["ridge_linear_probe", "nearest_centroid_probe"], "embedding_manifest": args.embedding_manifest, "trained": True, "aggregate_only": True, "n_embedding_rows": len(rows), "n_labeled_blocks": len(labels)}
    write_json(outdir / "train-manifest.json", manifest)
    write_json(outdir / "prediction-manifest.json", {"created_utc": now_utc(), "dry_run": False, "aggregate_only": True, "metrics": all_metrics})
    md = ["# v0A FlatASCEND embedding probes", "", f"Embedding rows: {len(rows)}", f"Labeled T0 blocks: {len(labels)}", ""]
    for m in all_metrics:
        if m["baseline"] in {"ridge_linear_probe", "nearest_centroid_probe"}:
            md.append(f"- {m['task']} / {m['pooling']} / {m['split']} / {m['baseline']}: top1={m['top1_accuracy']:.3f}, n={m['n_evaluated']}, classes={m['n_classes_train']}")
    (outdir / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"train_manifest": str(outdir / "train-manifest.json"), "n_metrics": len(all_metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
