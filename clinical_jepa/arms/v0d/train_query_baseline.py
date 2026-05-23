from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json


def _load_id_to_token(vocab_json_path: str | None) -> dict[int, str]:
    if not vocab_json_path:
        return {}
    p = Path(vocab_json_path)
    raw = json.loads(p.read_text())
    if "id_to_token" in raw:
        return {int(k): str(v) for k, v in raw["id_to_token"].items()}
    if "token_to_id" in raw:
        return {int(v): str(k) for k, v in raw["token_to_id"].items()}
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def _med_family(tok: str) -> str | None:
    if not tok.startswith("MED:"):
        return None
    parts = tok.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else tok


def _lab_family(tok: str) -> str | None:
    if not tok.startswith("LAB:"):
        return None
    parts = tok.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else tok


def _state_family(tok: str) -> str | None:
    if not tok.startswith("STATE:"):
        return None
    parts = tok.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else tok


def _first_label(tokens: list[str], fn) -> str | None:
    for tok in tokens:
        lab = fn(tok)
        if lab:
            return lab
    return None


def _mode_label(tokens: list[str], fn) -> str | None:
    c = collections.Counter(lab for tok in tokens if (lab := fn(tok)))
    return c.most_common(1)[0][0] if c else None


def _topk(counter: collections.Counter[str], k: int) -> list[str]:
    return [x for x, _ in counter.most_common(k)]


def _metric_record(task: str, split: str, baseline: str, n: int, correct1: int, correct5: int, coverage: int) -> dict[str, Any]:
    denom = max(n, 1)
    return {
        "task": task,
        "split": split,
        "baseline": baseline,
        "n_evaluated": n,
        "label_coverage": coverage,
        "top1_accuracy": correct1 / denom,
        "top5_accuracy": correct5 / denom,
    }


def _evaluate_real(dataset_cfg: dict[str, Any], target_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py

    id_to_token = _load_id_to_token(dataset_cfg.get("vocabulary", {}).get("vocab_json_path"))
    file_cache: dict[str, Any] = {}

    def get_tokens(block: dict[str, Any], start_key: str, end_key: str) -> list[str]:
        path = str(block["sequence_file"])
        if path not in file_cache:
            file_cache[path] = h5py.File(path, "r")
        h5 = file_cache[path]
        group = str(block.get("sequence_group") or block.get("sequence_id"))
        arr = h5[group]["token_ids"][:]
        start = max(0, int(block[start_key]))
        end = min(len(arr) - 1, int(block[end_key]))
        if end < start:
            return []
        return [id_to_token.get(int(t), "") for t in arr[start : end + 1]]

    try:
        rows: list[dict[str, Any]] = []
        for b in target_manifest.get("blocks", []):
            if b.get("target_type") != "T0":
                continue
            context = get_tokens(b, "context_start_ref", "context_end_ref")
            target = get_tokens(b, "target_start_ref", "target_end_ref")
            rows.append({
                "split": b.get("split"),
                "med_label": _first_label(target, _med_family),
                "med_context": _mode_label(context, _med_family),
                "lab_label": _first_label(target, _lab_family),
                "lab_context": _mode_label(context, _lab_family),
                "state_label": _first_label(target, _state_family),
                "state_context": _mode_label(context, _state_family),
            })
    finally:
        for f in file_cache.values():
            f.close()

    task_specs = [
        ("next_med_family", "med_label", "med_context"),
        ("next_lab_family", "lab_label", "lab_context"),
        ("next_state_family", "state_label", "state_context"),
    ]
    metrics: list[dict[str, Any]] = []
    label_summaries: dict[str, Any] = {}
    for task, label_key, context_key in task_specs:
        train_labels = [r[label_key] for r in rows if r["split"] == "train" and r[label_key]]
        prior = collections.Counter(train_labels)
        top1 = _topk(prior, 1)
        top5 = set(_topk(prior, 5))
        label_summaries[task] = {
            "train_label_coverage": len(train_labels),
            "n_train_classes": len(prior),
            "top_train_labels": prior.most_common(10),
        }
        for split in ["dev", "test"]:
            eval_rows = [r for r in rows if r["split"] == split and r[label_key]]
            n = len(eval_rows)
            metrics.append(_metric_record(task, split, "empirical_train_prior", n, sum(1 for r in eval_rows if top1 and r[label_key] == top1[0]), sum(1 for r in eval_rows if r[label_key] in top5), n))
            context_correct1 = 0
            context_correct5 = 0
            for r in eval_rows:
                pred = r[context_key] or (top1[0] if top1 else None)
                context_correct1 += int(pred == r[label_key])
                # Context-repeat has one prediction; top5 falls back to train prior support.
                context_correct5 += int((pred == r[label_key]) or (r[label_key] in top5))
            metrics.append(_metric_record(task, split, "context_mode_or_prior", n, context_correct1, context_correct5, n))
    aggregate = {
        "n_t0_blocks": len(rows),
        "label_summaries": label_summaries,
        "aggregate_only": True,
        "notes": "Labels are token-family aggregates derived from re-keyed blocks; no token sequences or patient examples are written.",
    }
    return metrics, aggregate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train/evaluate v0D query baseline")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--query-descriptors", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-blocks", help="Required for real aggregate baseline evaluation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    require_pass_leakage(args.leakage_report)
    dataset_cfg = load_yaml(args.dataset_config)
    desc = read_json(args.query_descriptors)
    outdir = ensure_dir(args.output_dir)

    if args.dry_run:
        metrics = []
        for q in desc.get("queries", []):
            metrics.append({"query_id": q["query_id"], "metric": "synthetic_macro_f1", "value": 0.2, "split": "dev", "leakage_audit_status": "pass"})
        aggregate = {"dry_run": True, "aggregate_only": True}
    else:
        if not args.target_blocks:
            raise SystemExit("--target-blocks is required for real v0D aggregate baseline evaluation")
        target_manifest = read_json(args.target_blocks)
        metrics, aggregate = _evaluate_real(dataset_cfg, target_manifest)

    write_json(outdir / "train-manifest.json", {
        "schema_version": "clinical-jepa-v0d-train-manifest-v0",
        "created_utc": now_utc(),
        "dry_run": args.dry_run,
        "n_queries": len(desc.get("queries", [])),
        "leakage_audit_status": "pass",
    })
    write_json(outdir / "prediction-manifest.json", {"created_utc": now_utc(), "dry_run": args.dry_run, "aggregate_only": True, "metrics": metrics, "aggregate": aggregate})
    summary = ["# v0D aggregate query baseline", "", f"Dry run: {args.dry_run}", f"Metrics: {len(metrics)}", ""]
    for m in metrics:
        if "top1_accuracy" in m:
            summary.append("- {} / {} / {}: top1={:.3f}, top5={:.3f}, n={}".format(m.get("task"), m.get("split"), m.get("baseline"), m.get("top1_accuracy", 0.0), m.get("top5_accuracy", 0.0), m.get("n_evaluated", 0)))
    (outdir / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({"prediction_manifest": str(outdir / "prediction-manifest.json"), "n_metrics": len(metrics), "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
