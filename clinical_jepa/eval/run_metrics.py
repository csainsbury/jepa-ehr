from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, read_json, require_pass_leakage, write_json
from clinical_jepa.validation import validate_artifact


def effective_rank(x: np.ndarray) -> float:
    _, s, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    p = s / (s.sum() + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(entropy))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run synthetic Clinical-JEPA metrics")
    ap.add_argument("--metrics-config", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--leakage-report", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    require_pass_leakage(args.leakage_report)
    _cfg = load_yaml(args.metrics_config)
    split_manifest = read_json(args.split_manifest)
    target_manifest = read_json(args.target_blocks)
    outdir = ensure_dir(args.output_dir)
    rng = np.random.default_rng(20260523)
    emb = rng.normal(size=(64, 16))
    diag = {
        "schema_version": "clinical-jepa-collapse-diagnostics-v0",
        "created_utc": now_utc(),
        "arm": "synthetic-baseline",
        "effective_rank": effective_rank(emb),
        "per_dimension_variance_mean": float(emb.var(axis=0).mean()),
        "nearest_neighbour_patient_diversity": 1.0,
        "average_target_predictor_delta": 0.1,
    }
    metrics = [
        {"arm": "empirical_prior", "metric": "Recall@10", "target_type": "T0", "split": "dev", "horizon": "short", "distractor_policy": "random_same_split", "leakage_audit_status": "pass", "value": 0.10, "confidence_interval": [0.08, 0.12]},
        {"arm": "bag_count_time", "metric": "macro_f1", "target_type": "T0", "split": "dev", "horizon": "short", "distractor_policy": "not_applicable", "leakage_audit_status": "pass", "value": 0.20, "confidence_interval": [0.16, 0.24]},
        {"arm": "utilisation_intensity", "metric": "AUROC", "target_type": "T1", "split": "dev", "horizon": "medium", "distractor_policy": "not_applicable", "leakage_audit_status": "pass", "value": 0.55, "confidence_interval": [0.50, 0.60]},
    ]
    baseline_bundle = {"created_utc": now_utc(), "metrics": metrics}
    retrieval_bundle = {"created_utc": now_utc(), "metrics": [metrics[0]], "dry_run": args.dry_run}
    validate_artifact("metrics-bundle", baseline_bundle)
    validate_artifact("metrics-bundle", retrieval_bundle)
    write_json(outdir / "baseline-results.json", baseline_bundle)
    write_json(outdir / "retrieval-results.json", retrieval_bundle)
    write_json(outdir / "collapse-diagnostics.json", diag)
    write_json(outdir / "utilisation-diagnostics.json", {"created_utc": now_utc(), "status": "synthetic", "features_checked": ["prefix_sequence_length", "lab_count", "medication_count"]})
    summary = f"# v0 synthetic metrics summary\n\nDataset: {split_manifest.get('dataset')}\n\nBlocks: {sum(sum(v.values()) for v in target_manifest.get('counts', {}).values())}\n\nLeakage audit: pass\n"
    (outdir / "summary.md").write_text(summary)
    print(json.dumps({"output_dir": str(outdir), "metrics": len(metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
