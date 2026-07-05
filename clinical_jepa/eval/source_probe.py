"""Source-prediction probe from latents (source-shortcut mask verification).

Rung guard (rung0_1_run_specs.md; design §4a change #3): after masking the
DATASET:SCID/MIMIC source token from the encoder/predictor input, a probe that
tries to predict the source from the pooled latents should sit **near the source
base-rate**. A probe accuracy well above base-rate means the source is still
leaking into the representation (mask ineffective or a residual source shortcut).

Numpy-only logistic-regression probe; reads a latent array + an index carrying
per-row ``source_dataset``. Output is aggregate-only (accuracies, base-rate, and
the above-base-rate delta) — never embeddings or per-row predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.utils import ensure_dir, now_utc, write_json


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int, lr: float, l2: float) -> np.ndarray:
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    w = np.zeros(xb.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = xb.T @ (p - y) / n
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return w


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
    z = xb @ w
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))) >= 0.5).astype(np.int64)


def source_prediction_probe(
    latents: np.ndarray,
    sources: list[str],
    *,
    seed: int = 20260523,
    test_fraction: float = 0.3,
    steps: int = 400,
    lr: float = 0.2,
    l2: float = 1e-3,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Fit a held-out logistic probe of source from latents.

    ``tolerance`` is the above-base-rate margin under which the mask is deemed to
    be working (probe cannot recover source beyond chance + tolerance).
    """
    latents = np.asarray(latents, dtype=np.float64)
    if latents.ndim != 2:
        raise ValueError("latents must be a 2D [n, dim] array")
    if len(sources) != latents.shape[0]:
        raise ValueError("sources length must match latent rows")

    counts = Counter(sources)
    majority_source, majority_count = counts.most_common(1)[0]
    n = latents.shape[0]

    result: dict[str, Any] = {
        "created_utc": now_utc(),
        "n": int(n),
        "dim": int(latents.shape[1]),
        "sources": {str(k): int(v) for k, v in counts.items()},
        "majority_source": str(majority_source),
        "majority_base_rate": float(majority_count / n) if n else 0.0,
        "tolerance": float(tolerance),
        # Pi round 2: this probe is an ALARM/REPORT ONLY. MIMIC and SCI-D differ
        # distributionally, so source can stay predictable even after a correct
        # mask; near-base-rate is NOT required as proof of no leakage. The hard
        # control is source-matched candidates (retrieval `same_source_*`), not
        # this probe. Nothing may gate pass/fail on `near_base_rate`.
        "role": "alarm_report_only",
        "is_pass_fail_gate": False,
        "near_base_rate_required": False,
        "aggregate_only": True,
    }

    if len(counts) < 2 or n < 8:
        result.update({
            "status": "not_applicable",
            "reason": "need >=2 source classes and >=8 rows",
            "source_prediction_accuracy": None,
            "balanced_accuracy": None,
            "above_base_rate_delta": None,
            "near_base_rate": None,
        })
        return result

    # Binary label: majority source vs rest (exact discrimination for 2 sources).
    y = np.asarray([0 if s == majority_source else 1 for s in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(2, int(round(n * test_fraction)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    if len(train_idx) < 2 or len(set(y[train_idx].tolist())) < 2:
        result.update({
            "status": "not_applicable",
            "reason": "degenerate train split (single class)",
            "source_prediction_accuracy": None,
            "balanced_accuracy": None,
            "above_base_rate_delta": None,
            "near_base_rate": None,
        })
        return result

    x_train, x_test = _standardize(latents[train_idx], latents[test_idx])
    w = _fit_logistic(x_train, y[train_idx], steps=steps, lr=lr, l2=l2)
    pred = _predict(x_test, w)
    y_test = y[test_idx].astype(np.int64)
    accuracy = float((pred == y_test).mean())

    # Balanced accuracy over the two classes present in test.
    per_class = []
    for cls in (0, 1):
        mask = y_test == cls
        if mask.sum() > 0:
            per_class.append(float((pred[mask] == cls).mean()))
    balanced = float(np.mean(per_class)) if per_class else accuracy

    # Test-set base-rate is the honest chance level for that split.
    test_majority = float(max(np.mean(y_test == 0), np.mean(y_test == 1)))
    delta = accuracy - test_majority
    result.update({
        "status": "ok",
        "n_test": int(n_test),
        "source_prediction_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "test_base_rate": test_majority,
        "above_base_rate_delta": float(delta),
        "near_base_rate": bool(delta <= tolerance),
    })
    return result


def _read_sources(index_path: str | Path, n_expected: int) -> list[str]:
    sources: list[str] = []
    with Path(index_path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                sources.append(str(row.get("source_dataset", "unknown")))
    if len(sources) != n_expected:
        raise ValueError(f"index rows ({len(sources)}) != latent rows ({n_expected})")
    return sources


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Source-prediction probe from latents (mask verification)")
    ap.add_argument("--latents", required=True, help="npy array [n, dim] of pooled context latents")
    ap.add_argument("--index", required=True, help="jsonl carrying per-row source_dataset (row-aligned)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args(argv)

    latents = np.load(args.latents)
    sources = _read_sources(args.index, latents.shape[0])
    report = source_prediction_probe(latents, sources, seed=args.seed, tolerance=args.tolerance)
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "source-prediction-probe.json", report)
    print(json.dumps({
        "output": str(outdir / "source-prediction-probe.json"),
        "accuracy": report.get("source_prediction_accuracy"),
        "base_rate": report.get("majority_base_rate"),
        "near_base_rate": report.get("near_base_rate"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
