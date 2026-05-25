from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import POLICIES, group_key, normalize, read_jsonl
from clinical_jepa.utils import ensure_dir, now_utc, write_json
from clinical_jepa.validation import validate_artifact

CONTROL_MODES = ("none", "query_shift", "target_shift", "time_shift", "all")


def _as_rollout(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must have shape (n, dim) or (n, horizon, dim); got {x.shape}")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _effective_rank(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    centered = x.astype(np.float32, copy=False) - x.astype(np.float32, copy=False).mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    total = float(s.sum())
    if total <= 1e-12:
        return 0.0
    p = s / total
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _summarize_ranks(rank_values: list[int], ks: tuple[int, ...]) -> dict[str, Any]:
    if not rank_values:
        return {"n": 0, "mrr": 0.0, "mean_rank": None, "median_rank": None, **{f"recall_at_{k}": 0.0 for k in ks}}
    arr = np.asarray(rank_values, dtype=np.float32)
    out: dict[str, Any] = {
        "n": len(rank_values),
        "mrr": float(np.mean(1.0 / arr)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
    }
    for k in ks:
        out[f"recall_at_{k}"] = float(np.mean(arr <= k))
    return out


def _slope(values: list[float | None]) -> float | None:
    y = np.asarray([v for v in values if v is not None], dtype=np.float32)
    if len(y) < 2:
        return None
    x = np.arange(len(y), dtype=np.float32)
    return float(np.polyfit(x, y, deg=1)[0])


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return np.sum(a * b, axis=1) / denom


def _candidate_groups(index: list[dict[str, Any]], policy: str) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(index):
        groups.setdefault(group_key(row, policy), []).append(i)
    return groups


def _matched_random_indices(index: list[dict[str, Any]], policy: str, rng: random.Random) -> tuple[np.ndarray, int]:
    """Return a within-policy shuffled target row for each query row."""
    groups = _candidate_groups(index, policy)
    control = np.arange(len(index), dtype=np.int64)
    small_groups = 0
    for indices in groups.values():
        if len(indices) <= 1:
            small_groups += len(indices)
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) > 1 and all(a == b for a, b in zip(indices, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
        for src, dst in zip(indices, shuffled):
            control[src] = dst
    return control, small_groups


def _requested_controls(control_mode: str) -> list[str]:
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"Unsupported control_mode={control_mode!r}; expected one of {CONTROL_MODES}")
    if control_mode == "none":
        return []
    if control_mode == "all":
        return ["query_shift", "target_shift", "time_shift"]
    return [control_mode]


def _control_summary(per_horizon: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_horizon:
        return {"n_horizons": 0}
    return {
        "n_horizons": len(per_horizon),
        "cosine_aligned_minus_control_mean": _mean([float(row["cosine_aligned_minus_control"]) for row in per_horizon]),
        "mrr_aligned_minus_control_mean": _mean([float(row["mrr_aligned_minus_control"]) for row in per_horizon]),
        "recall_at_10_aligned_minus_control_mean": _mean([float(row["recall_at_10_aligned_minus_control"]) for row in per_horizon]),
    }


def _retrieval_for_horizon(
    pred_h: np.ndarray,
    target_h: np.ndarray,
    index: list[dict[str, Any]],
    *,
    distractor_policy: str,
    ks: tuple[int, ...],
    max_candidates_per_group: int,
    min_candidates_per_group: int,
    rng: random.Random,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_norm = normalize(pred_h)
    target_norm = normalize(target_h)
    groups = _candidate_groups(index, distractor_policy)
    ranks: list[int] = []
    candidate_counts: list[int] = []
    skipped_no_candidates = 0

    for q_indices in groups.values():
        group_targets = list(q_indices)
        if len(group_targets) <= 1 or (min_candidates_per_group and len(group_targets) < min_candidates_per_group):
            skipped_no_candidates += len(q_indices)
            continue
        if max_candidates_per_group and len(group_targets) > max_candidates_per_group:
            base = list(group_targets)
            rng.shuffle(base)
            base = base[:max_candidates_per_group]
        else:
            base = group_targets
        candidate_counts.append(len(base))
        for start in range(0, len(q_indices), batch_size):
            batch_q_indices = q_indices[start : start + batch_size]
            candidates = list(dict.fromkeys([*base, *batch_q_indices]))
            candidate_pos = {idx: pos for pos, idx in enumerate(candidates)}
            cand = np.asarray(candidates, dtype=np.int64)
            sims = pred_norm[np.asarray(batch_q_indices, dtype=np.int64)] @ target_norm[cand].T
            true_cols = np.asarray([candidate_pos[idx] for idx in batch_q_indices], dtype=np.int64)
            true_sims = sims[np.arange(len(batch_q_indices)), true_cols]
            batch_ranks = (sims > true_sims[:, None]).sum(axis=1).astype(np.int64) + 1
            ranks.extend(int(r) for r in batch_ranks)

    retrieval = _summarize_ranks(ranks, ks)
    candidate_summary = {
        "groups": len(groups),
        "skipped_no_candidates": skipped_no_candidates,
        "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
        "candidate_count_median": _median([float(x) for x in candidate_counts]),
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
    }
    return retrieval, candidate_summary


def _control_metrics_for_horizon(
    control_pred_h: np.ndarray,
    control_target_h: np.ndarray,
    index: list[dict[str, Any]],
    *,
    aligned_cosine_mean: float,
    aligned_l2_mean: float,
    aligned_mae_mean: float,
    aligned_retrieval: dict[str, Any],
    distractor_policy: str,
    ks: tuple[int, ...],
    max_candidates_per_group: int,
    min_candidates_per_group: int,
    rng: random.Random,
    batch_size: int,
) -> dict[str, Any]:
    control_cos = _safe_cosine(control_pred_h, control_target_h)
    control_l2 = np.linalg.norm(control_pred_h - control_target_h, axis=1)
    control_mae = np.mean(np.abs(control_pred_h - control_target_h), axis=1)
    control_retrieval, control_candidate_summary = _retrieval_for_horizon(
        control_pred_h,
        control_target_h,
        index,
        distractor_policy=distractor_policy,
        ks=ks,
        max_candidates_per_group=max_candidates_per_group,
        min_candidates_per_group=min_candidates_per_group,
        rng=rng,
        batch_size=batch_size,
    )
    aligned_recall_10 = float(aligned_retrieval.get("recall_at_10", 0.0) or 0.0)
    control_recall_10 = float(control_retrieval.get("recall_at_10", 0.0) or 0.0)
    return {
        "cosine_mean": float(np.mean(control_cos)),
        "cosine_aligned_minus_control": float(aligned_cosine_mean - np.mean(control_cos)),
        "l2_mean": float(np.mean(control_l2)),
        "l2_aligned_minus_control": float(aligned_l2_mean - np.mean(control_l2)),
        "mae_mean": float(np.mean(control_mae)),
        "mae_aligned_minus_control": float(aligned_mae_mean - np.mean(control_mae)),
        "retrieval": control_retrieval,
        "candidate_summary": control_candidate_summary,
        "mrr_aligned_minus_control": float(aligned_retrieval.get("mrr", 0.0) - control_retrieval.get("mrr", 0.0)),
        "recall_at_10_aligned_minus_control": float(aligned_recall_10 - control_recall_10),
    }


def compute_autoregression_readiness_report(
    predicted_rollout: np.ndarray,
    target_rollout: np.ndarray,
    index: list[dict[str, Any]],
    *,
    scenario_id: str = "unspecified_autoregression_scenario",
    ks: tuple[int, ...] = (1, 5, 10),
    distractor_policy: str = "same_split_target_type_len_seq_util_bin",
    max_candidates_per_group: int = 0,
    min_candidates_per_group: int = 0,
    seed: int = 20260523,
    batch_size: int = 256,
    control_mode: str = "none",
) -> dict[str, Any]:
    """Aggregate-only latent autoregression readiness metrics.

    `predicted_rollout` and `target_rollout` must be aligned by row and horizon.
    Arrays may be 2D `(n, dim)` for one-step checks or 3D `(n, horizon, dim)`
    for true autoregressive rollout checks. The output intentionally contains no
    row identifiers, patient hashes, examples, raw tokens, or embeddings.
    """
    pred = _as_rollout(predicted_rollout, "predicted_rollout").astype(np.float32, copy=False)
    target = _as_rollout(target_rollout, "target_rollout").astype(np.float32, copy=False)
    if pred.shape != target.shape:
        raise ValueError(f"predicted and target rollout shapes differ: {pred.shape} vs {target.shape}")
    if len(index) != pred.shape[0]:
        raise ValueError(f"index length {len(index)} does not match rollout rows {pred.shape[0]}")
    if pred.shape[0] == 0:
        raise ValueError("rollout arrays must contain at least one row")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("rollout arrays contain non-finite values")

    rng = random.Random(seed)
    control_rng = random.Random(seed + 1009)
    n, horizons, dim = pred.shape
    per_horizon: list[dict[str, Any]] = []
    warnings: list[str] = []
    requested_controls = _requested_controls(control_mode)
    shift_controls: dict[str, Any] = {"mode": control_mode, "controls": {}}
    for name in requested_controls:
        shift_controls["controls"][name] = {"per_horizon": []}
    query_shift_indices: np.ndarray | None = None
    target_shift_indices: np.ndarray | None = None
    query_shift_small_groups = 0
    target_shift_small_groups = 0
    if "query_shift" in requested_controls:
        query_shift_indices, query_shift_small_groups = _matched_random_indices(index, distractor_policy, control_rng)
    if "target_shift" in requested_controls:
        target_shift_indices, target_shift_small_groups = _matched_random_indices(index, distractor_policy, control_rng)
    if "time_shift" in requested_controls and horizons < 2:
        warnings.append("time_shift_control_requires_multiple_horizons")

    for h in range(horizons):
        pred_h = pred[:, h, :]
        target_h = target[:, h, :]
        cos = _safe_cosine(pred_h, target_h)
        l2 = np.linalg.norm(pred_h - target_h, axis=1)
        mae = np.mean(np.abs(pred_h - target_h), axis=1)
        retrieval, candidate_summary = _retrieval_for_horizon(
            pred_h,
            target_h,
            index,
            distractor_policy=distractor_policy,
            ks=ks,
            max_candidates_per_group=max_candidates_per_group,
            min_candidates_per_group=min_candidates_per_group,
            rng=rng,
            batch_size=batch_size,
        )
        control_indices, control_small_groups = _matched_random_indices(index, distractor_policy, rng)
        control_target_h = target_h[control_indices]
        control_cos = _safe_cosine(pred_h, control_target_h)
        control_l2 = np.linalg.norm(pred_h - control_target_h, axis=1)
        control_mae = np.mean(np.abs(pred_h - control_target_h), axis=1)
        aligned_cosine_mean = float(np.mean(cos))
        aligned_l2_mean = float(np.mean(l2))
        aligned_mae_mean = float(np.mean(mae))
        per_horizon.append({
            "horizon_index": h,
            "n": int(n),
            "cosine_mean": aligned_cosine_mean,
            "cosine_median": float(np.median(cos)),
            "l2_mean": aligned_l2_mean,
            "mae_mean": aligned_mae_mean,
            "pred_norm_mean": float(np.mean(np.linalg.norm(pred_h, axis=1))),
            "target_norm_mean": float(np.mean(np.linalg.norm(target_h, axis=1))),
            "pred_effective_rank": _effective_rank(pred_h),
            "target_effective_rank": _effective_rank(target_h),
            "retrieval": retrieval,
            "candidate_summary": candidate_summary,
            "matched_random_control": {
                "policy": distractor_policy,
                "small_group_rows": int(control_small_groups),
                "cosine_mean": float(np.mean(control_cos)),
                "cosine_delta": float(np.mean(cos) - np.mean(control_cos)),
                "l2_mean": float(np.mean(control_l2)),
                "l2_delta": float(np.mean(l2) - np.mean(control_l2)),
                "mae_mean": float(np.mean(control_mae)),
                "mae_delta": float(np.mean(mae) - np.mean(control_mae)),
            },
        })

        if query_shift_indices is not None:
            metrics = _control_metrics_for_horizon(
                pred_h[query_shift_indices],
                target_h,
                index,
                aligned_cosine_mean=aligned_cosine_mean,
                aligned_l2_mean=aligned_l2_mean,
                aligned_mae_mean=aligned_mae_mean,
                aligned_retrieval=retrieval,
                distractor_policy=distractor_policy,
                ks=ks,
                max_candidates_per_group=max_candidates_per_group,
                min_candidates_per_group=min_candidates_per_group,
                rng=control_rng,
                batch_size=batch_size,
            )
            metrics.update({"horizon_index": h, "query_shift_small_group_rows": int(query_shift_small_groups)})
            shift_controls["controls"]["query_shift"]["per_horizon"].append(metrics)

        if target_shift_indices is not None:
            metrics = _control_metrics_for_horizon(
                pred_h,
                target_h[target_shift_indices],
                index,
                aligned_cosine_mean=aligned_cosine_mean,
                aligned_l2_mean=aligned_l2_mean,
                aligned_mae_mean=aligned_mae_mean,
                aligned_retrieval=retrieval,
                distractor_policy=distractor_policy,
                ks=ks,
                max_candidates_per_group=max_candidates_per_group,
                min_candidates_per_group=min_candidates_per_group,
                rng=control_rng,
                batch_size=batch_size,
            )
            metrics.update({"horizon_index": h, "target_shift_small_group_rows": int(target_shift_small_groups)})
            shift_controls["controls"]["target_shift"]["per_horizon"].append(metrics)

        if "time_shift" in requested_controls and horizons >= 2:
            target_horizon_index = int((h + 1) % horizons)
            metrics = _control_metrics_for_horizon(
                pred_h,
                target[:, target_horizon_index, :],
                index,
                aligned_cosine_mean=aligned_cosine_mean,
                aligned_l2_mean=aligned_l2_mean,
                aligned_mae_mean=aligned_mae_mean,
                aligned_retrieval=retrieval,
                distractor_policy=distractor_policy,
                ks=ks,
                max_candidates_per_group=max_candidates_per_group,
                min_candidates_per_group=min_candidates_per_group,
                rng=control_rng,
                batch_size=batch_size,
            )
            metrics.update({"horizon_index": h, "target_horizon_index": target_horizon_index, "mode": "cyclic_next_horizon"})
            shift_controls["controls"]["time_shift"]["per_horizon"].append(metrics)

    for control in shift_controls["controls"].values():
        control["summary"] = _control_summary(control.get("per_horizon", []))

    transitions: list[dict[str, Any]] = []
    if horizons < 2:
        warnings.append("single_horizon_no_autoregressive_transition_check")
    else:
        for h in range(1, horizons):
            pred_delta = pred[:, h, :] - pred[:, h - 1, :]
            target_delta = target[:, h, :] - target[:, h - 1, :]
            direction = _safe_cosine(pred_delta, target_delta)
            pred_norm = np.linalg.norm(pred_delta, axis=1)
            target_norm = np.linalg.norm(target_delta, axis=1)
            transitions.append({
                "from_horizon_index": h - 1,
                "to_horizon_index": h,
                "direction_cosine_mean": float(np.mean(direction)),
                "direction_cosine_median": float(np.median(direction)),
                "pred_step_norm_mean": float(np.mean(pred_norm)),
                "target_step_norm_mean": float(np.mean(target_norm)),
                "step_norm_ratio_mean": float(np.mean(pred_norm / np.maximum(target_norm, 1e-8))),
            })

    cosine_values = [h["cosine_mean"] for h in per_horizon]
    mrr_values = [h["retrieval"]["mrr"] for h in per_horizon]
    recall10_values = [h["retrieval"].get("recall_at_10") for h in per_horizon]
    report = {
        "schema_version": "clinical-jepa-autoregression-readiness-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "n_sequences": int(n),
        "n_horizons": int(horizons),
        "embedding_dim": int(dim),
        "retrieval_policy": distractor_policy,
        "max_candidates_per_group": int(max_candidates_per_group),
        "min_candidates_per_group": int(min_candidates_per_group),
        "control_mode": control_mode,
        "per_horizon": per_horizon,
        "shift_controls": shift_controls,
        "transition_dynamics": transitions,
        "rollout_summary": {
            "cosine_first": cosine_values[0] if cosine_values else None,
            "cosine_last": cosine_values[-1] if cosine_values else None,
            "cosine_last_minus_first": (cosine_values[-1] - cosine_values[0]) if len(cosine_values) >= 2 else None,
            "cosine_slope_per_horizon": _slope(cosine_values),
            "mrr_first": mrr_values[0] if mrr_values else None,
            "mrr_last": mrr_values[-1] if mrr_values else None,
            "mrr_last_minus_first": (mrr_values[-1] - mrr_values[0]) if len(mrr_values) >= 2 else None,
            "mrr_slope_per_horizon": _slope(mrr_values),
            "recall_at_10_first": recall10_values[0] if recall10_values else None,
            "recall_at_10_last": recall10_values[-1] if recall10_values else None,
            "recall_at_10_last_minus_first": (recall10_values[-1] - recall10_values[0]) if len(recall10_values) >= 2 else None,
            "recall_at_10_slope_per_horizon": _slope(recall10_values),
        },
        "warnings": warnings,
        "notes": "Latent autoregression readiness only: predicted rollout embeddings are compared with aligned observed future embeddings, within-policy matched-random controls, and optional query/target/time-shift controls. This is not explicit event generation, not external transfer, and not treatment-effect estimation. No row IDs, patient hashes, raw tokens, examples, or embeddings are emitted.",
    }
    validate_artifact("autoregression-readiness", report)
    return report


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Clinical-JEPA autoregression readiness",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Policy: `{report['retrieval_policy']}`",
        f"Sequences: {report['n_sequences']}",
        f"Horizons: {report['n_horizons']}",
        "",
        "Latent autoregression readiness only: this is not explicit event generation, external-transfer evidence, or treatment-effect estimation.",
        "",
        "## Horizon metrics",
        "",
        "| Horizon | Cosine mean | Matched-random cosine | Cosine delta | Recall@10 | MRR | Pred rank | Target rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_horizon"]:
        retrieval = row["retrieval"]
        control = row["matched_random_control"]
        lines.append(
            f"| {row['horizon_index']} | {row['cosine_mean']:.4f} | {control['cosine_mean']:.4f} | {control['cosine_delta']:.4f} | {retrieval.get('recall_at_10', 0.0):.4f} | {retrieval.get('mrr', 0.0):.4f} | {row['pred_effective_rank']:.2f} | {row['target_effective_rank']:.2f} |"
        )
    controls = report.get("shift_controls", {}).get("controls", {})
    if controls:
        lines.extend([
            "",
            "## Shift controls",
            "",
            "| Control | Horizon | Cosine mean | Cosine delta | Recall@10 | Recall delta | MRR | MRR delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, control in controls.items():
            for row in control.get("per_horizon", []):
                retrieval = row["retrieval"]
                lines.append(
                    f"| `{name}` | {row['horizon_index']} | {row['cosine_mean']:.4f} | {row['cosine_aligned_minus_control']:.4f} | {retrieval.get('recall_at_10', 0.0):.4f} | {row['recall_at_10_aligned_minus_control']:.4f} | {retrieval.get('mrr', 0.0):.4f} | {row['mrr_aligned_minus_control']:.4f} |"
                )
    summary = report["rollout_summary"]
    lines.extend([
        "",
        "## Rollout summary",
        "",
        f"- Cosine last-minus-first: {summary['cosine_last_minus_first']}",
        f"- Recall@10 last-minus-first: {summary['recall_at_10_last_minus_first']}",
        f"- MRR last-minus-first: {summary['mrr_last_minus_first']}",
        f"- Warnings: {', '.join(report['warnings']) if report['warnings'] else 'none'}",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate-only latent autoregression readiness metrics")
    ap.add_argument("--predicted-rollout", required=True)
    ap.add_argument("--target-rollout", required=True)
    ap.add_argument("--index", required=True, help="JSONL rows aligned to rollout row order")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="unspecified_autoregression_scenario")
    ap.add_argument("--distractor-policy", default="same_split_target_type_len_seq_util_bin", choices=POLICIES)
    ap.add_argument("--max-candidates-per-group", type=int, default=0)
    ap.add_argument("--min-candidates-per-group", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--control-mode", default="none", choices=CONTROL_MODES, help="Optional aggregate shift controls to compute")
    args = ap.parse_args(argv)

    report = compute_autoregression_readiness_report(
        np.load(args.predicted_rollout),
        np.load(args.target_rollout),
        read_jsonl(args.index),
        scenario_id=args.scenario_id,
        distractor_policy=args.distractor_policy,
        max_candidates_per_group=args.max_candidates_per_group,
        min_candidates_per_group=args.min_candidates_per_group,
        batch_size=args.batch_size,
        seed=args.seed,
        control_mode=args.control_mode,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "autoregression-readiness.json", report)
    (outdir / "summary.md").write_text(_summary_md(report))
    print(json.dumps({"output": str(outdir / "autoregression-readiness.json"), "sequences": report["n_sequences"], "horizons": report["n_horizons"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
