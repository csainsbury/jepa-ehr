from __future__ import annotations

import argparse
import collections
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

SAFE_COUNT_FIELDS = (
    "target_med_count",
    "target_lab_count",
    "target_state_count",
    "context_med_count",
    "context_lab_count",
    "context_state_count",
)


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _boolish(row: dict[str, Any], key: str) -> bool | None:
    if key not in row:
        return None
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "present", "pass"}
    return None


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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _pct(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _safe_group_name(key: tuple[Any, ...]) -> str:
    return "|".join(str(x) for x in key)


def compute_pseudo_rendering_report(
    query_embeddings: np.ndarray,
    query_index: list[dict[str, Any]],
    target_embeddings: np.ndarray,
    target_index: list[dict[str, Any]],
    *,
    scenario_id: str = "unspecified_scenario",
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 5, 10),
    distractor_policy: str = "same_split_target_type_len_seq_util_bin",
    max_candidates_per_group: int = 0,
    min_candidates_per_group: int = 0,
    seed: int = 20260523,
    batch_size: int = 256,
    consistency_field: str = "scenario_consistent",
    negative_control_field: str = "negative_control_present",
    exclude_true_target_from_analogues: bool = True,
    exclude_same_patient_analogues: bool = True,
) -> dict[str, Any]:
    """Aggregate-only pseudo-rendering readout over nearest retrieved target blocks.

    The function treats the top-k retrieved observed target blocks as future analogues.
    It never returns block IDs, patient hashes, token strings, or per-query examples.
    """
    if len(query_embeddings) != len(query_index):
        raise ValueError("query embedding rows and query index length differ")
    if len(target_embeddings) != len(target_index):
        raise ValueError("target embedding rows and target index length differ")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    q = normalize(query_embeddings)
    t = normalize(target_embeddings)
    target_by_block = {str(row.get("block_id")): i for i, row in enumerate(target_index)}
    target_groups: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    for i, row in enumerate(target_index):
        target_groups[group_key(row, distractor_policy)].append(i)

    query_groups: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    skipped_no_target = 0
    for i, row in enumerate(query_index):
        block_id = str(row.get("block_id"))
        if block_id not in target_by_block:
            skipped_no_target += 1
            continue
        query_groups[group_key(row, distractor_policy)].append(i)

    query_patient_key_available = any(row.get("patient_hash") for row in query_index)
    target_patient_key_available = any(row.get("patient_hash") for row in target_index)
    same_patient_exclusion_applied = bool(exclude_same_patient_analogues and query_patient_key_available and target_patient_key_available)

    rng = random.Random(seed)
    ranks: list[int] = []
    retrieved_count = 0
    retrieved_subject_hashes: set[str] = set()
    retrieved_count_fields: dict[str, list[float]] = {field: [] for field in SAFE_COUNT_FIELDS}
    consistency_values: list[bool] = []
    negative_values: list[bool] = []
    control_consistency_values: list[bool] = []
    control_negative_values: list[bool] = []
    control_retrieved_count = 0
    candidate_counts: list[int] = []
    skipped_no_candidates = 0
    per_group: dict[str, dict[str, Any]] = {}

    for key, q_indices in sorted(query_groups.items(), key=lambda kv: _safe_group_name(kv[0])):
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
        group_ranks: list[int] = []
        group_retrieved = 0
        group_consistency: list[bool] = []
        group_negative: list[bool] = []
        group_control_consistency: list[bool] = []
        group_control_negative: list[bool] = []

        for start in range(0, len(q_indices), batch_size):
            batch_q_indices = q_indices[start : start + batch_size]
            true_target_indices = [target_by_block[str(query_index[qi].get("block_id"))] for qi in batch_q_indices]
            candidates = list(dict.fromkeys([*base, *true_target_indices]))
            candidate_pos = {idx: pos for pos, idx in enumerate(candidates)}
            cand = np.asarray(candidates, dtype=np.int64)
            sims = q[np.asarray(batch_q_indices, dtype=np.int64)] @ t[cand].T
            true_cols = np.asarray([candidate_pos[idx] for idx in true_target_indices], dtype=np.int64)
            true_sims = sims[np.arange(len(batch_q_indices)), true_cols]
            batch_ranks = (sims > true_sims[:, None]).sum(axis=1).astype(np.int64) + 1
            ranks.extend(int(r) for r in batch_ranks)
            group_ranks.extend(int(r) for r in batch_ranks)

            # Aggregate pseudo-rendered analogues separately from true-target rank.
            # By default, exclude the query's own true target and any same-patient
            # target rows so analogue plausibility is not tautological.
            sorted_cols = np.argsort(-sims, axis=1)
            for row_offset, row_cols in enumerate(sorted_cols):
                query_row = query_index[batch_q_indices[row_offset]]
                query_patient = str(query_row.get("patient_hash")) if query_row.get("patient_hash") else None
                true_target_idx = true_target_indices[row_offset]
                eligible_cols: list[int] = []
                for col in row_cols:
                    target_idx = candidates[int(col)]
                    if exclude_true_target_from_analogues and target_idx == true_target_idx:
                        continue
                    target_row = target_index[target_idx]
                    target_patient = str(target_row.get("patient_hash")) if target_row.get("patient_hash") else None
                    if exclude_same_patient_analogues and query_patient and target_patient and query_patient == target_patient:
                        continue
                    eligible_cols.append(int(col))

                for col in eligible_cols[:top_k]:
                    target_row = target_index[candidates[int(col)]]
                    target_patient = str(target_row.get("patient_hash")) if target_row.get("patient_hash") else None
                    retrieved_count += 1
                    group_retrieved += 1
                    if target_patient:
                        retrieved_subject_hashes.add(target_patient)
                    for field in SAFE_COUNT_FIELDS:
                        retrieved_count_fields[field].append(_num(target_row, field, 0.0))
                    cval = _boolish(target_row, consistency_field)
                    if cval is not None:
                        consistency_values.append(cval)
                        group_consistency.append(cval)
                    nval = _boolish(target_row, negative_control_field)
                    if nval is not None:
                        negative_values.append(nval)
                        group_negative.append(nval)

                if eligible_cols:
                    sample_cols = rng.sample(eligible_cols, k=min(top_k, len(eligible_cols)))
                    for col in sample_cols:
                        target_row = target_index[candidates[int(col)]]
                        control_retrieved_count += 1
                        cval = _boolish(target_row, consistency_field)
                        if cval is not None:
                            control_consistency_values.append(cval)
                            group_control_consistency.append(cval)
                        nval = _boolish(target_row, negative_control_field)
                        if nval is not None:
                            control_negative_values.append(nval)
                            group_control_negative.append(nval)
        candidate_counts.append(len(base))
        group_name = _safe_group_name(key)
        per_group[group_name] = {
            "n_queries": len(group_ranks),
            "retrieved_topk_rows": group_retrieved,
            "candidate_count": len(base),
            "retrieval": _summarize_ranks(group_ranks, ks),
            "spec_consistency_rate": _pct(sum(group_consistency), len(group_consistency)) if group_consistency else None,
            "negative_control_rate": _pct(sum(group_negative), len(group_negative)) if group_negative else None,
            "matched_random_control": {
                "spec_consistency_rate": _pct(sum(group_control_consistency), len(group_control_consistency)) if group_control_consistency else None,
                "negative_control_rate": _pct(sum(group_control_negative), len(group_control_negative)) if group_control_negative else None,
            },
        }

    overall = _summarize_ranks(ranks, ks)
    report = {
        "schema_version": "clinical-jepa-pseudo-rendering-readiness-v0",
        "created_utc": now_utc(),
        "scenario_id": scenario_id,
        "aggregate_only": True,
        "retrieval_policy": distractor_policy,
        "top_k": top_k,
        "max_candidates_per_group": max_candidates_per_group,
        "min_candidates_per_group": min_candidates_per_group,
        "query_rows": len(query_index),
        "target_rows": len(target_index),
        "exclude_true_target_from_analogues": bool(exclude_true_target_from_analogues),
        "exclude_same_patient_analogues": bool(exclude_same_patient_analogues),
        "same_patient_exclusion_applied": same_patient_exclusion_applied,
        "queries_evaluated": overall["n"],
        "retrieved_topk_rows": retrieved_count,
        "skipped_no_target": skipped_no_target,
        "skipped_no_candidates": skipped_no_candidates,
        "candidate_count_summary": {
            "min": int(min(candidate_counts)) if candidate_counts else 0,
            "median": _median([float(x) for x in candidate_counts]),
            "max": int(max(candidate_counts)) if candidate_counts else 0,
        },
        "retrieval": overall,
        "spec_consistency_rate": _pct(sum(consistency_values), len(consistency_values)) if consistency_values else None,
        "negative_control_rate": _pct(sum(negative_values), len(negative_values)) if negative_values else None,
        "matched_random_control": {
            "retrieved_topk_rows": control_retrieved_count,
            "spec_consistency_rate": _pct(sum(control_consistency_values), len(control_consistency_values)) if control_consistency_values else None,
            "negative_control_rate": _pct(sum(control_negative_values), len(control_negative_values)) if control_negative_values else None,
            "spec_consistency_delta": None if not consistency_values or not control_consistency_values else _pct(sum(consistency_values), len(consistency_values)) - _pct(sum(control_consistency_values), len(control_consistency_values)),
            "negative_control_delta": None if not negative_values or not control_negative_values else _pct(sum(negative_values), len(negative_values)) - _pct(sum(control_negative_values), len(control_negative_values)),
        },
        "retrieved_subject_diversity_ratio": _pct(len(retrieved_subject_hashes), retrieved_count),
        "retrieved_count_summaries": {
            field: {"mean": _mean(values), "median": _median(values)} for field, values in retrieved_count_fields.items()
        },
        "groups": per_group,
        "warnings": [],
        "notes": "Top-k retrieved observed target blocks are pseudo-rendered future analogues, not generated sequences or treatment effects. No block IDs, patient hashes, raw tokens, or examples are emitted.",
    }
    if exclude_same_patient_analogues and not same_patient_exclusion_applied:
        report["warnings"].append("same_patient_exclusion_key_missing")
    elif exclude_same_patient_analogues and (not all(row.get("patient_hash") for row in query_index) or not all(row.get("patient_hash") for row in target_index)):
        report["warnings"].append("same_patient_exclusion_key_partially_missing")
    if consistency_values == []:
        report["warnings"].append("scenario_consistency_field_missing")
    if negative_values == []:
        report["warnings"].append("negative_control_field_missing")
    if retrieved_count and len(retrieved_subject_hashes) <= max(1, retrieved_count // max(top_k, 1) // 10):
        report["warnings"].append("low_retrieved_subject_diversity")
    validate_artifact("pseudo-rendering-readiness", report)
    return report


def _summary_md(report: dict[str, Any]) -> str:
    retrieval = report["retrieval"]
    lines = [
        "# Clinical-JEPA pseudo-rendering readiness",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Policy: `{report['retrieval_policy']}`",
        f"Top-k: {report['top_k']}",
        "",
        "Retrieved target blocks are observed future analogues, not generated sequences.",
        "",
        "## Overall",
        "",
        f"- Queries evaluated: {report['queries_evaluated']}",
        f"- Recall@10: {retrieval.get('recall_at_10', 0.0):.4f}",
        f"- MRR: {retrieval.get('mrr', 0.0):.4f}",
        f"- Spec consistency rate: {report['spec_consistency_rate']}",
        f"- Matched-random spec consistency rate: {report['matched_random_control']['spec_consistency_rate']}",
        f"- Negative-control rate: {report['negative_control_rate']}",
        f"- Matched-random negative-control rate: {report['matched_random_control']['negative_control_rate']}",
        f"- Retrieved subject diversity ratio: {report['retrieved_subject_diversity_ratio']:.4f}",
        f"- Warnings: {', '.join(report['warnings']) if report['warnings'] else 'none'}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate-only pseudo-rendering readiness from Clinical-JEPA embeddings")
    ap.add_argument("--query-embeddings", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--target-embeddings", required=True)
    ap.add_argument("--target-index", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenario-id", default="unspecified_scenario")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--distractor-policy", default="same_split_target_type_len_seq_util_bin", choices=POLICIES)
    ap.add_argument("--max-candidates-per-group", type=int, default=0)
    ap.add_argument("--min-candidates-per-group", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--include-true-target-analogues", action="store_true", help="Allow top-k analogue summaries to include each query's true target block")
    ap.add_argument("--include-same-patient-analogues", action="store_true", help="Allow top-k analogue summaries to include same-patient target blocks")
    args = ap.parse_args(argv)

    report = compute_pseudo_rendering_report(
        np.load(args.query_embeddings),
        read_jsonl(args.query_index),
        np.load(args.target_embeddings),
        read_jsonl(args.target_index),
        scenario_id=args.scenario_id,
        top_k=args.top_k,
        distractor_policy=args.distractor_policy,
        max_candidates_per_group=args.max_candidates_per_group,
        min_candidates_per_group=args.min_candidates_per_group,
        batch_size=args.batch_size,
        seed=args.seed,
        exclude_true_target_from_analogues=not args.include_true_target_analogues,
        exclude_same_patient_analogues=not args.include_same_patient_analogues,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "pseudo-rendering-readiness.json", report)
    (outdir / "summary.md").write_text(_summary_md(report))
    print(json.dumps({"output": str(outdir / "pseudo-rendering-readiness.json"), "queries": report["queries_evaluated"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
