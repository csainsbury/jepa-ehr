"""Rung-0 retrieval ranker — patient-disjoint, exactly-frozen candidate groups (Pi R5 C4).

For each query (a CONTEXT-ONLY predicted latent), rank its TRUE target latent against
a candidate pool drawn from the SAME frozen group, with distractors that are
PATIENT-DISJOINT (a query's own patient's blocks never distract it — no linkage
shortcut) and an equal candidate budget across paired channels. Emits per-query records
(patient, rank, occupancy, source, window_days, subwindow_k, granularity) for the paired
bootstrap; aggregate-only downstream. Equal N gives equal chance, not equal difficulty —
target-geometry diagnostics live in rung0_stats.

Frozen group key (C4): source + split + EXACT window_days + target_type + granularity +
subwindow_k + occupancy stratum + context length bin + context event-rate bin.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import count_bin, length_bin, normalize, occupancy_class


def context_rate_bin(row: dict[str, Any]) -> str:
    """Coarse context event-rate bin (C4). Uses an explicit context_rate_count if the
    export provides one, else the summed context family counts, else 'na'."""
    for key in ("context_rate_count", "context_event_count"):
        if row.get(key) is not None:
            return count_bin(row.get(key))
    fams = [row.get("context_med_count"), row.get("context_lab_count"), row.get("context_state_count")]
    if any(f is not None for f in fams):
        return count_bin(int(sum(int(f or 0) for f in fams)))
    return "rate_na"


def frozen_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_dataset"), row.get("split"), row.get("window_days"),
        row.get("target_type", "T0"), row.get("granularity"), row.get("subwindow_k"),
        occupancy_class(row), length_bin(row.get("context_len")), context_rate_bin(row),
    )


def rung0_rank(
    queries: np.ndarray,
    targets: np.ndarray,
    index: list[dict[str, Any]],
    *,
    max_candidates: int = 200,
    min_candidates: int = 2,
    seed: int = 20260523,
    patient_field: str = "patient_hash",
) -> dict[str, Any]:
    """Rank each row's true target among patient-disjoint distractors from its frozen
    group. queries[i] is the predicted latent for block i; targets[i] its true latent.
    Returns {records, n_ranked, skipped_no_candidates, candidate_count_summary}."""
    if len(queries) != len(index) or len(targets) != len(index):
        raise ValueError("queries, targets, index must be the same length")
    q = normalize(np.asarray(queries))
    t = normalize(np.asarray(targets))

    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(index):
        groups[frozen_group_key(row)].append(i)

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    skipped = 0
    cand_counts: list[int] = []
    for members in groups.values():
        m = len(members)
        if m < 2:
            skipped += m
            continue
        idx = np.asarray(members)
        pats = np.array([str(index[i].get(patient_field)) for i in members])
        # Shared distractor pool per group (fixed-seed), scored VECTORIZED — avoids the
        # O(m^2) full group similarity matrix (a 43k-block group would be ~7B entries).
        if max_candidates and m > max_candidates:
            pool_pos = np.sort(rng.choice(m, size=max_candidates, replace=False))
        else:
            pool_pos = np.arange(m)
        Tp = t[idx[pool_pos]]                         # [C, D]
        Q = q[idx]                                    # [m, D]
        sims = Q @ Tp.T                               # [m, C]  cos(query a, pool distractor)
        true_sim = (q[idx] * t[idx]).sum(axis=1)      # [m]     cos(query a, its true target)
        # patient-disjoint + not-self mask over the shared pool.
        eligible = (pats[:, None] != pats[pool_pos][None, :]) & (idx[:, None] != idx[pool_pos][None, :])
        n_cand = eligible.sum(axis=1) + 1             # + the true target
        ranks = 1 + ((sims > true_sim[:, None]) & eligible).sum(axis=1)
        for a in range(m):
            if n_cand[a] < max(2, min_candidates):
                skipped += 1
                continue
            row = index[members[a]]
            cand_counts.append(int(n_cand[a]))
            records.append({
                "patient": str(row.get(patient_field)),
                "rank": int(ranks[a]),
                "occupancy": occupancy_class(row),
                "source": row.get("source_dataset"),
                "window_days": row.get("window_days"),
                "subwindow_k": row.get("subwindow_k"),
                "granularity": row.get("granularity"),
                "n_candidates": int(n_cand[a]),
            })
    arr = np.asarray(cand_counts) if cand_counts else np.asarray([0])
    return {
        "records": records,
        "n_ranked": len(records),
        "skipped_no_candidates": skipped,
        "candidate_count_summary": {"min": int(arr.min()), "median": float(np.median(arr)), "max": int(arr.max())},
        "n_groups": len(groups),
    }
