"""Aggregate-real calibration extraction (Pi C=5 micro-gate submission).

Maps the TRAIN-only tokenised SCID/MIMIC substrate to the frozen ``AggregateStats`` marginals that the
calibration realism envelope consumes. **This module is the reviewed extraction implementation; it is NOT
run until the calibration micro-gate PASSes.** It reads governed HDF5 only when ``extract_source`` is
called with an explicit approval token, refuses any TEST path, and emits ONLY aggregate marginals — no
patient/sequence IDs, rows, tokens, timestamps, maps, HDF5 paths, embeddings, or checkpoints.

Governance: the DERIVED aggregates are Chris's explicit route-specific safe clearance
(``explicitly_cleared_safe_aggregate_only_no_patient_rows``); the INPUT bundle and execution environment
remain governed/local-only. Outputs go to a gitignored path; nothing here is committed or synced.

Field semantics (C=5, matched to the synthetic generator and the CALIBRATION_SPEC aggregate fields):

* structural tokens are EXCLUDED from every clinical class / count / timing quantity — ``special [0,4)``
  and ``dataset_context [1048,1050)`` (the masked ``[BOS] + DATASET:X`` source prefix). Only content
  tokens (the five clinical families) participate.
* class map: each content token id → class 0..4 via ``ORACLE_ENV_CLASS_FAMILIES`` half-open ranges
  (demographic [4,51) / diagnosis [51,91) / lab [91,951) / medication [951,1032) / state [1032,1048)).
* per sequence, the time channel ``cumulative_days`` is aligned index-for-index with ``token_ids`` and
  restricted to the content positions.
* a CLUSTER is a maximal run of content events sharing one ``cumulative_days`` value (Δt=0 multiplicity);
  ``events_per_sequence`` = number of distinct clusters; ``sequence_length`` = number of content tokens.
* an inter-event GAP is the ``cumulative_days`` difference between consecutive CLUSTERS; a positive gap has
  Δt>0; ``delta_t_zero_fraction`` = (# adjacent content-token pairs with Δt=0) / (# adjacent content-token
  pairs) — the simultaneity/multiplicity rate.

Aggregate fields (per source, over TRAIN sequences with ≥1 content token):

* ``n_sequences`` = sequences with ≥1 content token.
* ``class_counts`` = length-5 total content tokens per class, in ``ORACLE_ENV_CLASS_FAMILIES`` order.
* ``n_events`` = sum(class_counts).
* ``n_clusters`` = Σ over sequences of the distinct-cluster count.
* ``n_positive_gaps`` = Σ over sequences of the number of positive inter-cluster gaps.
* ``mean_occupancy_fraction`` = mean over sequences of (distinct classes present / 5).
* ``length_ecdf`` = ECDF of per-sequence content-token counts (``sequence_length``).
* ``count_ecdf`` = ECDF of per-sequence distinct-cluster counts (``events_per_sequence``).
* ``positive_gap_ecdf`` = ECDF of positive inter-cluster gaps (``cumulative_days`` units), support =
  ascending-unique exact observed values (Chris's explicit unbucketed-ECDF clearance).

Below ``ORACLE_ENV_MIN_DENOM`` on any denominator → the source is NOT_EVALUABLE (never zero-filled).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_calibration import AggregateStats, NOT_EVALUABLE, validate_aggregate_input
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_CLASS_FAMILIES, ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES, ORACLE_ENV_STRUCTURAL_RANGES,
)

# The explicit token the caller must pass to actually READ governed HDF5. Its presence in a call is the
# machine-checkable assertion that the calibration micro-gate has PASSed for this run. Absent it, every
# read path refuses — the module can be imported, reviewed, and unit-tested on synthetic fixtures without
# ever touching a governed bundle.
MICRO_GATE_APPROVAL_TOKEN = "CALIBRATION_MICRO_GATE_PASSED"
TIME_CHANNEL = "cumulative_days"
TRAIN_SPLIT_ONLY = "train"


class ExtractionRefused(RuntimeError):
    """Raised when a read would violate the TRAIN-only / approved-run / aggregate-only boundary."""


def _class_of(token_ids: np.ndarray) -> np.ndarray:
    """Map token ids → class 0..4, or -1 for structural/out-of-range (excluded). Vectorized, no Python
    loop over tokens."""
    cls = np.full(token_ids.shape, -1, dtype=np.int64)
    for c, (_, lo, hi) in enumerate(ORACLE_ENV_CLASS_FAMILIES):
        cls[(token_ids >= lo) & (token_ids < hi)] = c
    return cls


def _ecdf(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    """ECDF as ((support_point, cdf), ...) on ASCENDING-UNIQUE support of EXACT observed values (the
    explicitly cleared unbucketed convention). CDF is non-decreasing, final mass 1."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ()
    supp, counts = np.unique(v, return_counts=True)                 # ascending-unique, exact values
    cdf = np.cumsum(counts) / counts.sum()
    return tuple((float(round(s, 8)), float(c)) for s, c in zip(supp, cdf))


def _sequence_features(token_ids: np.ndarray, cdays: np.ndarray):
    """Per-sequence content-only features. Returns None if the sequence has no content token."""
    cls = _class_of(token_ids)
    keep = cls >= 0
    if not keep.any():
        return None
    c = cls[keep]
    t = np.asarray(cdays, dtype=float)[keep] if cdays is not None else None
    length = int(c.size)
    per_class = np.bincount(c, minlength=ORACLE_ENV_N_CLASSES)[:ORACLE_ENV_N_CLASSES]
    occupancy = float((per_class > 0).sum()) / ORACLE_ENV_N_CLASSES
    if t is None or t.size == 0:
        return {"length": length, "per_class": per_class, "occupancy": occupancy,
                "n_clusters": length, "n_pos_gaps": 0, "n_zero_adj": 0, "n_adj": max(0, length - 1),
                "pos_gaps": np.empty(0)}
    # clusters = maximal runs of equal consecutive timestamps; adjacency stats over content tokens.
    dt = np.diff(t)
    n_adj = int(dt.size)
    n_zero_adj = int((dt == 0).sum())
    boundaries = np.concatenate([[True], dt > 0]) if t.size else np.array([True])
    cluster_times = t[boundaries]
    n_clusters = int(cluster_times.size)
    inter = np.diff(cluster_times)
    pos_gaps = inter[inter > 0]
    return {"length": length, "per_class": per_class, "occupancy": occupancy,
            "n_clusters": n_clusters, "n_pos_gaps": int(pos_gaps.size),
            "n_zero_adj": n_zero_adj, "n_adj": n_adj, "pos_gaps": pos_gaps}


def aggregate_from_sequences(source: str, sequences) -> AggregateStats | str:
    """Build ``AggregateStats`` for one source from an ITERABLE of (token_ids, cumulative_days) content
    sequences. PURE — no I/O — so it is unit-tested on synthetic fixtures. Returns NOT_EVALUABLE if any
    denominator falls below the floor (never zero-filled)."""
    lengths, cluster_counts, occ = [], [], []
    class_tot = np.zeros(ORACLE_ENV_N_CLASSES, dtype=np.int64)
    n_clusters = n_pos_gaps = n_zero_adj = n_adj = 0
    pos_gap_pool: list[np.ndarray] = []
    for token_ids, cdays in sequences:
        f = _sequence_features(np.asarray(token_ids), cdays)
        if f is None:
            continue
        lengths.append(f["length"]); cluster_counts.append(f["n_clusters"]); occ.append(f["occupancy"])
        class_tot += f["per_class"]
        n_clusters += f["n_clusters"]; n_pos_gaps += f["n_pos_gaps"]
        n_zero_adj += f["n_zero_adj"]; n_adj += f["n_adj"]
        if f["pos_gaps"].size:
            pos_gap_pool.append(f["pos_gaps"])
    n_seq = len(lengths)
    if n_seq == 0:
        return NOT_EVALUABLE
    n_events = int(class_tot.sum())
    if min(n_seq, n_clusters, n_events, n_pos_gaps) < ORACLE_ENV_MIN_DENOM:
        return NOT_EVALUABLE                                        # below floor => not evaluable, no fill
    pos_gaps = np.concatenate(pos_gap_pool) if pos_gap_pool else np.empty(0)
    agg = AggregateStats(
        source=source, n_sequences=n_seq, n_events=n_events, n_clusters=n_clusters,
        n_positive_gaps=n_pos_gaps, class_counts=tuple(int(x) for x in class_tot),
        delta_t_zero_fraction=float(n_zero_adj / n_adj) if n_adj else 0.0,
        length_ecdf=_ecdf(np.asarray(lengths)), positive_gap_ecdf=_ecdf(pos_gaps),
        count_ecdf=_ecdf(np.asarray(cluster_counts)),
        mean_occupancy_fraction=float(np.mean(occ)) if occ else 0.0)
    ok, reason = validate_aggregate_input(agg)
    return agg if ok else reason


def _iter_h5_content_sequences(h5_path: str):
    """Yield (token_ids, cumulative_days) for every group in a TRAIN h5. Governed read — reached only
    from ``extract_source`` after the approval + TRAIN-only guards pass."""
    import h5py
    with h5py.File(h5_path, "r") as h5:
        for group in h5.keys():
            grp = h5.get(group)
            if grp is None or "token_ids" not in grp:
                continue
            tok = grp["token_ids"][:]
            cdays = grp[TIME_CHANNEL][:] if TIME_CHANNEL in grp else None
            yield tok, cdays


def extract_source(source: str, h5_path: str, split: str, *, approval_token: str) -> AggregateStats | str:
    """Fail-closed governed extraction for one source. Refuses unless (a) the approval token matches — the
    machine-checkable assertion that the micro-gate PASSed — (b) the split is TRAIN, and (c) the path does
    not look like a TEST/sealed bundle. Emits only AggregateStats (or NOT_EVALUABLE)."""
    if approval_token != MICRO_GATE_APPROVAL_TOKEN:
        raise ExtractionRefused("aggregate-real read refused: micro-gate approval token absent/incorrect")
    if split != TRAIN_SPLIT_ONLY:
        raise ExtractionRefused(f"aggregate-real read refused: split={split!r} is not TRAIN")
    low = h5_path.lower()
    if "test" in low or "sealed" in low:
        raise ExtractionRefused(f"aggregate-real read refused: path looks like TEST/sealed: {h5_path!r}")
    return aggregate_from_sequences(source, _iter_h5_content_sequences(h5_path))


def sanitized_output(aggs: dict[str, AggregateStats | str]) -> dict[str, Any]:
    """The ONLY thing written to disk / eligible to be committed: per-source aggregate marginals, scanned
    for forbidden row-level keys before return (fail-closed on any leak)."""
    from dataclasses import asdict
    from clinical_jepa.validation import _scan_forbidden_aggregate_keys
    out: dict[str, Any] = {"governance_class": "explicitly_cleared_safe_aggregate_only_no_patient_rows",
                           "split": TRAIN_SPLIT_ONLY, "n_classes": ORACLE_ENV_N_CLASSES, "sources": {}}
    for src, a in aggs.items():
        out["sources"][src] = a if isinstance(a, str) else asdict(a)
    leaks = _scan_forbidden_aggregate_keys(out)
    if leaks:
        raise ExtractionRefused("aggregate output failed the forbidden-key scan: " + "; ".join(leaks))
    return out
