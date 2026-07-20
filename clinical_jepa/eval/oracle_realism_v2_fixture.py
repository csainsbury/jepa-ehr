"""Independent reference/control fixture constructor for the realism-v2 verifier (rebuild step 3).

Implements the canonical maximal-run fixture law (design `FIXTURE_LAW`) as a CLOSED-FORM, INDEPENDENT
constructor: it shares no code path with any (future) M2 candidate adapter and performs NO governed read (it
uses only cleared development-seen aggregate CONSTANTS — the length scale — carried in the profile). This module
is the baseline (dependence = 0); the exact coupling constructions land separately under the frozen
`COUPLING_PROTOCOL` and must not be encoded here.

Also provides the derive-not-trust `SequenceRecord` derivation and the EXACT registered six-marginal estimators
(pooled class_tv, pooled dt0, positive-gap ECDF, per-sequence length/count ECDFs, equal-sequence occupancy).

Synthetic-only. Pi binding conditions: exact raw `dt==0` cluster semantics; `L==1` dt0 excluded; malformed /
non-finite / bool input refusal; structural-zero preservation; short-sequence handling. Invariants are tested
directly (see tests/test_oracle_v2_fixture.py).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
from clinical_jepa.eval.rung2_contract import ORACLE_ENV_N_CLASSES

C = ORACLE_ENV_N_CLASSES     # 5


class MalformedRecord(ValueError):
    """A SequenceRecord violated the derive-not-trust input schema."""


@dataclasses.dataclass(frozen=True)
class SequenceRecord:
    """A full content-token sequence (emission-side). Only source/class_ids/timestamps are trusted; the rest
    are DERIVED under exact raw-timestamp-equality cluster semantics."""
    source: str
    class_ids: np.ndarray          # (L,) ints in [0, C)
    timestamps: np.ndarray         # (L,) nondecreasing floats; Δt==0 => same cluster
    cluster_ids: np.ndarray        # (L,) derived canonical contiguous run ids
    L_total: int
    K: int                         # number of maximal runs
    positions: np.ndarray          # (L,) normalized index/(L-1); 0.0 if L==1

    @property
    def block_count_B(self) -> int:
        return self.L_total // 8

    @property
    def residual_R(self) -> int:
        return self.L_total % 8


def _is_int_array(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_


def derive_record(source: str, class_ids, timestamps) -> SequenceRecord:
    """Validate + derive a SequenceRecord (derive-not-trust). Clusters are maximal runs under EXACT raw
    timestamp equality (dt==0); every positive adjacency must satisfy t[i+1] > t[i]. Refuses malformed input."""
    if source not in REQUIRED_SOURCES:
        raise MalformedRecord(f"source {source!r} not in {REQUIRED_SOURCES}")
    ci = np.asarray(class_ids)
    ts = np.asarray(timestamps)
    if ci.ndim != 1 or ts.ndim != 1 or ci.shape[0] != ts.shape[0]:
        raise MalformedRecord("class_ids and timestamps must be equal-length 1-D arrays")
    L = int(ci.shape[0])
    if L < 1:
        raise MalformedRecord("empty sequence forbidden (L_total >= 1)")
    if ci.dtype == np.bool_ or not _is_int_array(ci):
        raise MalformedRecord("class_ids must be non-boolean integers")
    if np.issubdtype(ts.dtype, np.bool_) or not np.issubdtype(ts.dtype, np.floating):
        raise MalformedRecord("timestamps must be floats (not bool/int)")
    if not np.all(np.isfinite(ts)):
        raise MalformedRecord("non-finite timestamp")
    if np.any((ci < 0) | (ci >= C)):
        raise MalformedRecord(f"class id out of range [0,{C})")
    if L > 1:
        dt = np.diff(ts)
        if np.any(dt < 0):
            raise MalformedRecord("timestamps must be nondecreasing")
        # canonical contiguous run ids under EXACT equality; boundary iff dt > 0 (strictly positive)
        boundary = dt > 0.0
        cluster_ids = np.concatenate([[0], np.cumsum(boundary)]).astype(int)
    else:
        cluster_ids = np.zeros(1, dtype=int)
    K = int(cluster_ids[-1] + 1)
    positions = (np.arange(L, dtype=float) / (L - 1)) if L > 1 else np.zeros(1)
    return SequenceRecord(source=source, class_ids=ci.astype(int), timestamps=ts.astype(float),
                          cluster_ids=cluster_ids, L_total=L, K=K, positions=positions)


# ---------------------------------------------------------------------------------------------------
# canonical maximal-run fixture law (baseline; dependence = 0)
# ---------------------------------------------------------------------------------------------------
def _sample_length(profile: dict, rng: np.random.Generator) -> int:
    lg = profile["length"]
    if lg["family"] != "discretized_lognormal":
        raise ValueError(f"unsupported length family {lg['family']!r}")
    val = int(round(float(np.exp(rng.normal(lg["mu"], lg["sigma"])))))
    return max(lg.get("min", 1), val)


def _sample_runs(L: int, profile: dict, rng: np.random.Generator) -> np.ndarray:
    """Maximal run sizes summing to EXACTLY L, geometric with a frozen terminal-truncation rule: draw runs
    until the cumulative sum reaches/exceeds L, then TRUNCATE the last run to hit L (drop it if truncation
    would be 0)."""
    cs = profile["cluster_size"]
    if cs["family"] != "geometric":
        raise ValueError(f"unsupported cluster_size family {cs['family']!r}")
    p = cs["p"]
    runs: list[int] = []
    total = 0
    while total < L:
        r = int(rng.geometric(p))            # >= 1
        if total + r >= L:
            r = L - total                    # terminal truncation to hit L exactly
        runs.append(r)
        total += r
    return np.asarray([r for r in runs if r > 0], dtype=int)


def _sample_class_prior(profile: dict) -> np.ndarray:
    prior = np.asarray(profile["class_prior"], dtype=float)
    zeros = profile.get("structural_zero_classes", [])
    if zeros:
        prior = prior.copy()
        prior[list(zeros)] = 0.0            # HARD structural zeros
    s = prior.sum()
    if s <= 0:
        raise ValueError("class prior sums to zero")
    return prior / s


def build_sequence(source: str, profile: dict, rng: np.random.Generator) -> SequenceRecord:
    """One baseline SequenceRecord via the canonical law (no coupling). Timestamps are cumulative with EXACT
    Δt==0 within a run and a strictly-positive inter-run gap (float-collapse deterministically nudged)."""
    L = _sample_length(profile, rng)
    runs = _sample_runs(L, profile, rng)
    K = int(runs.shape[0])
    prior = _sample_class_prior(profile)
    class_ids = rng.choice(C, size=L, p=prior)
    gl = profile["gap"]
    ts = np.zeros(L, dtype=float)
    idx = int(runs[0])
    t = 0.0
    for k in range(1, K):
        gap = float(np.exp(rng.normal(gl["mu"], gl["sigma"])))
        new_t = t + gap
        if not (new_t > t):                  # float-collapse guard => deterministic nudge
            new_t = np.nextafter(t, np.inf)
        t = new_t
        r = int(runs[k])
        ts[idx:idx + r] = t
        idx += r
    return derive_record(source, class_ids, ts)


def sample_fixture(source: str, profile: dict, n: int, *, seed: int) -> list[SequenceRecord]:
    """A synthetic reference/control sample of ``n`` independent sequences (baseline)."""
    rng = np.random.default_rng(seed)
    return [build_sequence(source, profile, rng) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------
# EXACT registered six-marginal estimators (Pi defect-1 semantics)
# ---------------------------------------------------------------------------------------------------
def reg_lengths(sample) -> np.ndarray:
    return np.asarray([r.L_total for r in sample], dtype=float)


def reg_cluster_counts(sample) -> np.ndarray:
    return np.asarray([r.K for r in sample], dtype=float)


def reg_class_tv_proportions(sample) -> np.ndarray:
    """POOLED C-class event-count proportions (class_counts / n_events) — the registered class_tv estimand."""
    counts = np.zeros(C, dtype=float)
    for r in sample:
        counts += np.bincount(r.class_ids, minlength=C)
    total = counts.sum()
    return counts / total if total > 0 else counts


def reg_occupancy_mean(sample) -> float:
    """Equal-sequence mean(distinct classes / C)."""
    return float(np.mean([len(np.unique(r.class_ids)) / C for r in sample]))


def reg_dt0_pooled(sample) -> float:
    """POOLED zero adjacencies / POOLED adjacencies. L==1 sequences contribute ZERO adjacencies (excluded)."""
    zero = 0
    adj = 0
    for r in sample:
        if r.L_total < 2:
            continue
        dt = np.diff(r.timestamps)
        zero += int(np.count_nonzero(dt == 0.0))
        adj += int(dt.shape[0])
    return (zero / adj) if adj > 0 else float("nan")


def reg_positive_gaps(sample) -> np.ndarray:
    """POOLED positive inter-cluster gaps, support rounded to 8dp (right-continuous ECDF convention applied by
    the KS consumer)."""
    gaps: list[float] = []
    for r in sample:
        if r.L_total < 2:
            continue
        dt = np.diff(r.timestamps)
        gaps.extend(np.round(dt[dt > 0.0], 8).tolist())
    return np.asarray(gaps, dtype=float)


# ---------------------------------------------------------------------------------------------------
# implementation identity (Pi binding condition 1) — the code-level constants of this baseline constructor
# ---------------------------------------------------------------------------------------------------
FIXTURE_IMPL = {
    "name": "realism_v2_reference_constructor_dev",
    "law": "canonical_maximal_run",
    "length_family": "discretized_lognormal(min>=1)",
    "run_family": "geometric with terminal truncation to hit L exactly (drop zero-truncation)",
    "class_family": "multinomial with hard structural zeros",
    "gap_family": "lognormal positive; float-collapse => np.nextafter nudge",
    "cluster_semantics": "maximal runs under EXACT raw dt==0 equality; boundary iff dt>0",
    "dt0": "pooled zero/pooled adjacencies; L==1 excluded",
    "positive_gap_support_round_dp": 8,
    "n_classes": C,
    "couplings": "NONE (baseline dependence=0; couplings land separately under COUPLING_PROTOCOL)",
    "independence": "no import/call of any M2 candidate adapter; no governed read",
}


def fixture_impl_identity() -> str:
    """Code-level identity of the baseline reference constructor (bumps when the executable behaviour lands)."""
    return canonical_hash(FIXTURE_IMPL)
