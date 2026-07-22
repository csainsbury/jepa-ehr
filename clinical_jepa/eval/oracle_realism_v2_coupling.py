"""Exact marginal-preserving coupling constructions for the realism-v2 fixture (rebuild step 3).

Implements the five D-component coupling laws under the frozen `COUPLING_PROTOCOL`. The load-bearing mechanism
is CYCLE ACTIVATION of a target permutation: any permutation preserves the underlying multiset exactly, so a
strength ``s`` in [0, 0.6] activates a deterministic fraction of the permutation's cycles (interpolating from
identity at s=0 to the full target). Each component pins its exact pre/post-state, integer ``s→units`` (cycle
count), a dedicated coupling RNG (seed derived from a base seed + component), and infeasibility behaviour.

Exact invariants (tested directly in tests/test_oracle_v2_coupling.py):
  * burst_count_length          — pooled L multiset AND pooled K multiset preserved exactly;
  * burst_timing                — per-sequence positive-gap multiset preserved exactly;
  * mark_burst_tie              — per-sequence class counts preserved exactly;
  * cluster_size_mark_diversity — per-sequence class counts AND cluster sizes preserved exactly;
  * length_class_mix            — pooled class counts (=> class_tv) preserved exactly.

Preservation of the OTHER registered marginals / S2 is EMPIRICALLY REQUIRED (tested >=24/25 in the ablation
battery), NOT asserted by construction; a preservation failure is a DESIGN FAIL / re-gate, never threshold
tuning. Synthetic-only; shares no code path with any (future) M2 candidate adapter.

DIRECT-TESTING FINDINGS (Pi binding condition 2 — surfaced BEFORE building the ablation battery; require a Pi
ruling as they change the ablation expectations):
  * F1 — `burst_count_length` is NOT an effective S1 mover. The baseline maximal-run process already induces a
    very strong L↔K coupling (abs tau(L,K) ~= 0.92), so the exact comonotone permutation has almost no
    headroom (moves S1_tau only ~0.02 at s=0.6), and S1_density (K/L) cannot be moved without breaking the S2
    run-size marginal (K, L, run-size are algebraically linked). => S1 is largely a MARGINAL/structural
    property, not a separable dependence. Options for Pi: drop burst_count_length + treat S1 as marginal/
    terminal; OR allow S2 movement for this component's ablation; OR accept only weak/near-threshold movement.
  * F2 — `length_class_mix` moves S6 (length-conditioned class MIX, TV ~0.15 at s=0.5 => FAIL) but NOT S5
    (occupancy by length): relabelling shifts the mix, not the distinct-class COUNT. Options: map S5 -> a
    distinct-occupancy-by-length component (or make S5 allowed-sensitive/terminal); keep S6 -> length_class_mix.
  * WORKING (strong exact movers, confirmed): burst_timing->S3 (S3_tau 0.50/S3_loggap 0.96 @0.6), mark_burst_tie
    ->S4 (0.38), cluster_size_mark_diversity->S7 (0.19); S4<->S7 CROSS-LOADING confirmed empirically (each also
    moves the other ~0.25-0.31) — recorded, not orthogonalized, per the design.
"""
from __future__ import annotations

import hashlib

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU, V2_FROZEN_BINS
from clinical_jepa.eval.oracle_realism_v2_fixture import C, SequenceRecord, derive_record

_LENGTH_BINS = V2_FROZEN_BINS["length"]


def _rng(seed: int, component: str) -> np.random.Generator:
    h = hashlib.sha256(f"coupling|{component}|{seed}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def _runs(rec) -> np.ndarray:
    return np.bincount(rec.cluster_ids).astype(int)


def _first_indices(rec) -> np.ndarray:
    return np.concatenate([[0], np.where(np.diff(rec.cluster_ids) == 1)[0] + 1]) if rec.L_total > 1 else np.array([0])


def _cycles(perm: np.ndarray) -> list[list[int]]:
    """Non-trivial cycles (len>1) of a permutation ``perm`` (new[i] = old[perm[i]])."""
    n = perm.shape[0]; seen = np.zeros(n, bool); out = []
    for i in range(n):
        if not seen[i]:
            c = []; j = i
            while not seen[j]:
                seen[j] = True; c.append(j); j = int(perm[j])
            if len(c) > 1:
                out.append(c)
    return out


def _activate(perm: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation that applies a fraction ``s`` of ``perm``'s cycles (identity elsewhere). Preserves
    the multiset of whatever it indexes (it is a permutation)."""
    cycles = _cycles(perm)
    out = np.arange(perm.shape[0])
    if not cycles:
        return out
    m = int(round(float(np.clip(s, 0.0, 0.6)) * len(cycles)))
    if m <= 0:
        return out
    chosen = rng.permutation(len(cycles))[:m]
    for ci in chosen:
        for idx in cycles[ci]:
            out[idx] = int(perm[idx])
    return out


def _perm_from_target(orig: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Permutation sigma with orig[sigma] == target (both the SAME label multiset). Matches equal labels in
    order, so cycle-activation interpolates original -> target while preserving counts exactly."""
    sigma = np.empty(orig.shape[0], dtype=int)
    for c in range(C):
        op = np.where(orig == c)[0]
        tp = np.where(target == c)[0]
        sigma[tp] = op
    return sigma


def _rebuild_timestamps(run_sizes: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    L = int(run_sizes.sum()); ts = np.zeros(L, dtype=float)
    idx = int(run_sizes[0]); t = 0.0
    for k in range(1, run_sizes.shape[0]):
        new_t = t + float(gaps[k - 1])
        if not (new_t > t):
            new_t = np.nextafter(t, np.inf)
        t = new_t
        r = int(run_sizes[k]); ts[idx:idx + r] = t; idx += r
    return ts


def _bin_of_length(L: int) -> int | None:
    for i, (lo, hi) in enumerate(_LENGTH_BINS):
        if L >= lo and (hi is None or L <= hi):
            return i
    return None


# ==================================================================================================
# burst_timing — per-sequence positive-gap multiset preserved; couple gap to preceding cluster size
# ==================================================================================================
def _couple_burst_timing(rec, s, rng):
    K = rec.K
    if K < 2:
        return rec
    runs = _runs(rec)
    prev_size = runs[:-1]
    ft = rec.timestamps[_first_indices(rec)]
    gaps = np.diff(ft)                                   # K-1 strictly-positive gaps
    order = np.argsort(prev_size, kind="stable")        # boundaries sorted by preceding size
    target = np.empty(K - 1, dtype=int)                 # comonotone: sorted gaps -> sorted boundaries
    target[order] = np.argsort(gaps, kind="stable")
    new_gaps = gaps[_activate(target, s, rng)]
    ts = _rebuild_timestamps(runs, new_gaps)
    return derive_record(rec.source, rec.class_ids.copy(), ts)


# ==================================================================================================
# mark_burst_tie — per-sequence class counts preserved; raise same-cluster same-class
# ==================================================================================================
def _homogeneous_target(rec) -> np.ndarray:
    """Target label array (permutation of the sequence's label multiset) that fills whole clusters with a
    single label where possible — raising same-CLUSTER same-class (S4). Reassigns labels ACROSS clusters."""
    counts = np.bincount(rec.class_ids, minlength=C).astype(int)
    runs = _runs(rec)
    order = np.argsort(-runs, kind="stable")               # largest clusters first
    target = np.empty(rec.L_total, dtype=int)
    for c in order:
        pos = np.where(rec.cluster_ids == c)[0]
        need = pos.shape[0]
        filled = 0
        while filled < need:
            cls = int(np.argmax(counts))                   # most abundant remaining label
            take = min(need - filled, counts[cls])
            target[pos[filled:filled + take]] = cls
            counts[cls] -= take; filled += take
    return target


def _couple_mark_burst_tie(rec, s, rng):
    if rec.K < 1 or rec.L_total < 2:
        return rec
    sigma = _perm_from_target(rec.class_ids, _homogeneous_target(rec))
    new_labels = rec.class_ids[_activate(sigma, s, rng)]    # permutation of labels => exact counts
    return derive_record(rec.source, new_labels, rec.timestamps.copy())


# ==================================================================================================
# cluster_size_mark_diversity — per-seq class counts + cluster sizes preserved; couple diversity to size
# ==================================================================================================
def _quartile_of(rec) -> np.ndarray:
    return np.minimum(3, (rec.positions * 4).astype(int))


def _diversity_target(rec) -> np.ndarray:
    """POSITION-BALANCED target labels (Pi F3): relabelling is constrained WITHIN each sequence x canonical
    position quartile, preserving the exact C=5 class-count vector in every quartile (=> S8_class invariant by
    construction) while, within that constraint, concentrating LARGE clusters / diversifying SMALL clusters
    (=> couples class diversity to cluster SIZE, S7). Clusters spanning a quartile boundary are handled
    per-segment (each item assigned by its own quartile)."""
    runs = _runs(rec)
    med = float(np.median(runs)) if runs.size else 1.0
    quart = _quartile_of(rec)
    target = np.full(rec.L_total, -1, dtype=int)
    for qi in range(4):
        qmask = quart == qi
        if not qmask.any():
            continue
        counts = np.bincount(rec.class_ids[qmask], minlength=C).astype(int)   # this quartile's own multiset
        # cluster segments intersecting this quartile, big (concentrate) after small (diversify)
        segs = []
        for c in range(rec.K):
            seg = np.where(qmask & (rec.cluster_ids == c))[0]
            if seg.shape[0] > 0:
                segs.append((runs[c] >= med, seg))
        for big in (False, True):
            for is_big, seg in [(b, s) for (b, s) in segs if b == big]:
                need = seg.shape[0]; filled = 0
                if not big:                                # diversify: cycle distinct available labels
                    while filled < need:
                        for cl in [x for x in np.argsort(-counts) if counts[x] > 0]:
                            if filled >= need:
                                break
                            target[seg[filled]] = cl; counts[cl] -= 1; filled += 1
                else:                                      # concentrate: whole segment one label if possible
                    while filled < need:
                        cl = int(np.argmax(counts)); take = min(need - filled, counts[cl])
                        target[seg[filled:filled + take]] = cl; counts[cl] -= take; filled += take
    return target


def _couple_cluster_size_mark_diversity(rec, s, rng):
    if rec.K < 1 or rec.L_total < 2:
        return rec
    quart = _quartile_of(rec)
    target = _diversity_target(rec)
    # build sigma WITHIN each quartile so cycle-activation never moves labels across quartiles => per-quartile
    # class counts are preserved at EVERY strength (S8_class invariant by construction).
    sigma = np.arange(rec.L_total)
    for qi in range(4):
        qidx = np.where(quart == qi)[0]
        if qidx.size == 0:
            continue
        o = rec.class_ids[qidx]; t = target[qidx]
        local = np.empty(qidx.size, dtype=int)
        for cl in range(C):
            local[np.where(t == cl)[0]] = np.where(o == cl)[0]
        sigma[qidx] = qidx[local]
    new_labels = rec.class_ids[_activate(sigma, s, rng)]
    return derive_record(rec.source, new_labels, rec.timestamps.copy())


# ==================================================================================================
# burst_count_length — pooled L + K multisets preserved; couple K to L (comonotone across sequences)
# ==================================================================================================
def _couple_burst_count_length(sample, s, rng):
    n = len(sample)
    L = np.asarray([r.L_total for r in sample]); K = np.asarray([r.K for r in sample])
    # comonotone target as a PERMUTATION sigma of sequence indices: assigned_K = K[sigma] gives the r-th
    # smallest K to the sequence with the r-th smallest L. K[sigma] is a permutation of K => multiset exact.
    rank_L = np.argsort(np.argsort(L, kind="stable"), kind="stable")
    order_K = np.argsort(K, kind="stable")
    sigma = order_K[rank_L]
    perm = _activate(sigma, s, rng)
    assigned_K = K[perm]
    p_hat = 1.0 / max(1.5, float(np.mean(np.concatenate([_runs(r) for r in sample]))))
    out = []
    pooled_gaps = np.concatenate([np.diff(r.timestamps[_first_indices(r)]) for r in sample if r.K > 1])
    for i, r in enumerate(sample):
        kt = int(assigned_K[i])
        if kt == r.K:
            out.append(r); continue
        runs = _recompose_runs(r.L_total, kt, p_hat, rng)
        gaps = rng.choice(pooled_gaps, size=kt - 1) if kt > 1 else np.array([])
        ts = _rebuild_timestamps(runs, gaps)
        out.append(derive_record(r.source, r.class_ids.copy(), ts))
    return out


def _recompose_runs(L: int, K: int, p_hat: float, rng) -> np.ndarray:
    """Partition L into exactly K positive runs, shape ~geometric(p_hat) (keeps S2 close). Deterministic
    fix-up to sum EXACTLY to L."""
    if K == 1:
        return np.asarray([L], dtype=int)
    g = rng.geometric(p_hat, size=K).astype(float)
    g = np.maximum(g, 1.0)
    sizes = np.maximum(1, np.round(g / g.sum() * L)).astype(int)
    diff = L - int(sizes.sum())
    j = 0
    while diff != 0:                                     # nudge to hit L exactly, keeping all >=1
        k = j % K
        if diff > 0:
            sizes[k] += 1; diff -= 1
        elif sizes[k] > 1:
            sizes[k] -= 1; diff += 1
        j += 1
    return sizes


# ==================================================================================================
# length_class_mix — pooled class counts preserved; shift conditional class-mix by length bin
# ==================================================================================================
def _couple_length_class_mix(sample, s, rng, *, target_class=0):
    """LABEL-LEVEL swaps between LONG-bin and SHORT-bin sequences: give long sequences more of ``target_class``
    and short sequences less, exchanging one target_class label (short) for one non-target label (long). Each
    swap preserves pooled class counts EXACTLY (long +target/-x; short -target/+x). Strength scales the number
    of label swaps against the smaller available pool, so the conditional class mix (S5/S6) moves materially."""
    bins = np.asarray([(_bin_of_length(r.L_total) or 0) for r in sample])
    lo, hi = int(bins.min()), int(bins.max())
    if hi == lo:
        return list(sample)
    thresh = (lo + hi + 1) // 2
    labels = [r.class_ids.copy() for r in sample]
    # candidate slots: non-target labels in LONG sequences; target labels in SHORT sequences
    long_slots = [(i, p) for i in range(len(sample)) if bins[i] >= thresh
                  for p in np.where(labels[i] != target_class)[0]]
    short_slots = [(i, p) for i in range(len(sample)) if bins[i] < thresh
                   for p in np.where(labels[i] == target_class)[0]]
    budget = min(len(long_slots), len(short_slots))
    m = int(round(float(np.clip(s, 0.0, 0.6)) * budget))
    if m == 0:
        return list(sample)
    li = rng.permutation(len(long_slots))[:m]; sj = rng.permutation(len(short_slots))[:m]
    for a, b in zip(li, sj):
        ia, pa = long_slots[a]; ib, pb = short_slots[b]
        x = int(labels[ia][pa])                          # non-target class in the long sequence
        labels[ia][pa] = target_class                    # long: +target, -x
        labels[ib][pb] = x                               # short: -target, +x  (pooled unchanged)
    return [derive_record(r.source, labels[i], r.timestamps.copy()) for i, r in enumerate(sample)]


# ==================================================================================================
# dispatch
# ==================================================================================================
_PER_SEQUENCE = {
    "burst_timing": _couple_burst_timing,
    "mark_burst_tie": _couple_mark_burst_tie,
    "cluster_size_mark_diversity": _couple_cluster_size_mark_diversity,
}
# burst_count_length is REJECTED as an active D component (Pi F1); its construction is retained above as a
# historical diagnostic but is NOT dispatched — apply_coupling raises KeyError (it is not in the menu).
_SAMPLE_LEVEL = {
    "length_class_mix": _couple_length_class_mix,
}


def apply_coupling(sample, component: str, strength: float, *, seed: int) -> list:
    """Apply a single D-component coupling at ``strength`` in [0,0.6]. ``seed`` derives the dedicated coupling
    RNG. s==0 is a no-op. Returns a new sample (records rebuilt via derive_record)."""
    if component not in V2_D_COMPONENT_MENU:
        raise KeyError(f"unknown D component {component!r}")
    s = float(strength)
    if not (0.0 <= s <= 0.6):
        raise ValueError("strength must be in [0, 0.6]")
    if s == 0.0:
        return list(sample)
    rng = _rng(seed, component)
    if component in _SAMPLE_LEVEL:
        return _SAMPLE_LEVEL[component](list(sample), s, rng)
    fn = _PER_SEQUENCE[component]
    return [fn(r, s, rng) for r in sample]


COUPLING_IMPL = {
    "name": "realism_v2_coupling_dev",
    "mechanism": "cycle activation of a target permutation (multiset-exact) + cross-sequence label swaps",
    "components": list(V2_D_COMPONENT_MENU),
    "strength_range": [0.0, 0.6],
    "rng": "sha256(coupling|component|seed) -> default_rng",
    "exact_invariants": {   # ACTIVE only — exactly V2_D_COMPONENT_MENU (Pi re-gate #4)
        "burst_timing": "per-sequence positive-gap multiset",
        "mark_burst_tie": "per-sequence class counts",
        "cluster_size_mark_diversity": "per-sequence x position-QUARTILE class counts + cluster sizes "
                                       "(position-balanced; S8_class invariant by construction, Pi F3)",
    },
    "rejected": {
        "burst_count_length": "Pi F1 — S1 is structural under the maximal-run law (not separable D)",
        "length_class_mix": "Pi step-4 result gate — drives terminal S5 (candidate-A S5_abs ~19 FAIL/25); dropped",
    },
    "empirically_required": "other registered marginals + S2 tested >=24/25; failure => DESIGN FAIL / re-gate",
}


def coupling_impl_identity() -> str:
    return canonical_hash(COUPLING_IMPL)
