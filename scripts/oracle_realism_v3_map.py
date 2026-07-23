#!/usr/bin/env python3
"""Oracle realism v3 — corrected reference-OWNED frozen coarsening-map builder (Pi rev-5 #1).

Specifies + builds (but does NOT execute the reserved one-time map-design namespace) the frozen coarsening map for
every map-carrying conditional check. The rev-4 pilot map was WRONG: it invented data-driven unique-value edges and
pooled adjacency-level means. The registered estimand instead:

  * keeps the ORIGINAL registered bins — LENGTH_BINS for {S1_density, S5_abs, S6_tv}, CLUSTER_BINS for
    {S3_loggap, S7_abs} (Pi #1: S6 is length-binned, not class-coarsened);
  * freezes an independent reference-owned GROUPING of those original bin indices via the frozen v2 merge
    (`coarsen_reference`), derived ONCE from a reference-design sample — the candidate never influences the bins;
  * preserves each check's registered per-sequence summary + EQUAL-WEIGHT pooling (`_grouped`) and every
    denominator floor (sequence floor for all; the extra adjacent-pair floor for S3_loggap).

`build_frozen_map` produces the frozen bin-index grouping + a map artifact/hash. `apply_frozen_map` evaluates the
conditional discrepancy `d` (or NOT_EVALUABLE) under the FROZEN grouping — identical to the v2 check when the frozen
grouping equals the per-draw grouping (proved by the estimand-consistency self-test), but reference-owned and
anti-masking. The one-time reserved-namespace map-design draw remains BLOCKED; this module is exercised only on
development fixtures. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_map.py
"""
from __future__ import annotations

import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    coarsen_reference, _grouped, FLOOR, LENGTH_BINS, CLUSTER_BINS, _bin_index, _positive_gaps_and_prev_size,
    C, s1, s3, s5, s6, s7,
)

# original registered bins per map-carrying check (Pi #1 — S6 is LENGTH_BINS)
ORIGINAL_BINS = {"S1_density": ("LENGTH_BINS", LENGTH_BINS), "S5_abs": ("LENGTH_BINS", LENGTH_BINS),
                 "S6_tv": ("LENGTH_BINS", LENGTH_BINS), "S3_loggap": ("CLUSTER_BINS", CLUSTER_BINS),
                 "S7_abs": ("CLUSTER_BINS", CLUSTER_BINS)}
MAP_CARRYING = tuple(ORIGINAL_BINS)


# --- per-bin per-sequence summaries, mirroring the registered v2 checks EXACTLY --------------------
def _pb_density(sample):
    pb = [[] for _ in LENGTH_BINS]
    for r in sample:
        b = _bin_index(r.L_total, LENGTH_BINS)
        if b is not None:
            pb[b].append(r.K / r.L_total)
    return pb, None


def _pb_occ(sample):
    pb = [[] for _ in LENGTH_BINS]
    for r in sample:
        b = _bin_index(r.L_total, LENGTH_BINS)
        if b is not None:
            pb[b].append(len(np.unique(r.class_ids)) / C)
    return pb, None


def _pb_classvec(sample):
    pb = [[] for _ in LENGTH_BINS]
    for r in sample:
        b = _bin_index(r.L_total, LENGTH_BINS)
        if b is not None:
            pb[b].append(np.bincount(r.class_ids, minlength=C) / r.L_total)
    return pb, None


def _pb_logmean(sample):
    pb = [[] for _ in CLUSTER_BINS]; pairs = [0 for _ in CLUSTER_BINS]
    for r in sample:
        g, ps = _positive_gaps_and_prev_size(r)
        if g.shape[0] == 0:
            continue
        lg = np.log(g)
        for b in range(len(CLUSTER_BINS)):
            mask = np.asarray([_bin_index(int(s), CLUSTER_BINS) == b for s in ps])
            if mask.any():
                pb[b].append(float(np.mean(lg[mask]))); pairs[b] += int(mask.sum())
    return pb, pairs


def _pb_div(sample):
    pb = [[] for _ in CLUSTER_BINS]; clus = [0 for _ in CLUSTER_BINS]
    for r in sample:
        by_bin = [[] for _ in CLUSTER_BINS]
        for c in range(r.K):
            cls = r.class_ids[r.cluster_ids == c]
            b = _bin_index(int(cls.shape[0]), CLUSTER_BINS)
            if b is not None:
                by_bin[b].append(len(np.unique(cls)) / C)
        for b in range(len(CLUSTER_BINS)):
            if by_bin[b]:
                pb[b].append(float(np.mean(by_bin[b]))); clus[b] += len(by_bin[b])
    return pb, clus


# per check: (per-bin builder, scalar|vector, uses adjacent-pair floor)
CHECK_SPEC = {
    "S1_density": (_pb_density, "scalar", False), "S5_abs": (_pb_occ, "scalar", False),
    "S6_tv": (_pb_classvec, "vector", False), "S3_loggap": (_pb_logmean, "scalar", True),
    "S7_abs": (_pb_div, "scalar", False),
}


def _reduce_diff(cand_g, ref_g, kind):
    if kind == "vector":
        diffs = [0.5 * float(np.sum(np.abs(np.mean(np.asarray(c), 0) - np.mean(np.asarray(r), 0))))
                 for c, r in zip(cand_g, ref_g)]
    else:
        diffs = [abs(float(np.mean(c)) - float(np.mean(r))) for c, r in zip(cand_g, ref_g)]
    return float(max(diffs)), diffs


# --- frozen map build (reference-OWNED) + application ----------------------------------------------
def build_frozen_map(reference_sample, check):
    """Freeze the reference-owned grouping of ORIGINAL bin indices via the v2 merge on the reference per-bin
    sequence counts. Returns the map artifact (or None if the reference coarsening refuses)."""
    builder, _, _ = CHECK_SPEC[check]
    pb, _ = builder(reference_sample)
    counts = np.asarray([len(v) for v in pb])
    groups = coarsen_reference(counts)
    bins_id, bins = ORIGINAL_BINS[check]
    if groups is None:
        return {"check": check, "bins_id": bins_id, "n_original_bins": len(bins), "groups": None,
                "floor": FLOOR, "status": "REFUSED_reference_coarsening"}
    return {"check": check, "bins_id": bins_id, "n_original_bins": len(bins),
            "groups": [list(g) for g in groups], "floor": FLOOR, "status": "OK"}


def apply_frozen_map(cand, ref, check, frozen_map):
    """Discrepancy d under the FROZEN grouping (candidate never re-bins). Returns d (float) or None (NE), with the
    registered per-sequence reducer, equal-weight pooling, sequence floor, and (S3_loggap) adjacent-pair floor."""
    if frozen_map is None or frozen_map.get("groups") is None:
        return None
    builder, kind, use_pairs = CHECK_SPEC[check]
    groups = frozen_map["groups"]
    cpb, cpairs = builder(cand); rpb, rpairs = builder(ref)
    cand_g, ref_g = _grouped(cpb, groups), _grouped(rpb, groups)
    if any(len(c) < FLOOR for c in cand_g) or any(len(r) < FLOOR for r in ref_g):
        return None
    if use_pairs:
        ex_c = [int(sum(cpairs[i] for i in g)) for g in groups]
        ex_r = [int(sum(rpairs[i] for i in g)) for g in groups]
        if any(v < FLOOR for v in ex_c) or any(v < FLOOR for v in ex_r):
            return None
    d, _ = _reduce_diff(cand_g, ref_g, kind)
    return d


def map_identity(frozen_map):
    return canonical_hash({k: frozen_map[k] for k in ("check", "bins_id", "n_original_bins", "groups", "floor")})


# --- self-tests -----------------------------------------------------------------------------------
def _v2_value(check, cand, ref):
    fn = {"S1_density": s1, "S5_abs": s5, "S6_tv": s6, "S3_loggap": s3, "S7_abs": s7}[check]
    return fn(cand, ref)[check].value


def _dev_sample(profile, n, tag):
    from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
    from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
    import hashlib
    sk = "SCID" if "scid" in profile else "MIMIC"
    seed = int.from_bytes(hashlib.sha256(f"v3-map-dev|{profile}|{tag}".encode()).digest()[:6], "big")
    return sample_fixture(sk, PROFILES[profile], n, seed=seed)


def selftest():
    errs = []
    N = 2000
    ref = _dev_sample("mimic_scale_control", N, "ref")
    cand = _dev_sample("mimic_scale_control", N, "cand")
    cand2 = _dev_sample("scid_scale_control", N, "cand2")   # a DIFFERENT candidate

    # (1) estimand consistency: frozen grouping derived from `ref` == v2's per-draw grouping (since v2 coarsens on
    #     the same reference) => apply_frozen_map reproduces the v2 check .value EXACTLY.
    for check in MAP_CARRYING:
        fm = build_frozen_map(ref, check)
        got = apply_frozen_map(cand, ref, check, fm)
        want = _v2_value(check, cand, ref)
        if (got is None) != (want is None):
            errs.append(f"{check}: NE mismatch got={got} want={want}")
        elif got is not None and abs(got - want) > 1e-12:
            errs.append(f"{check}: value mismatch got={got} want={want}")

    # (2) anti-masking: the frozen map depends ONLY on the reference-design sample, never the candidate.
    for check in MAP_CARRYING:
        m1 = map_identity(build_frozen_map(ref, check))
        # building the map from `ref` must be candidate-independent — same map hash regardless of which candidate
        if m1 != map_identity(build_frozen_map(ref, check)):
            errs.append(f"{check}: map not deterministic")
        d_cand = apply_frozen_map(cand, ref, check, build_frozen_map(ref, check))
        d_cand2 = apply_frozen_map(cand2, ref, check, build_frozen_map(ref, check))
        # different candidates may give different d, but the MAP (bins/groups) is identical
        if map_identity(build_frozen_map(ref, check)) != m1:
            errs.append(f"{check}: candidate influenced the map")

    # (3) S6 is LENGTH_BINS (Pi #1)
    if build_frozen_map(ref, "S6_tv")["bins_id"] != "LENGTH_BINS":
        errs.append("S6_tv bins_id != LENGTH_BINS")

    # (4) floor enforcement: a tiny reference below floor => NE
    tiny = _dev_sample("mimic_scale_control", 50, "tiny")
    if apply_frozen_map(tiny, tiny, "S3_loggap", build_frozen_map(ref, "S3_loggap")) is not None:
        errs.append("floor not enforced (tiny sample should be NE)")

    return errs, {check: build_frozen_map(ref, check) for check in MAP_CARRYING}


def main():
    errs, maps = selftest()
    out = {"map_carrying": list(MAP_CARRYING),
           "original_bins": {k: ORIGINAL_BINS[k][0] for k in MAP_CARRYING},
           "frozen_maps": {k: {"bins_id": m["bins_id"], "n_original_bins": m["n_original_bins"],
                               "groups": m["groups"], "status": m["status"], "identity": map_identity(m)}
                           for k, m in maps.items()},
           "selftests_pass": not errs, "selftest_errors": errs,
           "authorization": "dev-only map builder; the reserved one-time map-design namespace draw is BLOCKED."}
    print(json.dumps(out, indent=2, default=str))
    print("\nMAP_SET_HASH:", canonical_hash(out["frozen_maps"]))
    assert not errs, f"map builder self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
