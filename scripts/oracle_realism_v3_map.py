#!/usr/bin/env python3
"""Oracle realism v3 — reference-OWNED frozen coarsening-map builder + ISSUANCE schema (Pi rev-5 #1 / rev-6 #3).

Builds (but does NOT execute the reserved one-time map-design namespace) the frozen coarsening map for every
map-carrying conditional check, preserving the REGISTERED estimand:

  * ORIGINAL registered bins — LENGTH_BINS for {S1_density, S5_abs, S6_tv} (S6 is length-binned), CLUSTER_BINS
    for {S3_loggap, S7_abs};
  * a reference-owned GROUPING of those original bin indices via the frozen v2 merge (`coarsen_reference`),
    derived ONCE from a reference-design sample — the candidate never influences the bins;
  * the registered per-sequence summary + EQUAL-WEIGHT pooling (`_grouped`) and every floor (sequence floor for
    all; the extra adjacent-pair floor for S3_loggap).

Pi rev-6 #3: the emitted artifact is ISSUANCE-COMPLETE — it binds a full trust root (profile, support regime,
map-design namespace, seed, N, estimator identity, original-bin identity, denominator policy, per-map provenance)
and is schema-validated. `apply_frozen_map` REFUSES a map applied under a different profile/regime/check identity.
The self-test proves exact v2-value reproduction at ADEQUATE support (every map `OK`, not `None==None`) and covers
malformed-artifact / cross-identity / floor refusals. The reserved-namespace map-design draw remains BLOCKED; this
module runs only on development fixtures. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_map.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    coarsen_reference, _grouped, FLOOR, LENGTH_BINS, CLUSTER_BINS, _bin_index, _positive_gaps_and_prev_size,
    C, s1, s3, s5, s6, s7,
)
from scripts.oracle_realism_v3_randomization import RefusalError

ORIGINAL_BINS = {"S1_density": "LENGTH_BINS", "S5_abs": "LENGTH_BINS", "S6_tv": "LENGTH_BINS",
                 "S3_loggap": "CLUSTER_BINS", "S7_abs": "CLUSTER_BINS"}
_BINS = {"LENGTH_BINS": LENGTH_BINS, "CLUSTER_BINS": CLUSTER_BINS}
MAP_CARRYING = tuple(ORIGINAL_BINS)
ESTIMATOR_ID = {
    "S1_density": "v2.cond_maxbin.mean(per_bin_density)@LENGTH_BINS[ref_coarsen]",
    "S5_abs": "v2.cond_maxbin.mean(per_bin_occupancy)@LENGTH_BINS[ref_coarsen]",
    "S6_tv": "v2.maxabs_tv(class_prior)@LENGTH_BINS[ref_coarsen]",
    "S3_loggap": "v2.cond_maxbin.maxabs(mean_log_positive_gap)@CLUSTER_BINS[ref_coarsen]",
    "S7_abs": "v2.cond_maxbin.mean(per_bin_distinct_class_frac)@CLUSTER_BINS[ref_coarsen]",
}
DENOM_POLICY = {c: ("seq_floor=%d + adjacent_pair_floor=%d per coarsened group, both arms" % (FLOOR, FLOOR)
                    if c == "S3_loggap" else "seq_floor=%d per coarsened group, both arms" % FLOOR)
                for c in MAP_CARRYING}


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


CHECK_SPEC = {"S1_density": (_pb_density, "scalar", False), "S5_abs": (_pb_occ, "scalar", False),
              "S6_tv": (_pb_classvec, "vector", False), "S3_loggap": (_pb_logmean, "scalar", True),
              "S7_abs": (_pb_div, "scalar", False)}


def _reduce_diff(cand_g, ref_g, kind):
    if kind == "vector":
        return float(max(0.5 * float(np.sum(np.abs(np.mean(np.asarray(c), 0) - np.mean(np.asarray(r), 0))))
                         for c, r in zip(cand_g, ref_g)))
    return float(max(abs(float(np.mean(c)) - float(np.mean(r))) for c, r in zip(cand_g, ref_g)))


# --- issuance artifact: schema + build + identity + validate ---------------------------------------
MAP_SCHEMA = {"check": str, "bins_id": str, "n_original_bins": int, "groups": (list, type(None)),
              "floor": int, "status": str, "profile": str, "regime": str, "namespace": str,
              "seed": (int, type(None)), "N": (int, type(None)), "estimator_identity": str,
              "original_bin_identity": str, "denominator_policy": str}
_VALID_STATUS = {"OK", "REFUSED_reference_coarsening"}


def build_frozen_map(reference_sample, check, *, profile, regime, namespace="v3-map-dev", seed=None, N=None,
                     floor=FLOOR):
    """Freeze the reference-owned grouping of ORIGINAL bin indices via the v2 merge on the reference per-bin
    sequence counts, and bind the full issuance trust root (Pi rev-6 #3). `floor` defaults to the registered
    FLOOR=500; a LOWER floor is only for LABELLED dev-scale demonstrations (recorded in the artifact)."""
    builder, _, _ = CHECK_SPEC[check]
    pb, _ = builder(reference_sample)
    counts = np.asarray([len(v) for v in pb])
    groups = coarsen_reference(counts, floor=floor)
    art = {"check": check, "bins_id": ORIGINAL_BINS[check], "n_original_bins": len(_BINS[ORIGINAL_BINS[check]]),
           "groups": ([list(g) for g in groups] if groups is not None else None), "floor": int(floor),
           "status": "OK" if groups is not None else "REFUSED_reference_coarsening",
           "profile": profile, "regime": regime, "namespace": namespace,
           "seed": (int(seed) if seed is not None else None), "N": (int(N) if N is not None else None),
           "estimator_identity": ESTIMATOR_ID[check], "original_bin_identity": ORIGINAL_BINS[check],
           "denominator_policy": DENOM_POLICY[check]}
    return art


def validate_map_artifact(art):
    """Refuse a malformed / incomplete map artifact BEFORE any application (fail-closed)."""
    if not isinstance(art, dict):
        raise RefusalError("map artifact is not a dict")
    keys, req = set(art), set(MAP_SCHEMA)
    if keys - req:
        raise RefusalError(f"map artifact has unknown fields {sorted(keys - req)}")
    if req - keys:
        raise RefusalError(f"map artifact missing fields {sorted(req - keys)}")
    for f, t in MAP_SCHEMA.items():
        if not isinstance(art[f], t):
            raise RefusalError(f"map field {f} type {type(art[f]).__name__} != {t}")
    if art["status"] not in _VALID_STATUS:
        raise RefusalError(f"map status {art['status']!r} invalid")
    if art["check"] not in MAP_CARRYING:
        raise RefusalError(f"map check {art['check']!r} not map-carrying")
    if art["bins_id"] != ORIGINAL_BINS[art["check"]]:
        raise RefusalError(f"map bins_id {art['bins_id']} != registered {ORIGINAL_BINS[art['check']]}")


def map_identity(art):
    """Trust-root hash over ALL issuance fields (Pi rev-6 #3)."""
    validate_map_artifact(art)
    return canonical_hash({k: art[k] for k in sorted(MAP_SCHEMA)})


def apply_frozen_map(cand, ref, check, art, *, expect_profile, expect_regime):
    """Discrepancy d under the FROZEN grouping. REFUSES if the artifact is malformed or its identity
    (check/profile/regime) does not match the application context (Pi rev-6 #3). Returns d (float) or None (NE)."""
    validate_map_artifact(art)
    if art["check"] != check:
        raise RefusalError(f"map check {art['check']} != applied check {check}")
    if art["profile"] != expect_profile:
        raise RefusalError(f"map profile {art['profile']} != context {expect_profile}")
    if art["regime"] != expect_regime:
        raise RefusalError(f"map regime {art['regime']} != context {expect_regime}")
    if art["groups"] is None:
        return None
    builder, kind, use_pairs = CHECK_SPEC[check]
    groups = art["groups"]; floor = art["floor"]
    cpb, cpairs = builder(cand); rpb, rpairs = builder(ref)
    cand_g, ref_g = _grouped(cpb, groups), _grouped(rpb, groups)
    if any(len(c) < floor for c in cand_g) or any(len(r) < floor for r in ref_g):
        return None
    if use_pairs:
        ex_c = [int(sum(cpairs[i] for i in g)) for g in groups]
        ex_r = [int(sum(rpairs[i] for i in g)) for g in groups]
        if any(v < floor for v in ex_c) or any(v < floor for v in ex_r):
            return None
    return _reduce_diff(cand_g, ref_g, kind)


# --- self-tests -----------------------------------------------------------------------------------
def _v2_value(check, cand, ref):
    return {"S1_density": s1, "S5_abs": s5, "S6_tv": s6, "S3_loggap": s3, "S7_abs": s7}[check](cand, ref)[check].value


def _dev_sample(profile, n, tag):
    from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
    from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
    sk = "SCID" if "scid" in profile else "MIMIC"
    seed = int.from_bytes(hashlib.sha256(f"v3-map-dev|{profile}|{tag}".encode()).digest()[:6], "big")
    return sample_fixture(sk, PROFILES[profile], n, seed=seed)


def selftest():
    errs = []
    N = 6000                                                  # ADEQUATE support so every map is OK (not None==None)
    prof, reg = "mimic_scale_control", "full"
    ref = _dev_sample(prof, N, "ref"); cand = _dev_sample(prof, N, "cand")
    cand2 = _dev_sample("scid_scale_control", N, "cand2")

    all_ok = True
    for check in MAP_CARRYING:
        art = build_frozen_map(ref, check, profile=prof, regime=reg, seed=1, N=N)
        if art["status"] != "OK":
            all_ok = False; errs.append(f"{check}: map REFUSED at adequate support N={N} (status {art['status']})")
            continue
        got = apply_frozen_map(cand, ref, check, art, expect_profile=prof, expect_regime=reg)
        want = _v2_value(check, cand, ref)
        if got is None or want is None:
            errs.append(f"{check}: NE at adequate support (got={got} want={want}) — not a real reproduction test")
        elif abs(got - want) > 1e-12:
            errs.append(f"{check}: value mismatch got={got} want={want}")
    if not all_ok:
        errs.append("adequate-support test did NOT exercise every map as OK (weak None==None case)")

    # anti-masking: map identity depends ONLY on the reference-design sample, never the candidate
    for check in MAP_CARRYING:
        m1 = map_identity(build_frozen_map(ref, check, profile=prof, regime=reg, seed=1, N=N))
        m2 = map_identity(build_frozen_map(ref, check, profile=prof, regime=reg, seed=1, N=N))
        if m1 != m2:
            errs.append(f"{check}: map identity not deterministic")

    # S6 is LENGTH_BINS
    if build_frozen_map(ref, "S6_tv", profile=prof, regime=reg)["bins_id"] != "LENGTH_BINS":
        errs.append("S6_tv bins_id != LENGTH_BINS")

    # floor enforcement: tiny reference => NE
    tiny = _dev_sample(prof, 50, "tiny")
    art_tiny = build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg)
    if apply_frozen_map(tiny, tiny, "S3_loggap", art_tiny, expect_profile=prof, expect_regime=reg) is not None:
        errs.append("floor not enforced (tiny sample should be NE)")

    # CROSS-IDENTITY refusal: a MIMIC/full map applied under SCID / different regime / different check must REFUSE
    art = build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=1, N=N)
    def _refused(**kw):
        try:
            apply_frozen_map(cand, ref, kw.get("check", "S3_loggap"), kw.get("art", art),
                             expect_profile=kw.get("expect_profile", prof), expect_regime=kw.get("expect_regime", reg))
            return False
        except RefusalError:
            return True
    for label, kw in (("wrong_profile", {"expect_profile": "scid_scale_control"}),
                      ("wrong_regime", {"expect_regime": "bounded"}),
                      ("wrong_check", {"check": "S7_abs"})):
        if not _refused(**kw):
            errs.append(f"cross-identity NOT refused: {label}")

    # MALFORMED-artifact refusal
    def _mrefused(bad):
        try:
            validate_map_artifact(bad); return False
        except RefusalError:
            return True
    missing = {k: v for k, v in art.items() if k != "seed"}
    extra = {**art, "sneaky": 1}
    mistyped = {**art, "floor": "500"}
    badbins = {**art, "bins_id": "LENGTH_BINS"}          # wrong bins_id for a CLUSTER_BINS check
    for label, bad in (("missing", missing), ("extra", extra), ("mistyped", mistyped), ("wrong_bins", badbins)):
        if not _mrefused(bad):
            errs.append(f"malformed-artifact NOT refused: {label}")

    return errs, {c: build_frozen_map(ref, c, profile=prof, regime=reg, seed=1, N=N) for c in MAP_CARRYING}


def main():
    errs, maps = selftest()
    out = {"map_carrying": list(MAP_CARRYING),
           "original_bins": {k: ORIGINAL_BINS[k] for k in MAP_CARRYING},
           "frozen_maps": {k: {"status": m["status"], "groups": m["groups"], "identity": map_identity(m)}
                           for k, m in maps.items()},
           "issuance_fields": sorted(MAP_SCHEMA),
           "selftests_pass": not errs, "selftest_errors": errs,
           "authorization": "dev-only map builder; the reserved one-time map-design namespace draw is BLOCKED."}
    print(json.dumps(out, indent=2, default=str))
    print("\nMAP_SET_HASH:", canonical_hash(out["frozen_maps"]))
    assert not errs, f"map builder self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
