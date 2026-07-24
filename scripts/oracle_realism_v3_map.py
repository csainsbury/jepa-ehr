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
def _denom_policy(check, floor):
    """Floor-CONSISTENT denominator identity (Pi rev-7 #6: a dev-floor map must NOT record the registered floor)."""
    if check == "S3_loggap":
        return f"seq_floor={floor} + adjacent_pair_floor={floor} per coarsened group, both arms"
    return f"seq_floor={floor} per coarsened group, both arms"


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


def build_frozen_map(reference_sample, check, *, profile, regime, seed, N, namespace="v3-map-dev", floor=FLOOR):
    """Freeze the reference-owned grouping of ORIGINAL bin indices via the v2 merge on the reference per-bin
    sequence counts, and bind the full issuance trust root (Pi rev-6 #3). `seed` and `N` are MANDATORY (Pi rev-9
    RC3): a builder must never emit an unvalidated artifact with a null provenance field. `floor` defaults to the
    registered FLOOR=500; a LOWER floor is only for LABELLED dev-scale demonstrations (recorded in the artifact)."""
    if check not in CHECK_SPEC:
        raise RefusalError(f"map builder unknown check {check!r}")
    # RC3: validate every provenance field BEFORE integer coercion (so True is NOT silently coerced to 1)
    for nm, v in (("seed", seed), ("N", N), ("floor", floor)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise RefusalError(f"map builder {nm} {v!r} must be a positive non-bool integer")
    for nm, v in (("profile", profile), ("regime", regime), ("namespace", namespace)):
        if not (isinstance(v, str) and v.strip()):
            raise RefusalError(f"map builder {nm} must be a nonempty string")
    if len(reference_sample) != N:                                 # provenance: the sample IS the declared N (Pi #4)
        raise RefusalError(f"reference_sample length {len(reference_sample)} != declared N {N}")
    builder, _, _ = CHECK_SPEC[check]
    pb, _ = builder(reference_sample)
    counts = np.asarray([len(v) for v in pb])
    groups = coarsen_reference(counts, floor=floor)
    art = {"check": check, "bins_id": ORIGINAL_BINS[check], "n_original_bins": len(_BINS[ORIGINAL_BINS[check]]),
           "groups": ([list(g) for g in groups] if groups is not None else None), "floor": int(floor),
           "status": "OK" if groups is not None else "REFUSED_reference_coarsening",
           "profile": profile, "regime": regime, "namespace": namespace,
           "seed": int(seed), "N": int(N),
           "estimator_identity": ESTIMATOR_ID[check], "original_bin_identity": ORIGINAL_BINS[check],
           "denominator_policy": _denom_policy(check, int(floor))}
    validate_map_artifact(art)                                     # RC3: never return an unvalidated artifact
    return art


def validate_map_artifact(art):
    """Refuse a malformed / incomplete / internally-inconsistent map artifact BEFORE any application
    (fail-closed; Pi rev-7 #6)."""
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
    chk = art["check"]
    # exact registered estimator / bin / n_original_bins / denominator identities
    if art["bins_id"] != ORIGINAL_BINS[chk]:
        raise RefusalError(f"map bins_id {art['bins_id']} != registered {ORIGINAL_BINS[chk]}")
    nbins = len(_BINS[ORIGINAL_BINS[chk]])
    if art["n_original_bins"] != nbins:
        raise RefusalError(f"map n_original_bins {art['n_original_bins']} != {nbins}")
    if art["estimator_identity"] != ESTIMATOR_ID[chk]:
        raise RefusalError(f"map estimator_identity mismatch for {chk}")
    if art["original_bin_identity"] != ORIGINAL_BINS[chk]:
        raise RefusalError(f"map original_bin_identity mismatch for {chk}")
    if art["denominator_policy"] != _denom_policy(chk, art["floor"]):   # floor-consistent (no floor-500 on dev-60)
        raise RefusalError(f"map denominator_policy inconsistent with floor {art['floor']} for {chk}")
    # N / seed / floor MANDATORY positive non-bool integers for every issued artifact (Pi rev-7 #4)
    for f in ("N", "seed", "floor"):
        v = art[f]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise RefusalError(f"map {f} {v!r} must be a positive non-bool integer")
    # nonempty profile / regime / namespace
    for f in ("profile", "regime", "namespace"):
        if not (isinstance(art[f], str) and art[f].strip()):
            raise RefusalError(f"map {f} must be a nonempty string")
    # status <-> groups: an OK map's groups are ORDERED, CONTIGUOUS, nonempty, disjoint, covering 0..n-1 exactly
    # (a frozen v2 merge yields ordered contiguous runs — an arbitrary mathematical partition is NOT valid; Pi #4)
    if art["status"] == "OK":
        groups = art["groups"]
        if not isinstance(groups, list) or not groups:
            raise RefusalError("map status OK but groups empty/None")
        for g in groups:
            if not isinstance(g, list) or not g or any(isinstance(i, bool) or not isinstance(i, int) for i in g):
                raise RefusalError(f"map group {g!r} must be a nonempty list of int bin indices")
        flat = [i for g in groups for i in g]
        if flat != list(range(nbins)):                          # ORDERED CONTIGUOUS (not merely sorted)
            raise RefusalError(f"map groups not an ordered contiguous partition of 0..{nbins - 1}: {flat}")
    elif art["groups"] is not None:
        raise RefusalError("map status REFUSED but groups non-null")


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
    if build_frozen_map(ref, "S6_tv", profile=prof, regime=reg, seed=1, N=N)["bins_id"] != "LENGTH_BINS":
        errs.append("S6_tv bins_id != LENGTH_BINS")

    # floor enforcement: tiny reference => NE (artifact carries mandatory seed/N)
    tiny = _dev_sample(prof, 50, "tiny")
    art_tiny = build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=7, N=N)
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
    # Pi rev-7 #4 reproduced-acceptance cases must now REFUSE
    ncb = len(_BINS[ORIGINAL_BINS["S3_loggap"]])
    noncontig = {**art, "groups": [[0, 2], [1, 3]] + [[i] for i in range(4, ncb)]}   # disjoint+complete but NOT ordered
    emptyfirst = {**art, "groups": [[]] + [[i] for i in range(ncb)]}                  # empty group prepended
    malformed = {
        "missing": {k: v for k, v in art.items() if k != "seed"},
        "extra": {**art, "sneaky": 1},
        "mistyped": {**art, "floor": "500"},
        "wrong_bins": {**art, "bins_id": "LENGTH_BINS"},
        "seed_none": {**art, "seed": None},
        "N_none": {**art, "N": None},
        "blank_profile": {**art, "profile": "  "},
        "non_contiguous_groups": noncontig,
        "empty_group": emptyfirst,
    }
    for label, bad in malformed.items():
        if not _mrefused(bad):
            errs.append(f"malformed-artifact NOT refused: {label}")

    # RC3 (Pi rev-9): the BUILDER refuses bad provenance BEFORE integer coercion and never returns an unvalidated
    # artifact (True must not be coerced to 1; N must equal the sample length; seed/N mandatory positive non-bool).
    builder_bad = {
        "bool_seed": lambda: build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=True, N=N),
        "bool_N": lambda: build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=1, N=True),
        "nonpos_seed": lambda: build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=0, N=N),
        "N_mismatch": lambda: build_frozen_map(ref, "S3_loggap", profile=prof, regime=reg, seed=1, N=N + 1),
        "blank_profile": lambda: build_frozen_map(ref, "S3_loggap", profile="  ", regime=reg, seed=1, N=N),
    }
    for label, fn in builder_bad.items():
        try:
            fn(); errs.append(f"builder did NOT refuse {label}")
        except RefusalError:
            pass

    return errs, {c: build_frozen_map(ref, c, profile=prof, regime=reg, seed=1, N=N) for c in MAP_CARRYING}


def main():
    errs, maps = selftest()
    out = {"map_carrying": list(MAP_CARRYING),
           "original_bins": {k: ORIGINAL_BINS[k] for k in MAP_CARRYING},
           "frozen_maps": {k: {"status": m["status"], "groups": m["groups"], "identity": map_identity(m)}
                           for k, m in maps.items()},
           "issuance_fields": sorted(MAP_SCHEMA),
           "map_set_hash_label": "DEVELOPMENT self-test map-set hash (mimic_scale_control@full, dev fixtures) — "
                                 "NOT the reserved registered map-set identity (that draw is BLOCKED) (Pi rev-9 RC3)",
           "selftests_pass": not errs, "selftest_errors": errs,
           "authorization": "dev-only map builder; the reserved one-time map-design namespace draw is BLOCKED."}
    print(json.dumps(out, indent=2, default=str))
    print("\nMAP_SET_HASH (dev self-test; NOT reserved):", canonical_hash(out["frozen_maps"]))
    assert not errs, f"map builder self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
