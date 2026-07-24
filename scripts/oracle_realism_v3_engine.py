#!/usr/bin/env python3
"""Oracle realism v3 — registry-OWNED heterogeneous estimator dispatcher + hardened group gate (Pi rev-6 #4).

The rev-6 gate took ONE caller-supplied `statfn` and assumed flat numeric values — it could not execute a real
registered group whose cells mix pooled-tau components, owner-indexed tied-KS, and frozen-map per-bin structures.
Here each cell is bound to a REGISTRY-OWNED estimator (precompute + per-permutation recompute + identity) selected
by its registered `check`; callers provide DATA (per-experiment candidate/reference sequences) and cell specs
(check + map artifact), never a statistic trust root. The gate:

  * validates through ONE trusted path BEFORE any statistic: exact registered cell ids + order (no missing/extra),
    MANDATORY per-map identity match + floor-policy, alpha_group in (0,1), positive-integer B, present seed,
    per-experiment EXECUTABLE fixture/coupling RNG identity (a hash of the seed-derivation, not a descriptive
    string);
  * constructs the product/stratified assignments through that same trusted path (never caller-injected);
  * uses the frozen NOT_EVALUABLE policy (observed NE -> group NE; permutation NE -> maximally extreme +inf, no
    zero-fill), dispatching each cell's recompute via the registry;
  * uses an IID-with-replacement Monte-Carlo permutation scheme in which duplicate assignments are VALID and bound
    (not spuriously prohibited).

Development-only. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_engine.py
"""
from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    _positive_gaps_and_prev_size, _bin_index, CLUSTER_BINS, LENGTH_BINS, _s4_contrast, C,
)
from scripts.oracle_realism_v3_randomization import cell_upper_p, RefusalError
from scripts.oracle_realism_v3_phase0_pilot import _seq_components
from scripts.oracle_realism_v3_map import validate_map_artifact, map_identity, FLOOR
import scripts.oracle_realism_v3_registry as REG

MAP_CHECKS = {"S3_loggap", "S5_abs", "S6_tv", "S7_abs", "S1_density"}   # map-carrying checks (engine-wired subset)

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"
_MAXSZ = 4096
_CLUSTER_LUT = np.array([(_bin_index(s, CLUSTER_BINS) if _bin_index(s, CLUSTER_BINS) is not None else -1)
                         for s in range(_MAXSZ)], int)


# --- registry-owned estimators: precompute(pool) -> pre ; recompute(pre, mask, groups) -> float|None ----
def _tau_pre(pool):
    return np.array([_seq_components(r) for r in pool])


def _tau_re(pre, mask, *, groups=None, floor=FLOOR):
    def t(Cm):
        s = Cm.sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
        return None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)
    a, b = t(pre[mask]), t(pre[~mask])
    return None if (a is None or b is None) else abs(a - b)


def _dt0_pre(pool):
    nz = np.array([max(0, r.L_total - r.K) for r in pool], float)
    na = np.array([max(0, r.L_total - 1) for r in pool], float)
    return np.stack([nz, na], 1)


def _dt0_re(pre, mask, *, groups=None, floor=FLOOR):
    a, b = pre[mask].sum(0), pre[~mask].sum(0)
    return None if (a[1] == 0 or b[1] == 0) else abs(a[0] / a[1] - b[0] / b[1])


def _gap_pre(pool):
    gaps, owner = [], []
    for i, r in enumerate(pool):
        y, _ = _positive_gaps_and_prev_size(r)
        for v in y:
            if v > 0:
                gaps.append(round(float(v), 8)); owner.append(i)   # ROUND to registered 8-dp support (Pi #4)
    gaps = np.asarray(gaps); owner = np.asarray(owner, int)
    uniq, inv = np.unique(gaps, return_inverse=True)               # unique SUPPORT values (post-rounding)
    return {"owner": owner, "inv": inv, "nu": len(uniq)}


def _gap_re(pre, mask, *, groups=None, floor=FLOOR):
    inA = mask[pre["owner"]]; nA = int(inA.sum()); nB = len(inA) - nA
    if nA == 0 or nB == 0:
        return None
    ca = np.bincount(pre["inv"][inA], minlength=pre["nu"]); cb = np.bincount(pre["inv"][~inA], minlength=pre["nu"])
    return float(np.max(np.abs(np.cumsum(ca) / nA - np.cumsum(cb) / nB)))


def _loggap_pre(pool):
    nb = len(CLUSTER_BINS); n = len(pool)
    sm = [np.full(n, np.nan) for _ in range(nb)]; sp = [np.zeros(n) for _ in range(nb)]
    for i, r in enumerate(pool):
        g, ps = _positive_gaps_and_prev_size(r)
        if g.shape[0] == 0:
            continue
        lg = np.log(g); bins = _CLUSTER_LUT[np.clip(ps.astype(int), 0, _MAXSZ - 1)]
        for b in range(nb):
            m = bins == b
            if m.any():
                sm[b][i] = float(lg[m].mean()); sp[b][i] = int(m.sum())
    return {"sm": sm, "sp": sp}


def _loggap_re(pre, mask, *, groups=None, floor=FLOOR):
    if groups is None:
        return None
    d = 0.0
    for grp in groups:
        cv, rv, cp, rp = [], [], 0.0, 0.0
        for b in grp:
            sm, sp = pre["sm"][b], pre["sp"][b]; pres = ~np.isnan(sm)
            cv.append(sm[pres & mask]); rv.append(sm[pres & ~mask])
            cp += sp[pres & mask].sum(); rp += sp[pres & ~mask].sum()
        cvv, rvv = np.concatenate(cv), np.concatenate(rv)
        if cvv.size < floor or rvv.size < floor or cp < floor or rp < floor:
            return None
        d = max(d, abs(cvv.mean() - rvv.mean()))
    return d


# --- class-mark group estimators (for mark_burst_tie / cluster_size_mark_diversity sensitivity; Pi #4) -----
def _s4_pre(pool):
    out = np.full((len(pool), 3), np.nan)          # [contrast, same_pairs, adj_pairs]
    for i, r in enumerate(pool):
        x = _s4_contrast(r)
        if x is not None:
            out[i] = [x[0], x[1], x[2]]
    return out


def _s4_re(pre, mask, *, groups=None, floor=FLOOR):
    pres = ~np.isnan(pre[:, 0]); ca = pres & mask; cr = pres & ~mask
    if int(ca.sum()) < floor or int(cr.sum()) < floor:
        return None
    if min(pre[ca, 1].sum(), pre[cr, 1].sum(), pre[ca, 2].sum(), pre[cr, 2].sum()) < floor:
        return None
    return abs(float(pre[ca, 0].mean()) - float(pre[cr, 0].mean()))


def _classtv_pre(pool):
    return np.array([np.bincount(r.class_ids, minlength=C)[:C] for r in pool], float)


def _classtv_re(pre, mask, *, groups=None, floor=FLOOR):
    a, b = pre[mask].sum(0), pre[~mask].sum(0)
    if a.sum() == 0 or b.sum() == 0:
        return None
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def _occ_pre(pool):
    return np.array([len(np.unique(r.class_ids)) / C for r in pool], float)


def _occ_re(pre, mask, *, groups=None, floor=FLOOR):
    a, b = pre[mask], pre[~mask]
    return None if (a.size < floor or b.size < floor) else abs(float(a.mean()) - float(b.mean()))


def _lenbin_scalar_pre(pool, val):
    nb = len(LENGTH_BINS); n = len(pool); sm = [np.full(n, np.nan) for _ in range(nb)]
    for i, r in enumerate(pool):
        b = _bin_index(r.L_total, LENGTH_BINS)
        if b is not None:
            sm[b][i] = val(r)
    return {"sm": sm}


def _s5_pre(pool):
    return _lenbin_scalar_pre(pool, lambda r: len(np.unique(r.class_ids)) / C)


def _s6_pre(pool):
    nb = len(LENGTH_BINS); n = len(pool); vec = [np.full((n, C), np.nan) for _ in range(nb)]
    for i, r in enumerate(pool):
        b = _bin_index(r.L_total, LENGTH_BINS)
        if b is not None:
            vec[b][i] = np.bincount(r.class_ids, minlength=C)[:C] / r.L_total
    return {"vec": vec}


def _s7_pre(pool):
    nb = len(CLUSTER_BINS); n = len(pool)
    sm = [np.full(n, np.nan) for _ in range(nb)]; cc = [np.zeros(n) for _ in range(nb)]
    for i, r in enumerate(pool):
        by = [[] for _ in range(nb)]
        for c in range(r.K):
            cls = r.class_ids[r.cluster_ids == c]
            b = _bin_index(int(cls.shape[0]), CLUSTER_BINS)
            if b is not None:
                by[b].append(len(np.unique(cls)) / C)
        for b in range(nb):
            if by[b]:
                sm[b][i] = float(np.mean(by[b])); cc[b][i] = len(by[b])
    return {"sm": sm, "cc": cc}


def _map_re_scalar(pre, mask, *, groups=None, extra_key=None, floor=FLOOR):
    if groups is None:
        return None
    sm = pre["sm"]; extra = pre.get(extra_key) if extra_key else None
    d = 0.0
    for grp in groups:
        cv, rv, ce, re_ = [], [], 0.0, 0.0
        for b in grp:
            s = sm[b]; pres = ~np.isnan(s)
            cv.append(s[pres & mask]); rv.append(s[pres & ~mask])
            if extra is not None:
                ce += extra[b][pres & mask].sum(); re_ += extra[b][pres & ~mask].sum()
        cvv, rvv = np.concatenate(cv), np.concatenate(rv)
        if cvv.size < floor or rvv.size < floor or (extra is not None and (ce < floor or re_ < floor)):
            return None
        d = max(d, abs(cvv.mean() - rvv.mean()))
    return d


def _map_re_vector(pre, mask, *, groups=None, floor=FLOOR):
    if groups is None:
        return None
    vec = pre["vec"]; d = 0.0
    for grp in groups:
        cvs, rvs = [], []
        for b in grp:
            v = vec[b]; pres = ~np.isnan(v[:, 0])
            cvs.append(v[pres & mask]); rvs.append(v[pres & ~mask])
        cv, rv = np.concatenate(cvs), np.concatenate(rvs)
        if cv.shape[0] < floor or rv.shape[0] < floor:
            return None
        d = max(d, 0.5 * float(np.abs(cv.mean(0) - rv.mean(0)).sum()))
    return d


ESTIMATORS = {
    "S3_tau": {"precompute": _tau_pre, "recompute": _tau_re, "map_carrying": False,
               "identity": "v3.pooled_tau_b.phasespanning_cap6::T_pool(cap6,quantile_spaced,tie_corrected)"},
    "delta_t_zero_abs": {"precompute": _dt0_pre, "recompute": _dt0_re, "map_carrying": False,
                         "identity": "v2.abs(P(delta_t=0))"},
    "positive_gap_ks": {"precompute": _gap_pre, "recompute": _gap_re, "map_carrying": False,
                        "identity": "v2.ks(positive_gap_ecdf@unique_support_8dp)"},
    "S3_loggap": {"precompute": _loggap_pre, "recompute": _loggap_re, "map_carrying": True,
                  "identity": "v2.cond_maxbin.maxabs(mean_log_positive_gap)@CLUSTER_BINS[ref_coarsen]"},
    "S4_abs": {"precompute": _s4_pre, "recompute": _s4_re, "map_carrying": False,
               "identity": "v2.abs(P(same|same_cluster)-P(same|adjacent))"},
    "class_tv": {"precompute": _classtv_pre, "recompute": _classtv_re, "map_carrying": False,
                 "identity": "v2.tv(class_prior)"},
    "occupancy_abs": {"precompute": _occ_pre, "recompute": _occ_re, "map_carrying": False,
                      "identity": "v2.abs(mean_occupancy)"},
    # RC1 (Pi rev-9): uniform keyword-only wrappers — `floor` is routed to the floor gate, never into `extra_key`.
    "S5_abs": {"precompute": _s5_pre,
               "recompute": (lambda pre, m, *, groups=None, floor=FLOOR: _map_re_scalar(pre, m, groups=groups, floor=floor)),
               "map_carrying": True, "identity": "v2.cond_maxbin.mean(occupancy)@LENGTH_BINS[ref_coarsen]"},
    "S6_tv": {"precompute": _s6_pre,
              "recompute": (lambda pre, m, *, groups=None, floor=FLOOR: _map_re_vector(pre, m, groups=groups, floor=floor)),
              "map_carrying": True, "identity": "v2.maxabs_tv(class_prior)@LENGTH_BINS[ref_coarsen]"},
    "S7_abs": {"precompute": _s7_pre,
               "recompute": (lambda pre, m, *, groups=None, floor=FLOOR: _map_re_scalar(pre, m, groups=groups, extra_key="cc", floor=floor)),
               "map_carrying": True, "identity": "v2.cond_maxbin.mean(distinct_class_frac)@CLUSTER_BINS[ref_coarsen]"},
}

# Estimator identities — SEMANTIC vs CODE are DISTINCT (Pi rev-11 Correction 2).
# SEMANTIC = calling convention + declared estimand identity STRINGS. It catches a protocol change or a re-declared
# estimand, but NOT a silent change to an estimator's implementation (the strings can stay fixed).
ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY = canonical_hash({
    "protocol": "recompute(pre, mask, *, groups, floor) keyword-only",
    "estimators": {k: v["identity"] for k, v in ESTIMATORS.items()}})

# CODE = a deterministic hash of the actual estimator IMPLEMENTATION source (precompute/recompute helpers + map
# reducers) plus the numpy version. A change to any estimator's code changes this hash even if every identity string
# is unchanged — this is what actually catches an altered implementation. Both identities are bound together.
_ESTIMATOR_IMPL_FNS = (_tau_pre, _tau_re, _dt0_pre, _dt0_re, _gap_pre, _gap_re, _loggap_pre, _loggap_re,
                       _s4_pre, _s4_re, _classtv_pre, _classtv_re, _occ_pre, _occ_re,
                       _lenbin_scalar_pre, _s5_pre, _s6_pre, _s7_pre, _map_re_scalar, _map_re_vector)
ESTIMATOR_CODE_IDENTITY = canonical_hash({
    "impl_src": {fn.__name__: inspect.getsource(fn) for fn in _ESTIMATOR_IMPL_FNS},
    "numpy": np.__version__})


# --- executable per-experiment RNG identity (a hash of the seed-derivation, not a string; Pi #4/#7) ------
def rng_identity(source_profile, replicate_seed, coupled_component):
    fixture = canonical_hash(["fixture", source_profile, replicate_seed, "candidate|reference"])
    coupling = (canonical_hash(["coupling", source_profile, coupled_component, replicate_seed, "candidate_D|reference"])
                if coupled_component is not None else None)
    return canonical_hash({"fixture_law": fixture, "coupling_law": coupling, "role_symmetric": True})


# --- the hardened gate --------------------------------------------------------------------------------
def _canonical_mask(strata):
    return np.concatenate([np.array([True] * nA + [False] * nB) for nA, nB in strata])


def _perm_mask(rng, strata):
    parts = []
    for nA, nB in strata:
        idx = rng.permutation(nA + nB); m = np.zeros(nA + nB, bool); m[idx[:nA]] = True; parts.append(m)
    return np.concatenate(parts)


def _validate(spec):
    reg = spec["registered"]
    a = reg.get("alpha_group")
    if not isinstance(a, float) or not (0.0 < a < 1.0):
        raise RefusalError(f"alpha_group {a!r} not in (0,1)")
    B = spec.get("B")
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise RefusalError(f"B {B!r} not a positive integer")
    if spec.get("seed") is None:
        raise RefusalError("missing seed")
    got_ids = [c["cell_id"] for c in spec["cells"]]
    if got_ids != list(reg["cell_ids"]):                            # exact ids AND order, no missing/extra
        raise RefusalError(f"cell ids/order mismatch: {got_ids} != {reg['cell_ids']}")
    if reg.get("floor_policy") != reg.get("expected_floor_policy", reg.get("floor_policy")):
        raise RefusalError("floor-policy mismatch")
    for c in spec["cells"]:
        chk = c["check"]
        if chk not in ESTIMATORS:
            raise RefusalError(f"unknown/unregistered estimator {chk}")
        if c["exp"] not in spec["experiments"]:
            raise RefusalError(f"cell {c['cell_id']} references unknown experiment {c['exp']}")
        if ESTIMATORS[chk]["map_carrying"]:
            art = c.get("map_art")
            if art is None:                                         # MANDATORY (not optional self-compare) (Pi #4)
                raise RefusalError(f"map-carrying cell {c['cell_id']} missing mandatory map_art")
            validate_map_artifact(art)
            if map_identity(art) != reg["map_hashes"].get(c["cell_id"]):
                raise RefusalError(f"map identity mismatch for {c['cell_id']}")
    # executable per-experiment RNG identity must match the registered value
    for e, ident in reg["rng_identities"].items():
        meta = spec["experiments"][e]
        if rng_identity(meta["source"], meta["replicate_seed"], meta["coupled_component"]) != ident:
            raise RefusalError(f"RNG identity mismatch for experiment {e}")


def _gate_group(spec):
    """PRIVATE / test-only low-level kernel (Pi rev-9 RC5): production callers use `gate_group_dev` (development)
    or `gate_group_registered` (registered, currently a blocked stub). It accepts a fully-formed trusted spec and
    is NOT a public entry point.
    spec: {cells:[{cell_id,exp,check,pre,[map_art]}], experiments:{e:{strata,source,replicate_seed,coupled_component}},
    registered:{cell_ids,map_hashes,rng_identities,alpha_group,floor_policy}, B, seed}. Fail-closed."""
    _validate(spec)                                                 # BEFORE any statistic
    reg = spec["registered"]; B = spec["B"]; floor = spec.get("floor", FLOOR)   # floor is a PARAM, not a global
    rng = np.random.default_rng(spec["seed"])
    exps = spec["experiments"]
    # ONE trusted assignment path (IID-with-replacement MC; duplicates VALID + bound)
    masks = {e: [_canonical_mask(exps[e]["strata"])] + [_perm_mask(rng, exps[e]["strata"]) for _ in range(B)]
             for e in exps}
    def d_of(c, m):
        est = ESTIMATORS[c["check"]]
        groups = c["map_art"]["groups"] if est["map_carrying"] else None
        return est["recompute"](c["pre"], m, groups=groups, floor=floor)
    for c in spec["cells"]:                                          # observed NE (or non-finite) -> group NE
        d0 = d_of(c, masks[c["exp"]][0])
        if d0 is None or not np.isfinite(d0):
            return {"verdict": NOT_EVALUABLE, "p_g": None, "reason": f"observed NE at {c['cell_id']}"}
    E = []
    for c in spec["cells"]:
        est = ESTIMATORS[c["check"]]; groups = c["map_art"]["groups"] if est["map_carrying"] else None
        ej = np.empty(B + 1)
        for j, m in enumerate(masks[c["exp"]]):
            d = est["recompute"](c["pre"], m, groups=groups, floor=floor)
            # NaN/Inf discrepancy is a support/precompute failure -> maximally extreme NE, NEVER zero-fill (Pi #4)
            ej[j] = np.inf if (d is None or not np.isfinite(d)) else max(0.0, d - c["delta"])
        E.append(ej)
    P = np.stack([cell_upper_p(e) for e in E], 0); S = P.min(0)
    p_g = float((S <= S[0]).sum() / len(S))
    return {"verdict": PASS if p_g > reg["alpha_group"] else FAIL, "p_g": p_g,
            "argmin_cell": spec["cells"][int(np.argmin(P[:, 0]))]["cell_id"]}


# ======================================================================================================
# DEVELOPMENT dispatcher boundary (Pi rev-7 #4; framing corrected rev-9 RC2): caller passes ONLY a group id
# + raw experiment pools + seed/B (+ dev map artifacts). The engine loads cell order / check / Delta / strata /
# map-carrying from the canonical registry, computes precompute ITSELF from the raw pools, and refuses / NEs any
# non-finite precompute or discrepancy. No caller-supplied check / Delta / registered / precompute is trusted.
# This is a DEV dispatcher — NOT a registered trusted execution boundary; registered mode is a blocked stub below.
# ======================================================================================================
def _build_canonical_groups():
    sd = REG.build_sd_cells(apply_uncalibratable_exemption=True)
    by_id = {c["cell_id"]: c for c in sd}
    groups = REG.build_groups(sd)
    out = {}
    for gid in ("G_full_burst_timing", "G_full_class_mark"):        # engine-wired full-support groups
        cells, exps = [], {}
        for cid in list(groups[gid]["cells"]):
            c = by_id[cid]; chk = c["statistic"]
            if chk not in ESTIMATORS:
                raise RefusalError(f"canonical group {gid} has unwired check {chk}")
            cells.append({"cell_id": cid, "exp": c["experiment_id"], "check": chk, "delta": float(c["delta"]),
                          "map_carrying": ESTIMATORS[chk]["map_carrying"]})
            strata = c["exchangeability_strata"]
            exps.setdefault(c["experiment_id"], {
                "source": c["source"], "condition": c["condition"], "coupled_component": c["coupled_component"],
                "stratum_ids": [s["stratum_id"] for s in strata],
                "registered_quota": [(s["n_candidate"], s["n_reference"]) for s in strata]})
        out[gid] = {"group_id": gid, "cells": cells, "experiments": exps}
    return out


CANONICAL_GROUPS = _build_canonical_groups()
CANONICAL_REGISTRY_HASH = canonical_hash(CANONICAL_GROUPS)
ALPHA_GROUP_EXACT = 0.04 / 6          # EXACT float (Pi rev-8 #2: bind exact 0.04/6, not a rounded value)


def _registry_identity():
    return {v: canonical_hash(REG._build_variant(u)[0]) for v, u in (("with", True), ("without", False))}


# The exact REGISTERED configuration — no mutable module globals carry registered semantics (Pi rev-8 #2).
REGISTERED = {"N_per_arm": 8000, "B": 20000, "floor": 500, "alpha_group": ALPHA_GROUP_EXACT,
              "registry_identity": _registry_identity(),
              "map_set_identity": "RESERVED_MAP_SET_NOT_DRAWN",       # reserved draw BLOCKED -> real registered run blocks
              "rng_manifest_identity": "RESERVED_RNG_MANIFEST_NOT_BOUND"}


def _is_int_arr(a):
    return isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_


def _validate_raw_records(pool):
    """Refuse a malformed RAW pool BEFORE precompute (Pi rev-8 #5): each record must satisfy the SequenceRecord
    invariants the estimators rely on (finite nondecreasing timestamps, class_ids in [0,C), cluster_ids in [0,K),
    consistent lengths). A hand-built / corrupt record thus refuses at the boundary instead of crashing (or silently
    mis-binning) inside a precompute. The engine derives nothing from a record it has not validated."""
    for i, r in enumerate(pool):
        for attr in ("L_total", "K", "class_ids", "timestamps", "cluster_ids"):
            if not hasattr(r, attr):
                raise RefusalError(f"raw record {i} missing attribute {attr}")
        L, K = r.L_total, r.K
        if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or int(L) < 1:
            raise RefusalError(f"raw record {i} L_total {L!r} must be a positive non-bool int")
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or not (1 <= int(K) <= int(L)):
            raise RefusalError(f"raw record {i} K {K!r} must be an int in [1, L_total]")
        L = int(L); K = int(K)
        ci, ts, cl = np.asarray(r.class_ids), np.asarray(r.timestamps), np.asarray(r.cluster_ids)
        if not (ci.ndim == ts.ndim == cl.ndim == 1 and ci.shape[0] == ts.shape[0] == cl.shape[0] == L):
            raise RefusalError(f"raw record {i} class_ids/timestamps/cluster_ids must be 1-D of length L_total")
        if not _is_int_arr(ci) or int(ci.min()) < 0 or int(ci.max()) >= C:
            raise RefusalError(f"raw record {i} class_ids must be non-bool ints in [0,{C})")
        if (ts.dtype == np.bool_ or not np.issubdtype(ts.dtype, np.floating) or not np.isfinite(ts).all()):
            raise RefusalError(f"raw record {i} timestamps must be finite floats")
        if L > 1 and bool(np.any(np.diff(ts) < 0)):
            raise RefusalError(f"raw record {i} timestamps must be nondecreasing")
        if not _is_int_arr(cl) or int(cl.min()) < 0 or int(cl.max()) >= K:
            raise RefusalError(f"raw record {i} cluster_ids must be non-bool ints in [0,K)")
    return pool


# --- per-estimator precompute SCHEMA (Pi rev-8 #5 / rev-10 #3): validate keys/shapes/dtypes/index-ranges and legal
#     NaN locations BEFORE any statistic. NaN is legal ONLY as the per-sequence 'absent-in-bin' sentinel (map/S4);
#     Inf is never legal; count/pair channels (sp, cc, class_tv, dt0, S4 pair-cols) must be finite + nonnegative. ---
def _pc_arr(a, name, check):
    if not isinstance(a, np.ndarray):
        raise RefusalError(f"{check} precompute {name} must be an ndarray, got {type(a).__name__}")
    return a


def _pc_no_inf(a, name, check):
    aa = np.asarray(a, float)
    if aa.size and bool(np.isinf(aa).any()):
        raise RefusalError(f"{check} precompute {name} contains Inf (never legal)")


def _pc_all_finite(a, name, check):
    aa = np.asarray(a, float)
    if aa.size and not bool(np.isfinite(aa).all()):
        raise RefusalError(f"{check} precompute {name} must be all-finite (no NaN/Inf here)")


def _pc_nonneg(a, name, check):
    aa = np.asarray(a, float); fin = np.isfinite(aa)
    if bool(fin.any()) and bool((aa[fin] < 0).any()):
        raise RefusalError(f"{check} precompute {name} must be nonnegative")


def _validate_precompute(pre, check, n):
    """Fail-closed per-estimator precompute schema. `n` is the pooled sequence count. Returns `pre` if valid, else
    raises RefusalError BEFORE any statistic runs."""
    if check not in ESTIMATORS:
        raise RefusalError(f"precompute schema: unknown/unregistered estimator {check}")
    nbL, nbC = len(LENGTH_BINS), len(CLUSTER_BINS)

    def col2d(a, name, cols, *, nan_ok):
        a = _pc_arr(a, name, check)
        if a.ndim != 2 or a.shape[0] != n or a.shape[1] != cols:
            raise RefusalError(f"{check} precompute {name} shape {a.shape} != ({n},{cols})")
        (_pc_no_inf if nan_ok else _pc_all_finite)(a, name, check)
        return a

    def row1d(a, name, *, nan_ok):
        a = _pc_arr(a, name, check)
        if a.ndim != 1 or a.shape[0] != n:
            raise RefusalError(f"{check} precompute {name} shape {a.shape} != ({n},)")
        (_pc_no_inf if nan_ok else _pc_all_finite)(a, name, check)
        return a

    def keys(d, want):
        if not isinstance(d, dict):
            raise RefusalError(f"{check} precompute must be a dict, got {type(d).__name__}")
        if set(d) != set(want):
            raise RefusalError(f"{check} precompute keys {sorted(d)} != {sorted(want)}")

    def binlist(x, name, nb, cols, *, nonneg=False):
        if not isinstance(x, list) or len(x) != nb:
            raise RefusalError(f"{check} precompute {name} must be a list of {nb} per-bin arrays (got {len(x) if isinstance(x, list) else type(x).__name__})")
        for b, arr in enumerate(x):
            a = row1d(arr, f"{name}[{b}]", nan_ok=True) if cols is None else col2d(arr, f"{name}[{b}]", cols, nan_ok=True)
            if nonneg:                                          # count channels are finite + nonneg (no NaN)
                _pc_all_finite(a, f"{name}[{b}]", check); _pc_nonneg(a, f"{name}[{b}]", check)

    if check == "S3_tau":
        a = _pc_arr(pre, "components", check)
        if a.ndim != 2 or a.shape[0] != n or a.shape[1] < 4:
            raise RefusalError(f"S3_tau precompute shape {a.shape} invalid (want (n,>=4))")
        _pc_all_finite(a, "components", check)
    elif check == "delta_t_zero_abs":
        _pc_nonneg(col2d(pre, "dt0", 2, nan_ok=False), "dt0", check)
    elif check == "positive_gap_ks":
        keys(pre, ("owner", "inv", "nu"))
        owner, inv, nu = pre["owner"], pre["inv"], pre["nu"]
        if not (_is_int_arr(np.asarray(owner)) and _is_int_arr(np.asarray(inv))):
            raise RefusalError("positive_gap_ks owner/inv must be integer arrays")
        owner, inv = np.asarray(owner), np.asarray(inv)
        if owner.ndim != 1 or inv.ndim != 1 or owner.shape[0] != inv.shape[0]:
            raise RefusalError("positive_gap_ks owner/inv must be equal-length 1-D arrays")
        if isinstance(nu, bool) or not isinstance(nu, (int, np.integer)) or int(nu) <= 0:
            raise RefusalError("positive_gap_ks nu must be a positive non-bool int")
        if owner.size and (int(owner.min()) < 0 or int(owner.max()) >= n):
            raise RefusalError("positive_gap_ks owner index out of [0,n)")
        if inv.size and (int(inv.min()) < 0 or int(inv.max()) >= int(nu)):
            raise RefusalError("positive_gap_ks inv index out of [0,nu)")
    elif check == "S3_loggap":
        keys(pre, ("sm", "sp")); binlist(pre["sm"], "sm", nbC, None); binlist(pre["sp"], "sp", nbC, None, nonneg=True)
    elif check == "S4_abs":
        a = col2d(pre, "s4", 3, nan_ok=True)                    # [contrast, same_pairs, adj_pairs]; NaN row = absent
        _pc_nonneg(a[:, 1:], "s4[pair-counts]", check)
    elif check == "class_tv":
        _pc_nonneg(col2d(pre, "class_tv", C, nan_ok=False), "class_tv", check)
    elif check == "occupancy_abs":
        row1d(pre, "occ", nan_ok=False)
    elif check == "S5_abs":
        keys(pre, ("sm",)); binlist(pre["sm"], "sm", nbL, None)
    elif check == "S6_tv":
        keys(pre, ("vec",)); binlist(pre["vec"], "vec", nbL, C)
    elif check == "S7_abs":
        keys(pre, ("sm", "cc")); binlist(pre["sm"], "sm", nbC, None); binlist(pre["cc"], "cc", nbC, None, nonneg=True)
    return pre


def _assemble_arms(arms_by_exp, canon, *, exact_counts):
    """STRUCTURED arms -> validated experiments with per-stratum (nA,nB) + a canonical-order pool
    [cand_s0, ref_s0, cand_s1, ref_s1, ...]. Fixes the flat-pool divisibility bug (Pi rev-8 #1): unequal registered
    strata (2667,2667,2666) are given explicitly, never guessed from a pool length."""
    if set(arms_by_exp) != set(canon["experiments"]):
        raise RefusalError(f"experiments {sorted(arms_by_exp)} != canonical {sorted(canon['experiments'])}")
    experiments = {}
    for e, meta in canon["experiments"].items():
        arms = arms_by_exp[e]
        if list(arms) != list(meta["stratum_ids"]):
            raise RefusalError(f"experiment {e} stratum ids/order {list(arms)} != canonical {meta['stratum_ids']}")
        strata, pool = [], []
        for sid, (qc, qr) in zip(meta["stratum_ids"], meta["registered_quota"]):
            arm = arms[sid]
            if not isinstance(arm, dict) or set(arm) != {"candidate", "reference"}:
                raise RefusalError(f"{e}/{sid} arm must be exactly {{candidate, reference}}")
            cand, ref = arm["candidate"], arm["reference"]
            if not isinstance(cand, list) or not isinstance(ref, list):
                raise RefusalError(f"{e}/{sid} candidate/reference must be lists")
            nc, nr = len(cand), len(ref)
            if exact_counts and (nc, nr) != (qc, qr):
                raise RefusalError(f"{e}/{sid} registered quota {(nc, nr)} != canonical {(qc, qr)}")
            if nc != nr or nc == 0:
                raise RefusalError(f"{e}/{sid} must be balanced non-empty")
            strata.append((nc, nr)); pool += list(cand) + list(ref)   # canonical order: cand then ref per stratum
        experiments[e] = {"strata": strata, "source": meta["source"], "replicate_seed": 0,
                          "coupled_component": meta["coupled_component"], "pool": pool}
    return experiments


def _gate_core(group_id, experiments, *, floor, B, seed, alpha_group, map_for_cell):
    """Common core: build cells from the canonical registry ONLY, compute precompute INTERNALLY, run the gate at
    the given `floor` (a parameter — no module-global mutation)."""
    canon = CANONICAL_GROUPS[group_id]; cells = []
    for cc in canon["cells"]:
        est = ESTIMATORS[cc["check"]]; pool = experiments[cc["exp"]]["pool"]
        _validate_raw_records(pool)                               # raw-record schema BEFORE precompute (Pi rev-8 #5)
        pre = _validate_precompute(est["precompute"](pool), cc["check"], len(pool))
        cell = {"cell_id": cc["cell_id"], "exp": cc["exp"], "check": cc["check"], "pre": pre, "delta": cc["delta"]}
        if cc["map_carrying"]:
            cell["map_art"] = map_for_cell(cc)                        # bound + validated by the caller's mode
        cells.append(cell)
    spec = {"cells": cells, "experiments": experiments, "B": B, "seed": seed, "floor": floor,
            "registered": {"cell_ids": [c["cell_id"] for c in cells], "alpha_group": alpha_group,
                           "floor_policy": f"floor_{floor}",
                           "map_hashes": {c["cell_id"]: (map_identity(c["map_art"]) if c.get("map_art") else None)
                                          for c in cells},
                           "rng_identities": {e: rng_identity(m["source"], m["replicate_seed"], m["coupled_component"])
                                              for e, m in experiments.items()}}}
    return _gate_group(spec)


def gate_group_dev(group_id, arms_by_exp, *, seed, B, floor, map_artifacts, dev_config_hash):
    """DEVELOPMENT entry point (Pi rev-8 #2): explicit hash-bound dev config (floor/B passed, NOT a mutated global).
    Strata STRUCTURE is validated (canonical stratum ids/order, balanced) but sizes are dev-scaled. Each map-cell's
    artifact is CONTEXT-BOUND to its (profile,regime,check)+dev floor/N, and one shared map identity per
    (profile,regime,check) is enforced (Pi rev-8 #3)."""
    if group_id not in CANONICAL_GROUPS:
        raise RefusalError(f"unknown/unwired canonical group {group_id}")
    canon = CANONICAL_GROUPS[group_id]
    # RC5 (Pi rev-9): positive non-bool B/floor and a non-bool integer seed
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise RefusalError("dev B must be a positive non-bool int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RefusalError("dev seed must be a non-bool int")
    if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
        raise RefusalError("dev floor must be a positive non-bool int")
    if dev_config_hash != canonical_hash({"mode": "dev", "floor": floor, "B": B, "group": group_id}):
        raise RefusalError("dev_config_hash does not bind (floor,B,group)")
    # RC5: reject EXTRA map artifacts (only the group's map-carrying cells may be supplied; no silent ignores)
    expected_map_cells = {cc["cell_id"] for cc in canon["cells"] if cc["map_carrying"]}
    extra = set(map_artifacts or {}) - expected_map_cells
    if extra:
        raise RefusalError(f"unexpected extra map artifacts {sorted(extra)}")
    experiments = _assemble_arms(arms_by_exp, canon, exact_counts=False)
    # CONTEXT-bind maps: one shared identity per (profile,regime,check); dev floor must match; profile/regime match
    per_ctx = {}; bound = {}
    def map_for_cell(cc):
        art = (map_artifacts or {}).get(cc["cell_id"])
        if art is None:
            raise RefusalError(f"map-carrying cell {cc['cell_id']} missing mandatory map artifact")
        validate_map_artifact(art)
        src = canon["experiments"][cc["exp"]]["source"]; ctx = (src, "full", cc["check"])
        if art["check"] != cc["check"] or art["profile"] != src or art["regime"] != "full":
            raise RefusalError(f"map context mismatch for {cc['cell_id']} (expected {ctx})")
        if art["floor"] != floor:
            raise RefusalError(f"map floor {art['floor']} != dev floor {floor} for {cc['cell_id']}")
        mid = map_identity(art)
        if per_ctx.setdefault(ctx, mid) != mid:
            raise RefusalError(f"cells sharing {ctx} received different maps")
        bound[cc["cell_id"]] = (mid, art["namespace"])
        return art
    res = _gate_core(group_id, experiments, floor=floor, B=B, seed=seed, alpha_group=ALPHA_GROUP_EXACT,
                     map_for_cell=map_for_cell)
    # RC5: persist a FULL dev-config reproducibility record — per-stratum counts, map identities/set identity,
    # code/registry identity, namespace, seed law — not merely (floor,B,group). A stable identity omits the seed.
    map_ids = {cid: mid for cid, (mid, _ns) in bound.items()}
    nss = sorted({ns for _mid, ns in bound.values()})
    if len(nss) > 1:
        raise RefusalError(f"dev maps span multiple namespaces {nss}")
    stable = {"mode": "dev", "group": group_id, "floor": floor, "B": B,
              "per_stratum_counts": {e: [list(s) for s in ex["strata"]] for e, ex in experiments.items()},
              "map_identities": map_ids,
              "map_set_identity": canonical_hash([map_ids[k] for k in sorted(map_ids)]),
              "registry_identity": CANONICAL_REGISTRY_HASH,
              # Pi rev-11 Correction 2: bind BOTH the semantic estimator identity AND the executable code identity.
              "estimator_protocol_semantic_identity": ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY,
              "estimator_code_identity": ESTIMATOR_CODE_IDENTITY,
              "namespace": (nss[0] if nss else None),
              "seed_law": "caller-supplied dev seed; assignments = numpy default_rng(seed) per-stratum permutation"}
    res["dev_config"] = {**stable, "seed": seed}
    res["dev_config_stable_identity"] = canonical_hash(stable)
    res["dev_config_identity"] = canonical_hash(res["dev_config"])
    return res


def gate_group_registered(group_id, arms_by_exp, *, seed, registry_identity, map_set_identity, rng_manifest_identity):
    """REGISTERED entry point — a BLOCKED STUB, not yet an executable evaluator (Pi rev-9 RC2). The registered
    invariants are DECLARED and the structured assembly + identity refusals are TESTED (registry identity, exact
    per-stratum quotas incl. the (2667,2667,2666) structural-zero, candidate N==8000, placeholder map-set / RNG
    identities), but the registered STATISTIC path is UNIMPLEMENTED: this function never calls `_gate_core`, never
    consumes a map set, never validates an RNG manifest, and never runs at B=20000/floor 500 — B/floor/alpha are not
    even call arguments here. It validates what it can, then UNCONDITIONALLY raises. Because the reserved map-set and
    RNG manifests are not drawn/bound, a real registered run is blocked; activation is a later reviewed change."""
    if group_id not in CANONICAL_GROUPS:
        raise RefusalError(f"unknown/unwired canonical group {group_id}")
    if seed is None:
        raise RefusalError("missing seed")
    if registry_identity != REGISTERED["registry_identity"]:
        raise RefusalError("registry identity mismatch")
    canon = CANONICAL_GROUPS[group_id]
    experiments = _assemble_arms(arms_by_exp, canon, exact_counts=True)     # exact (2667,2667,2666) etc.
    for e, ex in experiments.items():
        if sum(nc for nc, _ in ex["strata"]) != REGISTERED["N_per_arm"]:
            raise RefusalError(f"experiment {e} candidate N != registered {REGISTERED['N_per_arm']}")
    if map_set_identity != REGISTERED["map_set_identity"]:
        raise RefusalError("map-set identity != approved (reserved map set not yet drawn)")
    if rng_manifest_identity != REGISTERED["rng_manifest_identity"]:
        raise RefusalError("RNG manifest identity != approved (reserved RNG manifest not yet bound)")
    raise RefusalError("registered run BLOCKED: reserved map-set + RNG manifest not yet drawn/bound (stop line)")


# --- self-tests ---------------------------------------------------------------------------------------
def _mk_spec(cells, exps, B=199, seed=1, alpha=0.00667):
    reg = {"cell_ids": [c["cell_id"] for c in cells], "alpha_group": alpha, "floor_policy": "F",
           "map_hashes": {c["cell_id"]: (map_identity(c["map_art"]) if c.get("map_art") else None) for c in cells},
           "rng_identities": {e: rng_identity(m["source"], m["replicate_seed"], m["coupled_component"])
                              for e, m in exps.items()}}
    return {"cells": cells, "experiments": exps, "registered": reg, "B": B, "seed": seed}


def selftest():
    from scripts.oracle_realism_v3_map import build_frozen_map
    errs = []
    rng = np.random.default_rng(0)
    prof = "mimic_scale_control"
    # a small heterogeneous group: 3 non-map cells in one experiment (fast) — map cell tested separately
    from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
    from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
    def draw(tag, n=1500):
        return sample_fixture("MIMIC", PROFILES[prof], n, seed=int.from_bytes(
            hashlib.sha256(f"eng|{tag}".encode()).digest()[:6], "big"))
    cand, ref = draw("cand"), draw("ref")
    pool = list(cand) + list(ref)
    exps = {"e0": {"strata": [(len(cand), len(ref))], "source": prof, "replicate_seed": 0, "coupled_component": None}}
    cells = [{"cell_id": "SD|e0|S3_tau", "exp": "e0", "check": "S3_tau", "pre": _tau_pre(pool), "delta": 0.05},
             {"cell_id": "SD|e0|delta_t_zero_abs", "exp": "e0", "check": "delta_t_zero_abs",
              "pre": _dt0_pre(pool), "delta": 0.02},
             {"cell_id": "SD|e0|positive_gap_ks", "exp": "e0", "check": "positive_gap_ks",
              "pre": _gap_pre(pool), "delta": 0.05}]

    r = _gate_group(_mk_spec(cells, exps))
    if r["verdict"] not in (PASS, FAIL):
        errs.append(f"heterogeneous gate produced {r['verdict']}")

    def refused(mut):
        spec = _mk_spec([dict(c) for c in cells], {e: dict(v) for e, v in exps.items()})
        mut(spec)
        try:
            _gate_group(spec); return False
        except RefusalError:
            return True
    checks = {
        "bad_alpha": lambda s: s["registered"].__setitem__("alpha_group", 1.5),
        "bad_B": lambda s: s.__setitem__("B", 0),
        "bool_B": lambda s: s.__setitem__("B", True),
        "missing_seed": lambda s: s.__setitem__("seed", None),
        "cell_order": lambda s: s.__setitem__("cells", list(reversed(s["cells"]))),
        "missing_cell": lambda s: s.__setitem__("cells", s["cells"][:2]),
        "extra_cell": lambda s: s["cells"].append({**cells[0], "cell_id": "SD|e0|EXTRA"}),
        "unknown_exp": lambda s: s["cells"][0].__setitem__("exp", "eX"),
        "rng_identity": lambda s: s["experiments"]["e0"].__setitem__("coupled_component", "burst_timing"),
    }
    for name, mut in checks.items():
        if not refused(mut):
            errs.append(f"refusal NOT raised: {name}")

    # MANDATORY map-hash: a map-carrying cell with a missing/mismatched map_art must refuse
    mcell = {"cell_id": "SD|e0|S3_loggap", "exp": "e0", "check": "S3_loggap", "pre": _loggap_pre(pool),
             "delta": 0.09531, "map_art": build_frozen_map(ref, "S3_loggap", profile=prof, regime="full", seed=11, N=1500)}
    mcells = cells + [mcell]
    if not refused_map(mcells, exps, drop_art=True):
        errs.append("missing mandatory map_art NOT refused")
    if not refused_map(mcells, exps, tamper_hash=True):
        errs.append("map-hash mismatch NOT refused")

    # real permutation-NE: a value-dependent support cell yields +inf on some non-observed assignments and the
    # gate stays conservative (size <= alpha over null trials).
    rej = 0; T = 200
    for t in range(T):
        c2, r2 = draw(("neA", t), 400), draw(("neB", t), 400)      # small N so S3_loggap NEs on some perms
        p2 = list(c2) + list(r2)
        e2 = {"e0": {"strata": [(len(c2), len(r2))], "source": prof, "replicate_seed": 0, "coupled_component": None}}
        art = build_frozen_map(r2, "S3_loggap", profile=prof, regime="full", seed=100 + t, N=400)
        cl = [{"cell_id": "SD|e0|S3_loggap", "exp": "e0", "check": "S3_loggap", "pre": _loggap_pre(p2),
               "delta": 0.09531, "map_art": art}]
        res = _gate_group(_mk_spec(cl, e2, B=99, seed=100 + t))
        if res["verdict"] == FAIL:
            rej += 1
    if rej / T > 0.1:                                               # conservative under null despite +inf NE perms
        errs.append(f"permutation-NE gate not conservative: size {rej/T}")

    # EXACT-ESTIMAND consistency: each engine per-perm recompute on the OBSERVED split reproduces the exact v2
    # value (so observed + permuted use the identical statistic — permutation-test validity). S3_tau is the v3
    # pooled-tau (validated vs scipy in the pilot), excluded from v2 equality.
    from clinical_jepa.eval.oracle_realism_v2_verifier import s3, s4, s5, s6, s7, marginal_route_checks
    cc, rr = draw("consC", 6000), draw("consR", 6000); poolc = list(cc) + list(rr)
    obs = np.array([True] * len(cc) + [False] * len(rr))
    v2 = {**s3(cc, rr), **s4(cc, rr), **s5(cc, rr), **s6(cc, rr), **s7(cc, rr), **marginal_route_checks(cc, rr)}
    for chk in ("delta_t_zero_abs", "positive_gap_ks", "S4_abs", "class_tv", "occupancy_abs",
                "S3_loggap", "S5_abs", "S6_tv", "S7_abs"):
        est = ESTIMATORS[chk]; pre = est["precompute"](poolc)
        g = build_frozen_map(rr, chk, profile=prof, regime="full", seed=13, N=6000)["groups"] if est["map_carrying"] else None
        got, want = est["recompute"](pre, obs, groups=g, floor=FLOOR), v2[chk].value
        ok = (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9)
        if not ok:
            errs.append(f"engine estimand != v2 for {chk}: {got} vs {want}")

    # RC1 (Pi rev-9): uniform keyword-only estimator protocol — `floor` must reach the floor gate for EVERY
    # binding (the rev-9 S5 bug routed the dev floor into `extra_key`, FAKING the class-mark NOT_EVALUABLE).
    # Controlled precomputes with EXACTLY 100 present per arm: each floor-gated estimator EVALUATES at floor 60 and
    # REFUSES (None) at floor 500; each floor-insensitive estimator returns the identical value at both floors.
    m2 = np.array([True] * 100 + [False] * 100)
    sm100 = np.concatenate([np.ones(100), np.full(100, 2.0)])          # 100 present/arm, distinct means
    occ100 = np.concatenate([np.full(100, 0.3), np.full(100, 0.1)])
    s4_100 = np.column_stack([occ100, np.ones(200), np.ones(200)])     # [contrast, same_pairs, adj_pairs]
    oh0 = np.zeros(C); oh0[0] = 1.0; oh1 = np.zeros(C); oh1[1] = 1.0
    s6_100 = np.concatenate([np.tile(oh0, (100, 1)), np.tile(oh1, (100, 1))])
    floor_pre = {"occupancy_abs": (occ100, None), "S4_abs": (s4_100, None),
                 "S3_loggap": ({"sm": [sm100], "sp": [np.ones(200)]}, [[0]]),
                 "S5_abs": ({"sm": [sm100]}, [[0]]),
                 "S6_tv": ({"vec": [s6_100]}, [[0]]),
                 "S7_abs": ({"sm": [sm100], "cc": [np.ones(200)]}, [[0]])}
    for chk, (fpre, fg) in floor_pre.items():
        rc = ESTIMATORS[chk]["recompute"]
        lo = rc(fpre, m2, groups=fg, floor=60); hi = rc(fpre, m2, groups=fg, floor=500)
        if lo is None:
            errs.append(f"RC1 {chk}: floor-60 (100 present/arm) should evaluate, got None (floor not routed to gate)")
        if hi is not None:
            errs.append(f"RC1 {chk}: floor-500 (100<500 present) should refuse (None), got {hi}")
    obs_pool = np.array([True] * len(cand) + [False] * len(ref))
    for chk in ("S3_tau", "delta_t_zero_abs", "positive_gap_ks", "class_tv"):
        rc = ESTIMATORS[chk]["recompute"]; fpre = ESTIMATORS[chk]["precompute"](pool)
        a = rc(fpre, obs_pool, groups=None, floor=60); b = rc(fpre, obs_pool, groups=None, floor=500)
        if not ((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12)):
            errs.append(f"RC1 {chk}: floor-insensitive estimator differs across floors: {a} vs {b}")

    # PRECOMPUTE + RAW-RECORD SCHEMA (Pi rev-8 #5 / rev-10 #3): every REAL precompute validates; each malformed
    # class refuses BEFORE any statistic; corrupt raw records refuse at the boundary (not deep inside a precompute).
    npool = len(pool)
    for chk in ("S3_tau", "delta_t_zero_abs", "positive_gap_ks", "S4_abs", "class_tv", "occupancy_abs",
                "S3_loggap", "S5_abs", "S6_tv", "S7_abs"):
        try:
            _validate_precompute(ESTIMATORS[chk]["precompute"](pool), chk, npool)
        except RefusalError as ex:
            errs.append(f"precompute schema rejected a VALID {chk}: {ex}")

    def _corrupt(a, fn):
        if isinstance(a, np.ndarray):
            b = a.copy()
        elif isinstance(a, dict):
            b = {k: ([x.copy() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list)
                     else (v.copy() if isinstance(v, np.ndarray) else v)) for k, v in a.items()}
        else:
            b = a
        fn(b); return b

    def _srefused(check, pre):
        try:
            _validate_precompute(pre, check, npool); return False
        except RefusalError:
            return True
    occ = _occ_pre(pool); ctv = _classtv_pre(pool); dt0 = _dt0_pre(pool)
    gp = _gap_pre(pool); lgp = _loggap_pre(pool); s4p = _s4_pre(pool); s6p = _s6_pre(pool); s7p = _s7_pre(pool)
    ar = np.arange                                                   # index-0 selector helper
    schema_cases = {
        "wrong_pooled_length": ("occupancy_abs", occ[:-1]),
        "inf_in_finite_field": ("occupancy_abs", _corrupt(occ, lambda b: b.__setitem__(0, np.inf))),
        "illegal_nan_finite_field": ("occupancy_abs", _corrupt(occ, lambda b: b.__setitem__(0, np.nan))),
        "wrong_class_width": ("class_tv", ctv[:, :C - 1]),
        "negative_count": ("class_tv", _corrupt(ctv, lambda b: b.__setitem__((0, 0), -1.0))),
        "dt0_wrong_cols": ("delta_t_zero_abs", dt0[:, :1]),
        "gap_missing_key": ("positive_gap_ks", {"owner": gp["owner"], "inv": gp["inv"]}),
        "gap_owner_inv_mismatch": ("positive_gap_ks", {**gp, "inv": gp["inv"][:-1]}),
        "gap_owner_out_of_range": ("positive_gap_ks", {**gp, "owner": np.where(ar(len(gp["owner"])) == 0, npool, gp["owner"])}),
        "gap_inv_out_of_range": ("positive_gap_ks", {**gp, "inv": np.where(ar(len(gp["inv"])) == 0, gp["nu"], gp["inv"])}),
        "loggap_wrong_nb": ("S3_loggap", {"sm": lgp["sm"][:-1], "sp": lgp["sp"]}),
        "loggap_sp_has_nan": ("S3_loggap", _corrupt(lgp, lambda b: b["sp"][0].__setitem__(0, np.nan))),
        "loggap_sm_has_inf": ("S3_loggap", _corrupt(lgp, lambda b: b["sm"][0].__setitem__(0, np.inf))),
        "s4_pair_negative": ("S4_abs", _corrupt(s4p, lambda b: b.__setitem__((0, 1), -1.0))),
        "s6_extra_key": ("S6_tv", {**s6p, "sneaky": 1}),
        "s6_vec_wrong_width": ("S6_tv", {"vec": [v[:, :C - 1] for v in s6p["vec"]]}),
        "s7_cc_has_inf": ("S7_abs", _corrupt(s7p, lambda b: b["cc"][0].__setitem__(0, np.inf))),
    }
    for label, (chk, bad) in schema_cases.items():
        if not _srefused(chk, bad):
            errs.append(f"precompute schema did NOT refuse {label}")

    # raw-record schema: the valid pool passes; hand-corrupted records refuse at the boundary
    import dataclasses as _dc
    if _validate_raw_records(pool) is not pool:
        errs.append("raw-record schema rejected a valid pool")
    g0 = pool[0]
    raw_bad = {
        "nonfinite_timestamp": _dc.replace(g0, timestamps=np.where(ar(g0.L_total) == 0, np.inf, g0.timestamps).astype(float)),
        "class_id_out_of_range": _dc.replace(g0, class_ids=np.where(ar(g0.L_total) == 0, C, g0.class_ids).astype(int)),
        "cluster_id_out_of_range": _dc.replace(g0, cluster_ids=np.where(ar(g0.L_total) == 0, g0.K, g0.cluster_ids).astype(int)),
        "length_mismatch": _dc.replace(g0, class_ids=g0.class_ids[:-1]),
    }
    for label, rec in raw_bad.items():
        try:
            _validate_raw_records([rec]); errs.append(f"raw-record schema did NOT refuse {label}")
        except RefusalError:
            pass

    # ADVERSARIAL: Pi rev-7 #4 fail-open cases now REFUSE / NE (were PASS p_g=1.0)
    # (a) all-NaN precompute -> observed discrepancy non-finite -> group NOT_EVALUABLE, NOT a zero-filled PASS
    nan_cells = [{"cell_id": "SD|e0|occupancy_abs", "exp": "e0", "check": "occupancy_abs",
                  "pre": np.full(len(cand) + len(ref), np.nan), "delta": 0.03}]
    if _gate_group(_mk_spec(nan_cells, exps))["verdict"] != NOT_EVALUABLE:
        errs.append("all-NaN precompute did not NE (fail-open persists)")
    # (b) DEV + REGISTERED entry points close caller injection + validate inputs (Pi rev-8 #2)
    def _refused(fn):
        try:
            fn(); return False
        except RefusalError:
            return True
    _dch_burst = canonical_hash({"mode": "dev", "floor": 60, "B": 99, "group": "G_full_burst_timing"})
    ref = {
        "dev_unknown_group": lambda: gate_group_dev("NOT_A_GROUP", {}, seed=1, B=99, floor=60,
                                                    map_artifacts={}, dev_config_hash="x"),
        "dev_bad_B": lambda: gate_group_dev("G_full_burst_timing", {}, seed=1, B=0, floor=60,
                                            map_artifacts={}, dev_config_hash="x"),
        "dev_bool_seed": lambda: gate_group_dev("G_full_burst_timing", {}, seed=True, B=99, floor=60,
                                                map_artifacts={}, dev_config_hash="x"),          # RC5: non-bool int seed
        "dev_bad_floor": lambda: gate_group_dev("G_full_burst_timing", {}, seed=1, B=99, floor=0,
                                                map_artifacts={}, dev_config_hash="x"),          # RC5: positive floor
        "dev_bad_config_hash": lambda: gate_group_dev("G_full_burst_timing", {}, seed=1, B=99, floor=60,
                                                      map_artifacts={}, dev_config_hash="WRONG"),
        "dev_extra_map": lambda: gate_group_dev("G_full_burst_timing", {}, seed=1, B=99, floor=60,   # RC5: no extras
                                                map_artifacts={"BOGUS_CELL": {}}, dev_config_hash=_dch_burst),
        "inf_precompute": lambda: _validate_precompute(np.array([1.0, np.inf, 2.0]), "occupancy_abs", 3),
        "unknown_check_schema": lambda: _validate_precompute(np.zeros(3), "NOT_A_CHECK", 3),
    }
    for name, fn in ref.items():
        if not _refused(fn):
            errs.append(f"entry refusal NOT raised: {name}")

    # (c) REGISTERED adversarial preflight (Pi rev-8 next-gate ask): structured registered strata ASSEMBLE with no
    # divisibility refusal; gate_group_registered refuses every deviation and blocks the (reserved) real run.
    burst = "G_full_burst_timing"
    canon = CANONICAL_GROUPS[burst]
    def _reg_arms(quota_override=None):                              # synthetic registered arms (dummy sequences)
        arms = {}
        for e, meta in canon["experiments"].items():
            arms[e] = {}
            for sid, (qc, qr) in zip(meta["stratum_ids"], meta["registered_quota"]):
                if quota_override and e == "structural_zero":
                    qc, qr = quota_override
                arms[e][sid] = {"candidate": [0] * qc, "reference": [0] * qr}
        return arms
    RID, MSI, RMI = REGISTERED["registry_identity"], REGISTERED["map_set_identity"], REGISTERED["rng_manifest_identity"]
    # structured registered structural-zero (2667,2667,2666) ASSEMBLES without the old flat-pool divisibility error
    try:
        _assemble_arms(_reg_arms(), canon, exact_counts=True)
    except RefusalError as ex:
        errs.append(f"registered structured-strata assembly refused (divisibility not fixed): {ex}")
    reg_refusals = {
        "reg_blocked_when_all_valid": lambda: gate_group_registered(burst, _reg_arms(), seed=1, registry_identity=RID,
                                                                    map_set_identity=MSI, rng_manifest_identity=RMI),
        "reg_wrong_registry": lambda: gate_group_registered(burst, _reg_arms(), seed=1, registry_identity="X",
                                                            map_set_identity=MSI, rng_manifest_identity=RMI),
        "reg_wrong_quota": lambda: gate_group_registered(burst, _reg_arms(quota_override=(2000, 2000)), seed=1,
                                                         registry_identity=RID, map_set_identity=MSI,
                                                         rng_manifest_identity=RMI),
        "reg_wrong_mapset": lambda: gate_group_registered(burst, _reg_arms(), seed=1, registry_identity=RID,
                                                          map_set_identity="X", rng_manifest_identity=RMI),
    }
    for name, fn in reg_refusals.items():
        if not _refused(fn):                                        # ALL must raise (incl. the reserved-run block)
            errs.append(f"registered refusal NOT raised: {name}")
    if REGISTERED["alpha_group"] != 0.04 / 6:
        errs.append("alpha_group not the exact 0.04/6 float")
    if set(CANONICAL_GROUPS) != {"G_full_burst_timing", "G_full_class_mark"}:
        errs.append("canonical groups drift")
    return errs


def refused_map(mcells, exps, *, drop_art=False, tamper_hash=False):
    cells = [dict(c) for c in mcells]
    reg = {"cell_ids": [c["cell_id"] for c in cells], "alpha_group": 0.00667, "floor_policy": "F",
           "map_hashes": {c["cell_id"]: (map_identity(c["map_art"]) if c.get("map_art") else None) for c in cells},
           "rng_identities": {e: rng_identity(m["source"], m["replicate_seed"], m["coupled_component"])
                              for e, m in exps.items()}}
    if drop_art:
        for c in cells:
            if c["check"] == "S3_loggap":
                c.pop("map_art", None)
    if tamper_hash:
        for cid in reg["map_hashes"]:
            if "S3_loggap" in cid:
                reg["map_hashes"][cid] = "TAMPERED"
    spec = {"cells": cells, "experiments": exps, "registered": reg, "B": 99, "seed": 1}
    try:
        _gate_group(spec); return False
    except RefusalError:
        return True


def main():
    errs = selftest()
    out = {"dispatcher": "registry-OWNED estimators keyed by registered check; callers provide DATA + cell specs, "
                         "never a statfn trust root.",
           "registered_estimators": {k: v["identity"] for k, v in ESTIMATORS.items()},
           "hardening": ["validate BEFORE any statistic", "exact cell ids+order / no missing+extra",
                         "MANDATORY per-map identity + floor-policy", "alpha_group in (0,1), positive-int B, seed",
                         "executable per-experiment RNG identity (hash, not string)",
                         "one trusted assignment path (IID-with-replacement; duplicates VALID+bound)",
                         "observed NE->group NE; permutation NE->maximally extreme (+inf), no zero-fill"],
           "selftests_pass": not errs, "selftest_errors": errs,
           "authorization": "dev-only engine; no map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(out, indent=2, default=str))
    assert not errs, f"engine self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
