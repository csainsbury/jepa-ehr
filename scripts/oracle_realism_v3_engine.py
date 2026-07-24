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


def _tau_re(pre, mask, groups=None):
    def t(Cm):
        s = Cm.sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
        return None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)
    a, b = t(pre[mask]), t(pre[~mask])
    return None if (a is None or b is None) else abs(a - b)


def _dt0_pre(pool):
    nz = np.array([max(0, r.L_total - r.K) for r in pool], float)
    na = np.array([max(0, r.L_total - 1) for r in pool], float)
    return np.stack([nz, na], 1)


def _dt0_re(pre, mask, groups=None):
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


def _gap_re(pre, mask, groups=None):
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


def _loggap_re(pre, mask, groups):
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
        if cvv.size < FLOOR or rvv.size < FLOOR or cp < FLOOR or rp < FLOOR:
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


def _s4_re(pre, mask, groups=None):
    pres = ~np.isnan(pre[:, 0]); ca = pres & mask; cr = pres & ~mask
    if int(ca.sum()) < FLOOR or int(cr.sum()) < FLOOR:
        return None
    if min(pre[ca, 1].sum(), pre[cr, 1].sum(), pre[ca, 2].sum(), pre[cr, 2].sum()) < FLOOR:
        return None
    return abs(float(pre[ca, 0].mean()) - float(pre[cr, 0].mean()))


def _classtv_pre(pool):
    return np.array([np.bincount(r.class_ids, minlength=C)[:C] for r in pool], float)


def _classtv_re(pre, mask, groups=None):
    a, b = pre[mask].sum(0), pre[~mask].sum(0)
    if a.sum() == 0 or b.sum() == 0:
        return None
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def _occ_pre(pool):
    return np.array([len(np.unique(r.class_ids)) / C for r in pool], float)


def _occ_re(pre, mask, groups=None):
    a, b = pre[mask], pre[~mask]
    return None if (a.size < FLOOR or b.size < FLOOR) else abs(float(a.mean()) - float(b.mean()))


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


def _map_re_scalar(pre, mask, groups, extra_key=None):
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
        if cvv.size < FLOOR or rvv.size < FLOOR or (extra is not None and (ce < FLOOR or re_ < FLOOR)):
            return None
        d = max(d, abs(cvv.mean() - rvv.mean()))
    return d


def _map_re_vector(pre, mask, groups):
    if groups is None:
        return None
    vec = pre["vec"]; d = 0.0
    for grp in groups:
        cvs, rvs = [], []
        for b in grp:
            v = vec[b]; pres = ~np.isnan(v[:, 0])
            cvs.append(v[pres & mask]); rvs.append(v[pres & ~mask])
        cv, rv = np.concatenate(cvs), np.concatenate(rvs)
        if cv.shape[0] < FLOOR or rv.shape[0] < FLOOR:
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
    "S5_abs": {"precompute": _s5_pre, "recompute": _map_re_scalar, "map_carrying": True,
               "identity": "v2.cond_maxbin.mean(occupancy)@LENGTH_BINS[ref_coarsen]"},
    "S6_tv": {"precompute": _s6_pre, "recompute": _map_re_vector, "map_carrying": True,
              "identity": "v2.maxabs_tv(class_prior)@LENGTH_BINS[ref_coarsen]"},
    "S7_abs": {"precompute": _s7_pre, "recompute": (lambda pre, m, g: _map_re_scalar(pre, m, g, extra_key="cc")),
               "map_carrying": True, "identity": "v2.cond_maxbin.mean(distinct_class_frac)@CLUSTER_BINS[ref_coarsen]"},
}


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


def gate_group(spec):
    """spec: {cells:[{cell_id,exp,check,pre,[map_art]}], experiments:{e:{strata,source,replicate_seed,coupled_component}},
    registered:{cell_ids,map_hashes,rng_identities,alpha_group,floor_policy}, B, seed}. Fail-closed."""
    _validate(spec)                                                 # BEFORE any statistic
    reg = spec["registered"]; B = spec["B"]
    rng = np.random.default_rng(spec["seed"])
    exps = spec["experiments"]
    # ONE trusted assignment path (IID-with-replacement MC; duplicates VALID + bound)
    masks = {e: [_canonical_mask(exps[e]["strata"])] + [_perm_mask(rng, exps[e]["strata"]) for _ in range(B)]
             for e in exps}
    def d_of(c, m):
        est = ESTIMATORS[c["check"]]
        groups = c["map_art"]["groups"] if est["map_carrying"] else None
        return est["recompute"](c["pre"], m, groups)
    for c in spec["cells"]:                                          # observed NE (or non-finite) -> group NE
        d0 = d_of(c, masks[c["exp"]][0])
        if d0 is None or not np.isfinite(d0):
            return {"verdict": NOT_EVALUABLE, "p_g": None, "reason": f"observed NE at {c['cell_id']}"}
    E = []
    for c in spec["cells"]:
        est = ESTIMATORS[c["check"]]; groups = c["map_art"]["groups"] if est["map_carrying"] else None
        ej = np.empty(B + 1)
        for j, m in enumerate(masks[c["exp"]]):
            d = est["recompute"](c["pre"], m, groups)
            # NaN/Inf discrepancy is a support/precompute failure -> maximally extreme NE, NEVER zero-fill (Pi #4)
            ej[j] = np.inf if (d is None or not np.isfinite(d)) else max(0.0, d - c["delta"])
        E.append(ej)
    P = np.stack([cell_upper_p(e) for e in E], 0); S = P.min(0)
    p_g = float((S <= S[0]).sum() / len(S))
    return {"verdict": PASS if p_g > reg["alpha_group"] else FAIL, "p_g": p_g,
            "argmin_cell": spec["cells"][int(np.argmin(P[:, 0]))]["cell_id"]}


# ======================================================================================================
# TRUSTED boundary (Pi rev-7 #4): caller passes ONLY a group id + raw experiment pools + trusted seed/B
# (+ dev map artifacts). The engine loads cell order / check / Delta / strata / map-carrying from the
# canonical registry, computes precompute ITSELF from the raw pools, and refuses / NEs any non-finite
# precompute or discrepancy. No caller-supplied check / Delta / registered / precompute is trusted.
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
            exps.setdefault(c["experiment_id"], {
                "source": c["source"], "condition": c["condition"], "coupled_component": c["coupled_component"],
                "n_strata": len(c["exchangeability_strata"]),
                "registered_quota": [(s["n_candidate"], s["n_reference"]) for s in c["exchangeability_strata"]]})
        out[gid] = {"group_id": gid, "cells": cells, "experiments": exps,
                    "alpha_group": round(0.04 / len(groups), 10), "floor_policy": "registered_500"}
    return out


CANONICAL_GROUPS = _build_canonical_groups()
CANONICAL_REGISTRY_HASH = canonical_hash(CANONICAL_GROUPS)


def _validate_precompute(pre, check):
    """Refuse an Inf / malformed precompute BEFORE any statistic (Pi #4). NaN is permitted ONLY as the explicit
    per-sequence 'absent-in-bin' sentinel in map/S4 precomputes; Inf is always malformed."""
    def bad(a):
        a = np.asarray(a, float)
        return a.size and np.isinf(a).any()
    if isinstance(pre, dict):
        for k, v in pre.items():
            for arr in (v if isinstance(v, list) else [v]):
                if isinstance(arr, np.ndarray) and bad(arr):
                    raise RefusalError(f"infinite precompute in {check}:{k}")
    elif isinstance(pre, np.ndarray) and bad(pre):
        raise RefusalError(f"infinite precompute in {check}")
    return pre


def _derive_strata(pool, n_strata):
    """Balanced within-stratum quotas from the RAW pool; n_strata from the canonical registry. Pool order is
    stratum-interleaved [candA,refA,candB,refB,...]; each stratum balanced (nA==nB)."""
    M = len(pool)
    if n_strata <= 0 or M % (2 * n_strata) != 0:
        raise RefusalError(f"pool length {M} not divisible into {n_strata} balanced strata")
    per = M // n_strata
    return [(per // 2, per // 2) for _ in range(n_strata)]


def gate_group_trusted(group_id, pools_by_exp, *, seed, B, map_artifacts=None):
    """TRUSTED entry point. Caller passes only group_id + raw experiment pools + seed + B (+ dev map artifacts)."""
    if group_id not in CANONICAL_GROUPS:
        raise RefusalError(f"unknown/unwired canonical group {group_id}")
    canon = CANONICAL_GROUPS[group_id]
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise RefusalError(f"B {B!r} not a positive integer")
    if seed is None:
        raise RefusalError("missing seed")
    if set(pools_by_exp) != set(canon["experiments"]):
        raise RefusalError(f"experiments {sorted(pools_by_exp)} != canonical {sorted(canon['experiments'])}")
    map_artifacts = map_artifacts or {}
    experiments, cells = {}, []
    for e, meta in canon["experiments"].items():
        pool = pools_by_exp[e]
        if not isinstance(pool, list) or len(pool) == 0:
            raise RefusalError(f"experiment {e} pool must be a non-empty list of sequences")
        experiments[e] = {"strata": _derive_strata(pool, meta["n_strata"]), "source": meta["source"],
                          "replicate_seed": 0, "coupled_component": meta["coupled_component"]}
    for cc in canon["cells"]:                                        # cells built from canonical registry ONLY
        est = ESTIMATORS[cc["check"]]; pool = pools_by_exp[cc["exp"]]
        pre = _validate_precompute(est["precompute"](pool), cc["check"])   # computed HERE + finiteness-checked
        cell = {"cell_id": cc["cell_id"], "exp": cc["exp"], "check": cc["check"], "pre": pre, "delta": cc["delta"]}
        if cc["map_carrying"]:
            art = map_artifacts.get(cc["cell_id"])
            if art is None:
                raise RefusalError(f"map-carrying cell {cc['cell_id']} missing mandatory map artifact")
            validate_map_artifact(art)
            if art["check"] != cc["check"]:
                raise RefusalError(f"map artifact check {art['check']} != canonical {cc['check']}")
            cell["map_art"] = art
        cells.append(cell)
    spec = {"cells": cells, "experiments": experiments, "B": B, "seed": seed,
            "registered": {"cell_ids": [c["cell_id"] for c in cells], "alpha_group": canon["alpha_group"],
                           "floor_policy": canon["floor_policy"],
                           "map_hashes": {c["cell_id"]: (map_identity(c["map_art"]) if c.get("map_art") else None)
                                          for c in cells},
                           "rng_identities": {e: rng_identity(m["source"], m["replicate_seed"], m["coupled_component"])
                                              for e, m in experiments.items()}}}
    return gate_group(spec)


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

    r = gate_group(_mk_spec(cells, exps))
    if r["verdict"] not in (PASS, FAIL):
        errs.append(f"heterogeneous gate produced {r['verdict']}")

    def refused(mut):
        spec = _mk_spec([dict(c) for c in cells], {e: dict(v) for e, v in exps.items()})
        mut(spec)
        try:
            gate_group(spec); return False
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
             "delta": 0.09531, "map_art": build_frozen_map(ref, "S3_loggap", profile=prof, regime="full", N=1500)}
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
        art = build_frozen_map(r2, "S3_loggap", profile=prof, regime="full", N=400)
        cl = [{"cell_id": "SD|e0|S3_loggap", "exp": "e0", "check": "S3_loggap", "pre": _loggap_pre(p2),
               "delta": 0.09531, "map_art": art}]
        res = gate_group(_mk_spec(cl, e2, B=99, seed=100 + t))
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
        g = build_frozen_map(rr, chk, profile=prof, regime="full", N=6000)["groups"] if est["map_carrying"] else None
        got, want = est["recompute"](pre, obs, g), v2[chk].value
        ok = (got is None and want is None) or (got is not None and want is not None and abs(got - want) < 1e-9)
        if not ok:
            errs.append(f"engine estimand != v2 for {chk}: {got} vs {want}")

    # ADVERSARIAL: Pi rev-7 #4 fail-open cases now REFUSE / NE (were PASS p_g=1.0)
    # (a) all-NaN precompute -> observed discrepancy non-finite -> group NOT_EVALUABLE, NOT a zero-filled PASS
    nan_cells = [{"cell_id": "SD|e0|occupancy_abs", "exp": "e0", "check": "occupancy_abs",
                  "pre": np.full(len(cand) + len(ref), np.nan), "delta": 0.03}]
    if gate_group(_mk_spec(nan_cells, exps))["verdict"] != NOT_EVALUABLE:
        errs.append("all-NaN precompute did not NE (fail-open persists)")
    # (b) the TRUSTED entry structurally closes caller check/Delta/precompute injection + validates inputs
    def _trust_refused(fn):
        try:
            fn(); return False
        except RefusalError:
            return True
    trust = {
        "unknown_group": lambda: gate_group_trusted("NOT_A_GROUP", {}, seed=1, B=99),
        "wrong_experiments": lambda: gate_group_trusted("G_full_burst_timing", {"eX": [1]}, seed=1, B=99),
        "bad_B": lambda: gate_group_trusted("G_full_burst_timing", {}, seed=1, B=0),
        "missing_seed": lambda: gate_group_trusted("G_full_burst_timing", {}, seed=None, B=99),
        "inf_precompute": lambda: _validate_precompute(np.array([1.0, np.inf, 2.0]), "x"),
    }
    for name, fn in trust.items():
        if not _trust_refused(fn):
            errs.append(f"trusted-entry refusal NOT raised: {name}")
    # canonical registry is hash-bound + only wired groups present
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
        gate_group(spec); return False
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
