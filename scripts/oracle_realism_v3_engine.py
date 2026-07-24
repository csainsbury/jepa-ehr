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
import platform

import numpy as np
from scipy.stats import kendalltau, ks_2samp

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    _positive_gaps_and_prev_size, _bin_index, CLUSTER_BINS, LENGTH_BINS, _s4_contrast, _seam_mask, C,
)
from clinical_jepa.eval.oracle_realism_v2_fixture import derive_record, MalformedRecord, REQUIRED_SOURCES
from scripts.oracle_realism_v3_randomization import cell_upper_p, RefusalError
from scripts.oracle_realism_v3_phase0_pilot import _seq_components
from scripts.oracle_realism_v3_map import validate_map_artifact, map_identity, FLOOR
import scripts.oracle_realism_v3_registry as REG
import sys as _sys
import scripts.oracle_realism_v3_map as _map_mod
import scripts.oracle_realism_v3_randomization as _rand_mod
import scripts.oracle_realism_v3_phase0_pilot as _pilot_mod
from clinical_jepa.eval import oracle_realism_v2_verifier as _v2ver_mod

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


# --- run-size KS (S2_ks): MEAN-over-sequences of the per-sequence run-size ECDF, max-abs KS on the pool support.
# Reproduces v2 s2 EXACTLY: support = pool run-size union (permutation-invariant); M[i,j] = (# runs of seq i <=
# support[j]) / n_runs_i; F = mean of M over the arm. Sequence floor AND cluster floor (both arms). (rev-14 wiring) ----
def _s2_pre(pool):
    runs = [np.sort(np.bincount(r.cluster_ids).astype(int)) for r in pool]
    support = np.unique(np.concatenate(runs)) if runs else np.array([1], int)
    M = np.empty((len(pool), support.shape[0])); nruns = np.empty(len(pool), int)
    for i, sr in enumerate(runs):
        nruns[i] = sr.shape[0]
        M[i] = np.searchsorted(sr, support, side="right") / sr.shape[0]
    return {"M": M, "nruns": nruns}


def _s2_re(pre, mask, *, groups=None, floor=FLOOR):
    A = mask; nAs = int(A.sum()); nBs = int((~A).sum())
    if nAs < floor or nBs < floor:                                # sequence floor, both arms
        return None
    cA = int(pre["nruns"][A].sum()); cB = int(pre["nruns"][~A].sum())
    if cA < floor or cB < floor:                                 # cluster floor, both arms
        return None
    return float(np.max(np.abs(pre["M"][A].mean(0) - pre["M"][~A].mean(0))))


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


# --- length_density group estimators (rev-16 wiring). S1_density is MAP-carrying (mirrors S5 but per-bin cluster
# density K/L instead of occupancy). S1_tau is SOURCE-LEVEL tau-b(L,K). count_ks/length_ks are two-sample KS on the
# per-sequence cluster count K / length L (matrix-ECDF reproduces scipy ks_2samp exactly). ------------------------
def _s1density_pre(pool):
    return _lenbin_scalar_pre(pool, lambda r: r.K / r.L_total)        # per-length-bin mean cluster density K/L


def _s1tau_pre(pool):
    return np.array([(float(r.L_total), float(r.K)) for r in pool])   # (n,2): [L, K] per sequence


def _s1tau_re(pre, mask, *, groups=None, floor=FLOOR):
    a, b = pre[mask], pre[~mask]
    if a.shape[0] < floor or b.shape[0] < floor:                      # source-level sequence floor, both arms
        return None
    tc = kendalltau(a[:, 0], a[:, 1])[0]; tr = kendalltau(b[:, 0], b[:, 1])[0]
    return None if not (np.isfinite(tc) and np.isfinite(tr)) else abs(float(tc) - float(tr))


def _scalar_ks_pre(vals):
    v = np.asarray(vals, float); support = np.unique(v)
    return {"M": (v[:, None] <= support[None, :]).astype(float)}      # (n,|support|) right-continuous ECDF basis


def _countks_pre(pool):
    return _scalar_ks_pre([r.K for r in pool])


def _lengthks_pre(pool):
    return _scalar_ks_pre([r.L_total for r in pool])


def _scalar_ks_re(pre, mask, *, groups=None, floor=FLOOR):
    nA = int(mask.sum()); nB = int((~mask).sum())
    if nA < floor or nB < floor:                                     # per-arm sequence floor
        return None
    return float(np.max(np.abs(pre["M"][mask].mean(0) - pre["M"][~mask].mean(0))))   # == ks_2samp statistic


# --- phase_seam group estimators (rev-17 wiring). S8 = within-sequence CENTERED quartile-phase nonstationarity (each
# sequence's per-quartile cluster-start density / class vector MINUS its own whole-sequence value; then max-over-4-
# quartiles two-sample mean diff, with per-quartile seq+item floors both arms). S9 = block-SEAM structure (adjacency i
# is a seam iff (i+1)%8==0): per-sequence seam-minus-nonseam Δt==0 fraction (S9_zero) / same-class fraction (S9_class),
# and seam-vs-nonseam positive-gap KS (S9_gap = max within-arm + cross-arm ks_2samp). Reproduce v2 s8/s9. -----------
def _s8density_pre(pool):
    n = len(pool); dens = [np.full(n, np.nan) for _ in range(4)]; items = [np.zeros(n) for _ in range(4)]
    for i, r in enumerate(pool):
        q = np.minimum(3, (r.positions * 4).astype(int))
        starts = np.concatenate([[True], np.diff(r.cluster_ids) == 1]) if r.L_total > 1 else np.array([True])
        base = r.K / r.L_total
        for qi in range(4):
            m = q == qi; nq = int(m.sum())
            if nq > 0:
                dens[qi][i] = float(starts[m].sum()) / nq - base; items[qi][i] = nq
    return {"dens": dens, "items": items}


def _s8class_pre(pool):
    n = len(pool); vec = [np.full((n, C), np.nan) for _ in range(4)]; items = [np.zeros(n) for _ in range(4)]
    for i, r in enumerate(pool):
        q = np.minimum(3, (r.positions * 4).astype(int))
        p_i = np.bincount(r.class_ids, minlength=C)[:C] / r.L_total
        for qi in range(4):
            m = q == qi; nq = int(m.sum())
            if nq > 0:
                vec[qi][i] = np.bincount(r.class_ids[m], minlength=C)[:C] / nq - p_i; items[qi][i] = nq
    return {"vec": vec, "items": items}


def _s8density_re(pre, mask, *, groups=None, floor=FLOOR):
    diffs = []
    for qi in range(4):
        pres = ~np.isnan(pre["dens"][qi]); Ap = pres & mask; Bp = pres & ~mask
        if int(Ap.sum()) < floor or int(Bp.sum()) < floor:
            return None
        if pre["items"][qi][Ap].sum() < floor or pre["items"][qi][Bp].sum() < floor:
            return None
        diffs.append(abs(float(pre["dens"][qi][Ap].mean()) - float(pre["dens"][qi][Bp].mean())))
    return float(max(diffs))


def _s8class_re(pre, mask, *, groups=None, floor=FLOOR):
    diffs = []
    for qi in range(4):
        v = pre["vec"][qi]; pres = ~np.isnan(v[:, 0]); Ap = pres & mask; Bp = pres & ~mask
        if int(Ap.sum()) < floor or int(Bp.sum()) < floor:
            return None
        if pre["items"][qi][Ap].sum() < floor or pre["items"][qi][Bp].sum() < floor:
            return None
        diffs.append(0.5 * float(np.abs(v[Ap].mean(0) - v[Bp].mean(0)).sum()))
    return float(max(diffs))


def _s9_scalar_pre(pool, which):
    n = len(pool); val = np.full(n, np.nan)
    sa = np.zeros(n); na = np.zeros(n); nsg = np.zeros(n); nng = np.zeros(n)
    for i, r in enumerate(pool):
        if r.L_total < 2:
            continue
        dt = np.diff(r.timestamps); seam = _seam_mask(r.L_total); non = ~seam
        if seam.sum() == 0 or non.sum() == 0:
            continue
        sa[i] = int(seam.sum()); na[i] = int(non.sum())
        nsg[i] = int((dt[seam] > 0).sum()); nng[i] = int((dt[non] > 0).sum())
        if which == "zero":
            val[i] = float((dt[seam] == 0).mean()) - float((dt[non] == 0).mean())
        else:
            same = (r.class_ids[:-1] == r.class_ids[1:])
            val[i] = float(same[seam].mean()) - float(same[non].mean())
    return {"val": val, "sa": sa, "na": na, "nsg": nsg, "nng": nng}


def _s9zero_pre(pool):
    return _s9_scalar_pre(pool, "zero")


def _s9classseam_pre(pool):
    return _s9_scalar_pre(pool, "class")


def _s9_scalar_re(pre, mask, *, groups=None, floor=FLOOR):
    pres = ~np.isnan(pre["val"]); Ap = pres & mask; Bp = pres & ~mask
    if int(Ap.sum()) < floor or int(Bp.sum()) < floor:                # contributing-sequence floor, both arms
        return None
    for k in ("sa", "na", "nsg", "nng"):                              # adjacency + positive-gap count floors
        if pre[k][Ap].sum() < floor or pre[k][Bp].sum() < floor:
            return None
    return abs(float(pre["val"][Ap].mean()) - float(pre["val"][Bp].mean()))


def _s9gap_pre(pool):
    n = len(pool); sa = np.zeros(n); na = np.zeros(n); contrib = np.zeros(n, bool)
    sg, sgo, ng, ngo = [], [], [], []
    for i, r in enumerate(pool):
        if r.L_total < 2:
            continue
        dt = np.diff(r.timestamps); seam = _seam_mask(r.L_total); non = ~seam
        if seam.sum() == 0 or non.sum() == 0:
            continue
        contrib[i] = True; sa[i] = int(seam.sum()); na[i] = int(non.sum())
        for v in dt[seam]:
            if v > 0:
                sg.append(float(v)); sgo.append(i)
        for v in dt[non]:
            if v > 0:
                ng.append(float(v)); ngo.append(i)
    return {"sg": np.asarray(sg, float), "sg_owner": np.asarray(sgo, int), "ng": np.asarray(ng, float),
            "ng_owner": np.asarray(ngo, int), "sa": sa, "na": na, "contrib": contrib}


def _s9gap_re(pre, mask, *, groups=None, floor=FLOOR):
    Ap = pre["contrib"] & mask; Bp = pre["contrib"] & ~mask
    if int(Ap.sum()) < floor or int(Bp.sum()) < floor:
        return None
    for k in ("sa", "na"):
        if pre[k][Ap].sum() < floor or pre[k][Bp].sum() < floor:
            return None
    insA = mask[pre["sg_owner"]]; innA = mask[pre["ng_owner"]]
    sgA, sgB = pre["sg"][insA], pre["sg"][~insA]; ngA, ngB = pre["ng"][innA], pre["ng"][~innA]
    if min(sgA.shape[0], sgB.shape[0], ngA.shape[0], ngB.shape[0]) < floor:   # positive-gap count floor, both arms
        return None
    return float(max(ks_2samp(sgA, ngA).statistic, ks_2samp(sgB, ngB).statistic,
                     ks_2samp(sgA, sgB).statistic, ks_2samp(ngA, ngB).statistic))


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
    "S2_ks": {"precompute": _s2_pre, "recompute": _s2_re, "map_carrying": False,
              "identity": "v2.ks(mean_per_sequence_run_size_ecdf@pool_support)"},
    "S1_density": {"precompute": _s1density_pre,
                   "recompute": (lambda pre, m, *, groups=None, floor=FLOOR: _map_re_scalar(pre, m, groups=groups, floor=floor)),
                   "map_carrying": True, "identity": "v2.cond_maxbin.mean(cluster_density=K/L)@LENGTH_BINS[ref_coarsen]"},
    "S1_tau": {"precompute": _s1tau_pre, "recompute": _s1tau_re, "map_carrying": False,
               "identity": "v2.abs(kendalltau_b(L,K)) source-level"},
    "count_ks": {"precompute": _countks_pre, "recompute": _scalar_ks_re, "map_carrying": False,
                 "identity": "v2.ks_2samp(cluster_count_K)"},
    "length_ks": {"precompute": _lengthks_pre, "recompute": _scalar_ks_re, "map_carrying": False,
                  "identity": "v2.ks_2samp(length_L)"},
    "S8_density": {"precompute": _s8density_pre, "recompute": _s8density_re, "map_carrying": False,
                   "identity": "v2.maxquartile.abs(mean(centered_phase_cluster_density))"},
    "S8_class": {"precompute": _s8class_pre, "recompute": _s8class_re, "map_carrying": False,
                 "identity": "v2.maxquartile.tv(mean(centered_phase_class_vector))"},
    "S9_zero": {"precompute": _s9zero_pre, "recompute": _s9_scalar_re, "map_carrying": False,
                "identity": "v2.abs(mean(seam_minus_nonseam(dt==0_fraction)))"},
    "S9_class": {"precompute": _s9classseam_pre, "recompute": _s9_scalar_re, "map_carrying": False,
                 "identity": "v2.abs(mean(seam_minus_nonseam(same_class_fraction)))"},
    "S9_gap": {"precompute": _s9gap_pre, "recompute": _s9gap_re, "map_carrying": False,
               "identity": "v2.max_ks_2samp(seam/nonseam positive gaps within+cross arm)"},
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

# FIVE explicit identity layers (Pi rev-13 #3). Pi's guidance: hashing REVIEWED FULL MODULE FILES is safer than
# manually chasing helper closure (it transitively covers e.g. phase_spanning_indices, verifier `_runs`,
# coarsen_reference) even though it re-mints on unrelated edits in those files. All SOURCE layers are deterministic
# (env-independent); the dependency layer is the only version-bearing one and never enters a "deterministic" hash.
def _modsrc(mod):
    try:
        return hashlib.sha256(inspect.getsource(mod).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return "SOURCE_UNAVAILABLE"


_MODULE_SRC = {name: _modsrc(mod) for name, mod in (
    ("engine", _sys.modules[__name__]), ("v2_verifier", _v2ver_mod), ("phase0_pilot", _pilot_mod),
    ("map", _map_mod), ("randomization", _rand_mod))}

# (1) SEMANTIC — calling convention + declared estimand identity STRINGS (catches a protocol/estimand re-declaration,
#     not an implementation change).
ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY = canonical_hash({
    "protocol": "recompute(pre, mask, *, groups, floor) keyword-only",
    "estimators": {k: v["identity"] for k, v in ESTIMATORS.items()}})
# (2) IMPLEMENTATION SOURCE — the modules that contain estimator implementations + ALL transitively-called helpers
#     (engine estimators; verifier `_positive_gaps_and_prev_size`/`_runs`/`_s4_contrast`/`_bin_index`/bins; pilot
#     `_seq_components`/`phase_spanning_indices`). Full-file, transitively complete.
ESTIMATOR_IMPL_SOURCE_IDENTITY = canonical_hash({m: _MODULE_SRC[m] for m in ("engine", "v2_verifier", "phase0_pilot")})
# (3) ENGINE CANONICALIZATION / SCHEMA / GATE source — the derive-not-trust, precompute-schema, and gate-kernel code
#     (engine module) + the randomization kernel it calls.
ENGINE_CANON_SCHEMA_GATE_IDENTITY = canonical_hash({m: _MODULE_SRC[m] for m in ("engine", "randomization")})
# (4) MAP BUILDER / APPLY source — the map-builder + reducer modules. A map artifact's `map_identity` binds its
#     OUTPUT + semantic fields, NOT the builder implementation (rev-13 #3); the builder source is bound HERE.
MAP_SOURCE_IDENTITY = canonical_hash({m: _MODULE_SRC[m] for m in ("map", "v2_verifier")})
# (5) DEPENDENCY / ENVIRONMENT — interpreter + library versions + estimand constants. The ONLY version-bearing layer;
#     never enters a deterministic hash — it lives in environment-dependent artifacts.
ESTIMATOR_DEPENDENCY_IDENTITY = canonical_hash({
    "python": platform.python_version(), "numpy": np.__version__,
    "constants": {"C": int(C), "FLOOR": int(FLOOR)}})

# The DETERMINISTIC source-identity bundle bound into config / dev-stable identities (all env-independent).
SOURCE_IDENTITY_BUNDLE = {"estimator_semantic": ESTIMATOR_PROTOCOL_SEMANTIC_IDENTITY,
                          "estimator_impl_source": ESTIMATOR_IMPL_SOURCE_IDENTITY,
                          "engine_canon_schema_gate": ENGINE_CANON_SCHEMA_GATE_IDENTITY,
                          "map_source": MAP_SOURCE_IDENTITY}


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
    for gid in ("G_full_burst_timing", "G_full_class_mark", "G_full_run_size",     # engine-wired full-support groups
                "G_full_length_density", "G_full_phase_seam"):
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


# Explicit canonical profile -> fixture skeleton map (rev-13 #4): fail-closed identity code must NOT map every
# unknown string to MIMIC via a substring heuristic. Every registered SD source is enumerated; unknowns refuse.
_PROFILE_SKELETON = {"mimic_scale_control": "MIMIC", "scid_scale_control": "SCID",
                     "structural_zero_control": "MIMIC", "boundary_short": "MIMIC"}


def _canonicalize_pool(pool, exp_source, exp_id):
    """DERIVE-NOT-TRUST canonicalization (Pi rev-12 #1/#4). The SequenceRecord contract trusts ONLY
    (source, class_ids, timestamps); cluster_ids/L_total/K/positions are DERIVED under exact dt==0 run semantics.
    So the engine REBUILDS each record via the repository's `derive_record` boundary from its trusted fields —
    discarding any caller-supplied derived fields (a record with real positive gaps but K=1/cluster_ids=0 is
    canonicalized, not silently trusted) — and binds the experiment's expected skeleton source. Run ONCE per
    experiment (not once per cell). Malformed TRUSTED fields (bad source, class id out of [0,C), non-finite or
    non-monotone timestamps) refuse via derive_record."""
    if exp_source not in _PROFILE_SKELETON:                        # explicit map; unknown profile refuses (rev-13 #4)
        raise RefusalError(f"{exp_id}: unknown profile {exp_source!r} has no canonical skeleton mapping")
    expect = _PROFILE_SKELETON[exp_source]
    if expect not in REQUIRED_SOURCES:
        raise RefusalError(f"{exp_id}: skeleton {expect!r} not in {REQUIRED_SOURCES}")
    out = []
    for i, r in enumerate(pool):
        for attr in ("source", "class_ids", "timestamps"):
            if not hasattr(r, attr):
                raise RefusalError(f"{exp_id} record {i} missing trusted field {attr}")
        if r.source != expect:
            raise RefusalError(f"{exp_id} record {i} source {r.source!r} != experiment-expected {expect!r}")
        try:
            out.append(derive_record(r.source, r.class_ids, r.timestamps))   # rebuild; discard caller derived fields
        except (MalformedRecord, ValueError) as ex:
            raise RefusalError(f"{exp_id} record {i} malformed trusted fields: {ex}")
    return out


# --- per-estimator precompute SCHEMA (Pi rev-8 #5 / rev-10 #3, exactness rev-12 #2): validate keys / shapes /
#     explicit numeric dtypes / index ranges / legal-value ranges / integer count channels / all-or-nothing absent
#     rows BEFORE any statistic. NaN is legal ONLY as the per-sequence absent sentinel (map scalars, per S4/S6 ROW).
#     Support-ABSENCE (e.g. no positive gaps -> nu=0) is a VALID state that yields NOT_EVALUABLE in the statistic;
#     malformed STRUCTURE is REFUSED. The two must NOT be conflated. Inf is never legal. ---
def _pc_arr(a, name, check):
    if not isinstance(a, np.ndarray):
        raise RefusalError(f"{check} precompute {name} must be an ndarray, got {type(a).__name__}")
    # REAL integer/floating only (rev-13 #1): np.number ADMITS complex, which would be silently cast to float with an
    # imaginary-part loss — refuse it. bool is also excluded.
    if a.dtype == np.bool_ or not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise RefusalError(f"{check} precompute {name} dtype {a.dtype} is not a real integer/floating non-bool type")
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


def _pc_range(a, name, check, lo, hi):
    aa = np.asarray(a, float); fin = np.isfinite(aa)
    if bool(fin.any()) and (bool((aa[fin] < lo - 1e-9).any()) or bool((aa[fin] > hi + 1e-9).any())):
        raise RefusalError(f"{check} precompute {name} has finite value(s) outside [{lo},{hi}]")


def _pc_int_valued(a, name, check):
    aa = np.asarray(a, float); fin = np.isfinite(aa)
    if bool(fin.any()) and bool((np.abs(aa[fin] - np.round(aa[fin])) > 1e-9).any()):
        raise RefusalError(f"{check} precompute {name} must be integer-valued")


def _pc_rows_all_or_nothing(a, name, check):
    """Each ROW is EITHER the absent sentinel (all-NaN) OR complete-finite (no partial-NaN rows; no Inf anywhere)."""
    aa = np.asarray(a, float)
    if aa.size:
        if bool(np.isinf(aa).any()):
            raise RefusalError(f"{check} precompute {name} contains Inf (never legal)")
        nan = np.isnan(aa)
        if not bool((nan.all(axis=1) | (~nan).all(axis=1)).all()):
            raise RefusalError(f"{check} precompute {name} has a partially-absent row (must be all-NaN or all-finite)")


def _pc_finite_iff_positive(sm_list, cnt_list, name, check):
    """CROSS-CHANNEL (rev-13 #2): per bin/sequence the summary `sm` is finite IFF its count channel is > 0 (a NaN
    summary with a positive count, or a finite summary with a zero count, is a structurally impossible state)."""
    for b, (sm, cnt) in enumerate(zip(sm_list, cnt_list)):
        fin = np.isfinite(np.asarray(sm, float)); pos = np.asarray(cnt, float) > 0
        if not bool(np.array_equal(fin, pos)):
            raise RefusalError(f"{check} precompute {name}[{b}]: summary finite must match count>0 exactly")


def _pc_exactly_one_length_bin(bin_arrays, name, check, n):
    """CROSS-CHANNEL (rev-13 #2): LENGTH_BINS partition every canonical sequence, so each of the n sequences is
    present in EXACTLY one length bin (col-0 finite for vectors; finite for scalars)."""
    present = np.zeros(n, int)
    for arr in bin_arrays:
        a = np.asarray(arr, float)
        present += np.isfinite(a if a.ndim == 1 else a[:, 0]).astype(int)
    if not bool((present == 1).all()):
        raise RefusalError(f"{check} precompute {name}: each sequence must be present in EXACTLY one length bin")


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

    def scalar_binlist(x, name, nb, *, count=False, rng=None):     # nb per-bin (n,) scalar arrays; NaN=absent legal
        if not isinstance(x, list) or len(x) != nb:
            raise RefusalError(f"{check} precompute {name} must be a list of {nb} per-bin (n,) arrays")
        for b, arr in enumerate(x):
            a = row1d(arr, f"{name}[{b}]", nan_ok=not count)       # count channels forbid NaN
            if count:
                _pc_nonneg(a, f"{name}[{b}]", check); _pc_int_valued(a, f"{name}[{b}]", check)
            if rng is not None:
                _pc_range(a, f"{name}[{b}]", check, rng[0], rng[1])

    if check == "S3_tau":                                          # EXACT (n,4): [C-D, n0(total), n1(ties_x), n2(ties_y)]
        a = col2d(pre, "components", 4, nan_ok=False)
        _pc_nonneg(a[:, 1:], "components[n0,n1,n2]", check)        # pair counts nonnegative
        _pc_int_valued(a[:, 1:], "components[n0,n1,n2]", check)    # ...and integer-valued (rev-13 #2)
        if bool((a[:, 1] < a[:, 2] - 1e-9).any()) or bool((a[:, 1] < a[:, 3] - 1e-9).any()):
            raise RefusalError("S3_tau: total pairs n0 must be >= ties n1, n2")
        if bool((np.abs(a[:, 0]) > a[:, 1] + 1e-9).any()):        # |C-D| <= n0 (concordance can't exceed total pairs)
            raise RefusalError("S3_tau: |C-D| must be <= n0 (total pairs)")
    elif check == "delta_t_zero_abs":                             # [nz=L-K, na=L-1]; nonneg ints, nz <= na
        a = col2d(pre, "dt0", 2, nan_ok=False); _pc_nonneg(a, "dt0", check); _pc_int_valued(a, "dt0", check)
        if bool((a[:, 0] > a[:, 1] + 1e-9).any()):
            raise RefusalError("delta_t_zero_abs: channel 0 (L-K) must be <= channel 1 (L-1)")
    elif check == "positive_gap_ks":
        keys(pre, ("owner", "inv", "nu"))
        owner, inv, nu = pre["owner"], pre["inv"], pre["nu"]
        if not (_is_int_arr(np.asarray(owner)) and _is_int_arr(np.asarray(inv))):
            raise RefusalError("positive_gap_ks owner/inv must be integer arrays")
        owner, inv = np.asarray(owner), np.asarray(inv)
        if owner.ndim != 1 or inv.ndim != 1 or owner.shape[0] != inv.shape[0]:
            raise RefusalError("positive_gap_ks owner/inv must be equal-length 1-D arrays")
        if isinstance(nu, bool) or not isinstance(nu, (int, np.integer)) or int(nu) < 0:
            raise RefusalError("positive_gap_ks nu must be a non-bool int >= 0")
        nu = int(nu)
        if nu == 0:                                               # VALID support-empty -> statistic NE (not a refusal)
            if owner.size or inv.size:
                raise RefusalError("positive_gap_ks nu=0 requires empty owner/inv")
        else:
            if owner.size == 0 or int(owner.min()) < 0 or int(owner.max()) >= n:
                raise RefusalError("positive_gap_ks owner index out of [0,n) (empty only if nu=0)")
            if int(inv.min()) < 0 or int(inv.max()) >= nu:
                raise RefusalError("positive_gap_ks inv index out of [0,nu)")
            if len(np.unique(inv)) != nu:                         # np.unique(...return_inverse) covers 0..nu-1 exactly
                raise RefusalError("positive_gap_ks inv must cover every support index 0..nu-1")
    elif check == "S2_ks":
        keys(pre, ("M", "nruns"))
        M = _pc_arr(pre["M"], "M", check)
        if M.ndim != 2 or M.shape[0] != n or M.shape[1] < 1:
            raise RefusalError(f"S2_ks precompute M shape {M.shape} invalid (want (n,>=1))")
        _pc_all_finite(M, "M", check); _pc_range(M, "M", check, 0.0, 1.0)   # per-seq ECDF fractions in [0,1]
        nr = row1d(pre["nruns"], "nruns", nan_ok=False); _pc_nonneg(nr, "nruns", check); _pc_int_valued(nr, "nruns", check)
    elif check in ("count_ks", "length_ks"):                      # two-sample KS: (n,|support|) 0/1 ECDF-indicator basis
        keys(pre, ("M",))
        M = _pc_arr(pre["M"], "M", check)
        if M.ndim != 2 or M.shape[0] != n or M.shape[1] < 1:
            raise RefusalError(f"{check} precompute M shape {M.shape} invalid (want (n,>=1))")
        _pc_all_finite(M, "M", check); _pc_range(M, "M", check, 0.0, 1.0); _pc_int_valued(M, "M", check)
    elif check == "S1_tau":                                        # source-level [L, K] per sequence; K in [1, L]
        a = col2d(pre, "LK", 2, nan_ok=False); _pc_int_valued(a, "LK", check)
        if bool((a[:, 0] < 1).any()) or bool((a[:, 1] < 1).any()):
            raise RefusalError("S1_tau: L and K must be >= 1")
        if bool((a[:, 1] > a[:, 0] + 1e-9).any()):
            raise RefusalError("S1_tau: K must be <= L")
    elif check == "S8_density":                                    # 4 per-quartile centered densities (NaN=absent) + item counts
        keys(pre, ("dens", "items"))
        scalar_binlist(pre["dens"], "dens", 4)                     # centered -> any sign; NaN=absent
        scalar_binlist(pre["items"], "items", 4, count=True)
        _pc_finite_iff_positive(pre["dens"], pre["items"], "dens/items", check)
    elif check == "S8_class":                                      # 4 per-quartile centered class vectors + item counts
        keys(pre, ("vec", "items"))
        if not isinstance(pre["vec"], list) or len(pre["vec"]) != 4:
            raise RefusalError("S8_class vec must be a list of 4 per-quartile arrays")
        for qi, arr in enumerate(pre["vec"]):
            v = _pc_arr(arr, f"vec[{qi}]", check)
            if v.ndim != 2 or v.shape[0] != n or v.shape[1] != C:
                raise RefusalError(f"S8_class vec[{qi}] shape {v.shape} != ({n},{C})")
            _pc_rows_all_or_nothing(v, f"vec[{qi}]", check)        # absent(all-NaN) or complete-finite (centered; any sign)
        scalar_binlist(pre["items"], "items", 4, count=True)
        _pc_finite_iff_positive([v[:, 0] for v in pre["vec"]], pre["items"], "vec/items", check)
    elif check in ("S9_zero", "S9_class"):                         # per-seq centered scalar (NaN=noncontrib) + count channels
        keys(pre, ("val", "sa", "na", "nsg", "nng"))
        row1d(pre["val"], "val", nan_ok=True)                      # centered -> any sign; NaN=non-contributing
        for k in ("sa", "na", "nsg", "nng"):
            kk = row1d(pre[k], k, nan_ok=False); _pc_nonneg(kk, k, check); _pc_int_valued(kk, k, check)
        _pc_finite_iff_positive([pre["val"]], [pre["sa"]], "val/sa", check)   # contributes IFF has seam adjacencies
    elif check == "S9_gap":                                        # seam/nonseam positive gaps w/ owners + count channels
        keys(pre, ("sg", "sg_owner", "ng", "ng_owner", "sa", "na", "contrib"))
        for vk, ok in (("sg", "sg_owner"), ("ng", "ng_owner")):
            v = _pc_arr(pre[vk], vk, check); ow = pre[ok]
            if v.ndim != 1 or not np.issubdtype(np.asarray(ow).dtype, np.integer) or np.asarray(ow).shape != v.shape:
                raise RefusalError(f"S9_gap {vk}/{ok} must be equal-length 1-D (values / int owners)")
            _pc_all_finite(v, vk, check)
            if v.size and bool((v <= 0).any()):
                raise RefusalError(f"S9_gap {vk} must be strictly positive gaps")
            if v.size and (int(np.asarray(ow).min()) < 0 or int(np.asarray(ow).max()) >= n):
                raise RefusalError(f"S9_gap {ok} out of [0,n)")
        for k in ("sa", "na"):
            kk = row1d(pre[k], k, nan_ok=False); _pc_nonneg(kk, k, check); _pc_int_valued(kk, k, check)
        cb = np.asarray(pre["contrib"])
        if cb.dtype != np.bool_ or cb.ndim != 1 or cb.shape[0] != n:
            raise RefusalError(f"S9_gap contrib must be a ({n},) bool array")
    elif check == "S3_loggap":
        keys(pre, ("sm", "sp"))
        scalar_binlist(pre["sm"], "sm", nbC)                      # mean log gap: unrestricted finite-or-NaN
        scalar_binlist(pre["sp"], "sp", nbC, count=True)          # pair counts: finite nonneg integer
        _pc_finite_iff_positive(pre["sm"], pre["sp"], "sm/sp", check)   # cross-channel: sm finite IFF sp>0
    elif check == "S4_abs":                                        # per row: absent(all-NaN) OR [contrast in[-1,1], >0 ints]
        a = _pc_arr(pre, "s4", check)
        if a.ndim != 2 or a.shape[0] != n or a.shape[1] != 3:
            raise RefusalError(f"S4_abs precompute shape {a.shape} != ({n},3)")
        _pc_rows_all_or_nothing(a, "s4", check)
        _pc_range(a[:, 0], "s4[contrast]", check, -1.0, 1.0)
        _pc_int_valued(a[:, 1:], "s4[pair-counts]", check)
        pres = np.isfinite(a[:, 0])                               # present rows require BOTH pair counts strictly > 0
        if bool(pres.any()) and (bool((a[pres, 1] <= 0).any()) or bool((a[pres, 2] <= 0).any())):
            raise RefusalError("S4_abs: present rows require same_pairs>0 AND adj_pairs>0")
    elif check == "class_tv":                                      # per-class counts: nonneg integer
        a = col2d(pre, "class_tv", C, nan_ok=False); _pc_nonneg(a, "class_tv", check); _pc_int_valued(a, "class_tv", check)
    elif check == "occupancy_abs":
        _pc_range(row1d(pre, "occ", nan_ok=False), "occ", check, 0.0, 1.0)
    elif check in ("S5_abs", "S1_density"):                        # per-bin scalar in [0,1] (occupancy / K-L density); NaN=absent
        keys(pre, ("sm",)); scalar_binlist(pre["sm"], "sm", nbL, rng=(0.0, 1.0))
        _pc_exactly_one_length_bin(pre["sm"], "sm", check, n)     # each sequence present in exactly one length bin
    elif check == "S6_tv":                                         # per-bin class-fraction rows: absent OR sum-to-1 nonneg
        keys(pre, ("vec",))
        if not isinstance(pre["vec"], list) or len(pre["vec"]) != nbL:
            raise RefusalError(f"S6_tv precompute vec must be a list of {nbL} per-bin arrays")
        for b, arr in enumerate(pre["vec"]):
            a = _pc_arr(arr, f"vec[{b}]", check)
            if a.ndim != 2 or a.shape[0] != n or a.shape[1] != C:
                raise RefusalError(f"S6_tv precompute vec[{b}] shape {a.shape} != ({n},{C})")
            _pc_rows_all_or_nothing(a, f"vec[{b}]", check); _pc_nonneg(a, f"vec[{b}]", check)
            rs = np.asarray(a, float).sum(axis=1); fin = np.isfinite(rs)
            if bool(fin.any()) and bool((np.abs(rs[fin] - 1.0) > 1e-6).any()):
                raise RefusalError(f"S6_tv vec[{b}] present rows must be class fractions summing to 1")
        _pc_exactly_one_length_bin(pre["vec"], "vec", check, n)   # each sequence present in exactly one length bin
    elif check == "S7_abs":                                        # per-bin distinct-class-frac in [0,1]; cc = counts
        keys(pre, ("sm", "cc"))
        scalar_binlist(pre["sm"], "sm", nbC, rng=(0.0, 1.0))
        scalar_binlist(pre["cc"], "cc", nbC, count=True)
        _pc_finite_iff_positive(pre["sm"], pre["cc"], "sm/cc", check)   # cross-channel: sm finite IFF cc>0
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
    # Pi rev-12 #1/#4: canonicalize each experiment's pool ONCE (derive-not-trust), then reuse for every cell.
    for e, ex in experiments.items():
        ex["pool"] = _canonicalize_pool(ex["pool"], canon["experiments"][e]["source"], e)
    for cc in canon["cells"]:
        est = ESTIMATORS[cc["check"]]; pool = experiments[cc["exp"]]["pool"]
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
              # Pi rev-12 #3 / rev-13 #3: the DETERMINISTIC stable identity carries the four SOURCE identity layers
              # only; the environment-dependent dependency identity is recorded in the full dev_config, not the hash.
              "source_identities": SOURCE_IDENTITY_BUNDLE,
              "namespace": (nss[0] if nss else None),
              "seed_law": "caller-supplied dev seed; assignments = numpy default_rng(seed) per-stratum permutation"}
    res["dev_config"] = {**stable, "seed": seed, "estimator_dependency_identity": ESTIMATOR_DEPENDENCY_IDENTITY}
    res["dev_config_stable_identity"] = canonical_hash(stable)     # deterministic (env-independent)
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
    from clinical_jepa.eval.oracle_realism_v2_verifier import (s1, s2, s3, s4, s5, s6, s7, s8, s9,
                                                               marginal_route_checks)
    cc, rr = draw("consC", 6000), draw("consR", 6000); poolc = list(cc) + list(rr)
    obs = np.array([True] * len(cc) + [False] * len(rr))
    v2 = {**s1(cc, rr), **s2(cc, rr), **s3(cc, rr), **s4(cc, rr), **s5(cc, rr), **s6(cc, rr), **s7(cc, rr),
          **s8(cc, rr), **s9(cc, rr), **marginal_route_checks(cc, rr)}
    for chk in ("delta_t_zero_abs", "positive_gap_ks", "S2_ks", "S1_density", "S1_tau", "count_ks", "length_ks",
                "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap",
                "S4_abs", "class_tv", "occupancy_abs", "S3_loggap", "S5_abs", "S6_tv", "S7_abs"):
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
    lk200 = np.column_stack([np.arange(1, 201, dtype=float), (np.arange(200) // 2 + 1).astype(float)])  # varying (L,K)
    floor_pre = {"occupancy_abs": (occ100, None), "S4_abs": (s4_100, None),
                 "S2_ks": ({"M": np.ones((200, 1)), "nruns": np.ones(200, int)}, None),
                 "count_ks": ({"M": np.ones((200, 1))}, None), "length_ks": ({"M": np.ones((200, 1))}, None),
                 "S1_tau": (lk200, None), "S1_density": ({"sm": [sm100]}, [[0]]),
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
    for chk in ("S3_tau", "delta_t_zero_abs", "positive_gap_ks", "S2_ks", "S1_density", "S1_tau", "count_ks", "length_ks",
                "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap",
                "S4_abs", "class_tv", "occupancy_abs", "S3_loggap", "S5_abs", "S6_tv", "S7_abs"):
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

    # rev-12 #2 exactness — cases Pi reproduced as WRONGLY ACCEPTED must now refuse:
    tau5 = np.hstack([_tau_pre(pool), np.zeros((npool, 1))])                       # (n,5), not exact (n,4)
    occ_obj = np.asarray(occ, dtype=object)                                        # object dtype (not numeric schema)
    occ_hi = _corrupt(occ, lambda b: b.__setitem__(0, 1.5))                        # occupancy outside [0,1]
    s4_partial = np.full((npool, 3), np.nan); s4_partial[0] = [0.5, np.nan, np.nan]   # partially-absent row
    s4_hi = np.full((npool, 3), np.nan); s4_hi[0] = [2.0, 1.0, 1.0]                # contrast outside [-1,1]
    s6_partial = {"vec": [np.full((npool, C), np.nan) for _ in range(len(LENGTH_BINS))]}
    s6_partial["vec"][0][0, 0] = 1.0                                              # row 0: one finite + rest NaN
    # rev-13 #1/#2 — real-dtype + cross-channel cases Pi reproduced as WRONGLY ACCEPTED:
    s4_zero_pair = np.full((npool, 3), np.nan); s4_zero_pair[0] = [0.5, 0.0, 1.0]                 # present but pair count 0
    exact_cases = {
        "s3_tau_extra_col": ("S3_tau", tau5),
        "occupancy_object_dtype": ("occupancy_abs", occ_obj),
        "occupancy_out_of_range": ("occupancy_abs", occ_hi),
        "s4_partial_nan_row": ("S4_abs", s4_partial),
        "s4_contrast_out_of_range": ("S4_abs", s4_hi),
        "s6_partial_nan_row": ("S6_tv", s6_partial),
        "complex_dtype": ("occupancy_abs", occ.astype(complex)),                                  # rev-13 #1: no complex
        "s3_tau_fractional_count": ("S3_tau", _corrupt(_tau_pre(pool), lambda b: b.__setitem__((0, 1), b[0, 1] + 0.5))),
        "s3_loggap_sm_nan_sp_positive": ("S3_loggap", _corrupt(lgp, lambda b: (b["sm"][0].__setitem__(0, np.nan),
                                                                               b["sp"][0].__setitem__(0, 1.0)))),
        "s7_sm_nan_cc_positive": ("S7_abs", _corrupt(s7p, lambda b: (b["sm"][0].__setitem__(0, np.nan),
                                                                     b["cc"][0].__setitem__(0, 1.0)))),
        "s4_present_zero_pair": ("S4_abs", s4_zero_pair),
    }
    for label, (chk, bad) in exact_cases.items():
        if not _srefused(chk, bad):
            errs.append(f"precompute schema did NOT refuse {label} (rev-12/rev-13)")
    # rev-13 #4 — _canonicalize_pool refuses an unknown profile (no MIMIC fallback):
    try:
        _canonicalize_pool(pool[:1], "NOT_A_PROFILE", "test"); errs.append("canonicalize did NOT refuse unknown profile")
    except RefusalError:
        pass
    # POSITIVE — valid support-empty positive_gap_ks (no positive gaps) is a NE state, NOT a schema refusal:
    try:
        _validate_precompute({"owner": np.array([], int), "inv": np.array([], int), "nu": 0}, "positive_gap_ks", npool)
    except RefusalError as ex:
        errs.append(f"positive_gap_ks nu=0 support-empty wrongly REFUSED (should be NE): {ex}")

    # VALID DEGENERATE POOLS (Pi rev-12 #2): support-absence must VALIDATE (recompute may be NE) — never a refusal.
    def _mk(source, cls, ts):
        return derive_record(source, np.asarray(cls, int), np.asarray(ts, float))
    allL1 = [_mk("MIMIC", [k % C], [float(k)]) for k in range(120)]                # every sequence length 1
    onecl = [_mk("MIMIC", list(range(min(6, C))), [5.0] * min(6, C)) for _ in range(120)]  # all dt==0 -> one cluster
    ALL_CHK = ("S3_tau", "delta_t_zero_abs", "positive_gap_ks", "S2_ks", "S1_density", "S1_tau", "count_ks", "length_ks",
               "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap",
               "S4_abs", "class_tv", "occupancy_abs", "S3_loggap", "S5_abs", "S6_tv", "S7_abs")
    for label, dpool in (("all_L1", allL1), ("one_cluster", onecl)):
        for chk in ALL_CHK:
            try:
                _validate_precompute(ESTIMATORS[chk]["precompute"](dpool), chk, len(dpool))
            except RefusalError as ex:
                errs.append(f"degenerate {label}: schema wrongly REFUSED valid {chk}: {ex}")

    # DERIVE-NOT-TRUST canonicalization (Pi rev-12 #1/#4): a valid pool canonicalizes; corrupt TRUSTED fields refuse;
    # corrupt DERIVED fields are REBUILT to canonical (not silently trusted). exp "mimic_scale_control" -> "MIMIC".
    import dataclasses as _dc
    canon_pool = _canonicalize_pool(pool, "mimic_scale_control", "test")
    if len(canon_pool) != len(pool):
        errs.append("canonicalization changed pool length")
    g0 = next((r for r in pool if r.K > 1 and r.L_total > 1), pool[0])
    truth = derive_record(g0.source, g0.class_ids, g0.timestamps)      # canonical rebuild of the trusted fields
    if truth.K <= 1:
        errs.append("derive-not-trust test needs a multi-cluster record; none found")
    # (a) corrupt DERIVED fields (Pi's repro: positive gaps but K=1/cluster_ids=0) -> REBUILT to canonical, not trusted
    corrupt_derived = _dc.replace(g0, K=1, cluster_ids=np.zeros(g0.L_total, int))
    rebuilt = _canonicalize_pool([corrupt_derived], "mimic_scale_control", "test")[0]
    if rebuilt.K != truth.K or not np.array_equal(rebuilt.cluster_ids, truth.cluster_ids):
        errs.append("derive-not-trust did NOT rebuild corrupt cluster_ids/K to canonical")
    # (b) corrupt TRUSTED fields (and source mismatch) REFUSE
    raw_bad = {
        "nonfinite_timestamp": _dc.replace(g0, timestamps=np.where(ar(g0.L_total) == 0, np.inf, g0.timestamps).astype(float)),
        "class_id_out_of_range": _dc.replace(g0, class_ids=np.where(ar(g0.L_total) == 0, C, g0.class_ids).astype(int)),
        "nonmonotone_timestamps": _dc.replace(g0, timestamps=np.arange(g0.L_total, 0, -1, dtype=float)),
    }
    for label, rec in raw_bad.items():
        try:
            _canonicalize_pool([rec], "mimic_scale_control", "test"); errs.append(f"canonicalization did NOT refuse {label}")
        except RefusalError:
            pass
    try:                                                              # experiment-source binding
        _canonicalize_pool([g0], "scid_scale_control", "test"); errs.append("source mismatch NOT refused")
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
    if set(CANONICAL_GROUPS) != {"G_full_burst_timing", "G_full_class_mark", "G_full_run_size",
                                 "G_full_length_density", "G_full_phase_seam"}:
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
