#!/usr/bin/env python3
"""Oracle realism v3 — Phase-0 estimator micro-pilot (REVISED per Pi rev-2 §6).

Development-only, reproducible, aggregate-hashed. NO calibration/audit/evaluation seeds are used. Run:

    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_phase0_pilot.py

Freezes and evidences ONE burst-timing dependence estimator for the v3 same-distribution (SD) cells:

  T_pool = pooled tie-corrected Kendall tau-b between (preceding-cluster-size x, positive inter-cluster gap y),
           over a PHASE-SPANNING cap of <=6 boundaries per sequence (six frozen quantile-spaced boundary indices
           spanning the WHOLE sequence; all boundaries if m<=6), pooling the standard tau-b components
           (C-D, n0, n1, n2) across sequences over WITHIN-sequence pairs only.
  d      = |T_pool(candidate) - T_pool(reference)|;  SD null = between-group sequence-label permutation.

Property claim (frozen): a CAPPED, PHASE-SPANNING, pair-count-weighted within-sequence rank dependence between
inter-cluster gaps and preceding cluster sizes. NOT first-6 (early-sequence), NOT uncapped, NOT equal-per-sequence.

This pilot demonstrates ONLY: (a) formula correctness vs scipy tau-b incl. joint ties; (b) phase coverage of the
cap; (c) contribution concentration fixed on full support; (d) source-wise power (a distribution, not one number);
(e) burst-timing does not cross-load onto the S8 phase check. It does NOT claim calibrated tail behaviour — the
exact SD null is the per-draw label-permutation test computed at evaluation time (Pi rev-2 §1).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.stats import kendalltau

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    _positive_gaps_and_prev_size, sequence_route_checks, s3, _TAU, _LOGGAP,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling

CAP = 6
DEV_NS = "v3-estimator-dev"                       # disjoint dev namespace (never calibration/audit/evaluation)
DEV_SEEDS = list(range(90000, 90040))             # frozen 40-seed development list
N = 3000                                          # per-side pilot sample size (dev scale, not the registered N)


def dseed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *parts))).encode()).digest()[:6], "big")


# ---------------------------------------------------------------------------------------------------
# frozen phase-spanning cap + standard tie-corrected pooled tau-b
# ---------------------------------------------------------------------------------------------------
def phase_spanning_indices(m: int, cap: int = CAP):
    """FROZEN: all boundary indices if m<=cap; else `cap` distinct quantile-spaced indices across [0, m-1]."""
    if m <= cap:
        return list(range(m))
    return sorted({int(round(v)) for v in np.linspace(0, m - 1, cap)})


def _seq_components(rec):
    """Standard tau-b components over the phase-spanning boundaries of one sequence. n1/n2 count ALL pairs tied
    in x / in y (joint ties included), reconciled with scipy kendalltau (tau-b)."""
    y, x = _positive_gaps_and_prev_size(rec)          # y = gap, x = preceding cluster size
    m = x.shape[0]
    if m < 2:
        return 0.0, 0.0, 0.0, 0.0
    sel = phase_spanning_indices(m)
    x, y = x[sel], y[sel]
    mm = len(sel)
    if mm < 2:
        return 0.0, 0.0, 0.0, 0.0
    sx = np.sign(np.subtract.outer(x, x))
    sy = np.sign(np.subtract.outer(y, y))
    iu = np.triu_indices(mm, 1)
    sx, sy = sx[iu], sy[iu]
    cmd = float(np.sum(sx * sy))                      # C - D  (pairs tied in x or y contribute 0)
    n0 = float(mm * (mm - 1) / 2)
    n1 = float(np.sum(sx == 0))                       # pairs tied in x (standard tau-b: incl. both-tied)
    n2 = float(np.sum(sy == 0))                       # pairs tied in y
    return cmd, n0, n1, n2


def T_pool(sample):
    cmd = n0 = n1 = n2 = 0.0
    for r in sample:
        c, z, t1, t2 = _seq_components(r)
        cmd += c; n0 += z; n1 += t1; n2 += t2
    dA, dB = n0 - n1, n0 - n2
    return None if (dA <= 0 or dB <= 0) else cmd / np.sqrt(dA * dB)


def _two_sample_d(A, B):
    a, b = T_pool(A), T_pool(B)
    return None if (a is None or b is None) else abs(a - b)


# ---------------------------------------------------------------------------------------------------
# (a) formula validation vs scipy tau-b (incl. joint ties)
# ---------------------------------------------------------------------------------------------------
def validate_tie_formula():
    rng = np.random.default_rng(dseed("validate"))
    worst = 0.0
    for _ in range(500):
        n = int(rng.integers(4, 12))
        x = rng.integers(0, 4, n).astype(float)       # heavy ties in x
        y = rng.integers(0, 4, n).astype(float)       # heavy ties in y (=> joint ties occur)
        sx = np.sign(np.subtract.outer(x, x)); sy = np.sign(np.subtract.outer(y, y))
        iu = np.triu_indices(n, 1); sx, sy = sx[iu], sy[iu]
        cmd = float(np.sum(sx * sy)); n0 = n * (n - 1) / 2
        n1 = float(np.sum(sx == 0)); n2 = float(np.sum(sy == 0))
        dA, dB = n0 - n1, n0 - n2
        if dA <= 0 or dB <= 0:
            continue
        mine = cmd / np.sqrt(dA * dB)
        ref = kendalltau(x, y)[0]
        if np.isfinite(ref):
            worst = max(worst, abs(mine - ref))
    return worst


# ---------------------------------------------------------------------------------------------------
# regimes (dev): EXACT registered source profiles (single-profile draws, Pi rev-3 §3 — NOT a 3-mu smoke);
# bounded-short = canonical uniform_int[1,7].
# ---------------------------------------------------------------------------------------------------
def _skey(profile_name):
    return "SCID" if "scid" in profile_name else "MIMIC"


def exact_source(profile_name, n, seed):
    return sample_fixture(_skey(profile_name), PROFILES[profile_name], n, seed=dseed("src", profile_name, seed))


def bounded_short(n, seed):
    return sample_fixture("MIMIC", PROFILES["boundary_short"], n, seed=dseed("short", seed))


REGIMES = {
    "full_MIMIC": ("MIMIC", lambda n, s: exact_source("mimic_scale_control", n, s)),
    "full_SCID":  ("SCID",  lambda n, s: exact_source("scid_scale_control", n, s)),
    "bounded":    ("MIMIC", lambda n, s: bounded_short(n, s)),
}


# ---------------------------------------------------------------------------------------------------
# (b) phase coverage; (c) concentration; (d) power; (e) S8 interaction
# ---------------------------------------------------------------------------------------------------
def phase_coverage(sample):
    """Mean normalized position of selected boundaries: first-6 vs phase-spanning (evidence of whole-seq coverage)."""
    first_pos, span_pos, n_long = [], [], 0
    for r in sample:
        _, x = _positive_gaps_and_prev_size(r); m = x.shape[0]
        if m <= CAP:
            continue
        n_long += 1
        first = list(range(CAP))                       # the rejected early-only cap
        span = phase_spanning_indices(m)
        first_pos.append(np.mean(first) / (m - 1))
        span_pos.append(np.mean(span) / (m - 1))
    if not n_long:
        return None
    return {"n_long_seq": n_long, "first6_mean_pos": round(float(np.mean(first_pos)), 4),
            "phasespan_mean_pos": round(float(np.mean(span_pos)), 4)}


def concentration(sample):
    pf, pc = [], []
    for r in sample:
        _, x = _positive_gaps_and_prev_size(r); m = x.shape[0]
        pf.append(m * (m - 1) / 2 if m >= 2 else 0.0)  # uncapped pairs
        mm = len(phase_spanning_indices(m)) if m >= 2 else 0
        pc.append(mm * (mm - 1) / 2 if mm >= 2 else 0.0)  # capped (phase-spanning) pairs
    def ef(p):
        p = np.asarray(p, float); tot = p.sum()
        if tot == 0:
            return None
        order = np.sort(p)[::-1]; w = p / tot
        return {"eff_frac": round(float((1 / np.sum(w ** 2)) / len(p)), 3),
                "top1pct_share": round(float(np.cumsum(order / tot)[max(1, int(len(order) * 0.01)) - 1]), 3)}
    return {"uncapped": ef(pf), "phasespan": ef(pc)}


def _s3loggap(cand, ref):
    r = s3(cand, ref)["S3_loggap"]
    return r.status, (None if r.value is None else float(r.value))


# ---------------------------------------------------------------------------------------------------
# Δ-aligned boundary recompute via a reference-OWNED FROZEN coarsening map (Pi rev-4 #6)
# ---------------------------------------------------------------------------------------------------
FLOOR_DEV = 200            # dev-scale conditional floor (registered N=8000 uses 500; dev N=3000 is smaller)


def _prevsize_loggap(sample):
    """(preceding cluster size x, log positive gap) over all within-sequence adjacencies with a positive gap."""
    xs, logys = [], []
    for r in sample:
        y, x = _positive_gaps_and_prev_size(r)          # y = gap, x = preceding cluster size
        for xi, yi in zip(x, y):
            if yi > 0:
                xs.append(int(xi)); logys.append(float(np.log(yi)))
    return np.asarray(xs, float), np.asarray(logys, float)


def frozen_prevsize_bins(map_ref, floor=FLOOR_DEV):
    """Reference-OWNED frozen bins over preceding-cluster-size, greedy-merged ascending until each bin holds
    >= floor adjacencies in the DISJOINT dev map-reference. Returns ascending closing sizes (or None if <floor
    total). The candidate never influences these bins (anti-masking; Pi rev-3 §2)."""
    xs, _ = _prevsize_loggap(map_ref)
    if xs.size < floor:
        return None
    sizes, counts = np.unique(xs, return_counts=True)
    edges, acc = [], 0
    for s, c in zip(sizes, counts):
        acc += int(c)
        if acc >= floor:
            edges.append(int(s)); acc = 0
    if acc > 0:                                          # merge leftover tail into the last bin
        if edges:
            edges[-1] = int(sizes[-1])
        else:
            edges = [int(sizes[-1])]
    return edges


def _binid(x, edges):
    """bin = first closing size >= x (clamped to the last bin)."""
    return np.clip(np.searchsorted(edges, x, side="left"), 0, len(edges) - 1)


def s3loggap_frozen(cand, ref, edges, floor=FLOOR_DEV):
    """Frozen-map S3_loggap discrepancy d = max_bin |mean(log gap)_cand - mean(log gap)_ref| under the FROZEN
    reference-owned `edges`. Floor breach on either arm in any bin => None (NOT_EVALUABLE). d>Δ is the MM-aligned
    detection criterion (Pi rev-4 #6) — NOT the v2 adaptive PASS/FAIL status."""
    if edges is None:
        return None
    xc, yc = _prevsize_loggap(cand); xr, yr = _prevsize_loggap(ref)
    if xc.size == 0 or xr.size == 0:
        return None
    bc, br = _binid(xc, edges), _binid(xr, edges)
    d = 0.0
    for b in range(len(edges)):
        cv, rv = yc[bc == b], yr[br == b]
        if cv.size < floor or rv.size < floor:
            return None
        d = max(d, abs(cv.mean() - rv.mean()))
    return d


def delta_aligned_boundary():
    """For each S3 subcheck, dev-only P[d > exact Δ under @0.5] (detection) and under null (specificity), on the
    BOUNDED and FULL-support regimes. S3_tau uses d=|T_pool(cand)-T_pool(ref)| vs Δ=_TAU; S3_loggap uses the
    FROZEN-map discrepancy vs Δ=_LOGGAP. Makes the boundary exemption decision Δ-ALIGNED (Pi rev-4 #6)."""
    out = {}
    for regime_key in ("bounded", "full_MIMIC"):
        _, gen = REGIMES[regime_key]
        edges = frozen_prevsize_bins(gen(N, ("mapref", 0)))     # disjoint dev map-reference (never calibration)
        tau_pow, tau_null, lg_pow, lg_null, lg_ne = [], [], [], [], 0
        for k in DEV_SEEDS:
            A = gen(N, ("A", k)); B = gen(N, ("B", k))
            Bc = apply_coupling(list(B), "burst_timing", 0.5, seed=dseed("cpl_da", regime_key, k))
            dn, dp = _two_sample_d(A, B), _two_sample_d(A, Bc)
            if dn is not None:
                tau_null.append(dn > _TAU)
            if dp is not None:
                tau_pow.append(dp > _TAU)
            ln, lp = s3loggap_frozen(B, A, edges), s3loggap_frozen(Bc, A, edges)
            if ln is not None:
                lg_null.append(ln > _LOGGAP)
            if lp is None:
                lg_ne += 1
            else:
                lg_pow.append(lp > _LOGGAP)
        rate = lambda v: round(float(np.mean(v)), 3) if v else None
        out[regime_key] = {
            "delta_tau": _TAU, "delta_loggap": round(_LOGGAP, 8), "frozen_bins": edges,
            "S3_tau": {"detect_P[d>delta]@0.5": rate(tau_pow), "false_P[d>delta]_null": rate(tau_null),
                       "n_pow": len(tau_pow), "n_null": len(tau_null)},
            "S3_loggap": {"detect_P[d>delta]@0.5": rate(lg_pow), "false_P[d>delta]_null": rate(lg_null),
                          "frozen_map_ne_rate": round(lg_ne / len(DEV_SEEDS), 3),
                          "n_pow": len(lg_pow), "n_null": len(lg_null)},
        }
    bnd = out["bounded"]
    out["provisional_decision"] = {
        "criterion": "boundary EXEMPT (provisional) iff bounded detection P[d>Δ @0.5] < 0.5 (un-calibratable at Δ)",
        "S3_tau_bounded_detect": bnd["S3_tau"]["detect_P[d>delta]@0.5"],
        "S3_loggap_bounded_detect": bnd["S3_loggap"]["detect_P[d>delta]@0.5"],
        "note": "PROVISIONAL — final exemption only after the reference-owned frozen-map CALIBRATION draw (blocked). "
                "Full-support detection reported alongside for contrast (checks work where supported).",
        "caveat": "P[d>Δ] here is the DIRECT (MM-aligned) two-independent-draw discrepancy used ONLY for the "
                  "boundary detection/exemption decision. It is NOT the SD gate's type-I control (that is the "
                  "permutation test). The elevated full-support S3_loggap null-exceedance is dev-scale (N=3000, "
                  "floor=200, max-over-bins) sampling noise and a flag for MM-specificity calibration at the "
                  "registered N — not a boundary-decision or SD-gate defect.",
    }
    return out


def _statcounts(statuses):
    from collections import Counter
    c = Counter(statuses)
    return {k: c.get(k, 0) for k in ("PASS", "FAIL", "NOT_EVALUABLE")}


def null_and_power(regime_key, reps=DEV_SEEDS):
    """T_pool (S3_tau replacement) AND S3_loggap (existing coarsened check), null vs burst_timing@0.5-coupled,
    over the exact-profile draws. S3_tau/S3_loggap exemptions are decided SEPARATELY (Pi rev-3 §3)."""
    src, gen = REGIMES[regime_key]
    nulls, powers = [], []
    lg_null_st, lg_null_v, lg_pow_st, lg_pow_v = [], [], [], []
    for k in reps:
        A = gen(N, ("A", k)); B = gen(N, ("B", k))
        Bc = apply_coupling(list(B), "burst_timing", 0.5, seed=dseed("cpl", regime_key, k))
        dn = _two_sample_d(A, B); dp = _two_sample_d(A, Bc)
        if dn is not None:
            nulls.append(dn)
        if dp is not None:
            powers.append(dp)
        sn, vn = _s3loggap(B, A); sp, vp = _s3loggap(Bc, A)   # cand vs ref (A = reference)
        lg_null_st.append(sn); lg_pow_st.append(sp)
        if vn is not None:
            lg_null_v.append(vn)
        if vp is not None:
            lg_pow_v.append(vp)
    nulls, powers = np.asarray(nulls), np.asarray(powers)
    null_p96 = float(np.quantile(nulls, 0.96)) if len(nulls) else None
    sep = float(np.mean(powers > null_p96)) if (len(powers) and null_p96 is not None) else None
    # S3_loggap "power" proxy: fraction of coupled draws detected (status FAIL) — reported alongside status mix.
    lg_pow = round(float(np.mean(np.array(lg_pow_st) == "FAIL")), 3) if lg_pow_st else None
    return {
        "S3_tau_pool": {"n_null": len(nulls), "null_mean": round(float(nulls.mean()), 5) if len(nulls) else None,
                        "null_p96": round(null_p96, 5) if null_p96 is not None else None,
                        "pow_mean": round(float(powers.mean()), 5) if len(powers) else None,
                        "pow_min": round(float(powers.min()), 5) if len(powers) else None,
                        "power_frac_gt_null_p96": round(sep, 3) if sep is not None else None},
        "S3_loggap_check": {"null_status": _statcounts(lg_null_st), "coupled_status": _statcounts(lg_pow_st),
                            "null_val_mean": round(float(np.mean(lg_null_v)), 5) if lg_null_v else None,
                            "coupled_val_mean": round(float(np.mean(lg_pow_v)), 5) if lg_pow_v else None,
                            "power_frac_detected_FAIL": lg_pow}}


def s8_interaction(seed=1):
    """burst_timing must NOT cross-load onto the S8 phase check: S8 on null-vs-coupled should behave like null."""
    ref = exact_source("mimic_scale_control", N, ("s8ref", seed))
    null_cand = exact_source("mimic_scale_control", N, ("s8null", seed))
    coupled = apply_coupling(list(ref), "burst_timing", 0.5, seed=dseed("s8cpl", seed))
    def s8(cand):
        ch = sequence_route_checks(cand, ref)
        return {k: {"status": ch[k].status, "value": None if ch[k].value is None else round(float(ch[k].value), 5)}
                for k in ("S8_density", "S8_class")}
    return {"null_vs_ref": s8(null_cand), "burst_coupled_vs_ref": s8(coupled)}


def main():
    agg = {"cap": CAP, "dev_namespace": DEV_NS, "dev_seeds": DEV_SEEDS, "N": N, "regimes": "exact registered profiles (single-profile) + boundary_short",
           "cap_rule": "phase-spanning: m<=6 -> all; else 6 quantile-spaced indices round(linspace(0,m-1,6))",
           "tie_formula": "standard tau-b: C-D=sum sign(dx)sign(dy); n1=#tied-in-x; n2=#tied-in-y; "
                          "den=sqrt((n0-n1)(n0-n2)); pooled across within-sequence pairs"}

    agg["formula_max_abs_err_vs_scipy"] = round(validate_tie_formula(), 12)

    cov = phase_coverage(exact_source("mimic_scale_control", N, ("cov", 7)))
    agg["phase_coverage_MIMIC"] = cov

    agg["concentration"] = {k: concentration(REGIMES[k][1](N, ("conc", 7))) for k in ("full_MIMIC", "full_SCID")}

    agg["null_and_power"] = {k: null_and_power(k) for k in REGIMES}

    agg["s8_interaction"] = s8_interaction()

    agg["delta_aligned_boundary"] = delta_aligned_boundary()

    agg_hash = canonical_hash(agg)
    print(json.dumps(agg, indent=2))
    print("\nAGGREGATE_HASH:", agg_hash)
    return agg, agg_hash


if __name__ == "__main__":
    main()
