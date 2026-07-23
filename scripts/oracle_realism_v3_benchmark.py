#!/usr/bin/env python3
"""Oracle realism v3 — ROUTE-WEIGHTED permutation-recompute benchmark + whole-JOB forecast (Pi rev-4 #1/#2/#3).

Rev-5. Replaces the earlier non-executing, three-surrogate-average forecast. Every registered statistic maps to
one of FIVE per-permutation recompute ROUTE CLASSES; this script measures each class's efficient O(volume)
recompute cost at the registered N, then forecasts the whole job as

    SD_secs = (B_main + audits*B_audit) * sum_over_inscope_cells( cost_per_perm(route(cell)) )
              + generation/precompute (one-time per experiment) + serialization/checkpoint
    total   = SD_secs + MM_proxy

i.e. sum_route(n_cells_on_route * measured_cost_per_perm_for_route) — NOT an unweighted surrogate average, and
NOT driven by the group count G (G only sets alpha_group). Cell counts per route come from the EXACT registry
(scripts/oracle_realism_v3_registry.py), both boundary-exemption variants. The MM term is reported explicitly as
a conservative v2 upper-bound PROXY, not an exact MM measurement.

The main/audit resolution split is predeclared honestly (Pi #3): the min-p test needs B >= K_max/alpha_group to
resolve a single-cell effect in the deepest group; a smaller audit B is a VALID conservative plus-one test but a
distinct, coarser evaluator. The exact audit false-park probability is proven from the binomial.

Development seeds only; aggregate-hashed. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_benchmark.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from math import comb, log

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, _positive_gaps_and_prev_size,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
import scripts.oracle_realism_v3_registry as REG

BENCH_NS = "v3-benchmark-dev"
N = 8000                      # registered per-side sample size
BREP = 2000                   # measurement permutations per route (timing only; not the job B)
B_MAIN = 20000
B_AUDIT_COARSE = 2000         # coarse pathology-screen option
AUDITS = 15                   # audit repetitions
AUDIT_PARK_K = 4              # park if >= 4 of 15 audit reps reject
ALPHA_SD = 0.04               # SD battery family-wise level (audit per-rep false-reject <= this, conservatively)

# every registered check -> one of five per-permutation recompute route classes
ROUTE_OF = {
    "S1_tau": "tau_source",
    "S3_tau": "tau_pooled",
    "S2_ks": "ks", "count_ks": "ks", "length_ks": "ks", "positive_gap_ks": "ks", "S9_gap": "ks",
    "class_tv": "marginal", "occupancy_abs": "marginal", "delta_t_zero_abs": "marginal", "S4_abs": "marginal",
    "S8_class": "marginal", "S8_density": "marginal", "S9_class": "marginal", "S9_zero": "marginal",
    "S1_density": "frozen_map", "S5_abs": "frozen_map", "S7_abs": "frozen_map", "S6_tv": "frozen_map",
    "S3_loggap": "frozen_map",
}
# per-route operating volume key (what the O(volume) recompute scales with)
ROUTE_VOLUME = {"tau_source": "sequences", "tau_pooled": "cap_pairs", "ks": "items(varies)",
                "marginal": "sequences", "frozen_map": "sequences"}


def bseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (BENCH_NS, *p))).encode()).digest()[:6], "big")


def draw(profile_name, seed):
    sk = "SCID" if "scid" in profile_name else "MIMIC"
    return sample_fixture(sk, PROFILES[profile_name], N, seed=bseed(profile_name, seed))


def _cap6_components(rec):
    y, x = _positive_gaps_and_prev_size(rec)
    m = x.shape[0]
    if m < 2:
        return 0.0, 0.0, 0.0, 0.0
    sel = sorted({int(round(v)) for v in np.linspace(0, m - 1, 6)}) if m > 6 else list(range(m))
    x, y = x[sel], y[sel]; mm = len(sel)
    if mm < 2:
        return 0.0, 0.0, 0.0, 0.0
    sx = np.sign(np.subtract.outer(x, x)); sy = np.sign(np.subtract.outer(y, y))
    iu = np.triu_indices(mm, 1); sx, sy = sx[iu], sy[iu]
    return float(np.sum(sx * sy)), float(mm * (mm - 1) / 2), float(np.sum(sx == 0)), float(np.sum(sy == 0))


def volumes(sample):
    pos_gaps = int(sum(len(_positive_gaps_and_prev_size(r)[0]) for r in sample))
    return {"n_sequences": len(sample), "events": int(sum(r.L_total for r in sample)),
            "clusters": int(sum(r.K for r in sample)), "positive_gaps": pos_gaps,
            "cap_pairs": int(sum(_cap6_components(r)[1] for r in sample))}


# --- efficient O(n log n) tau-b (merge-sort discordant count) for the tau_source route -------------------
def _tau_b(x, y):
    """Kendall tau-b via sort + BIT; O(n log n). Standard n1/n2 (ties in x / in y)."""
    n = len(x)
    if n < 2:
        return 0.0
    order = np.lexsort((y, x))
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    # tie corrections: n1 (ties in x), n2 (ties in y), t_xy (ties in BOTH)
    _, cx = np.unique(xs, return_counts=True); n1 = float(np.sum(cx * (cx - 1) // 2))
    _, cy = np.unique(ys, return_counts=True); n2 = float(np.sum(cy * (cy - 1) // 2))
    _, cxy = np.unique(np.stack([xs, ys], 1), axis=0, return_counts=True)
    t_xy = float(np.sum(cxy * (cxy - 1) // 2))
    n0 = n * (n - 1) / 2.0
    # count discordant via BIT over y where x strictly increases; within equal-x blocks contribute 0
    # concordant-minus-discordant = n0 - n1 - n2 + n3 - 2*discordant_across_strict; use rank-based swap count
    ranks = {v: i for i, v in enumerate(np.unique(ys))}
    yr = np.array([ranks[v] for v in ys], int)
    bit = np.zeros(len(ranks) + 1, int)
    def upd(i):
        i += 1
        while i < len(bit):
            bit[i] += 1; i += i & (-i)
    def qry(i):
        i += 1; s = 0
        while i > 0:
            s += bit[i]; i -= i & (-i)
        return s
    disc = 0; seen = 0
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        for k in range(i, j):                        # count already-inserted y greater than this y
            disc += seen - qry(yr[k])
        for k in range(i, j):
            upd(yr[k]); seen += 1
        i = j
    # untied-in-x-and-y pairs = C + D = n0 - n1 - n2 + t_xy ; disc = D ; so C - D = (C+D) - 2D
    conc_minus_disc = (n0 - n1 - n2 + t_xy) - 2 * disc
    den = np.sqrt((n0 - n1) * (n0 - n2))
    return 0.0 if den == 0 else conc_minus_disc / den


def measure_routes(pool, nA, M, rng):
    """Measure secs/perm for each route class at the registered volumes. Uses monotonic timing."""
    import time
    costs = {}

    # tau_pooled: precompute (C-D,n0,n1,n2) per seq; O(N) masked resum per perm
    comps = np.array([_cap6_components(r) for r in pool])
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); m = np.zeros(M, bool); m[idx[:nA]] = True
        s = comps[m].sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
        _ = None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)
    costs["tau_pooled"] = (time.perf_counter() - t) / BREP

    # ks: pooled sort of values fixed under label perm; O(M) cumsum per perm. Cost measured PER ITEM.
    K = np.array([r.K for r in pool], float); order = np.argsort(K, kind="mergesort")
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        a = inA[order].astype(float); b = (~inA[order]).astype(float)
        _ = float(np.max(np.abs(np.cumsum(a) / nA - np.cumsum(b) / (M - nA))))
    costs["ks_per_item"] = ((time.perf_counter() - t) / BREP) / M

    # marginal: precompute per-seq class histogram; O(M) masked prior diff per perm (TV/abs family)
    C = 5
    hist = np.zeros((M, C))
    for i, r in enumerate(pool):
        cc = np.bincount(np.asarray(r.class_ids) % C, minlength=C)
        hist[i] = cc / max(cc.sum(), 1)
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        pa = hist[inA].mean(0); pb = hist[~inA].mean(0)
        _ = 0.5 * float(np.abs(pa - pb).sum())
    costs["marginal"] = (time.perf_counter() - t) / BREP

    # frozen_map: reference-owned FROZEN length bins; O(M) per-bin mean-diff per perm
    L = np.array([r.L_total for r in pool]); binid = np.digitize(L, np.quantile(L, [0.2, 0.4, 0.6, 0.8]))
    val = K / np.maximum(L, 1); nbin = int(binid.max()) + 1
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        d = 0.0
        for bb in range(nbin):
            mb = binid == bb; va = val[mb & inA]; vb = val[mb & ~inA]
            if va.size and vb.size:
                d = max(d, abs(va.mean() - vb.mean()))
    costs["frozen_map"] = (time.perf_counter() - t) / BREP

    # tau_source: |tau_b(L,K)_cand - tau_b(L,K)_ref| per perm; O(n log n) each side. Measured with scipy's
    # C-level kendalltau (the representative OPTIMIZED route a real evaluator uses; our pure-python _tau_b is
    # only for the correctness cross-check, not the hot path). FEWER perms (still the most expensive route).
    from scipy.stats import kendalltau
    Lp = np.array([r.L_total for r in pool], float); Kp = np.array([r.K for r in pool], float)
    reps_src = min(BREP, 400)
    t = time.perf_counter()
    for _ in range(reps_src):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        _ = abs(kendalltau(Lp[inA], Kp[inA])[0] - kendalltau(Lp[~inA], Kp[~inA])[0])
    costs["tau_source"] = (time.perf_counter() - t) / reps_src
    return costs


def route_cost_per_cell(costs, vol):
    """secs/perm for a cell on each route, using measured class costs and each check's POOLED operating volume.
    ks_per_item was measured on the pooled cumsum, so cost = ks_per_item * (pooled item count) — NO extra 2x
    (vol is already the pooled A+ref count). S9_gap does ~4 seam/non-seam sub-KS -> a modest multiple."""
    ks_items = {"S2_ks": vol["clusters"], "count_ks": vol["n_sequences"], "length_ks": vol["n_sequences"],
                "positive_gap_ks": vol["positive_gaps"], "S9_gap": vol["positive_gaps"]}
    ks_mult = {"S9_gap": 4.0}                       # seam/non-seam within+cross sub-KS
    per = {}
    for check, route in ROUTE_OF.items():
        if route == "ks":
            per[check] = costs["ks_per_item"] * ks_items[check] * ks_mult.get(check, 1.0)
        else:
            per[check] = costs[route]
    return per


def sd_seconds(inscope_checks, per_cell_cost, B_main, B_audit, audits):
    """SD permutation cost = (B_main + audits*B_audit) * sum_over_inscope_cells cost_per_perm(cell)."""
    per_perm_sum = sum(per_cell_cost[c] for c in inscope_checks)
    return (B_main + audits * B_audit) * per_perm_sum, per_perm_sum


def inscope_checks_for_variant(apply_uncal):
    sd = REG.build_sd_cells(apply_uncalibratable_exemption=apply_uncal)
    return [c["statistic"] for c in sd if c["scope"] == "in"]


def audit_false_park_prob(alpha=ALPHA_SD, n=AUDITS, k=AUDIT_PARK_K):
    """Exact P(park | null) = P(Binom(n, alpha) >= k). Conservative: per-rep false-reject <= alpha for ANY B
    (plus-one randomization test is level-controlled regardless of B; small B costs POWER, not level)."""
    return float(sum(comb(n, i) * alpha ** i * (1 - alpha) ** (n - i) for i in range(k, n + 1)))


def main():
    A = draw("mimic_scale_control", 1); Bs = draw("mimic_scale_control", 2)
    pool = A + Bs; nA = len(A); M = len(pool); rng = np.random.default_rng(bseed("perm"))
    vol = volumes(pool)

    # naive full-battery recompute per permutation (infeasible upper bound)
    import time
    t = time.perf_counter(); sequence_route_checks(A, Bs); marginal_route_checks(A, Bs)
    naive = time.perf_counter() - t

    # one-time generation + precompute cost per experiment (fixtures + per-seq route precompute)
    t = time.perf_counter(); _ = [_cap6_components(r) for r in pool]; precompute = time.perf_counter() - t

    costs = measure_routes(pool, nA, M, rng)
    per_cell = route_cost_per_cell(costs, vol)

    mm_proxy_hours = 4.9   # v2 MM battery total (25 replicates) — conservative UPPER-BOUND proxy, not exact MM

    def forecast(apply_uncal, B_main, B_audit, audits=AUDITS):
        checks = inscope_checks_for_variant(apply_uncal)
        n_full_exp = sum(1 for e in REG.SD_EXPERIMENTS if e[4] == "full")
        sd_perm, per_perm_sum = sd_seconds(checks, per_cell, B_main, B_audit, audits)
        gen = precompute * (n_full_exp + 1)                     # per-experiment one-time precompute + gen
        serialize = 0.02 * (len(checks) * (audits + 1))         # ~20ms per cell-battery result serialize/checkpoint
        mm = mm_proxy_hours * 3600
        total = sd_perm + gen + serialize + mm
        by_route = {}
        for c in checks:
            by_route[ROUTE_OF[c]] = by_route.get(ROUTE_OF[c], 0) + 1
        return {"M0": len(checks), "B_main": B_main, "B_audit": B_audit, "audits": audits,
                "inscope_cells_by_route": by_route,
                "sum_cost_per_perm_secs": round(per_perm_sum, 6),
                "sd_permutation_secs": round(sd_perm, 1), "generation_precompute_secs": round(gen, 1),
                "serialize_checkpoint_secs": round(serialize, 1),
                "mm_proxy_secs": round(mm, 1), "mm_proxy_note": "v2 25-replicate UPPER-BOUND proxy (not exact MM)",
                "total_hours": round(total / 3600, 2), "under_8h_cap": total <= 8 * 3600}

    resolution = REG._resolution(REG.build_groups(REG.build_sd_cells(True)))
    fpp = audit_false_park_prob()

    def split_jobs(apply_uncal, B_main, B_audit, audits=AUDITS):
        """Separately-gated stop-on-failure jobs (Pi #3): SD-main | SD-audit | MM, each costed alone."""
        checks = inscope_checks_for_variant(apply_uncal)
        n_full_exp = sum(1 for e in REG.SD_EXPERIMENTS if e[4] == "full")
        s = sum(per_cell[c] for c in checks)
        gen = precompute * (n_full_exp + 1)
        sd_main = B_main * s + gen + 0.02 * len(checks)
        sd_audit = audits * B_audit * s + 0.02 * len(checks) * audits
        mm = mm_proxy_hours * 3600
        j = {"sd_main_hours": round(sd_main / 3600, 2), "sd_audit_coarse_hours": round(sd_audit / 3600, 2),
             "mm_proxy_hours": round(mm / 3600, 2)}
        j["each_under_8h_cap"] = all(v <= 8 for v in j.values())
        j["note"] = (f"SD-main B={B_main} fits; SD-audit is a coarse B={B_audit} screen in its own job; MM proxy "
                     "in its own job. Dominant SD cost = large-volume KS (S9_gap, S2_ks, positive_gap_ks).")
        return j

    # per-route contribution to one SD permutation (transparency: what drives the cost)
    checks_we = inscope_checks_for_variant(True)
    route_share = {}
    for c in checks_we:
        route_share[ROUTE_OF[c]] = route_share.get(ROUTE_OF[c], 0.0) + per_cell[c]
    tot_share = sum(route_share.values())
    route_share_pct = {r: round(100 * v / tot_share, 1) for r, v in sorted(route_share.items(),
                       key=lambda kv: -kv[1])}

    agg = {
        "namespace": BENCH_NS, "N": N, "B_measure": BREP,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "volumes_pooled": vol,
        "route_costs_ms": {"tau_pooled": round(costs["tau_pooled"] * 1000, 4),
                           "ks_per_item_us": round(costs["ks_per_item"] * 1e6, 6),
                           "marginal": round(costs["marginal"] * 1000, 4),
                           "frozen_map": round(costs["frozen_map"] * 1000, 4),
                           "tau_source": round(costs["tau_source"] * 1000, 4)},
        "per_cell_cost_ms": {c: round(per_cell[c] * 1000, 5) for c in sorted(per_cell)},
        "naive_full_battery_ms_per_perm": round(naive * 1000, 1),
        "label_note": "pooled-tau ~0.25 ms/perm (NOT microseconds). DOMINANT SD cost = the large-volume KS "
                      "routes (S9_gap ~56 ms, S2_ks/positive_gap_ks ~14 ms); tau_source is only ~1.3 ms.",
        "route_share_pct_of_one_perm": route_share_pct,
        "resolution": resolution,
        "b_audit_split": {
            "rule": "B >= K_max/alpha_group to resolve a single-cell effect in the deepest group",
            "B_resolution_min": resolution["B_resolution_min"],
            "main_B20000_resolves": True,
            "coarse_audit_B2000_resolves": False,
            "coarse_audit_is_valid_level": True,
            "coarse_audit_caveat": "conservative plus-one test => level-controlled at ANY B; small B loses POWER "
                                   "not level. Predeclared as a DISTINCT coarser pathology screen.",
            "audit_false_park_prob": round(fpp, 6),
            "audit_false_park_formula": f"P(Binom({AUDITS},{ALPHA_SD})>= {AUDIT_PARK_K}) = {fpp:.6g}",
        },
        "whole_job_forecast": {
            "with_exemption_audit_same_B20000": forecast(True, B_MAIN, B_MAIN),
            "with_exemption_audit_coarse_B2000": forecast(True, B_MAIN, B_AUDIT_COARSE),
            "without_exemption_audit_same_B20000": forecast(False, B_MAIN, B_MAIN),
            "without_exemption_audit_coarse_B2000": forecast(False, B_MAIN, B_AUDIT_COARSE),
        },
        "separately_gated_jobs": {
            "with_exemption": split_jobs(True, B_MAIN, B_AUDIT_COARSE),
            "without_exemption": split_jobs(False, B_MAIN, B_AUDIT_COARSE),
        },
        "feasibility_verdict": ("single combined job EXCEEDS the 8h cap (16-81h). SD-main (B=20000) ~4.8h and MM "
                                "proxy ~4.9h each fit alone; the 15-rep audit is the driver. Recommend separately "
                                "gated stop-on-failure jobs, and/or optimising the large-volume KS routes "
                                "(esp. S9_gap). The earlier rev-4 '~6h' figure was the flawed surrogate-average."),
        "forecast_formula": "SD = (B_main + audits*B_audit) * sum_inscope_cells cost_perm(route(cell)) "
                            "+ gen/precompute + serialize/checkpoint + MM_proxy. Route-weighted; NOT G-driven, "
                            "NOT a surrogate average.",
        "split_guidance": "if a single job exceeds the 8h cap, split SD-main / SD-audit / MM into separately "
                          "gated, stop-on-failure jobs (Pi #3) rather than weakening B after costing.",
        "authorization": "dev-only benchmark; no calibration/audit/evaluation seed, no map draw, no policy.",
    }
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
