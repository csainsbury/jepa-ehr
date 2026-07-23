#!/usr/bin/env python3
"""Oracle realism v3 — PER-PROFILE route-weighted whole-job benchmark + per-group job plan (Pi rev-5 #5/#6).

Rev-6. Two corrections over rev-5:
  * PER-PROFILE volumes (Pi #6): route costs are per-ITEM (profile-independent), but the whole-job forecast sums
    over EACH experiment using THAT experiment's exact profile/regime volume — SCID-scale experiments have larger
    event/cluster/gap volumes than MIMIC. The forecast is `sum_experiment sum_cell cost(route, profile volume)`,
    NOT one MIMIC cost times M0. So the claim "SD-main fits one 8h job" is NOT assumed — a PER-GROUP job plan is
    emitted so groups can be separately gated.
  * NO coarse scientific audit (Pi #5, preferred): the main conditional-randomization battery already has exact
    SD level; the 15-replicate coarse audit was level-valid but could not resolve a single-cell effect in the
    deepest group, so it is REMOVED (mechanical/exhaustive implementation tests are retained). alpha_main_SD=0.04
    <= 0.05 with unused margin; no audit term in the forecast.

Two identities are reported: a DETERMINISTIC config/forecast identity (formula + routes + per-experiment cell
routing + per-profile volumes + B — all seed-deterministic) and a separate TIMING artifact (environment-dependent
route costs / hours). Serialization cost is MEASURED, not assumed. Development seeds only. Run:
    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_benchmark.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    sequence_route_checks, marginal_route_checks, _positive_gaps_and_prev_size,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from scipy.stats import kendalltau
import scripts.oracle_realism_v3_registry as REG

BENCH_NS = "v3-benchmark-dev"
N = 8000
BREP = 2000
B_MAIN = 20000

ROUTE_OF = {
    "S1_tau": "tau_source", "S3_tau": "tau_pooled",
    "S2_ks": "ks", "count_ks": "ks", "length_ks": "ks", "positive_gap_ks": "ks", "S9_gap": "ks",
    "class_tv": "marginal", "occupancy_abs": "marginal", "delta_t_zero_abs": "marginal", "S4_abs": "marginal",
    "S8_class": "marginal", "S8_density": "marginal", "S9_class": "marginal", "S9_zero": "marginal",
    "S1_density": "frozen_map", "S5_abs": "frozen_map", "S7_abs": "frozen_map", "S6_tv": "frozen_map",
    "S3_loggap": "frozen_map",
}
# which pooled volume each KS check operates on
KS_VOLUME = {"S2_ks": "clusters", "count_ks": "n_sequences", "length_ks": "n_sequences",
             "positive_gap_ks": "positive_gaps", "S9_gap": "positive_gaps"}
KS_MULT = {"S9_gap": 4.0}
# experiment source profile -> the profile whose volume it uses (structural_zero and boundary are their own)
PROFILE_FOR = {"scid_scale_control": "scid_scale_control", "mimic_scale_control": "mimic_scale_control",
               "structural_zero_control": "structural_zero_control", "boundary_short": "boundary_short"}


def bseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (BENCH_NS, *p))).encode()).digest()[:6], "big")


def draw(profile, seed):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=bseed(profile, seed))


def _cap6_pairs(rec):
    y, x = _positive_gaps_and_prev_size(rec)
    m = x.shape[0]
    if m < 2:
        return 0.0
    sel = sorted({int(round(v)) for v in np.linspace(0, m - 1, 6)}) if m > 6 else list(range(m))
    mm = len(sel)
    return float(mm * (mm - 1) / 2) if mm >= 2 else 0.0


def volumes(sample):
    pos_gaps = int(sum(len(_positive_gaps_and_prev_size(r)[0]) for r in sample))
    return {"n_sequences": len(sample), "events": int(sum(r.L_total for r in sample)),
            "clusters": int(sum(r.K for r in sample)), "positive_gaps": pos_gaps,
            "cap_pairs": int(sum(_cap6_pairs(r) for r in sample))}


def measure_route_costs(pool, nA, M, rng):
    """secs/perm per route class (timing; profile-independent per-item where relevant)."""
    costs = {}
    comps = np.array([[*(_seqcomp(r))] for r in pool])
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); m = np.zeros(M, bool); m[idx[:nA]] = True
        s = comps[m].sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
        _ = None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)
    costs["tau_pooled"] = (time.perf_counter() - t) / BREP

    K = np.array([r.K for r in pool], float); order = np.argsort(K, kind="mergesort")
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        a = inA[order].astype(float); b = (~inA[order]).astype(float)
        _ = float(np.max(np.abs(np.cumsum(a) / nA - np.cumsum(b) / (M - nA))))
    costs["ks_per_item"] = ((time.perf_counter() - t) / BREP) / M

    Cn = 5; hist = np.zeros((M, Cn))
    for i, r in enumerate(pool):
        cc = np.bincount(np.asarray(r.class_ids) % Cn, minlength=Cn); hist[i] = cc / max(cc.sum(), 1)
    t = time.perf_counter()
    for _ in range(BREP):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        _ = 0.5 * float(np.abs(hist[inA].mean(0) - hist[~inA].mean(0)).sum())
    costs["marginal"] = (time.perf_counter() - t) / BREP

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

    Lp = np.array([r.L_total for r in pool], float); Kp = np.array([r.K for r in pool], float)
    reps = min(BREP, 400)
    t = time.perf_counter()
    for _ in range(reps):
        idx = rng.permutation(M); inA = np.zeros(M, bool); inA[idx[:nA]] = True
        _ = abs(kendalltau(Lp[inA], Kp[inA])[0] - kendalltau(Lp[~inA], Kp[~inA])[0])
    costs["tau_source"] = (time.perf_counter() - t) / reps
    return costs


def _seqcomp(rec):
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


def cell_cost(stat, costs, prof_vol):
    """secs/perm for a cell of `stat` at a given profile's pooled volume."""
    route = ROUTE_OF[stat]
    if route == "ks":
        items = 2 * prof_vol[KS_VOLUME[stat]]                 # pooled cand+ref
        return costs["ks_per_item"] * items * KS_MULT.get(stat, 1.0)
    return costs[route]


def main():
    A = draw("mimic_scale_control", 1); Bs = draw("mimic_scale_control", 2)
    pool = A + Bs; nA = len(A); M = len(pool); rng = np.random.default_rng(bseed("perm"))
    costs = measure_route_costs(pool, nA, M, rng)

    # PER-PROFILE volumes (Pi #6): measure every source profile the registry uses
    prof_vol = {p: volumes(draw(p, 9)) for p in
                ("scid_scale_control", "mimic_scale_control", "structural_zero_control", "boundary_short")}

    # measured serialization cost (Pi #6: not assumed) — hash a representative per-cell result dict
    rep = {"cell_id": "SD|x|y", "p_g": 0.5, "S_null_summary": [0.0] * 32, "argmin": "z"}
    t = time.perf_counter()
    for _ in range(2000):
        canonical_hash(rep)
    serialize_per_cell = (time.perf_counter() - t) / 2000

    mm_proxy_hours = 4.9   # v2 MM battery (25 replicates) — conservative UPPER-BOUND proxy, not exact MM

    def forecast(apply_uncal, B_main):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=apply_uncal)
        groups = REG.build_groups(sd)
        inscope = [c for c in sd if c["scope"] == "in"]
        # per-cell cost at ITS experiment's profile volume (Pi #6)
        per_group_secs, per_profile_cells = {}, {}
        total_perm = 0.0
        for c in inscope:
            pv = prof_vol[PROFILE_FOR[c["source"]]]
            cs = cell_cost(c["statistic"], costs, pv) * B_main
            total_perm += cs
            per_group_secs[c["group_id"]] = per_group_secs.get(c["group_id"], 0.0) + cs
            per_profile_cells[c["source"]] = per_profile_cells.get(c["source"], 0) + 1
        gen = 0.0
        serialize = serialize_per_cell * len(inscope)
        # per-GROUP job plan: each group + its share of gen/serialize + (MM only once)
        group_jobs = {g: round((per_group_secs[g] + serialize * (len(v["cells"]) / len(inscope))) / 3600, 3)
                      for g, v in groups.items()}
        sd_main_hours = round((total_perm + gen + serialize) / 3600, 2)
        mm_hours = mm_proxy_hours
        return {"M0": len(inscope), "B_main": B_main, "audit": "REMOVED (Pi #5)",
                "per_profile_inscope_cells": per_profile_cells,
                "sd_main_total_hours": sd_main_hours, "mm_proxy_hours": mm_hours,
                "per_group_job_hours": group_jobs,
                "deepest_group_job_hours": round(max(group_jobs.values()), 3),
                "sd_main_fits_one_8h_job": sd_main_hours <= 8,
                "measured_serialize_secs_per_cell": round(serialize_per_cell, 8)}

    forecasts = {"with_exemption": forecast(True, B_MAIN), "without_exemption": forecast(False, B_MAIN)}

    # --- WIRED-ENGINE measurement (Pi rev-6 #6): the ACTUAL engine's MEASURED generation (precompute) + per-perm
    #     recompute + serialization for the burst-timing group, with a CONSERVATIVE cap margin (not merely <8h) ---
    from scripts.oracle_realism_v3_engine import ESTIMATORS as _EST
    from scripts.oracle_realism_v3_map import build_frozen_map as _bfm
    poolw = A + Bs; nAw = len(A); Mw = len(poolw); BW = 500
    wired = {}
    for chk in ("S3_tau", "delta_t_zero_abs", "positive_gap_ks", "S3_loggap"):
        est = _EST[chk]
        t = time.perf_counter(); pre = est["precompute"](poolw); gen = time.perf_counter() - t     # MEASURED generation
        groups = (_bfm(Bs, chk, profile="mimic_scale_control", regime="full", N=N)["groups"]
                  if est["map_carrying"] else None)
        t = time.perf_counter()
        for _ in range(BW):
            idx = rng.permutation(Mw); m = np.zeros(Mw, bool); m[idx[:nAw]] = True
            est["recompute"](pre, m, groups)
        wired[chk] = {"generation_secs": round(gen, 3), "per_perm_ms": round((time.perf_counter() - t) / BW * 1000, 4)}
    BT_CELLS, BT_EXP, MARGIN = 36, 9, 1.5
    B_bt = int(np.ceil(BT_CELLS / (0.04 / 6) / 100.0)) * 100                                       # 5400
    gen_total = sum(w["generation_secs"] for w in wired.values()) * BT_EXP
    perperm_sum = sum(w["per_perm_ms"] / 1000 for w in wired.values()) * BT_EXP
    bt_secs = B_bt * perperm_sum + gen_total + serialize_per_cell * BT_CELLS
    bt_h = bt_secs / 3600
    wired_engine = {"group": "burst_timing", "cells": BT_CELLS, "B": B_bt, "per_check": wired,
                    "measured_generation_secs_total": round(gen_total, 1),
                    "measured_serialize_secs_per_cell": round(serialize_per_cell, 8),
                    "job_hours_measured": round(bt_h, 3), "margin": MARGIN,
                    "job_hours_with_margin": round(bt_h * MARGIN, 3), "fits_8h_with_margin": bt_h * MARGIN <= 8,
                    "note": "ACTUAL engine estimators (generation + per-perm recompute) MEASURED at the REGISTERED "
                            "N=8000; the class-mark group's generation is heavier (S4/S6/S7 precompute) — the full "
                            "per-group wired benchmark follows once all five groups' estimators are wired. "
                            "Conservative margin 1.5x required, not merely <8h."}

    # DETERMINISTIC config identity (no timing): formula + routes + per-experiment routing + per-profile volumes + B
    sd = REG.build_sd_cells(True)
    routing = {c["cell_id"]: [ROUTE_OF[c["statistic"]], PROFILE_FOR[c["source"]]]
               for c in sd if c["scope"] == "in"}
    config_identity = canonical_hash({
        "formula": "sum_experiment sum_cell cost(route, profile_volume) * B_main + serialize + MM_proxy; NO audit",
        "route_of": ROUTE_OF, "ks_volume": KS_VOLUME, "ks_mult": KS_MULT,
        "profile_volumes": prof_vol, "cell_routing": routing, "B_main": B_MAIN})

    timing_artifact = {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "route_costs_ms": {"tau_pooled": round(costs["tau_pooled"] * 1e3, 4),
                           "ks_per_item_us": round(costs["ks_per_item"] * 1e6, 6),
                           "marginal": round(costs["marginal"] * 1e3, 4),
                           "frozen_map": round(costs["frozen_map"] * 1e3, 4),
                           "tau_source": round(costs["tau_source"] * 1e3, 4)},
        "forecasts_hours": forecasts,
        "note": "route costs + hours are TIMING/ENVIRONMENT dependent (NOT reproducible); the deterministic "
                "identity is config_identity. KS routes dominate; per-profile volumes make SCID experiments "
                "heavier than MIMIC.",
    }
    out = {"namespace": BENCH_NS, "N": N, "B_main": B_MAIN, "audit": "REMOVED (Pi #5 preferred)",
           "config_identity_deterministic": config_identity,
           "wired_engine_measurement": wired_engine,
           "profile_volumes": prof_vol, "timing_artifact": timing_artifact,
           "feasibility": ("SD-main is costed PER PROFILE (SCID heavier than MIMIC); do NOT assume one 8h job — "
                           "see per_group_job_hours and sd_main_fits_one_8h_job. If it does not fit, run SEPARATELY "
                           "GATED per-group SD jobs (stop-on-failure) + a separate MM job. No audit job."),
           "authorization": "dev-only benchmark; no calibration/eval seed, no map draw, no policy."}
    print(json.dumps(out, indent=2, default=str))
    print("\nCONFIG_IDENTITY (deterministic):", config_identity)
    print("TIMING_ARTIFACT_HASH (environment-dependent):", canonical_hash(timing_artifact))
    return out


if __name__ == "__main__":
    main()
