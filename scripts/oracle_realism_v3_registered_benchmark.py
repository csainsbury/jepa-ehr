#!/usr/bin/env python3
"""Oracle realism v3 — FULL PER-GROUP REGISTERED-SCALE (N=8000, B=20000) benchmark + cap / checkpoint /
persistence plan.

This supersedes the ROUTE-SURROGATE forecast in `oracle_realism_v3_benchmark.py` for the SD-permutation cost.
That module priced each cell through one of five hand-rolled surrogate kernels (`tau_pooled`, `ks_per_item`,
`marginal`, `frozen_map`, `tau_source`) because only four estimators were wired. All twenty canonical estimators
across all six groups are now wired, so every in-scope cell is priced by MEASURING ITS ACTUAL ENGINE ESTIMATOR
at the registered pooled volume of its own experiment's profile. The surrogate module is left in place as a
separate artifact; the reconciliation between its forecast and this one is reported in the contract, not here.

It also prices the three registered-scale stages the surrogate benchmark never costed at all — they are invisible
below B~5400 and each is a hard blocker or a large constant at B=20000:

  * ASSIGNMENT MATERIALISATION. `_gate_group` builds `[canonical] + [perm]*B` boolean masks for EVERY experiment
    before any statistic. At M=16000 and B=20000 that is (B+1)*M = 320 MB per experiment and 2.88 GB for a
    nine-experiment full-support group, held live across the whole recompute phase.
  * MIN-P RANKING. `cell_upper_p` forms an A x A boolean comparison matrix (A = B+1). At A=20001 that is an
    exact 400.0 MB transient allocation and ~4.0e8 comparisons PER CELL; the deepest group has K=54 cells.
    A sort/`searchsorted` formulation returns BIT-IDENTICAL ranks in O(A log A) time and O(A) memory; the
    equality is proven here over an adversarial battery (ties, +inf NE sentinels, all-constant vectors) and
    both costs are reported.
  * PER-REPLICATE ASSIGNMENT DERIVATION. The current law draws every mask from ONE sequential
    `default_rng(seed)` stream, so permutation block k cannot be regenerated without replaying blocks 0..k-1.
    That makes a checkpoint/resume job plan impossible as written. The cost of a per-replicate seed-derived law
    (block-addressable, streamable at O(M) memory) is measured here so the plan's forecast is honest about it.
    The law CHANGE ITSELF IS NOT APPLIED — it is a design decision bound by the RNG manifest and is routed for
    review, not adopted unilaterally.

DEVELOPMENT ONLY, and nothing here is a draw. Fixtures come from the benchmark's own dev namespace; the only
maps built are TIMING-ONLY artifacts carrying the EXACT fixture seed that generated the reference arm they were
built from (the Pi rev-11 correction), explicitly namespaced apart from any reserved map-design artifact. No
calibration/evaluation seed is used, no reserved map-set is drawn, no manifest is populated or frozen, no policy
is written and no result is persisted to `state/`.

Two identities are emitted: a DETERMINISTIC config/forecast identity (formula, cell routing, per-profile volumes,
B, cost model, plan parameters, source identity layers) and a separate ENVIRONMENT-DEPENDENT timing artifact
(measured seconds, hours, RAM). Only the former is reproducible across machines.

Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/oracle_realism_v3_registered_benchmark.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES

import scripts.oracle_realism_v3_registry as REG
from scripts.oracle_realism_v3_engine import (
    ESTIMATORS, CANONICAL_GROUPS, CANONICAL_REGISTRY_HASH, ALPHA_GROUP_EXACT, REGISTERED,
    SOURCE_IDENTITY_BUNDLE, ESTIMATOR_DEPENDENCY_IDENTITY, _canonicalize_pool, _validate_precompute,
)
from scripts.oracle_realism_v3_map import build_frozen_map, FLOOR
from scripts.oracle_realism_v3_randomization import cell_upper_p, _canonical_mask, _perm_mask

# ---------------------------------------------------------------------------------------------------
# fixed benchmark configuration
# ---------------------------------------------------------------------------------------------------
BENCH_NS = "v3-registered-benchmark-dev"           # fixture namespace: dev only, disjoint from calib/eval
TIMING_MAP_NS = "v3-registered-benchmark-timing-map"   # TIMING-ONLY maps; NOT the reserved map-design namespace
N_PER_ARM = REGISTERED["N_per_arm"]                # 8000
B_MAIN = REGISTERED["B"]                           # 20000
FLOOR_REG = REGISTERED["floor"]                    # 500
CAP_HOURS = 8.0                                    # per-job wall-clock cap
MARGIN = 1.5                                       # conservative forecast margin (Pi rev-6: not merely <8h)
CHECKPOINT_BLOCK = 1000                            # permutation replicates per checkpoint block

# adaptive timing controls
TARGET_SECS = 0.60                                 # per-measurement timing budget
MIN_REPS, MAX_REPS = 8, 300
MASK_CYCLE = 8                                     # distinct pre-generated masks cycled during recompute timing

_PAGE = 4096


def _rss_bytes():
    """Current resident set size (not a high-water mark, so deltas are meaningful)."""
    try:
        with open("/proc/self/statm", "r") as fh:
            return int(fh.read().split()[1]) * _PAGE
    except OSError:
        return 0


def bseed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (BENCH_NS, *parts))).encode()).digest()[:6], "big")


def _skeleton(profile):
    return "SCID" if "scid" in profile else "MIMIC"


def draw_arm(profile, role):
    return sample_fixture(_skeleton(profile), PROFILES[profile], N_PER_ARM, seed=bseed(profile, role))


def timed(fn, *, target=TARGET_SECS, min_reps=MIN_REPS, max_reps=MAX_REPS):
    """Adaptive repetition timing: one warm-up, one probe, then enough reps to fill `target` seconds."""
    fn()
    t = time.perf_counter(); fn(); probe = time.perf_counter() - t
    reps = int(np.clip(int(target / max(probe, 1e-9)), min_reps, max_reps))
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps, reps


def _strata_key(strata):
    """Single canonical key for a stratum layout, used by both the measurement and the forecast."""
    return json.dumps([[int(a), int(b)] for a, b in strata])


def _nbytes(pre):
    """Exact numpy footprint of a precompute payload (ndarray, dict of arrays, or list of arrays)."""
    if isinstance(pre, np.ndarray):
        return int(pre.nbytes)
    if isinstance(pre, dict):
        return int(sum(_nbytes(v) for v in pre.values()))
    if isinstance(pre, (list, tuple)):
        return int(sum(_nbytes(v) for v in pre))
    return 0


# ---------------------------------------------------------------------------------------------------
# the O(A log A) rank formulation (equality-proven below; NOT yet substituted into the engine)
# ---------------------------------------------------------------------------------------------------
def cell_upper_p_sorted(e):
    """p^(j) = #{b : e[b] >= e[j]} / A, computed by sorting instead of by an A x A comparison matrix.

    #{b : e[b] >= e[j]} = A - #{b : e[b] < e[j]}, and `searchsorted(sorted_e, e, 'left')` is exactly
    #{b : e[b] < e[j]}. Ties and the +inf NE sentinel are handled by construction, so the result is
    bit-identical to `cell_upper_p` while using O(A) memory instead of O(A^2)."""
    e = np.asarray(e, float)
    A = e.shape[0]
    lt = np.searchsorted(np.sort(e), e, side="left")
    return (A - lt) / A


def prove_rank_equality(rng, trials=400):
    """Adversarial equality battery: heavy ties, +inf NE sentinels, all-constant, single-element, two-valued."""
    cases = 0
    for t in range(trials):
        n = int(rng.integers(1, 60))
        kind = t % 5
        if kind == 0:
            e = rng.normal(size=n)
        elif kind == 1:
            e = rng.integers(0, 3, size=n).astype(float)          # heavy ties
        elif kind == 2:
            e = np.full(n, 1.25)                                   # all constant
        elif kind == 3:
            e = rng.integers(0, 4, size=n).astype(float)
            e[rng.integers(0, n)] = np.inf                         # NE sentinel present
        else:
            e = np.where(rng.random(n) < 0.5, 0.0, np.inf)         # all-or-nothing NE
        if not np.array_equal(cell_upper_p(e), cell_upper_p_sorted(e)):
            raise AssertionError(f"rank formulations disagree on {e!r}")
        cases += 1
    return cases


# ---------------------------------------------------------------------------------------------------
# the block-addressable assignment law whose cost the plan needs (measured, NOT adopted here)
# ---------------------------------------------------------------------------------------------------
def _replicate_rng(namespace, group_id, exp_id, replicate_index):
    """Proposed per-replicate seed derivation, so permutation block k is reproducible without replaying 0..k-1."""
    key = "|".join((namespace, group_id, exp_id, str(int(replicate_index)))).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


# ---------------------------------------------------------------------------------------------------
# registry-derived work plan
# ---------------------------------------------------------------------------------------------------
def registry_view():
    """Per-variant in-scope cells, the (profile, statistic) pairs that must be measured, and per-experiment meta."""
    variants, needed, exp_meta = {}, set(), {}
    for label, apply_uncal in (("with_exemption", True), ("without_exemption", False)):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=apply_uncal)
        groups = REG.build_groups(sd)
        by_id = {c["cell_id"]: c for c in sd}
        gv = {}
        for gid, g in groups.items():
            cells = []
            for cid in g["cells"]:
                c = by_id[cid]
                cells.append({"cell_id": cid, "exp": c["experiment_id"], "stat": c["statistic"],
                              "profile": c["source"], "regime": c["support_regime"],
                              "map_carrying": ESTIMATORS[c["statistic"]]["map_carrying"]})
                needed.add((c["source"], c["statistic"]))
                strata = [(s["n_candidate"], s["n_reference"]) for s in c["exchangeability_strata"]]
                exp_meta[c["experiment_id"]] = {"profile": c["source"], "regime": c["support_regime"],
                                                "strata": strata, "M": sum(a + b for a, b in strata)}
            gv[gid] = cells
        variants[label] = gv
    return variants, sorted(needed), exp_meta


# ---------------------------------------------------------------------------------------------------
# measurement stages
# ---------------------------------------------------------------------------------------------------
def build_timing_map(reference_arm, seed, profile, regime, stat, log):
    """A TIMING-ONLY frozen map bound to the EXACT fixture seed of the reference arm it is built from."""
    art = build_frozen_map(reference_arm, stat, profile=profile, regime=regime, seed=seed, N=N_PER_ARM,
                           namespace=TIMING_MAP_NS, floor=FLOOR_REG)
    if art["seed"] != seed:
        raise AssertionError("timing map seed must equal the fixture seed that drew the reference arm")
    if art.get("status") != "OK":
        log(f"[map] {profile}/{regime}/{stat}: status={art.get('status')} (recorded; timing still measured)")
    return art


def measure_profiles_and_cells(needed, exp_meta, log):
    """One pass per profile: draw both arms, canonicalise once, price every statistic that profile needs, then
    RELEASE the pool before the next profile. Holding all four pooled pools (64k records) at once is avoidable
    and is the largest avoidable RAM spike in the benchmark itself."""
    regime_of, strata_of = {}, {}
    for m in exp_meta.values():
        regime_of.setdefault(m["profile"], m["regime"])
        strata_of.setdefault(m["profile"], m["strata"])
    stats_of = {}
    for profile, stat in needed:
        stats_of.setdefault(profile, []).append(stat)

    prof_cost, cell_cost = {}, {}
    for profile in sorted(stats_of):
        log(f"[profile] {profile}: drawing two arms of N={N_PER_ARM}")
        rss0 = _rss_bytes()
        t = time.perf_counter(); cand = draw_arm(profile, "candidate"); draw_secs = time.perf_counter() - t
        ref_seed = bseed(profile, "reference")
        ref = draw_arm(profile, "reference")
        t = time.perf_counter()
        pool = _canonicalize_pool(list(cand) + list(ref), profile, f"bench_{profile}")
        canon_secs = time.perf_counter() - t
        del cand
        rss1 = _rss_bytes()
        prof_cost[profile] = {
            "draw_secs_per_arm": round(draw_secs, 4),
            "canonicalize_secs_pooled": round(canon_secs, 4),
            "M_pooled": len(pool),
            "volumes": {"n_sequences": len(pool),
                        "events": int(sum(r.L_total for r in pool)),
                        "clusters": int(sum(r.K for r in pool))},
            "pool_rss_bytes_measured": max(0, rss1 - rss0),
        }
        log(f"[profile] {profile}: draw {draw_secs:.2f}s/arm, canon {canon_secs:.2f}s, "
            f"events {prof_cost[profile]['volumes']['events']}, "
            f"RSS +{prof_cost[profile]['pool_rss_bytes_measured']/1e6:.0f} MB")

        strata = strata_of[profile]
        if sum(a + b for a, b in strata) != len(pool):
            raise AssertionError(f"{profile}: registry strata total {sum(a + b for a, b in strata)} "
                                 f"!= pooled draw {len(pool)}")
        for stat in sorted(stats_of[profile]):
            est = ESTIMATORS[stat]
            t = time.perf_counter()
            pre = _validate_precompute(est["precompute"](pool), stat, len(pool))
            pre_secs = time.perf_counter() - t

            groups, map_status = None, None
            if est["map_carrying"]:
                art = build_timing_map(ref, ref_seed, profile, regime_of[profile], stat, log)
                groups, map_status = art["groups"], art.get("status")

            rng = np.random.default_rng(bseed("perm", profile, stat))
            masks = [_perm_mask(rng, strata) for _ in range(MASK_CYCLE)]
            state = {"i": 0}

            def one():
                m = masks[state["i"] % MASK_CYCLE]; state["i"] += 1
                est["recompute"](pre, m, groups=groups, floor=FLOOR_REG)

            per_perm, reps = timed(one)
            probe = est["recompute"](pre, masks[0], groups=groups, floor=FLOOR_REG)
            cell_cost[(profile, stat)] = {
                "precompute_secs": round(pre_secs, 4),
                "precompute_bytes": _nbytes(pre),
                "per_perm_secs": per_perm,
                "timing_reps": reps,
                "map_carrying": bool(est["map_carrying"]),
                "map_status": map_status,
                "recompute_defined_on_probe": bool(probe is not None and np.isfinite(probe)),
            }
            log(f"[cell] {profile:24s} {stat:16s} pre {pre_secs:7.2f}s "
                f"({_nbytes(pre)/1e6:7.1f} MB)  perm {per_perm*1e3:8.3f} ms  x{reps}"
                + ("" if probe is not None and np.isfinite(probe) else "   [NE on probe]"))
            del pre, masks
        del pool, ref
    return prof_cost, cell_cost


def measure_assignments(exp_meta, log):
    """Mask generation cost per stratum layout, plus the block-addressable law's overhead."""
    layouts = {}
    for e, m in exp_meta.items():
        layouts.setdefault(tuple(m["strata"]), []).append(e)
    out = {}
    for strata, exps in layouts.items():
        rng = np.random.default_rng(bseed("mask", str(strata)))
        secs, reps = timed(lambda: _perm_mask(rng, list(strata)))
        idx = [0]

        def derived():
            idx[0] += 1
            _perm_mask(_replicate_rng(BENCH_NS, "G", exps[0], idx[0]), list(strata))

        derived_secs, dreps = timed(derived)
        M = sum(a + b for a, b in strata)
        out[_strata_key(strata)] = {
            "experiments": sorted(exps), "n_strata": len(strata), "M": M,
            "sequential_law_secs_per_mask": secs, "sequential_reps": reps,
            "block_addressable_law_secs_per_mask": derived_secs, "block_addressable_reps": dreps,
            "derivation_overhead_pct": round(100 * (derived_secs - secs) / max(secs, 1e-12), 1),
            "materialised_bytes_per_experiment": (B_MAIN + 1) * M,
            "streamed_bytes_per_experiment": M,
        }
        log(f"[mask] strata={len(strata)} M={M}: sequential {secs*1e6:.1f} us, "
            f"block-addressable {derived_secs*1e6:.1f} us "
            f"(+{out[_strata_key(strata)]['derivation_overhead_pct']}%)")
    return out


def measure_ranking(rng, log):
    """Reference O(A^2) vs sorted O(A log A) ranking, measured directly at the registered A = B+1."""
    A_reg = B_MAIN + 1
    proven = prove_rank_equality(rng)
    log(f"[rank] rank-formulation equality proven on {proven} adversarial cases (ties / +inf / constant)")
    ladder = {}
    for A in (2001, 5401, 10001, A_reg):
        e = rng.normal(size=A)
        ref_secs, ref_reps = timed(lambda: cell_upper_p(e), target=0.3, min_reps=3, max_reps=20)
        fast_secs, fast_reps = timed(lambda: cell_upper_p_sorted(e), target=0.3, min_reps=5, max_reps=200)
        ladder[A] = {"reference_secs": ref_secs, "reference_reps": ref_reps,
                     "reference_transient_bytes_exact": A * A,          # numpy bool comparison matrix, 1 byte/elem
                     "sorted_secs": fast_secs, "sorted_reps": fast_reps,
                     "sorted_transient_bytes_exact": A * 8 * 2,
                     "speedup": round(ref_secs / max(fast_secs, 1e-12), 1)}
        log(f"[rank] A={A:6d}  reference {ref_secs*1e3:9.2f} ms / {A*A/1e6:7.1f} MB transient   "
            f"sorted {fast_secs*1e3:7.3f} ms   speedup {ladder[A]['speedup']}x")
    return {"equality_cases_proven": proven, "ladder": {str(k): v for k, v in ladder.items()},
            "registered_A": A_reg}


def measure_persistence(log):
    """Aggregate-only result payload and checkpoint-block payload: measured hash cost and exact byte sizes."""
    K_max = max(len(v) for v in CANONICAL_GROUPS.values() for v in [v["cells"]])
    record = {
        "group_id": "G_full_class_mark", "verdict": "PASS", "p_g": 0.5137, "alpha_group": ALPHA_GROUP_EXACT,
        "B": B_MAIN, "K": K_max, "argmin_cell": "SD|null_scid|S5_abs",
        "observed_cell_p": [0.5] * K_max, "observed_cell_e": [0.0] * K_max,
        "null_S_quantiles": {str(q): 0.0 for q in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "registry_identity": CANONICAL_REGISTRY_HASH, "source_identities": SOURCE_IDENTITY_BUNDLE,
    }
    secs, reps = timed(lambda: canonical_hash(record), target=0.3)
    payload_bytes = len(json.dumps(record, sort_keys=True, default=str).encode())
    log(f"[persist] aggregate record {payload_bytes} bytes, canonical_hash {secs*1e6:.1f} us x{reps}")
    return {"aggregate_record_bytes": payload_bytes, "canonical_hash_secs": secs,
            "checkpoint_block_bytes_per_group": K_max * CHECKPOINT_BLOCK * 8,
            "full_null_matrix_bytes_deepest_group": K_max * (B_MAIN + 1) * 8,
            "K_max": K_max}


# ---------------------------------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------------------------------
def forecast_group(cells, cell_cost, prof_cost, exp_meta, mask_cost, rank, persist):
    """Total wall-clock seconds and peak RAM for ONE separately-gated per-group SD job at B=20000."""
    A = B_MAIN + 1
    exps = sorted({c["exp"] for c in cells})

    draw = sum(2 * prof_cost[exp_meta[e]["profile"]]["draw_secs_per_arm"] for e in exps)
    canon = sum(prof_cost[exp_meta[e]["profile"]]["canonicalize_secs_pooled"] for e in exps)
    precompute = sum(cell_cost[(c["profile"], c["stat"])]["precompute_secs"] for c in cells)
    recompute = A * sum(cell_cost[(c["profile"], c["stat"])]["per_perm_secs"] for c in cells)
    masks = A * sum(mask_cost[_strata_key(exp_meta[e]["strata"])]["sequential_law_secs_per_mask"]
                    for e in exps)
    K = len(cells)
    rank_ref = K * rank["ladder"][str(A)]["reference_secs"]
    rank_sorted = K * rank["ladder"][str(A)]["sorted_secs"]
    persist_secs = persist["canonical_hash_secs"] * K

    base = draw + canon + precompute + recompute + masks + persist_secs
    total_ref = base + rank_ref
    total_sorted = base + rank_sorted

    # which cells actually drive the bill (the plan must NAME them, not just report a group total)
    drivers = sorted((
        {"cell_id": c["cell_id"], "profile": c["profile"], "stat": c["stat"],
         "hours": round(A * cell_cost[(c["profile"], c["stat"])]["per_perm_secs"] / 3600.0, 3),
         "per_perm_ms": round(cell_cost[(c["profile"], c["stat"])]["per_perm_secs"] * 1e3, 3)}
        for c in cells), key=lambda d: -d["hours"])

    # SPLIT: permutation replicates are divisible work; the fixed per-job overhead (draw + canonicalise +
    # precompute) is paid by EVERY split job, so the split is solved against the per-job budget including it.
    budget_secs = CAP_HOURS * 3600.0 / MARGIN
    fixed = draw + canon + precompute
    divisible = recompute + masks
    if total_sorted * MARGIN / 3600.0 <= CAP_HOURS:
        jobs = 1
    else:
        jobs = 1
        while jobs < 512 and fixed + divisible / jobs + rank_sorted > budget_secs:
            jobs += 1
    replicates_per_job = math.ceil(B_MAIN / jobs)
    per_job_secs = fixed + divisible / jobs + (rank_sorted if jobs == 1 else 0.0)

    precompute_bytes = sum(cell_cost[(c["profile"], c["stat"])]["precompute_bytes"] for c in cells)
    pool_bytes = sum(prof_cost[exp_meta[e]["profile"]]["pool_rss_bytes_measured"] for e in exps)
    masks_materialised = sum(mask_cost[_strata_key(exp_meta[e]["strata"])]["materialised_bytes_per_experiment"]
                             for e in exps)
    masks_streamed = sum(mask_cost[_strata_key(exp_meta[e]["strata"])]["streamed_bytes_per_experiment"]
                         for e in exps)
    null_matrix = K * A * 8

    def hours(x):
        return round(x / 3600.0, 3)

    return {
        "K_cells": K, "n_experiments": len(exps), "B": B_MAIN, "A": A,
        "seconds": {"draw": round(draw, 1), "canonicalize": round(canon, 1),
                    "precompute": round(precompute, 1), "mask_generation": round(masks, 1),
                    "recompute": round(recompute, 1), "ranking_reference": round(rank_ref, 1),
                    "ranking_sorted": round(rank_sorted, 3), "persist": round(persist_secs, 4)},
        "hours_current_impl": hours(total_ref),
        "hours_sorted_ranking": hours(total_sorted),
        "hours_with_margin": hours(total_sorted * MARGIN),
        "fits_cap_with_margin": (total_sorted * MARGIN / 3600.0) <= CAP_HOURS,
        "recompute_share_pct": round(100 * recompute / max(total_sorted, 1e-9), 1),
        "peak_ram_bytes": {
            "pools_measured": pool_bytes, "precompute": precompute_bytes,
            "assignments_materialised_current_impl": masks_materialised,
            "assignments_streamed": masks_streamed,
            "null_e_matrix": null_matrix,
            "ranking_transient_reference": rank["ladder"][str(A)]["reference_transient_bytes_exact"],
            "ranking_transient_sorted": rank["ladder"][str(A)]["sorted_transient_bytes_exact"],
            "total_current_impl": (pool_bytes + precompute_bytes + masks_materialised + null_matrix
                                   + rank["ladder"][str(A)]["reference_transient_bytes_exact"]),
            "total_streamed_sorted": (precompute_bytes + masks_streamed + null_matrix
                                      + rank["ladder"][str(A)]["sorted_transient_bytes_exact"]),
        },
        "blocks_required": math.ceil(B_MAIN / CHECKPOINT_BLOCK),
        "hours_per_block_sorted": hours(recompute / max(math.ceil(B_MAIN / CHECKPOINT_BLOCK), 1)),
        "split_plan": {
            "jobs_required": jobs,
            "replicates_per_job": replicates_per_job,
            "fixed_overhead_hours_per_job": hours(fixed),
            "hours_per_split_job": hours(per_job_secs),
            "hours_per_split_job_with_margin": hours(per_job_secs * MARGIN),
            "split_job_fits_cap_with_margin": (per_job_secs * MARGIN / 3600.0) <= CAP_HOURS,
            "final_ranking_job_hours": hours(rank_sorted) if jobs > 1 else 0.0,
            "note": ("permutation replicates are the divisible unit; draw + canonicalise + precompute is FIXED "
                     "overhead repaid by every split job, and the min-p ranking runs ONCE at the end over the "
                     "assembled E matrix, so a split group needs jobs_required compute jobs plus one cheap "
                     "ranking/aggregation job."),
        },
        "top_cost_drivers": drivers[:5],
    }


def build_plan(forecasts):
    """Cap / checkpoint / persistence plan, stated as executable parameters rather than prose."""
    over = sorted({g for v in forecasts.values() for g, f in v.items() if not f["fits_cap_with_margin"]})
    return {
        "job_architecture": "one SEPARATELY GATED job per group per variant; stop-on-failure preserved; MM is a "
                            "separate job (unchanged); no audit job (removed at Pi rev-5 #5).",
        "cap_hours": CAP_HOURS, "margin": MARGIN,
        "groups_over_cap_with_margin": over,
        "split_rule": ("a group whose single-job forecast exceeds cap_hours/margin is split into the smallest "
                       "number of SEQUENTIALLY GATED permutation-replicate jobs such that "
                       "fixed_overhead + divisible/jobs fits that budget, plus one final ranking/aggregation "
                       "job; stop-on-failure applies across the split exactly as across groups"),
        "split_jobs_by_group": {label: {gid: f["split_plan"]["jobs_required"] for gid, f in v.items()}
                                for label, v in forecasts.items()},
        "resolution": REG._resolution(REG.build_groups(REG.build_sd_cells(True))),
        "checkpoint": {
            "unit": "permutation replicate block",
            "block_size": CHECKPOINT_BLOCK,
            "blocks_per_group": math.ceil(B_MAIN / CHECKPOINT_BLOCK),
            "state_per_block": "the block's columns of the per-cell discrepancy matrix E[K, block] only",
            "resume_rule": "a resumed job recomputes ONLY missing blocks; it must re-verify the bound identity "
                           "set (registry, source identities, map-set identity, RNG manifest identity, floor, B, "
                           "alpha_group, group id) and REFUSE on any mismatch rather than merge across identities",
            "REQUIRED_DESIGN_CHANGE": "block-addressable assignments. The current law draws every mask from one "
                                      "sequential default_rng(seed) stream, so block k is not reproducible without "
                                      "replaying 0..k-1 and checkpoint/resume is impossible. The measured "
                                      "alternative derives a per-replicate seed from "
                                      "sha256(namespace|group|experiment|replicate_index). This CHANGES the bound "
                                      "assignment RNG law and is therefore a reviewer decision — it is measured "
                                      "here and NOT applied.",
        },
        "persistence": {
            "policy": "AGGREGATE ONLY. No per-permutation statistic, mask, pool or precompute array is written to "
                      "the result artifact.",
            "result_artifact": ["group_id", "verdict", "p_g", "alpha_group", "B", "K", "argmin_cell",
                               "observed per-cell p", "observed per-cell e", "null S quantiles",
                               "bound identity set", "block manifest with per-block assignment digests"],
            "checkpoint_artifact": "E[K, block] float64 per block, plus the block's assignment digest; discarded "
                                   "once the group's aggregate result is written and verified.",
            "explicitly_not_persisted": ["per-permutation E columns beyond the live checkpoint",
                                         "assignment masks", "canonicalised pools", "precompute payloads"],
        },
        "ram_note": "peak RAM is reported per group for the CURRENT implementation and for the streamed/sorted "
                    "alternative; the current implementation's assignment materialisation dominates it.",
    }


# ---------------------------------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------------------------------
def selftest(variants, cell_cost, forecasts, exp_meta):
    errs = []
    # every in-scope (profile, statistic) pair is measured — no cell priced by a surrogate or a default
    for label, gv in variants.items():
        for gid, cells in gv.items():
            for c in cells:
                if (c["profile"], c["stat"]) not in cell_cost:
                    errs.append(f"{label}/{gid}: unmeasured cell cost for {c['profile']}/{c['stat']}")
    # forecast covers every group of every variant, and cell counts agree with the registry
    for label, gv in variants.items():
        if set(forecasts[label]) != set(gv):
            errs.append(f"{label}: forecast groups {sorted(forecasts[label])} != registry {sorted(gv)}")
        for gid, cells in gv.items():
            if forecasts[label][gid]["K_cells"] != len(cells):
                errs.append(f"{label}/{gid}: forecast K {forecasts[label][gid]['K_cells']} != {len(cells)}")
    # M0 conservation: the per-group cell counts partition the variant's in-scope cell total
    for label, apply_uncal in (("with_exemption", True), ("without_exemption", False)):
        m0 = sum(1 for c in REG.build_sd_cells(apply_uncal) if c["scope"] == "in")
        got = sum(f["K_cells"] for f in forecasts[label].values())
        if got != m0:
            errs.append(f"{label}: sum K_g {got} != M0 {m0}")
    # every experiment is the registered size
    for e, m in exp_meta.items():
        if m["M"] != 2 * N_PER_ARM:
            errs.append(f"{e}: pooled M {m['M']} != 2*{N_PER_ARM}")
    # ranking equality is a hard precondition for reporting the sorted forecast at all
    if not np.array_equal(cell_upper_p(np.array([0.0, np.inf, 0.0, 1.0])),
                          cell_upper_p_sorted(np.array([0.0, np.inf, 0.0, 1.0]))):
        errs.append("rank formulations disagree on the NE-sentinel case")
    # the split plan must actually discharge the cap, and must partition B without losing replicates
    for label, v in forecasts.items():
        for gid, f in v.items():
            sp = f["split_plan"]
            if not sp["split_job_fits_cap_with_margin"]:
                errs.append(f"{label}/{gid}: split into {sp['jobs_required']} jobs still exceeds the cap "
                            f"({sp['hours_per_split_job_with_margin']}h > {CAP_HOURS}h)")
            if sp["jobs_required"] * sp["replicates_per_job"] < B_MAIN:
                errs.append(f"{label}/{gid}: split covers {sp['jobs_required']*sp['replicates_per_job']} "
                            f"replicates < B={B_MAIN}")
            if f["fits_cap_with_margin"] and sp["jobs_required"] != 1:
                errs.append(f"{label}/{gid}: fits the cap but was split into {sp['jobs_required']} jobs")
    return errs


def main():
    t_start = time.perf_counter()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    variants, needed, exp_meta = registry_view()
    profiles = sorted({p for p, _ in needed})
    log(f"[plan] {len(needed)} (profile, statistic) pairs over {len(profiles)} profiles; "
        f"{len(exp_meta)} experiments at M={2*N_PER_ARM}")

    prof_cost, cell_cost = measure_profiles_and_cells(needed, exp_meta, log)
    mask_cost = measure_assignments(exp_meta, log)
    rng = np.random.default_rng(bseed("rank"))
    rank = measure_ranking(rng, log)
    persist = measure_persistence(log)

    forecasts = {label: {gid: forecast_group(cells, cell_cost, prof_cost, exp_meta, mask_cost, rank, persist)
                         for gid, cells in gv.items()}
                 for label, gv in variants.items()}
    plan = build_plan(forecasts)
    errs = selftest(variants, cell_cost, forecasts, exp_meta)

    # ---- deterministic config identity: formula + routing + plan parameters + source identity layers ----
    cell_routing = {label: {gid: sorted((c["cell_id"], c["profile"], c["stat"]) for c in cells)
                            for gid, cells in gv.items()}
                    for label, gv in variants.items()}
    config_identity = canonical_hash({
        "formula": ("per-group job seconds = draw + canonicalize + sum_cells precompute "
                    "+ (B+1)*(sum_cells measured_per_perm + sum_experiments mask) "
                    "+ K*rank(A) + K*persist; each cell measured on ITS OWN wired estimator at ITS experiment's "
                    "registered profile volume; no route surrogate"),
        "cell_routing": cell_routing,
        "experiments": {e: {"profile": m["profile"], "regime": m["regime"],
                            "strata": [list(s) for s in m["strata"]], "M": m["M"]} for e, m in exp_meta.items()},
        "B_main": B_MAIN, "N_per_arm": N_PER_ARM, "floor": FLOOR_REG,
        "alpha_group": ALPHA_GROUP_EXACT, "cap_hours": CAP_HOURS, "margin": MARGIN,
        "checkpoint_block": CHECKPOINT_BLOCK,
        "split_rule": plan["split_rule"],
        "split_jobs_by_group": plan["split_jobs_by_group"],
        "rank_formulation": "sorted searchsorted-left complement; bit-identical to the A x A reference",
        "registry_identity": CANONICAL_REGISTRY_HASH,
        "source_identities": SOURCE_IDENTITY_BUNDLE,
    })

    timing_artifact = {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "estimator_dependency_identity": ESTIMATOR_DEPENDENCY_IDENTITY,
        "profile_costs": prof_cost,
        "cell_costs": {f"{p}|{s}": v for (p, s), v in sorted(cell_cost.items())},
        "assignment_costs": mask_cost,
        "ranking": rank,
        "persistence": persist,
        "forecasts": forecasts,
        "benchmark_wall_secs": round(time.perf_counter() - t_start, 1),
        "note": "seconds / hours / RAM are ENVIRONMENT DEPENDENT and are NOT reproducible across machines; the "
                "reproducible identity is config_identity_deterministic.",
    }

    out = {
        "namespace": BENCH_NS,
        "N_per_arm": N_PER_ARM, "B_main": B_MAIN, "floor": FLOOR_REG,
        "config_identity_deterministic": config_identity,
        "plan": plan,
        "forecast_summary": {
            label: {gid: {"K": f["K_cells"],
                          "hours_current_impl": f["hours_current_impl"],
                          "hours_sorted_ranking": f["hours_sorted_ranking"],
                          "hours_with_margin": f["hours_with_margin"],
                          "fits_8h_with_margin": f["fits_cap_with_margin"],
                          "split_jobs_required": f["split_plan"]["jobs_required"],
                          "hours_per_split_job_with_margin": f["split_plan"]["hours_per_split_job_with_margin"],
                          "top_cost_driver": (f["top_cost_drivers"][0]["cell_id"]
                                              if f["top_cost_drivers"] else None),
                          "peak_ram_GB_current": round(f["peak_ram_bytes"]["total_current_impl"] / 1e9, 3),
                          "peak_ram_GB_streamed": round(f["peak_ram_bytes"]["total_streamed_sorted"] / 1e9, 3)}
                    for gid, f in v.items()}
            for label, v in forecasts.items()
        },
        "timing_artifact": timing_artifact,
        "selftests_pass": not errs,
        "selftest_errors": errs,
        "authorization": ("DEV-ONLY benchmark. No reserved map-set draw, no calibration/evaluation seed, no "
                          "manifest population or freeze, no policy write, no persisted run artifact. Maps built "
                          f"here are TIMING-ONLY in namespace {TIMING_MAP_NS!r} carrying the exact fixture seed of "
                          "the reference arm they were built from."),
    }
    print(json.dumps(out, indent=2, default=str))
    print("\nCONFIG_IDENTITY (deterministic):", config_identity)
    print("TIMING_ARTIFACT_HASH (environment-dependent):", canonical_hash(timing_artifact))
    print("selftests_pass:", json.dumps(not errs))
    if errs:
        for e in errs:
            print("  SELFTEST ERROR:", e)
    return out


if __name__ == "__main__":
    main()
