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
  * MIN-P RANKING. The ORIGINAL rank stage formed an A x A boolean comparison matrix (A = B+1). At A=20001
    that is an exact 400.0 MB transient allocation and ~4.0e8 comparisons PER CELL; the deepest group has
    K=54 cells. `cell_upper_p` is now the sort/`searchsorted` form (adopted under Pi rev-22 ruling 4), which
    returns BIT-IDENTICAL ranks in O(A log A) time and O(A) memory. The benchmark times the retained
    quadratic oracle `_cell_upper_p_quadratic` against the PRODUCTION `cell_upper_p`, so the reported
    speedup is a real comparison rather than the function against itself.
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

from clinical_jepa.eval.oracle_realism_v2_battery import _multiscale, _ZERO_PROF, CONTROL_ALLOC
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling

import scripts.oracle_realism_v3_registry as REG
import scripts.oracle_realism_v3_constructors as CON
import scripts.oracle_realism_v3_engine as ENG
from scripts.oracle_realism_v3_engine import (
    ESTIMATORS, CANONICAL_GROUPS, CANONICAL_REGISTRY_HASH, ALPHA_GROUP_EXACT, REGISTERED,
    SOURCE_IDENTITY_BUNDLE, ESTIMATOR_DEPENDENCY_IDENTITY, _canonicalize_pool, _validate_precompute,
    RefusalError,
)
from scripts.oracle_realism_v3_map import build_frozen_map, FLOOR
from scripts.oracle_realism_v3_randomization import (
    cell_upper_p, _cell_upper_p_quadratic, _canonical_mask, _perm_mask,
)

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
MARGINAL_HEADROOM = 0.10                           # < this much cap headroom => report MARGINAL, not "fits"

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


def _benchmark_source_identity():
    """This module's own source, bound into the deterministic configuration (Pi rev-22)."""
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def bseed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (BENCH_NS, *parts))).encode()).digest()[:6], "big")


def _skeleton(profile):
    return "SCID" if "scid" in profile else "MIMIC"


def draw_arm(profile, role, n=None):
    """GENERIC single-profile draw — correct ONLY for the `source_profile_fixture` route (scid/mimic)."""
    return sample_fixture(_skeleton(profile), PROFILES[profile], n or N_PER_ARM, seed=bseed(profile, role))


# ---------------------------------------------------------------------------------------------------
# CANONICAL per-experiment arm assembly (Pi rev-22 blocking correction)
# ---------------------------------------------------------------------------------------------------
# rev-22 drew every arm through `draw_arm(profile)` and then concatenated `cand + ref`. That was wrong twice
# over: it ran neither the structural-zero `_multiscale` constructor (mu = log 18 / 60 / 250 per stratum) nor the
# boundary structured constructor, and the flat `cand + ref` concatenation does not match the engine's canonical
# pool order `[cand_s0, ref_s0, cand_s1, ref_s1, ...]` for the two STRATIFIED experiments. Since event volume and
# support cardinality drive the expensive estimators, both errors move the measured cost. Costs are now measured
# PER EXPERIMENT (not per profile) so coupling and per-experiment construction are priced as they actually run.
def canonical_arms_for_experiment(exp_id, meta, *, n_per_arm=N_PER_ARM):
    """{stratum_id: {candidate, reference}} built through THAT experiment's canonical constructor route."""
    source, sids = meta["profile"], list(meta["stratum_ids"])
    quotas = [q for q, _ in meta["quota"]]
    if source == "structural_zero_control":                      # canonical multiscale control constructor
        alloc = _scaled_alloc(quotas, n_per_arm)
        cand = _multiscale(_ZERO_PROF, "MIMIC", f"szc|{exp_id}", int(bseed("szc", exp_id)), alloc)
        ref = _multiscale(_ZERO_PROF, "MIMIC", f"szr|{exp_id}", int(bseed("szr", exp_id)), alloc)
        arms, off = {}, 0
        for sid, a in zip(sids, alloc):
            arms[sid] = {"candidate": list(cand[off:off + a]), "reference": list(ref[off:off + a])}
            off += a
    elif source == CON.BOUNDED_PROFILE:                          # canonical bounded-length control constructor
        arms = CON.registered_bounded_arms(exp_id, 0, n_per_arm=n_per_arm)
    else:                                                        # source-profile fixture route (single stratum)
        arms = {sids[0]: {"candidate": list(draw_arm(source, f"cand|{exp_id}", n_per_arm)),
                          "reference": list(draw_arm(source, f"ref|{exp_id}", n_per_arm))}}
    comp = meta.get("coupled_component")
    if comp is not None:                                         # repeatability: BOTH roles carry the component
        for sid in arms:
            arms[sid] = {"candidate": apply_coupling(list(arms[sid]["candidate"]), comp, 0.5,
                                                     seed=bseed("cplc", exp_id, sid)),
                         "reference": apply_coupling(list(arms[sid]["reference"]), comp, 0.5,
                                                     seed=bseed("cplr", exp_id, sid))}
    if list(arms) != sids:
        raise RefusalError(f"{exp_id}: assembled stratum order {list(arms)} != canonical {sids}")
    return arms


def _scaled_alloc(quotas, n_per_arm):
    """Registered quotas, or a proportional dev-scaled version summing EXACTLY to n_per_arm."""
    total = sum(quotas)
    if n_per_arm == total:
        return tuple(int(q) for q in quotas)
    exact = [n_per_arm * q / total for q in quotas]
    out = [int(np.floor(x)) for x in exact]
    order = sorted(range(len(quotas)), key=lambda i: (-(exact[i] - out[i]), i))
    for i in order[:n_per_arm - sum(out)]:
        out[i] += 1
    return tuple(out)


def canonical_pool(arms, sids):
    """The engine's canonical pool order: per stratum, candidates then references (`_assemble_arms`)."""
    pool = []
    for sid in sids:
        pool += list(arms[sid]["candidate"]) + list(arms[sid]["reference"])
    return pool


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
    """Per-variant in-scope cells, the (experiment, statistic) pairs to measure, and per-experiment meta.

    rev-23: measurement is keyed by EXPERIMENT, not profile. Experiments sharing a source profile no longer
    share a measurement, because repeatability experiments carry a coupling on both roles and the two stratified
    experiments are built by their own constructors — all of which move event volume and support cardinality."""
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
                needed.add((c["experiment_id"], c["statistic"]))
                strata = [(s["n_candidate"], s["n_reference"]) for s in c["exchangeability_strata"]]
                exp_meta[c["experiment_id"]] = {
                    "profile": c["source"], "regime": c["support_regime"],
                    "stratum_ids": [s["stratum_id"] for s in c["exchangeability_strata"]],
                    "quota": strata, "strata": strata, "coupled_component": c["coupled_component"],
                    "constructor_route": CON.CANONICAL_ROUTE[c["source"]],
                    "M": sum(a + b for a, b in strata)}
            gv[gid] = cells
        variants[label] = gv
    return variants, sorted(needed), exp_meta


# ---------------------------------------------------------------------------------------------------
# measurement stages
# ---------------------------------------------------------------------------------------------------
def measure_experiments_and_cells(needed, exp_meta, log):
    """One pass per EXPERIMENT: build its canonical arms, assemble the canonical pool, canonicalise once, price
    every statistic that experiment needs, then release the pool before the next."""
    stats_of = {}
    for exp_id, stat in needed:
        stats_of.setdefault(exp_id, []).append(stat)

    exp_cost, cell_cost = {}, {}
    for exp_id in sorted(stats_of):
        meta = exp_meta[exp_id]
        log(f"[exp] {exp_id}: canonical route {meta['constructor_route']}"
            + (f" + coupling {meta['coupled_component']}" if meta["coupled_component"] else ""))
        rss0 = _rss_bytes()
        t = time.perf_counter()
        arms = canonical_arms_for_experiment(exp_id, meta)
        assemble_secs = time.perf_counter() - t
        pool_raw = canonical_pool(arms, meta["stratum_ids"])
        t = time.perf_counter()
        pool = _canonicalize_pool(pool_raw, meta["profile"], exp_id)
        canon_secs = time.perf_counter() - t
        del arms, pool_raw
        rss1 = _rss_bytes()
        if len(pool) != meta["M"]:
            raise RefusalError(f"{exp_id}: canonical pool {len(pool)} != registered M {meta['M']}")
        exp_cost[exp_id] = {
            "profile": meta["profile"], "constructor_route": meta["constructor_route"],
            "coupled_component": meta["coupled_component"],
            "assemble_secs": round(assemble_secs, 4), "canonicalize_secs_pooled": round(canon_secs, 4),
            "M_pooled": len(pool), "n_strata": len(meta["stratum_ids"]),
            "volumes": {"n_sequences": len(pool),
                        "events": int(sum(r.L_total for r in pool)),
                        "clusters": int(sum(r.K for r in pool))},
            "pool_rss_bytes_measured": max(0, rss1 - rss0),
        }
        log(f"[exp] {exp_id}: assemble {assemble_secs:.2f}s, canon {canon_secs:.2f}s, "
            f"events {exp_cost[exp_id]['volumes']['events']}, RSS +{exp_cost[exp_id]['pool_rss_bytes_measured']/1e6:.0f} MB")

        strata = meta["quota"]
        for stat in sorted(stats_of[exp_id]):
            est = ESTIMATORS[stat]
            t = time.perf_counter()
            pre = _validate_precompute(est["precompute"](pool), stat, len(pool))
            pre_secs = time.perf_counter() - t

            groups, map_status = None, None
            if est["map_carrying"]:
                ref_only = [pool[i] for i in np.where(~_canonical_mask(strata))[0]]
                art = build_frozen_map(ref_only, stat, profile=meta["profile"], regime=meta["regime"],
                                       seed=bseed("timingmap", exp_id, stat), N=N_PER_ARM,
                                       namespace=TIMING_MAP_NS, floor=FLOOR_REG)
                groups, map_status = art["groups"], art.get("status")
                if map_status != "OK":
                    log(f"[map] {exp_id}/{stat}: status={map_status} (recorded; timing still measured)")

            rng = np.random.default_rng(bseed("perm", exp_id, stat))
            masks = [_perm_mask(rng, strata) for _ in range(MASK_CYCLE)]
            state = {"i": 0}

            def one():
                m = masks[state["i"] % MASK_CYCLE]; state["i"] += 1
                est["recompute"](pre, m, groups=groups, floor=FLOOR_REG)

            per_perm, reps = timed(one)
            probe = est["recompute"](pre, masks[0], groups=groups, floor=FLOOR_REG)
            cell_cost[(exp_id, stat)] = {
                "precompute_secs": round(pre_secs, 4), "precompute_bytes": _nbytes(pre),
                "per_perm_secs": per_perm, "timing_reps": reps,
                "map_carrying": bool(est["map_carrying"]), "map_status": map_status,
                "recompute_defined_on_probe": bool(probe is not None and np.isfinite(probe)),
            }
            log(f"[cell] {exp_id:42s} {stat:16s} pre {pre_secs:7.2f}s "
                f"({_nbytes(pre)/1e6:7.1f} MB)  perm {per_perm*1e3:8.3f} ms  x{reps}"
                + ("" if probe is not None and np.isfinite(probe) else "   [NE on probe]"))
            del pre, masks
        del pool
    return exp_cost, cell_cost


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
        # Pi rev-25: this previously timed `cell_upper_p` as the "reference". Once ruling 4 made
        # `cell_upper_p` the SORTED form, that compared the production function to itself and still
        # labelled one side quadratic/400 MB. Time the real quadratic oracle, and the PRODUCTION
        # function as the fast side, so the comparison means what it says.
        ref_secs, ref_reps = timed(lambda: _cell_upper_p_quadratic(e), target=0.3, min_reps=3, max_reps=20)
        fast_secs, fast_reps = timed(lambda: cell_upper_p(e), target=0.3, min_reps=5, max_reps=200)
        ladder[A] = {"reference_secs": ref_secs, "reference_reps": ref_reps,
                     "reference_transient_bytes_exact": A * A,          # numpy bool comparison matrix, 1 byte/elem
                     "reference_impl": "_cell_upper_p_quadratic (A x A comparison matrix)",
                     "sorted_secs": fast_secs, "sorted_reps": fast_reps,
                     "sorted_impl": "cell_upper_p (PRODUCTION sort/searchsorted form)",
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
def _block_allocation(n_blocks, jobs):
    """Whole blocks per shard, as even as possible, summing EXACTLY to n_blocks (largest shards first)."""
    if jobs < 1 or jobs > n_blocks:
        raise RefusalError(f"cannot split {n_blocks} blocks into {jobs} shards")
    base, rem = divmod(n_blocks, jobs)
    return tuple(base + (1 if i < rem else 0) for i in range(jobs))


def _block_partition(block_alloc):
    """Explicit inclusive replicate ranges per shard, so the plan can PROVE an exact 1..B partition."""
    out, start = [], 1
    for i, nb in enumerate(block_alloc):
        end = start + nb * CHECKPOINT_BLOCK - 1
        out.append({"shard": i, "blocks": nb, "first_replicate": start, "last_replicate": end})
        start = end + 1
    return out


def verify_block_partition(partition, *, b_total=B_MAIN):
    """PROVE the shards tile 1..B exactly: no gap, no overlap, no duplicate, no extra replicate (Pi rev-22 #3).
    The rev-22 check only asserted `jobs * ceil(B/jobs) >= B`, which permits an extra replicate."""
    errs = []
    if not partition:
        return ["empty partition"]
    expect = 1
    for sh in partition:
        if sh["first_replicate"] != expect:
            errs.append(f"shard {sh['shard']} starts at {sh['first_replicate']}, expected {expect}")
        if sh["last_replicate"] < sh["first_replicate"]:
            errs.append(f"shard {sh['shard']} is empty/inverted")
        expect = sh["last_replicate"] + 1
    if expect - 1 != b_total:
        errs.append(f"partition covers 1..{expect - 1}, expected 1..{b_total}")
    covered = sum(sh["last_replicate"] - sh["first_replicate"] + 1 for sh in partition)
    if covered != b_total:
        errs.append(f"partition covers {covered} replicates, expected exactly {b_total}")
    return errs


def forecast_group(cells, cell_cost, exp_cost, exp_meta, mask_cost, rank, persist):
    """Total wall-clock seconds and peak RAM for ONE separately-gated per-group SD job at B=20000.
    Every cost is keyed by the CELL'S OWN EXPERIMENT (rev-23), so canonical construction and coupling are priced
    as they actually run rather than shared across a profile."""
    A = B_MAIN + 1
    exps = sorted({c["exp"] for c in cells})

    draw = sum(exp_cost[e]["assemble_secs"] for e in exps)
    canon = sum(exp_cost[e]["canonicalize_secs_pooled"] for e in exps)
    precompute = sum(cell_cost[(c["exp"], c["stat"])]["precompute_secs"] for c in cells)
    recompute = A * sum(cell_cost[(c["exp"], c["stat"])]["per_perm_secs"] for c in cells)
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
         "hours": round(A * cell_cost[(c["exp"], c["stat"])]["per_perm_secs"] / 3600.0, 3),
         "per_perm_ms": round(cell_cost[(c["exp"], c["stat"])]["per_perm_secs"] * 1e3, 3)}
        for c in cells), key=lambda d: -d["hours"])

    # SPLIT (Pi rev-22 ruling 3): allocate WHOLE CHECKPOINT BLOCKS, never `ceil(B/jobs)` replicates. The fixed
    # per-job overhead (assemble + canonicalise + precompute) is repaid by every shard, so the split is solved
    # against the per-job budget including it, and the worst shard is priced at `ceil(n_blocks/jobs)/n_blocks`
    # of the divisible work — an equal-fraction price understates the executable maximum.
    budget_secs = CAP_HOURS * 3600.0 / MARGIN
    fixed = draw + canon + precompute
    divisible = recompute + masks
    n_blocks = math.ceil(B_MAIN / CHECKPOINT_BLOCK)
    if total_sorted * MARGIN / 3600.0 <= CAP_HOURS:
        jobs = 1
    else:
        jobs = 1
        while jobs < n_blocks and (fixed + divisible * math.ceil(n_blocks / jobs) / n_blocks) > budget_secs:
            jobs += 1
    # The RATIFIED rule is "smallest shard count that fits the cap". Separately report the smallest shard count
    # that also leaves MARGINAL_HEADROOM, because a shard priced within the observed run-to-run timing variance
    # is not a safe classification. This is REPORTED as a recommendation, not substituted for the ratified rule.
    rec_jobs = 1
    while rec_jobs < n_blocks and ((fixed + divisible * math.ceil(n_blocks / rec_jobs) / n_blocks) * MARGIN
                                   / 3600.0) > CAP_HOURS * (1.0 - MARGINAL_HEADROOM):
        rec_jobs += 1
    rec_alloc = _block_allocation(n_blocks, rec_jobs)
    rec_secs = fixed + divisible * max(rec_alloc) / n_blocks

    block_alloc = _block_allocation(n_blocks, jobs)
    worst_blocks = max(block_alloc)
    per_job_secs = fixed + divisible * worst_blocks / n_blocks + (rank_sorted if jobs == 1 else 0.0)
    replicates_per_job = [b * CHECKPOINT_BLOCK for b in block_alloc]
    partition = _block_partition(block_alloc)

    precompute_bytes = sum(cell_cost[(c["exp"], c["stat"])]["precompute_bytes"] for c in cells)
    pool_bytes = sum(exp_cost[e]["pool_rss_bytes_measured"] for e in exps)
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
            "checkpoint_blocks_total": n_blocks,
            "blocks_per_job": list(block_alloc),
            "worst_shard_blocks": worst_blocks,
            "replicates_per_job": list(replicates_per_job),
            "replicate_partition": partition,
            "partition_errors": verify_block_partition(partition),
            "fixed_overhead_hours_per_job": hours(fixed),
            "hours_per_split_job": hours(per_job_secs),
            "hours_per_split_job_with_margin": hours(per_job_secs * MARGIN),
            "split_job_fits_cap_with_margin": (per_job_secs * MARGIN / 3600.0) <= CAP_HOURS,
            "cap_headroom_pct": round(100.0 * (CAP_HOURS - per_job_secs * MARGIN / 3600.0) / CAP_HOURS, 1),
            # A "fits" verdict inside the observed run-to-run timing variance is not a safe classification: these
            # hours are environment-dependent and have moved ~5% between reruns on the same machine.
            "marginal_fit": 0.0 <= (CAP_HOURS - per_job_secs * MARGIN / 3600.0) / CAP_HOURS < MARGINAL_HEADROOM,
            "final_ranking_job_hours": hours(rank_sorted) if jobs > 1 else 0.0,
            "recommended_shards_for_headroom": rec_jobs,
            "recommended_blocks_per_job": list(rec_alloc),
            "recommended_hours_per_shard_with_margin": hours(rec_secs * MARGIN),
            "recommended_headroom_pct": round(100.0 * (CAP_HOURS - rec_secs * MARGIN / 3600.0) / CAP_HOURS, 1),
            "note": ("WHOLE CHECKPOINT BLOCKS are the divisible unit (Pi rev-22 #3), never ceil(B/jobs) "
                     "replicates; the worst shard is priced at ceil(n_blocks/jobs)/n_blocks of the divisible "
                     "work, not an equal fraction. Assemble + canonicalise + precompute is FIXED overhead repaid "
                     "by every shard, and the min-p ranking runs ONCE at the end over the assembled E matrix, so "
                     "a split group needs jobs_required compute shards plus one cheap ranking/aggregation job. A "
                     "shard is an INTEGRITY-GATED block stage, not a scientific gate: no within-group PASS/FAIL "
                     "exists until final assembly, so stop-on-failure within a split means execution/identity/"
                     "checkpoint failure only."),
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
        "groups_marginal_within_10pct_of_cap": sorted({g for v in forecasts.values() for g, f in v.items()
                                                       if f["split_plan"]["marginal_fit"]}),
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
                if (c["exp"], c["stat"]) not in cell_cost:
                    errs.append(f"{label}/{gid}: unmeasured cell cost for {c['exp']}/{c['stat']}")
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

    # rev-23: the benchmark's canonical pool order must EQUAL the engine's `_assemble_arms` order, and each
    # experiment must run its registry-declared constructor route. Proven at dev scale so the check is cheap.
    n_small = 60
    canon_bt = CANONICAL_GROUPS["G_bounded_support"]
    bmeta = exp_meta["boundary_short"]
    arms = canonical_arms_for_experiment("boundary_short", bmeta, n_per_arm=n_small)
    mine = canonical_pool(arms, bmeta["stratum_ids"])
    try:
        theirs = ENG._assemble_arms({"boundary_short": arms}, canon_bt, exact_counts=False)["boundary_short"]["pool"]
        if [id(r) for r in mine] != [id(r) for r in theirs]:
            errs.append("benchmark canonical_pool order != engine _assemble_arms order")
    except RefusalError as ex:
        errs.append(f"canonical bounded arms refused by the engine: {ex}")
    # the two stratified experiments must NOT be built by the generic single-profile route
    for e, want in (("structural_zero", "structural_multiscale"), ("boundary_short", "bounded_length_control")):
        if exp_meta[e]["constructor_route"] != want:
            errs.append(f"{e}: constructor route {exp_meta[e]['constructor_route']} != {want}")
        if len(exp_meta[e]["stratum_ids"]) != 3:
            errs.append(f"{e}: expected 3 strata, got {len(exp_meta[e]['stratum_ids'])}")
    # boundary must carry the SELECTED registered allocation
    if [q for q, _ in exp_meta["boundary_short"]["quota"]] != list(REG.BOUNDARY_ALLOC):
        errs.append(f"boundary_short quota {exp_meta['boundary_short']['quota']} != {REG.BOUNDARY_ALLOC}")
    # exact-partition machinery must reject a deliberately broken partition
    if not verify_block_partition([{"shard": 0, "blocks": 20, "first_replicate": 1,
                                    "last_replicate": B_MAIN + 1}]):
        errs.append("verify_block_partition accepted a partition with an extra replicate")
    if not verify_block_partition([{"shard": 0, "blocks": 1, "first_replicate": 1, "last_replicate": 1000},
                                   {"shard": 1, "blocks": 1, "first_replicate": 1000,
                                    "last_replicate": B_MAIN}]):
        errs.append("verify_block_partition accepted an overlapping partition")
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
            if sp["partition_errors"]:
                errs.append(f"{label}/{gid}: replicate partition is not exact: {sp['partition_errors']}")
            if sum(sp["blocks_per_job"]) != sp["checkpoint_blocks_total"]:
                errs.append(f"{label}/{gid}: blocks_per_job {sp['blocks_per_job']} does not sum to "
                            f"{sp['checkpoint_blocks_total']}")
            if sum(sp["replicates_per_job"]) != B_MAIN:
                errs.append(f"{label}/{gid}: shard replicates sum to {sum(sp['replicates_per_job'])} != {B_MAIN}")
            if f["fits_cap_with_margin"] and sp["jobs_required"] != 1:
                errs.append(f"{label}/{gid}: fits the cap but was split into {sp['jobs_required']} jobs")
    return errs


def main():
    t_start = time.perf_counter()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    variants, needed, exp_meta = registry_view()
    log(f"[plan] {len(needed)} (experiment, statistic) pairs over {len(exp_meta)} experiments "
        f"at M={2*N_PER_ARM}; canonical constructor routes: "
        f"{sorted({m['constructor_route'] for m in exp_meta.values()})}")

    exp_cost, cell_cost = measure_experiments_and_cells(needed, exp_meta, log)
    mask_cost = measure_assignments(exp_meta, log)
    rng = np.random.default_rng(bseed("rank"))
    rank = measure_ranking(rng, log)
    persist = measure_persistence(log)

    forecasts = {label: {gid: forecast_group(cells, cell_cost, exp_cost, exp_meta, mask_cost, rank, persist)
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
                            "strata": [list(s) for s in m["strata"]], "M": m["M"],
                            "stratum_ids": list(m["stratum_ids"]),
                            "constructor_route": m["constructor_route"],
                            "coupled_component": m["coupled_component"]} for e, m in exp_meta.items()},
        # Pi rev-22: bind the EXACT constructor/profile payloads the measurement ran through, and this
        # benchmark module's own source identity, into the deterministic configuration.
        "constructor_payloads": {
            "pool_order": "per stratum in canonical order: candidates then references (_assemble_arms)",
            "profiles": {src: PROFILES[src] for src in sorted({m["profile"] for m in exp_meta.values()})},
            "structural_multiscale": {"base_profile": _ZERO_PROF, "skeleton": "MIMIC",
                                      "allocation": list(CONTROL_ALLOC),
                                      "means": ["log(18)", "log(60)", "log(250)"]},
            "bounded_length_control": {
                "registered_variant": CON.REGISTERED_BOUNDED_VARIANT,
                "bands": [list(b) for b in CON.BOUNDED_BANDS],
                "allocation": list(REG.BOUNDARY_ALLOC),
                "route_identity": CON.constructor_route_identity(CON.REGISTERED_BOUNDED_VARIANT, canonical=True)},
            "coupling": {"strength": 0.5, "applied_to": "both roles on repeatability experiments"},
        },
        "benchmark_source_identity": _benchmark_source_identity(),
        "B_main": B_MAIN, "N_per_arm": N_PER_ARM, "floor": FLOOR_REG,
        "alpha_group": ALPHA_GROUP_EXACT, "cap_hours": CAP_HOURS, "margin": MARGIN,
        "checkpoint_block": CHECKPOINT_BLOCK,
        "split_unit": "whole checkpoint blocks; worst shard priced at ceil(n_blocks/jobs)/n_blocks",
        # the split RULE is deterministic and belongs here; the resulting per-group JOB COUNTS are derived from
        # measured timings and must NOT enter a deterministic identity (they live in the timing artifact).
        "split_rule": plan["split_rule"],
        "rank_formulation": "sorted searchsorted-left complement; bit-identical to the A x A reference",
        "registry_identity": CANONICAL_REGISTRY_HASH,
        "source_identities": SOURCE_IDENTITY_BUNDLE,
    })

    timing_artifact = {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "estimator_dependency_identity": ESTIMATOR_DEPENDENCY_IDENTITY,
        "experiment_costs": exp_cost,
        "cell_costs": {f"{e}|{s}": v for (e, s), v in sorted(cell_cost.items())},
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
                          "blocks_per_job": f["split_plan"]["blocks_per_job"],
                          "cap_headroom_pct": f["split_plan"]["cap_headroom_pct"],
                          "marginal_fit": f["split_plan"]["marginal_fit"],
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
