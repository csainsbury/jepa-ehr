#!/usr/bin/env python3
"""Oracle realism v3 — SPARSE, trusted-boundary group-power demonstration (Pi rev-7 #1/#3/#4).

Rebuilds the rev-7 demo, whose perturbation was DENSE (all nine experiments) and which went through the porous
caller boundary. Fixes:
  * SPARSE alternative — perturb EXACTLY one experiment (`perturb_exp_id`); assert every OTHER experiment pool is
    content-identical to its null construction (executable, hashed);
  * TRUSTED boundary — evaluate ONLY via `engine.gate_group_trusted(group_id, pools_by_exp, ...)`: the engine loads
    the canonical registry, computes precompute itself, and refuses/NEs non-finite values (no caller check/Δ/pre);
  * CANONICAL structural-zero — the registered multiscale control constructor (means 18/60/250, allocation split),
    not three same-profile draws;
  * MAP PROVENANCE — each map artifact records the EXACT seed/namespace of the sample that built it;
  * argmin reported.

The permutation count here is a LABELLED mechanical development B (not the registered B=20000 evaluator). All three
active components get group-level sensitivity evidence. Development-only. Run:
    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_group_power.py
"""
from __future__ import annotations

import hashlib
import json
from math import sqrt

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_battery import _multiscale, _ZERO_PROF
import scripts.oracle_realism_v3_engine as ENG
from scripts.oracle_realism_v3_engine import gate_group_trusted
from scripts.oracle_realism_v3_map import build_frozen_map, map_identity
import scripts.oracle_realism_v3_registry as REG

DEV_NS = "v3-grouppower-dev"
N = 810                             # dev per-side, divisible by 3 for balanced structural-zero strata (registered N=8000)
DEV_FLOOR = 60                      # LABELLED dev floor (registered 500)
ZALLOC = (N // 3, N // 3, N // 3)   # dev structural-zero per-stratum allocation (equal; registered is 2667/2667/2666)
B_MECH = 999                        # LABELLED mechanical dev permutation count (NOT the registered B=20000)
PRIMARY = {"burst_timing": ["S3_tau", "S3_loggap"], "mark_burst_tie": ["S4_abs"],
           "cluster_size_mark_diversity": ["S7_abs"]}
COMPONENT_GROUP = {"burst_timing": "G_full_burst_timing", "mark_burst_tie": "G_full_class_mark",
                   "cluster_size_mark_diversity": "G_full_class_mark"}


def dseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *p))).encode()).digest()[:6], "big")


def _draw(profile, *tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, *tag)), int(dseed(profile, *tag))


def draw_experiment_pool(exp_id, meta, trial, perturb_component=None):
    """Stratum-interleaved pool [candS0,refS0,candS1,refS1,...] for one experiment. exp_id in every seed."""
    cond, src, n_strata = meta["condition"], meta["source"], meta["n_strata"]
    if cond == "structural_zero":
        cand = _multiscale(_ZERO_PROF, "MIMIC", f"zc|{exp_id}", int(dseed("zc", exp_id, trial)), ZALLOC)
        ref = _multiscale(_ZERO_PROF, "MIMIC", f"zr|{exp_id}", int(dseed("zr", exp_id, trial)), ZALLOC)
        if perturb_component:
            cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
        pool, off = [], 0
        for a in ZALLOC:
            pool += cand[off:off + a] + ref[off:off + a]; off += a
        return pool
    cand, _ = _draw(src, exp_id, "cand", trial); ref, _ = _draw(src, exp_id, "ref", trial)
    if cond == "repeatability":
        comp = meta["coupled_component"]
        ref = apply_coupling(list(ref), comp, 0.5, seed=dseed("cpl_ref", exp_id, trial))
        cand = apply_coupling(list(cand), comp, 0.5, seed=dseed("cpl_cand", exp_id, trial))
    if perturb_component:
        cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
    return list(cand) + list(ref)


def _pool_hash(pool):
    return canonical_hash([[int(r.L_total), int(r.K), r.class_ids.tolist(), r.cluster_ids.tolist(),
                            np.asarray(r.timestamps).tolist()] for r in pool])   # incl. timestamps (burst_timing)


def build_pools(group_id, trial, perturb_exp_id=None, perturb_component=None):
    canon = ENG.CANONICAL_GROUPS[group_id]
    return {e: draw_experiment_pool(e, m, trial, perturb_component if e == perturb_exp_id else None)
            for e, m in canon["experiments"].items()}


def _independent_map(source, check):
    """INDEPENDENT dev map-design sample per (source, check); the artifact records the EXACT seed/namespace used."""
    sk = "SCID" if "scid" in source else "MIMIC"
    s = int(dseed(source, "MAPDESIGN", check))                 # the ACTUAL draw seed (Pi #6 provenance)
    map_ref = sample_fixture(sk, PROFILES[source], N, seed=s)
    return build_frozen_map(map_ref, check, profile=source, regime="full", namespace=DEV_NS, seed=s, N=N,
                            floor=DEV_FLOOR)


def build_map_artifacts(group_id):
    canon = ENG.CANONICAL_GROUPS[group_id]; cache, arts = {}, {}
    for cc in canon["cells"]:
        if cc["map_carrying"]:
            src = canon["experiments"][cc["exp"]]["source"]; key = (src, cc["check"])
            cache.setdefault(key, _independent_map(src, cc["check"]))
            arts[cc["cell_id"]] = cache[key]
    return arts


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def component_demo(component, T, perturb_exp_id="null_mimic"):
    """MECHANICAL demonstration (B<resolution): shows the CORRECTED sparse + trusted machinery, argmin ATTRIBUTION
    to the perturbed experiment's registered primary cell, and direction (perturbed p_g < null). Rejection/power at
    alpha_group needs B>=K_g/alpha_group -> the registered-scale run (NOT authorized)."""
    gid = COMPONENT_GROUP[component]; arts = build_map_artifacts(gid)
    null = gate_group_trusted(gid, build_pools(gid, trial=0), seed=int(dseed("null", component)), B=B_MECH,
                              map_artifacts=arts)
    trials, unchanged_ok = [], True
    for t in range(1, T + 1):
        base = build_pools(gid, trial=t)
        pert = build_pools(gid, trial=t, perturb_exp_id=perturb_exp_id, perturb_component=component)
        for e in base:                                           # SPARSE: non-target pools content-identical to null
            if e != perturb_exp_id and _pool_hash(base[e]) != _pool_hash(pert[e]):
                unchanged_ok = False
        r = gate_group_trusted(gid, pert, seed=int(dseed("pow", component, t)), B=B_MECH, map_artifacts=arts)
        am = r.get("argmin_cell")
        attributes = bool(am and f"|{perturb_exp_id}|" in am and any(am.endswith(p) for p in PRIMARY[component]))
        trials.append({"perturbed_p_g": r.get("p_g"), "argmin": am,
                       "argmin_attributes_to_perturbed_primary": attributes,
                       "direction_ok": r.get("p_g") is not None and null.get("p_g") is not None
                       and r["p_g"] < null["p_g"]})
    return {"group": gid, "component": component, "perturbed_exp_id": perturb_exp_id,
            "primary_cells": PRIMARY[component], "null_verdict": null["verdict"], "null_p_g": null.get("p_g"),
            "T": T, "trials": trials, "sparse_nontarget_pools_unchanged": unchanged_ok,
            "argmin_attribution_rate": round(sum(x["argmin_attributes_to_perturbed_primary"] for x in trials) / T, 3),
            "direction_rate": round(sum(x["direction_ok"] for x in trials) / T, 3),
            "mode": "MECHANICAL (B<resolution); power qualification = registered-scale run, NOT authorized"}


def main():
    ENG.FLOOR = DEV_FLOOR
    demos = {"burst_timing": component_demo("burst_timing", T=4),
             "mark_burst_tie": component_demo("mark_burst_tie", T=2),
             "cluster_size_mark_diversity": component_demo("cluster_size_mark_diversity", T=2)}
    variants = {}
    for name, u in (("with_exemption", True), ("without_exemption", False)):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=u); g = REG.build_groups(sd)
        variants[name] = {"M0": len([c for c in sd if c["scope"] == "in"]),
                          "G_bounded_support": len(g["G_bounded_support"]["cells"])}
    agg = {"dev_namespace": DEV_NS, "N": N, "dev_floor": DEV_FLOOR, "B_mechanical": B_MECH,
           "canonical_registry_hash": ENG.CANONICAL_REGISTRY_HASH,
           "fixes": ["SPARSE single-experiment perturbation + executable non-target unchanged assertion",
                     "TRUSTED engine boundary (gate_group_trusted): canonical registry loaded internally, "
                     "precompute computed internally, non-finite -> NE",
                     "CANONICAL multiscale structural-zero constructor", "map provenance records exact seed/namespace",
                     "argmin reported"],
           "component_group_sensitivity": demos,
           "exemption_reporting": {"note": "full-support groups are exemption-INVARIANT; registered-N preflight "
                                   "decided both S3 exemptions on detection (0.0<0.5). Reported with/without each.",
                                   "variants": variants},
           "B_note": f"B={B_MECH} is a LABELLED MECHANICAL dev test, NOT the registered B=20000 evaluator (Pi #3).",
           "scale_note": f"dev floor={DEV_FLOOR}/N={N} LABELLED dev scale; exact size from randomization theory.",
           "authorization": "dev-only; no reserved map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
