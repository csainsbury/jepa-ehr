#!/usr/bin/env python3
"""Oracle realism v3 — CORRECTED exact-group product/min-p power demonstration via the engine (Pi rev-6 #1/#4).

Rebuilds the rev-6 demo, which was NOT exchangeability-conformant. Fixes:
  * MAP LEAKAGE -> each S3_loggap/S5/S6/S7 cell's frozen map is drawn from an INDEPENDENT dev map-design sample per
    (profile, check), applied unchanged to observed + permuted assignments (never the observed reference arm);
  * CROSS-EXPERIMENT DEPENDENCE -> `exp_id` is in every fixture seed, so the nine experiment pairs are independent;
  * STRATA -> the structural-zero experiment uses its exact within-stratum quotas (3 strata), not one pooled split;
  * ATTRIBUTION -> a burst_timing@0.5 perturbation is a SPARSE one-experiment / two-primary-cell alternative
    (S3_tau AND S3_loggap), and all THREE active components get group-level sensitivity evidence (burst_timing via
    the burst-timing group; mark_burst_tie and cluster_size_mark_diversity via the class-mark group);
  * TIED-KS -> positive gaps rounded to the registered 8-dp support before unique-support KS (engine estimator).

Runs through the registry-owned engine (scripts/oracle_realism_v3_engine.py) — callers provide DATA + cell specs,
never a statfn. Development-only; dev floor/N are LABELLED (exact SD size is from randomization theory + the
exhaustive tests, not this run). NO reserved map draw / calibration / eval seed. Run:
    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_group_power.py
"""
from __future__ import annotations

import hashlib
import json
from math import sqrt, ceil

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_verifier import _TAU, _KS, _DT0, _LOGGAP, _OCC, _TV
import scripts.oracle_realism_v3_engine as ENG
from scripts.oracle_realism_v3_engine import ESTIMATORS, gate_group, rng_identity
from scripts.oracle_realism_v3_map import build_frozen_map, map_identity
import scripts.oracle_realism_v3_registry as REG

DEV_NS = "v3-grouppower-dev"
N = 800                             # dev per-side sample (LABELLED dev scale; registered N=8000)
# NOTE: engine floors are the REGISTERED 500; for this LABELLED dev demonstration we lower BOTH the map-coarsening
# floor (build_frozen_map floor=DEV_FLOOR) AND the engine per-perm floor (ENG.FLOOR) to DEV_FLOOR so the machinery
# is exercised end-to-end at dev scale. Exact SD size is from randomization theory + the exhaustive tests.
DEV_FLOOR = 60
ALPHA_SD, G = 0.04, 6
ALPHA_GROUP = ALPHA_SD / G
DELTA = {"S3_tau": _TAU, "S3_loggap": _LOGGAP, "delta_t_zero_abs": _DT0, "positive_gap_ks": _KS,
         "S4_abs": _OCC, "S5_abs": _OCC, "S6_tv": _TV, "S7_abs": _OCC, "class_tv": _TV, "occupancy_abs": _OCC}
GROUPS = {"burst_timing": ["S3_tau", "S3_loggap", "delta_t_zero_abs", "positive_gap_ks"],
          "class_mark": ["S4_abs", "S5_abs", "S6_tv", "S7_abs", "class_tv", "occupancy_abs"]}
PRIMARY = {"burst_timing": ["S3_tau", "S3_loggap"], "mark_burst_tie": ["S4_abs"],
           "cluster_size_mark_diversity": ["S7_abs"]}


def dseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *p))).encode()).digest()[:6], "big")


def _draw(profile, *tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, *tag))


def _draw_strat(profile, alloc, *tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    out = []
    for i, a in enumerate(alloc):
        out += sample_fixture(sk, PROFILES[profile], a, seed=dseed(profile, "strat", i, *tag))
    return out


def draw_experiment(exp_id, cond, src, comp, trial, perturb_component=None):
    """(pool_sequences_in_mask_order, strata) for one full SD experiment. exp_id is in EVERY seed (Pi #1). The
    structural-zero experiment uses 3 within-stratum quotas. A perturbation applies `perturb_component`@0.5 to
    THIS experiment's candidate arm only (a sparse one-experiment alternative)."""
    if cond == "structural_zero":
        alloc = (N // 3, N // 3, N - 2 * (N // 3))
        cand = _draw_strat(src, alloc, exp_id, "cand", trial)
        ref = _draw_strat(src, alloc, exp_id, "ref", trial)
        if perturb_component:
            cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
        # interleave per stratum: cand_s0, ref_s0, cand_s1, ref_s1, cand_s2, ref_s2
        pool, strata, off = [], [], 0
        for a in alloc:
            pool += cand[off:off + a] + ref[off:off + a]; strata.append((a, a)); off += a
        return pool, strata
    ref = _draw(src, exp_id, "ref", trial); cand = _draw(src, exp_id, "cand", trial)
    if cond == "repeatability":
        ref = apply_coupling(list(ref), comp, 0.5, seed=dseed("cpl_ref", exp_id, trial))
        cand = apply_coupling(list(cand), comp, 0.5, seed=dseed("cpl_cand", exp_id, trial))
    if perturb_component:
        cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
    return list(cand) + list(ref), [(len(cand), len(ref))]


def _independent_map(profile, regime, check):
    """INDEPENDENT dev map-design sample per (profile, check) — NOT the observed reference arm (fixes leakage)."""
    map_ref = _draw(profile, "MAPDESIGN", check)               # disjoint seed namespace tag
    return build_frozen_map(map_ref, check, profile=profile, regime=regime,
                            seed=int(dseed("mapdesign", profile, check)), N=N, floor=DEV_FLOOR)


def build_group(group, trial, perturb_component=None):
    full = [e for e in REG.SD_EXPERIMENTS if e[4] == "full"]
    cells, experiments = [], {}
    map_cache = {}
    for exp_id, cond, src, comp, _ in full:
        pool, strata = draw_experiment(exp_id, cond, src, comp, trial, perturb_component)
        experiments[exp_id] = {"strata": strata, "source": src, "replicate_seed": trial,
                               "coupled_component": comp}
        for chk in group:
            est = ESTIMATORS[chk]; pre = est["precompute"](pool)
            cell = {"cell_id": f"SD|{exp_id}|{chk}", "exp": exp_id, "check": chk, "pre": pre, "delta": DELTA[chk]}
            if est["map_carrying"]:
                key = (src, "full", chk)
                if key not in map_cache:
                    map_cache[key] = _independent_map(src, "full", chk)
                cell["map_art"] = map_cache[key]
            cells.append(cell)
    registered = {"cell_ids": [c["cell_id"] for c in cells], "alpha_group": ALPHA_GROUP, "floor_policy": "dev",
                  "map_hashes": {c["cell_id"]: (map_identity(c["map_art"]) if c.get("map_art") else None) for c in cells},
                  "rng_identities": {e: rng_identity(m["source"], m["replicate_seed"], m["coupled_component"])
                                     for e, m in experiments.items()}}
    return {"cells": cells, "experiments": experiments, "registered": registered}


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _resolve_B(group):
    K_g = len(group) * sum(1 for e in REG.SD_EXPERIMENTS if e[4] == "full")
    return K_g, int(ceil(K_g / ALPHA_GROUP / 100.0)) * 100


def component_demo(group_name, perturb_component, T):
    group = GROUPS[group_name]; K_g, B = _resolve_B(group)
    null_spec = build_group(group, trial=0)
    null_spec.update(B=B, seed=dseed("null", group_name))
    null = gate_group(null_spec)
    hits, evald = 0, 0
    for t in range(1, T + 1):
        spec = build_group(group, trial=t, perturb_component=perturb_component)
        spec.update(B=B, seed=dseed("pow", group_name, perturb_component, t))
        r = gate_group(spec)
        if r["verdict"] != "NOT_EVALUABLE":
            evald += 1; hits += (r["verdict"] == "FAIL")
    return {"group": group_name, "K_g": K_g, "B": B, "perturbed_component": perturb_component,
            "primary_cells": PRIMARY[perturb_component], "null_verdict": null["verdict"], "null_p_g": null.get("p_g"),
            "T": T, "evaluated": evald, "detections": hits,
            "power": round(hits / evald, 3) if evald else None, "wilson95": _wilson(hits, evald)}


def main():
    ENG.FLOOR = DEV_FLOOR                                       # dev-scale floor (LABELLED); registered floor is 500

    # burst_timing group: a small power CI (4 trials). class_mark group builds are ~45s each (54 heavy cells x 9
    # experiments), so the two class-mark components get a SINGLE-DRAW sensitivity check (null + 1 perturbed),
    # LABELLED as such (not an empirical power CI).
    demos = {
        "burst_timing": component_demo("burst_timing", "burst_timing", T=4),
        "mark_burst_tie": component_demo("class_mark", "mark_burst_tie", T=1),
        "cluster_size_mark_diversity": component_demo("class_mark", "cluster_size_mark_diversity", T=1),
    }
    variants = {}
    for name, u in (("with_exemption", True), ("without_exemption", False)):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=u); groups = REG.build_groups(sd)
        variants[name] = {"M0": len([c for c in sd if c["scope"] == "in"]),
                          "G_bounded_support": len(groups["G_bounded_support"]["cells"])}

    agg = {"dev_namespace": DEV_NS, "N": N, "dev_floor": DEV_FLOOR,
           "engine": "registry-owned dispatch (scripts/oracle_realism_v3_engine.py); callers provide DATA only",
           "fixes": ["independent per-(profile,check) map-design sample (no leakage)", "exp_id in every fixture seed",
                     "structural-zero 3 within-stratum quotas", "sparse one-exp/two-primary-cell attribution",
                     "all three active components", "tied-KS on 8-dp-rounded unique support"],
           "component_group_sensitivity": demos,
           "exemption_reporting": {"note": "the full-support groups are exemption-INVARIANT (the S3 exemption only "
                                   "removes cells from the BOUNDED group). Registered-N preflight decided both S3 "
                                   "exemptions on the DETECTION criterion (bounded detect 0.0<0.5), NOT structural "
                                   "un-calibratability (the map IS OK at N=8000). Reported with AND without each.",
                                   "variants": variants},
           "scale_note": f"dev floor={DEV_FLOOR}/N={N} LABELLED dev scale; exact SD size from randomization theory + "
                         "exhaustive tests, not this run.",
           "authorization": "dev-only; no reserved map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
