#!/usr/bin/env python3
"""Oracle realism v3 — SPARSE, PAIRED, dev-boundary group-power demonstration (Pi rev-7 #1/#3, rev-8 #7).

Runs ONLY through the explicit development boundary `engine.gate_group_dev` (no module-global mutation; floor is a
parameter). Fixes over rev-8:
  * STRUCTURED arms — each experiment is `{stratum_id: {candidate, reference}}`; structural-zero uses its canonical
    multiscale constructor (data AND map-design sample), not a single-profile draw;
  * SPARSE — perturb EXACTLY one experiment; assert (hashed) every non-target arm is identical to null AND the
    target arm changed;
  * PAIRED direction (Pi rev-8 #7) — for each trial, evaluate the base pool AND its one-experiment perturbation
    under the SAME permutation seed, and compare `p_g(perturbed)` with `p_g(base)`; argmin reported;
  * map artifacts are CONTEXT-bound (dev floor + profile/regime) and one shared identity per (profile,regime,check).

The permutation count is a LABELLED MECHANICAL dev test (B<resolution); a rejection/power qualification needs the
registered B=20000 run (NOT authorized). Development-only. Run:
    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_group_power.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_battery import _multiscale, _ZERO_PROF
import scripts.oracle_realism_v3_engine as ENG
from scripts.oracle_realism_v3_engine import gate_group_dev
from scripts.oracle_realism_v3_map import build_frozen_map
import scripts.oracle_realism_v3_registry as REG

DEV_NS = "v3-grouppower-dev"
N = 810                             # dev per-side; divisible by 3 for balanced structural-zero strata (registered N=8000)
DEV_FLOOR = 60                      # dev floor (registered 500) — passed as a PARAMETER, not a mutated global
ZALLOC = (N // 3, N // 3, N // 3)   # dev structural-zero per-stratum allocation (equal; registered 2667/2667/2666)
B_MECH = 999                        # LABELLED mechanical dev permutation count (NOT registered B=20000)
PRIMARY = {"burst_timing": ["S3_tau", "S3_loggap"], "mark_burst_tie": ["S4_abs"],
           "cluster_size_mark_diversity": ["S7_abs"]}
COMPONENT_GROUP = {"burst_timing": "G_full_burst_timing", "mark_burst_tie": "G_full_class_mark",
                   "cluster_size_mark_diversity": "G_full_class_mark"}
PERTURB_EXP = "null_mimic"


def dseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *p))).encode()).digest()[:6], "big")


def _draw(profile, *tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, *tag))


def draw_experiment_arms(exp_id, meta, trial, perturb_component=None):
    """{stratum_id: {candidate, reference}} for one experiment. exp_id in every seed. structural-zero uses the
    canonical multiscale constructor split into its registered strata."""
    cond, src, sids = meta["condition"], meta["source"], meta["stratum_ids"]
    if cond == "structural_zero":
        cand = _multiscale(_ZERO_PROF, "MIMIC", f"zc|{exp_id}", int(dseed("zc", exp_id, trial)), ZALLOC)
        ref = _multiscale(_ZERO_PROF, "MIMIC", f"zr|{exp_id}", int(dseed("zr", exp_id, trial)), ZALLOC)
        if perturb_component:
            cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
        arms, off = {}, 0
        for sid, a in zip(sids, ZALLOC):
            arms[sid] = {"candidate": list(cand[off:off + a]), "reference": list(ref[off:off + a])}; off += a
        return arms
    cand = _draw(src, exp_id, "cand", trial); ref = _draw(src, exp_id, "ref", trial)
    if cond == "repeatability":
        comp = meta["coupled_component"]
        ref = apply_coupling(list(ref), comp, 0.5, seed=dseed("cpl_ref", exp_id, trial))
        cand = apply_coupling(list(cand), comp, 0.5, seed=dseed("cpl_cand", exp_id, trial))
    if perturb_component:
        cand = apply_coupling(list(cand), perturb_component, 0.5, seed=dseed("perturb", exp_id, trial))
    return {sids[0]: {"candidate": list(cand), "reference": list(ref)}}


def build_arms(group_id, trial, perturb_exp_id=None, perturb_component=None):
    canon = ENG.CANONICAL_GROUPS[group_id]
    return {e: draw_experiment_arms(e, m, trial, perturb_component if e == perturb_exp_id else None)
            for e, m in canon["experiments"].items()}


def _arms_hash(arms):
    def seqrep(r):
        return [int(r.L_total), int(r.K), r.class_ids.tolist(), r.cluster_ids.tolist(),
                np.asarray(r.timestamps).tolist()]
    return canonical_hash({sid: {"candidate": [seqrep(r) for r in a["candidate"]],
                                 "reference": [seqrep(r) for r in a["reference"]]} for sid, a in arms.items()})


def _independent_map(source, check):
    """INDEPENDENT dev map-design sample per (source, check); records exact seed/namespace/N. structural-zero uses
    the CANONICAL multiscale constructor for the map-design sample too (Pi rev-7 #4/#8)."""
    s = int(dseed(source, "MAPDESIGN", check))
    if source == "structural_zero_control":
        map_ref = _multiscale(_ZERO_PROF, "MIMIC", f"mapdesign|{check}", s, ZALLOC)
    else:
        sk = "SCID" if "scid" in source else "MIMIC"
        map_ref = sample_fixture(sk, PROFILES[source], N, seed=s)
    return build_frozen_map(map_ref, check, profile=source, regime="full", namespace=DEV_NS, seed=s,
                            N=len(map_ref), floor=DEV_FLOOR)


def build_map_artifacts(group_id):
    canon = ENG.CANONICAL_GROUPS[group_id]; cache, arts = {}, {}
    for cc in canon["cells"]:
        if cc["map_carrying"]:
            src = canon["experiments"][cc["exp"]]["source"]; cache.setdefault((src, cc["check"]),
                                                                              _independent_map(src, cc["check"]))
            arts[cc["cell_id"]] = cache[(src, cc["check"])]
    return arts


def component_demo(component, T):
    gid = COMPONENT_GROUP[component]; arts = build_map_artifacts(gid)
    dch = canonical_hash({"mode": "dev", "floor": DEV_FLOOR, "B": B_MECH, "group": gid})
    trials, unchanged_ok, target_changed_ok = [], True, True
    dev_cfg_id = None; stable_id_paired_ok = True
    for t in range(1, T + 1):
        base = build_arms(gid, trial=t)
        pert = build_arms(gid, trial=t, perturb_exp_id=PERTURB_EXP, perturb_component=component)
        for e in base:                                              # sparse: non-target identical, target changed
            same = _arms_hash(base[e]) == _arms_hash(pert[e])
            if e != PERTURB_EXP and not same:
                unchanged_ok = False
            if e == PERTURB_EXP and same:
                target_changed_ok = False
        seed = int(dseed("pair", component, t))                     # PAIRED: same permutation seed for base + perturbed
        rb = gate_group_dev(gid, base, seed=seed, B=B_MECH, floor=DEV_FLOOR, map_artifacts=arts, dev_config_hash=dch)
        rp = gate_group_dev(gid, pert, seed=seed, B=B_MECH, floor=DEV_FLOOR, map_artifacts=arts, dev_config_hash=dch)
        dev_cfg_id = rb.get("dev_config_stable_identity")           # RC5: stable across trials (seed-invariant)
        # RC5 follow-through (Pi rev-10): base and perturbation share the SAME dev config (same counts/maps/registry/
        # protocol/namespace; only sequence CONTENT differs) -> their stable identities MUST be equal every trial.
        if rb.get("dev_config_stable_identity") != rp.get("dev_config_stable_identity"):
            stable_id_paired_ok = False
        am = rp.get("argmin_cell")
        trials.append({"p_g_base": rb.get("p_g"), "p_g_perturbed": rp.get("p_g"),
                       "base_verdict": rb["verdict"], "perturbed_verdict": rp["verdict"], "argmin": am,
                       "argmin_attributes_to_perturbed_primary":
                           bool(am and f"|{PERTURB_EXP}|" in am and any(am.endswith(p) for p in PRIMARY[component])),
                       "paired_direction_ok": rb.get("p_g") is not None and rp.get("p_g") is not None
                       and rp["p_g"] < rb["p_g"]})            # RC4 (Pi rev-9): STRICT < (a tie is no direction evidence)
    evald = [x for x in trials if x["perturbed_verdict"] != "NOT_EVALUABLE"]
    return {"group": gid, "component": component, "perturbed_exp_id": PERTURB_EXP, "primary_cells": PRIMARY[component],
            "T": T, "sparse_nontarget_unchanged": unchanged_ok, "sparse_target_changed": target_changed_ok,
            "dev_config_stable_identity": dev_cfg_id,               # RC5: full dev-config reproducibility identity
            "stable_id_base_eq_perturbed": stable_id_paired_ok,     # RC5 follow-through: identical config each paired trial
            "trials": trials, "evaluated": len(evald),
            "argmin_attribution_rate": round(sum(x["argmin_attributes_to_perturbed_primary"] for x in evald)
                                             / len(evald), 3) if evald else None,
            "paired_direction_rate": round(sum(x["paired_direction_ok"] for x in evald) / len(evald), 3) if evald else None,
            "mode": "MECHANICAL PAIRED (B<resolution); power qualification = registered B=20000 run, NOT authorized"}


def main():
    demos = {"burst_timing": component_demo("burst_timing", T=4),
             "mark_burst_tie": component_demo("mark_burst_tie", T=2),
             "cluster_size_mark_diversity": component_demo("cluster_size_mark_diversity", T=2)}
    variants = {}
    for name, u in (("with_exemption", True), ("without_exemption", False)):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=u); g = REG.build_groups(sd)
        variants[name] = {"M0": len([c for c in sd if c["scope"] == "in"]),
                          "G_bounded_support": len(g["G_bounded_support"]["cells"])}
    agg = {"dev_namespace": DEV_NS, "N": N, "dev_floor": DEV_FLOOR, "B_mechanical": B_MECH,
           "engine_boundary": "gate_group_dev (explicit dev config; floor is a PARAM; no module-global mutation)",
           "canonical_registry_hash": ENG.CANONICAL_REGISTRY_HASH,
           "fixes": ["structured arms per stratum", "canonical multiscale structural-zero (data + map)",
                     "sparse single-experiment + hashed non-target-unchanged + target-changed assertions",
                     "PAIRED base-vs-perturbation under the SAME permutation seed (Pi rev-8 #7)",
                     "context-bound map artifacts (dev floor + profile/regime; one identity per (profile,regime,check))"],
           "component_group_sensitivity": demos,
           "exemption_reporting": {"note": "full-support groups exemption-INVARIANT; registered-N preflight decided "
                                   "both S3 exemptions on detection 0.0<0.5; reported with/without each.",
                                   "variants": variants},
           "scale_note": f"dev floor={DEV_FLOOR}/N={N} LABELLED dev scale; exact size from randomization theory.",
           "authorization": "dev-only; no reserved map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
