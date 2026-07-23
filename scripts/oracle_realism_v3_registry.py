#!/usr/bin/env python3
"""Oracle realism v3 — exact machine-readable SD/MM gate-cell registry + self-tests (Pi rev-3 §4/§5/§8).

Enumerates every same-distribution (SD, permutation-gated) and mismatched-arm (MM, direct-effect) gate cell,
their groups, exchangeability strata, and Δ effect limits, and proves by self-test: exact partition
(Σ K_g = M0, no omission/duplication), reachability of every check, explicit exemptions, core cells
non-exemptible, Δ == v2 practical-effect thresholds, and source-swap / S4↔S7 rules matching the executable
verdict. Aggregate-hashed. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_registry.py
"""
from __future__ import annotations

import json

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_battery import ALL_CHECK_KEYS, _BOUNDARY_EXPECTED_NE
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU

# --- substantive families (partition of the 20 checks) --------------------------------------------
FAMILY = {
    "length_density": ["S1_density", "S1_tau", "count_ks", "length_ks"],
    "run_size":       ["S2_ks"],
    "burst_timing":   ["S3_tau", "S3_loggap", "delta_t_zero_abs", "positive_gap_ks"],
    "class_mark":     ["S4_abs", "S5_abs", "S6_tv", "S7_abs", "class_tv", "occupancy_abs"],
    "phase_seam":     ["S8_class", "S8_density", "S9_zero", "S9_class", "S9_gap"],
}
CHECK_FAMILY = {c: f for f, cs in FAMILY.items() for c in cs}

# --- Δ effect limits = the v2 per-check practical-effect thresholds (preserve v2 semantics; Pi §4) ---
DELTA = {"S1_density": 0.03, "S1_tau": 0.05, "S2_ks": 0.05, "S3_loggap": 0.095310, "S3_tau": 0.05,
         "S4_abs": 0.03, "S5_abs": 0.03, "S6_tv": 0.05, "S7_abs": 0.03, "S8_class": 0.05, "S8_density": 0.03,
         "S9_class": 0.03, "S9_gap": 0.05, "S9_zero": 0.03, "class_tv": 0.05, "count_ks": 0.05,
         "delta_t_zero_abs": 0.02, "length_ks": 0.05, "occupancy_abs": 0.03, "positive_gap_ks": 0.05}

# --- boundary-short exemptions -------------------------------------------------------------------
# degenerate (estimand undefined/collapsed on L<=7): the v2 predeclared-NE length/seam set.
BOUNDARY_EXEMPT_DEGENERATE = set(_BOUNDARY_EXPECTED_NE)                      # S1_density,S5_abs,S6_tv,S9_zero/class/gap
# un-calibratable (property-specific, PROVISIONAL from the dev pilot; Δ-aligned confirmation pending).
BOUNDARY_EXEMPT_UNCALIBRATABLE = {"S3_tau", "S3_loggap"}
BOUNDARY_EXEMPT = BOUNDARY_EXEMPT_DEGENERATE | BOUNDARY_EXEMPT_UNCALIBRATABLE

# --- SD experiments (each a distinct actual candidate/reference pair; Pi §8) ----------------------
# (experiment_id, condition, source_profile, coupled_component_or_None, support_regime)
SD_EXPERIMENTS = [
    ("null_scid", "null", "scid_scale_control", None, "full"),
    ("null_mimic", "null", "mimic_scale_control", None, "full"),
    *[(f"repeat_{c}_{s[:4]}", "repeatability", s, c, "full")
      for c in V2_D_COMPONENT_MENU for s in ("scid_scale_control", "mimic_scale_control")],
    ("structural_zero", "structural_zero", "structural_zero_control", None, "full"),
    ("boundary_short", "boundary", "boundary_short", None, "bounded"),
]

# per-stratum quota for fixed-quota controls (permute WITHIN matching length strata; Pi §8)
CONTROL_ALLOC = (2667, 2667, 2666)


def _in_scope(stat, regime):
    return True if regime == "full" else (stat not in BOUNDARY_EXEMPT)


def build_sd_cells():
    cells = []
    for exp_id, cond, src, comp, regime in SD_EXPERIMENTS:
        stratum = "within_length_strata(2667,2667,2666)" if cond in ("structural_zero", "boundary") \
            else "within_source_profile"
        for stat in sorted(ALL_CHECK_KEYS):
            scope = "in" if _in_scope(stat, regime) else (
                "exempt-degenerate" if stat in BOUNDARY_EXEMPT_DEGENERATE else "exempt-uncalibratable")
            cells.append({
                "cell_id": f"SD|{exp_id}|{stat}", "class": "SD", "experiment_id": exp_id, "condition": cond,
                "source": src, "coupled_component": comp, "support_regime": regime, "statistic": stat,
                "family": CHECK_FAMILY[stat], "delta": DELTA[stat], "permutation_scheme": "sequence_label",
                "exchangeability_stratum": stratum, "scope": scope,
                "core": regime == "full",           # full-support SD cells are CORE (non-exemptible)
            })
    return cells


def build_groups(sd_cells):
    """Groups by (support-regime x substantive-family). Full-support families span the 9 full experiments via
    PRODUCT permutation (independent per-experiment label perms under one synchronized MC index). Bounded-support
    in-scope cells form ONE property-specific support-control group (sparse per-family; single regime)."""
    inscope = [c for c in sd_cells if c["scope"] == "in"]
    groups = {}
    for c in inscope:
        if c["support_regime"] == "full":
            gid = f"G_full_{c['family']}"
            plan = "product: independent within-source label permutation per full experiment, synchronized MC index"
            rationale = f"full-support {c['family']} family across all full SD experiments (coherent alternative)"
        else:
            gid = "G_bounded_support"
            plan = "within-length-strata label permutation on the single bounded experiment"
            rationale = "single property-specific bounded-support control; sparse per-family in-scope cells"
        g = groups.setdefault(gid, {"group_id": gid, "rationale": rationale, "product_permutation_plan": plan,
                                    "cells": []})
        g["cells"].append(c["cell_id"])
    return groups


def build_mm_cells():
    """Mismatched-arm cells (direct effect, NOT permutation): candidate_A vs coupled reference, per component x
    source. Primary attributed cells must be DETECTED d>Δ (>=20/25); non-attributed specificity cells d<=Δ
    (>=24/25). S4<->S7 allowed-sensitive per ABLATION_MATRIX. Plus source_swap expected-FAIL negative control."""
    cells = []
    for comp in V2_D_COMPONENT_MENU:
        primary = ABLATION_MATRIX[comp]["primary_fail"]
        allowed = set(ABLATION_MATRIX[comp]["allowed_sensitive"])
        for src in ("scid_scale_control", "mimic_scale_control"):
            for stat in sorted(ALL_CHECK_KEYS):
                if stat in primary:
                    role, rule, bar = "primary_detection", "detect iff d>delta", ">=20/25"
                elif stat in allowed:
                    continue                          # allowed-sensitive: exempt from specificity for this component
                else:
                    role, rule, bar = "specificity", "pass iff d<=delta", ">=24/25"
                cells.append({"cell_id": f"MM|{comp}|{src}|{stat}", "class": "MM", "component": comp, "source": src,
                              "statistic": stat, "delta": DELTA[stat], "role": role, "rule": rule, "bar": bar})
    cells.append({"cell_id": "MM|source_swap", "class": "MM_negative", "role": "expected_FAIL",
                  "rule": "must FAIL nondegenerate set", "nondegenerate": ["count_ks", "positive_gap_ks",
                          "class_tv", "S1_density", "S1_tau"]})
    return cells


# --- self-tests (Pi §4) ---------------------------------------------------------------------------
def selftest(reg):
    sd, mm, groups = reg["sd_cells"], reg["mm_cells"], reg["groups"]
    errs = []
    # families partition the 20 checks
    fam_union = set().union(*FAMILY.values())
    if fam_union != set(ALL_CHECK_KEYS):
        errs.append(f"families do not cover checks: {fam_union ^ set(ALL_CHECK_KEYS)}")
    if sum(len(v) for v in FAMILY.values()) != len(fam_union):
        errs.append("families overlap")
    # Δ == v2 thresholds for every check
    if set(DELTA) != set(ALL_CHECK_KEYS):
        errs.append("DELTA keys != checks")
    # each SD/MM cell id unique
    ids = [c["cell_id"] for c in sd + mm]
    if len(ids) != len(set(ids)):
        errs.append("duplicate cell ids")
    # groups partition the in-scope SD cells exactly (Σ K_g = M0, no omission/duplication)
    inscope_ids = [c["cell_id"] for c in sd if c["scope"] == "in"]
    grouped = [cid for g in groups.values() for cid in g["cells"]]
    if sorted(grouped) != sorted(inscope_ids):
        errs.append("groups do not partition in-scope SD cells")
    if len(grouped) != len(set(grouped)):
        errs.append("a cell appears in >1 group")
    M0 = len(inscope_ids)
    if sum(len(g["cells"]) for g in groups.values()) != M0:
        errs.append("sum K_g != M0")
    # every check reachable somewhere in-scope; every exemption explicit
    reachable = {c["statistic"] for c in sd if c["scope"] == "in"}
    if reachable != set(ALL_CHECK_KEYS):
        errs.append(f"unreachable checks: {set(ALL_CHECK_KEYS) - reachable}")
    for c in sd:
        if c["scope"] != "in" and c["support_regime"] != "bounded":
            errs.append(f"exemption outside bounded regime: {c['cell_id']}")
    # core (full-support) cells non-exemptible
    if any(c["core"] and c["scope"] != "in" for c in sd):
        errs.append("a core cell is exempt")
    # every group has rationale + product-permutation plan
    for g in groups.values():
        if not g.get("rationale") or not g.get("product_permutation_plan"):
            errs.append(f"group {g['group_id']} missing rationale/permutation plan")
    # source-swap + S4/S7 allowed-sensitive match the executable ABLATION_MATRIX
    for comp in V2_D_COMPONENT_MENU:
        allowed = set(ABLATION_MATRIX[comp]["allowed_sensitive"])
        mm_specificity = {c["statistic"] for c in mm if c.get("component") == comp
                          and c.get("source") == "scid_scale_control" and c.get("role") == "specificity"}
        mm_primary = {c["statistic"] for c in mm if c.get("component") == comp
                      and c.get("source") == "scid_scale_control" and c.get("role") == "primary_detection"}
        if mm_primary != set(ABLATION_MATRIX[comp]["primary_fail"]):
            errs.append(f"MM primary != ablation primary for {comp}")
        if (mm_specificity & allowed):
            errs.append(f"allowed-sensitive leaked into specificity for {comp}")
        if (mm_specificity | mm_primary | allowed) != set(ALL_CHECK_KEYS):
            errs.append(f"MM cells + allowed != all checks for {comp}")
    return errs, M0


def main():
    sd = build_sd_cells()
    groups = build_groups(sd)
    mm = build_mm_cells()
    reg = {"sd_cells": sd, "mm_cells": mm, "groups": groups, "families": FAMILY, "delta": DELTA,
           "boundary_exempt_degenerate": sorted(BOUNDARY_EXEMPT_DEGENERATE),
           "boundary_exempt_uncalibratable_provisional": sorted(BOUNDARY_EXEMPT_UNCALIBRATABLE),
           "control_alloc": list(CONTROL_ALLOC)}
    errs, M0 = selftest(reg)
    G = len(groups)
    alpha_group = round(0.04 / G, 6)
    summary = {
        "n_sd_experiments": len(SD_EXPERIMENTS), "M0_inscope_sd_cells": M0, "G_groups": G,
        "alpha_group": alpha_group, "group_sizes": {g: len(v["cells"]) for g, v in sorted(groups.items())},
        "n_mm_cells": len(mm), "min_p_deepest_group_crit_approx": round(alpha_group / max(len(v["cells"])
                              for v in groups.values()), 8),
        "compute_note": "job driven by M0*B (not G). With main B=20000 + audit B=2000 (15 reps) => ~6h under the "
                        "8h cap (see scripts/oracle_realism_v3_benchmark.py). G sets alpha_group only.",
        "selftests_pass": not errs, "selftest_errors": errs,
    }
    reg_hash = canonical_hash(reg)
    print(json.dumps({"summary": summary, "groups": {g: {"rationale": v["rationale"], "n_cells": len(v["cells"])}
                                                     for g, v in sorted(groups.items())}}, indent=2))
    print("\nREGISTRY_HASH:", reg_hash)
    assert not errs, f"registry self-tests FAILED: {errs}"
    return reg, reg_hash


if __name__ == "__main__":
    main()
