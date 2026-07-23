#!/usr/bin/env python3
"""Oracle realism v3 — EXACT machine-readable SD/MM gate-cell registry + schema-validated self-tests.

Rev-5 (Pi rev-4 gate defects #4/#5/#6). Every same-distribution (SD, permutation-gated) and mismatched-arm
(MM, direct-effect) gate cell binds its EXACT identity fields and is SCHEMA-VALIDATED (missing/extra/unknown
fields REFUSE). The self-tests prove, on executable evidence rather than prose:

  * Δ == the LIVE registered verifier `.threshold` for every check (exact float equality; no rounding), with
    per-Δ provenance pinned to the constant symbol; a Δ-table hash is pinned (defect #5).
  * every contract-required field is present, typed, and non-empty; group_id is bound DIRECTLY on every
    in-scope SD cell; map_identity / floor_policy / RNG-law / stratum-quota / expected-status /
    malformed-input behaviour are all bound (defect #4).
  * BOTH boundary-exemption variants are emitted and partition exactly (with and without the PROVISIONAL
    uncalibratable S3 exemption) so an exemption cannot silently shrink M0 or a group (defect #6).
  * the min-p resolution requirement B >= K_max/alpha_group is computed from the deepest group and the main
    B is checked against it (audit B is a SEPARATE coarser evaluator — see the benchmark).
  * map-carrying checks are detected by EXECUTABLE evidence (reference coarsening in CheckResult.detail).

Development-only, synthetic-only; no governed read, no calibration/audit/evaluation seed, no map draw. Run:
    PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_registry.py
"""
from __future__ import annotations

import json
from math import ceil

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_battery import (
    ALL_CHECK_KEYS, _BOUNDARY_EXPECTED_NE, _evidence_map, _multiscale, _MIMIC_PROF, CONTROL_ALLOC, REGISTERED_N,
)
from clinical_jepa.eval.oracle_realism_v2_verifier_design import ABLATION_MATRIX
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
# EXACT registered constants (single source of truth — imported, never transcribed):
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS, ORACLE_ENV_MIN_DENOM,
)
from clinical_jepa.eval.oracle_realism_v2_verifier import _TAU as V2_TAU, _LOGGAP as V2_LOGGAP

_KS, _TV, _OCC, _DT0 = ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS

# --- substantive families (partition of the 20 checks) --------------------------------------------
FAMILY = {
    "length_density": ["S1_density", "S1_tau", "count_ks", "length_ks"],
    "run_size":       ["S2_ks"],
    "burst_timing":   ["S3_tau", "S3_loggap", "delta_t_zero_abs", "positive_gap_ks"],
    "class_mark":     ["S4_abs", "S5_abs", "S6_tv", "S7_abs", "class_tv", "occupancy_abs"],
    "phase_seam":     ["S8_class", "S8_density", "S9_zero", "S9_class", "S9_gap"],
}
CHECK_FAMILY = {c: f for f, cs in FAMILY.items() for c in cs}

# --- Δ effect limits: bound to the EXACT registered verifier constant symbols (NOT rounded literals) -----
# Each check -> (exact value object, provenance symbol). The value objects are the SAME python floats the
# verifier thresholds on, so Δ == threshold by construction; the self-test re-proves it against a live run.
_DELTA_SRC = {
    "S1_density": (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S1_tau":     (V2_TAU, "oracle_realism_v2_verifier._TAU"),
    "S2_ks":      (_KS, "rung2_contract.ORACLE_ENV_KS"),
    "S3_tau":     (V2_TAU, "oracle_realism_v2_verifier._TAU"),
    "S3_loggap":  (V2_LOGGAP, "oracle_realism_v2_verifier._LOGGAP=log(1.10)"),
    "S4_abs":     (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S5_abs":     (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S6_tv":      (_TV, "rung2_contract.ORACLE_ENV_TV"),
    "S7_abs":     (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S8_class":   (_TV, "rung2_contract.ORACLE_ENV_TV"),
    "S8_density": (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S9_class":   (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "S9_gap":     (_KS, "rung2_contract.ORACLE_ENV_KS"),
    "S9_zero":    (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "class_tv":   (_TV, "rung2_contract.ORACLE_ENV_TV"),
    "count_ks":   (_KS, "rung2_contract.ORACLE_ENV_KS"),
    "delta_t_zero_abs": (_DT0, "rung2_contract.ORACLE_ENV_DT0_ABS"),
    "length_ks":  (_KS, "rung2_contract.ORACLE_ENV_KS"),
    "occupancy_abs": (_OCC, "rung2_contract.ORACLE_ENV_OCCUPANCY_ABS"),
    "positive_gap_ks": (_KS, "rung2_contract.ORACLE_ENV_KS"),
}
DELTA = {k: v for k, (v, _) in _DELTA_SRC.items()}
DELTA_PROVENANCE = {k: p for k, (_, p) in _DELTA_SRC.items()}

# --- exact estimator identity per statistic (route + reducer). S3_tau is the v3 pilot-frozen estimator; all
#     others are the registered v2 verifier route.  Cross-checked: map-carrying set == verifier coarsen callers.
ESTIMATOR = {
    "S1_density": "v2.cond_maxbin.mean(per_bin_density)@LENGTH_BINS[ref_coarsen]",
    "S1_tau":     "v2.source_tau_b(length,cluster_count)",
    "S2_ks":      "v2.ks(cluster_size_ecdf)",
    "S3_tau":     "v3.pooled_tau_b.phasespanning_cap6(prev_cluster_size,positive_gap)"
                  "@pilot=oracle_realism_v3_phase0_pilot.py#0e1680fc",
    "S3_loggap":  "v2.cond_maxbin.maxabs(mean_log_positive_gap)@CLUSTER_BINS[ref_coarsen]",
    "S4_abs":     "v2.abs(P(same_class|same_cluster)-P(same_class|adjacent))",
    "S5_abs":     "v2.cond_maxbin.mean(per_bin_occupancy)@LENGTH_BINS[ref_coarsen]",
    "S6_tv":      "v2.maxabs_tv(class_prior)@ref_class_coarsen",
    "S7_abs":     "v2.cond_maxbin.mean(per_bin_distinct_class_frac)@CLUSTER_BINS[ref_coarsen]",
    "S8_class":   "v2.maxabs_tv(class_prior|phase_quartile)[terminal]",
    "S8_density": "v2.maxabs(density|phase_quartile)[terminal]",
    "S9_class":   "v2.abs(class_prior@terminal_seam)[terminal]",
    "S9_gap":     "v2.ks(positive_gap: seam_vs_nonseam)[terminal]",
    "S9_zero":    "v2.abs(delta_t_zero_frac@terminal_seam)[terminal]",
    "class_tv":   "v2.tv(5_class_prior)",
    "count_ks":   "v2.ks(per_seq_event_count_ecdf)",
    "delta_t_zero_abs": "v2.abs(P(delta_t=0))",
    "length_ks":  "v2.ks(length_ecdf)",
    "occupancy_abs": "v2.abs(mean_occupancy=distinct_classes/5)",
    "positive_gap_ks": "v2.ks(positive_gap_ecdf)",
}
# required-property id: the realism property each check certifies (stable id; audits reference this, not the label)
REQUIRED_PROPERTY = {
    "S1_density": "PROP.burst_density_vs_length", "S1_tau": "PROP.length_count_rank_coupling",
    "S2_ks": "PROP.burst_size_law", "S3_tau": "PROP.burst_timing_rank_coupling",
    "S3_loggap": "PROP.burst_gap_magnitude_vs_size", "S4_abs": "PROP.mark_burst_tie",
    "S5_abs": "PROP.composition_length_coupling", "S6_tv": "PROP.length_dependent_class_mix",
    "S7_abs": "PROP.class_diversity_across_burst_size", "S8_class": "PROP.phase_stationarity_class",
    "S8_density": "PROP.phase_stationarity_density", "S9_class": "PROP.terminal_seam_class",
    "S9_gap": "PROP.seam_gap_law", "S9_zero": "PROP.terminal_seam_zero_gap",
    "class_tv": "PROP.class_marginal", "count_ks": "PROP.event_count_marginal",
    "delta_t_zero_abs": "PROP.zero_gap_marginal", "length_ks": "PROP.length_marginal",
    "occupancy_abs": "PROP.occupancy_marginal", "positive_gap_ks": "PROP.positive_gap_marginal",
}
# reference-OWNED coarsening-map carriers (verifier coarsen_reference / _conditional_maxbin callers).
# Verified by executable evidence in the self-test (CheckResult.detail carries 'per_group' iff map-carrying).
MAP_BINS = {"S1_density": "LENGTH_BINS", "S5_abs": "LENGTH_BINS", "S3_loggap": "CLUSTER_BINS",
            "S7_abs": "CLUSTER_BINS", "S6_tv": "REF_CLASS_COARSEN"}
MAP_CARRYING = set(MAP_BINS)
FLOOR_POLICY = f"denom<{ORACLE_ENV_MIN_DENOM}(ORACLE_ENV_MIN_DENOM)=>NOT_EVALUABLE; never zero-fill"

# --- RNG-law identity (how each SD source's candidate/reference draws + any coupling are seeded) -----------
RNG_LAW = {
    "fixture": "seed=sha256(('fixture',source_profile,replicate_seed,role))  [role in {candidate,reference}]",
    "coupling": "seed=sha256(('coupling',source_key,component,replicate_seed,role))",
    "permutation": "within-stratum label permutation; MC index synchronized across a group's experiments",
}

# --- SD experiments (each a distinct candidate/reference pair) --------------------------------------------
# (experiment_id, condition, source_profile, coupled_component_or_None, support_regime)
SD_EXPERIMENTS = [
    ("null_scid", "null", "scid_scale_control", None, "full"),
    ("null_mimic", "null", "mimic_scale_control", None, "full"),
    *[(f"repeat_{c}_{s[:4]}", "repeatability", s, c, "full")
      for c in V2_D_COMPONENT_MENU for s in ("scid_scale_control", "mimic_scale_control")],
    ("structural_zero", "structural_zero", "structural_zero_control", None, "full"),
    ("boundary_short", "boundary", "boundary_short", None, "bounded"),
]

# --- boundary-short exemptions -----------------------------------------------------------------------------
BOUNDARY_EXEMPT_DEGENERATE = set(_BOUNDARY_EXPECTED_NE)          # v2 predeclared-NE: estimand degenerate on L<=7
BOUNDARY_EXEMPT_UNCALIBRATABLE = {"S3_tau", "S3_loggap"}         # PROVISIONAL (pending Δ-aligned confirmation)


def _strata(condition):
    """Exact exchangeability strata (ids + per-stratum candidate/reference quotas), NOT prose."""
    if condition in ("structural_zero", "boundary"):
        return [{"stratum_id": f"len_{i}", "n_candidate": a, "n_reference": a}
                for i, a in enumerate(CONTROL_ALLOC)]           # permute WITHIN each length stratum
    return [{"stratum_id": "pooled_source", "n_candidate": REGISTERED_N, "n_reference": REGISTERED_N}]


def _exemption_permission(stat, regime):
    if regime == "full":
        return "NONE:core_non_exemptible"
    if stat in BOUNDARY_EXEMPT_DEGENERATE:
        return "PERMITTED:degenerate_estimand_L<=7(v2_predeclared_NE)"
    if stat in BOUNDARY_EXEMPT_UNCALIBRATABLE:
        return "PROVISIONAL:pending_Delta_aligned_confirmation"
    return "NONE:in_scope_bounded"


# --- SD cell schema (exact required-field set; the validator REFUSES missing/extra/unknown) ---------------
SD_SCHEMA = {
    "cell_id": str, "class": str, "experiment_id": str, "condition": str, "source": str,
    "source_role_rng_law": str, "coupled_component": (str, type(None)), "support_regime": str,
    "statistic": str, "family": str, "estimator_identity": str, "required_property": str,
    "delta": float, "delta_provenance": str, "permutation_scheme": str, "exchangeability_strata": list,
    "map_identity": str, "floor_policy": str, "group_id": (str, type(None)), "scope": str,
    "exemption_permission": str, "core": bool, "expected_status_rule": str, "malformed_input_behavior": str,
}
MM_SCHEMA = {
    "cell_id": str, "class": str, "component": (str, type(None)), "source": (str, type(None)),
    "statistic": (str, type(None)), "delta": (float, type(None)), "role": str, "rule": str,
    "bar": (str, type(None)), "estimator_identity": (str, type(None)),
    "required_property": (str, type(None)), "delta_provenance": (str, type(None)),
    "malformed_input_behavior": str, "nondegenerate": (list, type(None)),
}


def _in_scope(stat, regime, apply_uncal):
    if regime == "full":
        return True
    exempt = BOUNDARY_EXEMPT_DEGENERATE | (BOUNDARY_EXEMPT_UNCALIBRATABLE if apply_uncal else set())
    return stat not in exempt


def build_sd_cells(apply_uncalibratable_exemption=True):
    cells = []
    for exp_id, cond, src, comp, regime in SD_EXPERIMENTS:
        strata = _strata(cond)
        for stat in sorted(ALL_CHECK_KEYS):
            inside = _in_scope(stat, regime, apply_uncalibratable_exemption)
            if inside:
                scope = "in"
            elif stat in BOUNDARY_EXEMPT_DEGENERATE:
                scope = "exempt-degenerate"
            else:
                scope = "exempt-uncalibratable"
            gid = f"G_full_{CHECK_FAMILY[stat]}" if (inside and regime == "full") else (
                "G_bounded_support" if inside else None)
            cells.append({
                "cell_id": f"SD|{exp_id}|{stat}", "class": "SD", "experiment_id": exp_id, "condition": cond,
                "source": src, "source_role_rng_law": RNG_LAW["fixture"], "coupled_component": comp,
                "support_regime": regime, "statistic": stat, "family": CHECK_FAMILY[stat],
                "estimator_identity": ESTIMATOR[stat], "required_property": REQUIRED_PROPERTY[stat],
                "delta": float(DELTA[stat]), "delta_provenance": DELTA_PROVENANCE[stat],
                "permutation_scheme": "within_stratum_sequence_label_permutation",
                "exchangeability_strata": strata,
                "map_identity": (f"REF_OWNED_COARSEN::{MAP_BINS[stat]}::PENDING_DRAW"
                                 if stat in MAP_CARRYING else "NONE"),
                "floor_policy": FLOOR_POLICY, "group_id": gid, "scope": scope,
                "exemption_permission": _exemption_permission(stat, regime),
                "core": regime == "full",
                "expected_status_rule": ("SD_same_distribution: permutation p_c NOT in upper tail; group "
                                         "min-p NOT below alpha_group (no rejection under a true null)"),
                "malformed_input_behavior": ("floor breach=>NOT_EVALUABLE; truncated/quota-mismatch/"
                                             "non-bijection/duplicate-index=>REFUSE"),
            })
    return cells


def build_groups(sd_cells):
    """Groups by (support-regime x substantive-family), read from each in-scope cell's DIRECT group_id."""
    groups = {}
    for c in sd_cells:
        if c["scope"] != "in":
            continue
        gid = c["group_id"]
        if gid.startswith("G_full_"):
            plan = "product: independent within-source label permutation per full experiment, synchronized MC index"
            rationale = f"full-support {c['family']} family across all full SD experiments (coherent alternative)"
        else:
            plan = "within-length-strata label permutation on the single bounded experiment"
            rationale = "single property-specific bounded-support control; sparse per-family in-scope cells"
        g = groups.setdefault(gid, {"group_id": gid, "rationale": rationale, "product_permutation_plan": plan,
                                    "cells": []})
        g["cells"].append(c["cell_id"])
    return groups


def build_mm_cells():
    """Mismatched-arm cells (direct effect d>Δ, NOT permutation). Primary DETECT >=20/25; specificity d<=Δ
    >=24/25. S4<->S7 allowed-sensitive per ABLATION_MATRIX. Plus a source_swap expected-FAIL negative control."""
    cells = []
    for comp in V2_D_COMPONENT_MENU:
        primary = ABLATION_MATRIX[comp]["primary_fail"]
        allowed = set(ABLATION_MATRIX[comp]["allowed_sensitive"])
        for src in ("scid_scale_control", "mimic_scale_control"):
            for stat in sorted(ALL_CHECK_KEYS):
                if stat in primary:
                    role, rule, bar = "primary_detection", "detect iff d>delta", ">=20/25"
                elif stat in allowed:
                    continue
                else:
                    role, rule, bar = "specificity", "pass iff d<=delta", ">=24/25"
                cells.append({"cell_id": f"MM|{comp}|{src}|{stat}", "class": "MM", "component": comp,
                              "source": src, "statistic": stat, "delta": float(DELTA[stat]), "role": role,
                              "rule": rule, "bar": bar, "estimator_identity": ESTIMATOR[stat],
                              "required_property": REQUIRED_PROPERTY[stat], "delta_provenance": DELTA_PROVENANCE[stat],
                              "malformed_input_behavior": "floor breach=>NOT_EVALUABLE(non-detect); else REFUSE",
                              "nondegenerate": None})
    cells.append({"cell_id": "MM|source_swap", "class": "MM_negative", "component": None, "source": None,
                  "statistic": None, "delta": None, "role": "expected_FAIL", "rule": "must FAIL nondegenerate set",
                  "bar": None, "estimator_identity": None, "required_property": None, "delta_provenance": None,
                  "malformed_input_behavior": "n/a(negative control)",
                  "nondegenerate": ["count_ks", "positive_gap_ks", "class_tv", "S1_density", "S1_tau"]})
    return cells


# --- schema validation: REFUSE missing / extra / mistyped fields (Pi defect #4) ---------------------------
def _validate_schema(cells, schema, label):
    errs = []
    required = set(schema)
    for c in cells:
        keys = set(c)
        missing, extra = required - keys, keys - required
        if missing:
            errs.append(f"{label} {c.get('cell_id','?')}: missing {sorted(missing)}")
        if extra:
            errs.append(f"{label} {c.get('cell_id','?')}: unknown/extra {sorted(extra)}")
        for f, t in schema.items():
            if f in c and not isinstance(c[f], t):
                errs.append(f"{label} {c.get('cell_id','?')}: field {f} type {type(c[f]).__name__} != {t}")
        # non-empty required strings (no silent blanks)
        for f, t in schema.items():
            if t is str and isinstance(c.get(f), str) and not c[f].strip():
                errs.append(f"{label} {c.get('cell_id','?')}: empty required string {f}")
    return errs


def _schema_refusal_proof_errors(sd_cells):
    """Executable proof the schema validator REFUSES malformed cells (missing / extra / mistyped / empty).
    A validator that fails to reject any of these is itself a defect."""
    base = dict(sd_cells[0])
    missing = dict(base); missing.pop("delta")
    extra = dict(base); extra["__sneaky__"] = 1
    mistyped = dict(base); mistyped["delta"] = "notafloat"
    empty = dict(base); empty["estimator_identity"] = "   "
    errs = []
    for label, bad in (("missing", missing), ("extra", extra), ("mistyped", mistyped), ("empty", empty)):
        if not _validate_schema([bad], SD_SCHEMA, "SD"):
            errs.append(f"schema validator FAILED to refuse malformed cell ({label})")
    return errs


def _delta_live_equality_errors():
    """Prove Δ == the LIVE verifier .threshold for every check (exact float equality). Thresholds are emitted
    regardless of support, so a tiny multiscale fixture suffices (synthetic-only, no governed read)."""
    ref = _multiscale(_MIMIC_PROF, "MIMIC", "delta_ref", 1, alloc=(120, 120, 120))
    cand = _multiscale(_MIMIC_PROF, "MIMIC", "delta_cand", 1, alloc=(120, 120, 120))
    ev = _evidence_map(cand, ref)
    errs = []
    for k in sorted(ALL_CHECK_KEYS):
        live = ev[k]["threshold"]
        if DELTA[k] != live:                                    # EXACT equality (not approx): rounding is a defect
            errs.append(f"Δ[{k}]={DELTA[k]!r} != live threshold {live!r}")
    return errs


def _map_carrying_evidence_errors():
    """Prove MAP_CARRYING == checks whose CheckResult.detail carries reference coarsening ('per_group'), at
    adequate support (>=FLOOR per length stratum). Executable evidence, not a declared string (Pi/Cog)."""
    ref = _multiscale(_MIMIC_PROF, "MIMIC", "map_ref", 1, alloc=(700, 700, 700))
    cand = _multiscale(_MIMIC_PROF, "MIMIC", "map_cand", 1, alloc=(700, 700, 700))
    ev = _evidence_map(cand, ref)
    errs = []
    for k in sorted(ALL_CHECK_KEYS):
        det = ev[k]["detail"]
        has_map = isinstance(det, dict) and "per_group" in det
        if has_map != (k in MAP_CARRYING):
            errs.append(f"map-carrying mismatch {k}: detail_has_map={has_map} declared={k in MAP_CARRYING} "
                        f"(status {ev[k]['status']})")
    return errs


def _resolution(groups):
    """min-p resolution requirement B >= K_max/alpha_group for the DEEPEST group (defect #3, reporting side)."""
    G = len(groups)
    alpha_sd = 0.04
    alpha_group = alpha_sd / G
    k_max = max(len(g["cells"]) for g in groups.values())
    b_res_min = ceil(k_max / alpha_group)
    return {"alpha_sd": alpha_sd, "G": G, "alpha_group": alpha_group, "K_max": k_max,
            "B_resolution_min": b_res_min, "rule": "B >= K_max/alpha_group (min-p floor-tie resolution)",
            "B_main_20000_resolves": 20000 >= b_res_min, "B_audit_2000_resolves": 2000 >= b_res_min,
            "audit_note": "audit B=2000 does NOT meet resolution => a SEPARATE coarser evaluator (see benchmark)"}


def selftest(reg):
    sd, mm, groups = reg["sd_cells"], reg["mm_cells"], reg["groups"]
    errs = []
    # families partition the 20 checks
    fam_union = set().union(*FAMILY.values())
    if fam_union != set(ALL_CHECK_KEYS):
        errs.append(f"families do not cover checks: {fam_union ^ set(ALL_CHECK_KEYS)}")
    if sum(len(v) for v in FAMILY.values()) != len(fam_union):
        errs.append("families overlap")
    # Δ keys == checks AND Δ == live registered thresholds (exact) + provenance present
    if set(DELTA) != set(ALL_CHECK_KEYS):
        errs.append("DELTA keys != checks")
    if set(DELTA_PROVENANCE) != set(ALL_CHECK_KEYS):
        errs.append("DELTA_PROVENANCE keys != checks")
    errs += _delta_live_equality_errors()
    # schema validation (refuse missing/extra/mistyped) + executable refusal proof
    errs += _validate_schema(sd, SD_SCHEMA, "SD")
    errs += _validate_schema(mm, MM_SCHEMA, "MM")
    errs += _schema_refusal_proof_errors(sd)
    # map-carrying by executable evidence
    errs += _map_carrying_evidence_errors()
    # unique ids
    ids = [c["cell_id"] for c in sd + mm]
    if len(ids) != len(set(ids)):
        errs.append("duplicate cell ids")
    # groups partition in-scope SD cells exactly (Σ K_g = M0); direct group_id agrees with membership
    inscope_ids = [c["cell_id"] for c in sd if c["scope"] == "in"]
    grouped = [cid for g in groups.values() for cid in g["cells"]]
    if sorted(grouped) != sorted(inscope_ids):
        errs.append("groups do not partition in-scope SD cells")
    if len(grouped) != len(set(grouped)):
        errs.append("a cell appears in >1 group")
    for c in sd:
        if c["scope"] == "in" and not c["group_id"]:
            errs.append(f"in-scope cell without group_id: {c['cell_id']}")
        if c["scope"] != "in" and c["group_id"] is not None:
            errs.append(f"exempt cell with group_id: {c['cell_id']}")
    M0 = len(inscope_ids)
    if sum(len(g["cells"]) for g in groups.values()) != M0:
        errs.append("sum K_g != M0")
    # every check reachable in-scope; exemptions only in the bounded regime
    reachable = {c["statistic"] for c in sd if c["scope"] == "in"}
    if reachable != set(ALL_CHECK_KEYS):
        errs.append(f"unreachable checks: {set(ALL_CHECK_KEYS) - reachable}")
    for c in sd:
        if c["scope"] != "in" and c["support_regime"] != "bounded":
            errs.append(f"exemption outside bounded regime: {c['cell_id']}")
    if any(c["core"] and c["scope"] != "in" for c in sd):
        errs.append("a core cell is exempt")
    for g in groups.values():
        if not g.get("rationale") or not g.get("product_permutation_plan"):
            errs.append(f"group {g['group_id']} missing rationale/permutation plan")
    # MM source-swap + S4/S7 allowed-sensitive match the executable ABLATION_MATRIX
    for comp in V2_D_COMPONENT_MENU:
        allowed = set(ABLATION_MATRIX[comp]["allowed_sensitive"])
        mm_spec = {c["statistic"] for c in mm if c.get("component") == comp
                   and c.get("source") == "scid_scale_control" and c.get("role") == "specificity"}
        mm_prim = {c["statistic"] for c in mm if c.get("component") == comp
                   and c.get("source") == "scid_scale_control" and c.get("role") == "primary_detection"}
        if mm_prim != set(ABLATION_MATRIX[comp]["primary_fail"]):
            errs.append(f"MM primary != ablation primary for {comp}")
        if mm_spec & allowed:
            errs.append(f"allowed-sensitive leaked into specificity for {comp}")
        if (mm_spec | mm_prim | allowed) != set(ALL_CHECK_KEYS):
            errs.append(f"MM cells + allowed != all checks for {comp}")
    return errs, M0


def _build_variant(apply_uncal):
    sd = build_sd_cells(apply_uncalibratable_exemption=apply_uncal)
    groups = build_groups(sd)
    mm = build_mm_cells()
    reg = {"sd_cells": sd, "mm_cells": mm, "groups": groups}
    errs, M0 = selftest(reg)
    return reg, errs, M0, groups


def main():
    # BOTH boundary-exemption variants (defect #6): an exemption must not silently shrink M0 or a group.
    variants = {}
    for name, apply_uncal in (("with_provisional_exemption", True), ("without_provisional_exemption", False)):
        reg, errs, M0, groups = _build_variant(apply_uncal)
        res = _resolution(groups)
        variants[name] = {
            "apply_uncalibratable_exemption": apply_uncal, "M0_inscope_sd_cells": M0, "G_groups": len(groups),
            "group_sizes": {g: len(v["cells"]) for g, v in sorted(groups.items())},
            "resolution": res, "selftests_pass": not errs, "selftest_errors": errs,
            "registry_hash": canonical_hash(reg),
        }

    delta_table = {"delta": DELTA, "provenance": DELTA_PROVENANCE}
    reg_default, errs_default, _, _ = _build_variant(True)
    n_mm = len(reg_default["mm_cells"])
    summary = {
        "n_sd_experiments": len(SD_EXPERIMENTS), "n_full_experiments": sum(1 for e in SD_EXPERIMENTS if e[4] == "full"),
        "n_components": len(V2_D_COMPONENT_MENU), "n_mm_cells": n_mm,
        "delta_table_hash": canonical_hash(delta_table),
        "delta_binds_to": "LIVE registered verifier thresholds (exact float equality; provenance per Δ)",
        "variants": variants,
        "compute_note": "job driven by M0*B (not G). Main B must meet B>=K_max/alpha_group (see resolution). "
                        "Audit is a SEPARATE coarser evaluator; route-weighted forecast in the benchmark.",
        "authorization": "dev-only registry; NO map draw, NO calibration/audit/evaluation seed, NO policy.",
    }
    print(json.dumps(summary, indent=2, default=str))
    print("\nDELTA_TABLE_HASH:", canonical_hash(delta_table))
    print("REGISTRY_HASH(with_exemption):", variants["with_provisional_exemption"]["registry_hash"])
    print("REGISTRY_HASH(without_exemption):", variants["without_provisional_exemption"]["registry_hash"])
    assert not errs_default, f"registry self-tests FAILED: {errs_default}"
    for name, v in variants.items():
        assert v["selftests_pass"], f"variant {name} self-tests FAILED: {v['selftest_errors']}"
    return summary


if __name__ == "__main__":
    main()
