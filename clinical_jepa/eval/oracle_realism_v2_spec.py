"""Oracle realism v2 — M3a verification-spec DRAFT (blueprint step 5; Pi P-B/P-D).

SUPERSEDED-PENDING-REBUILD (Pi M3a gate, thread thr-20260720T143304Z): REVISE — do NOT freeze. This draft's
length unit (restriction of one L=8 block) and length bins capped at 8 are INFEASIBLE against the spent
full-sequence target (SCID/MIMIC P(L<=8)=0). The realism unit must become the full-sequence multi-block unit
and the verifier must be EXECUTABLE (not descriptive dicts). See `docs/oracle-realism-v2-blueprint.md`
§ Realism-unit vs certification-unit and § M3a REVISE. Kept only as the superseded reference identity
(m3a_spec_dev_hash 57ecfc93…) until the rebuild lands; it does NOT gate anything.

This module declares the verification spec that M3a will FREEZE **before** any M2 fitting: the six marginal
checks plus cross-statistics S1–S7 (exact bins / thresholds / denominator floors / refusal), the
parameter→statistic attribution table, the identifiability battery spec (Jacobian full-column-rank +
grid-recovery + collision search), a per-check power statement, the source-conjunction rule, and the
immutable escalation ledger (component→check attribution, tie rule, iteration cap).

STATUS: **DRAFT / DEV** — `m3a_spec_dev_hash()` is a development identity. The FINAL frozen M3a identity is
minted only after Pi rules on this draft (the numeric thresholds/bins/power below are PROPOSED and explicitly
flagged for Pi's ruling). NO fitting, sampling law, or target comparison is implied here — this is the
verifier, declared before the generator, per the freeze-before-fit discipline (Cog D2: a predeclared family of
probes + controls, every artifact frozen before evaluation).

Governance: synthetic-only. Admissible claim after any pass is ONLY "matches the declared marginal +
cross-statistic envelope," never identification of the joint event process. No latent/AR/transfer/causal claim.
"""
from __future__ import annotations

from typing import Any

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS,
    ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES,
)

# --- FROZEN HISTORICAL SNAPSHOTS (Pi: preserve the old dev hash 57ecfc93 as provenance) ---------------
# This superseded draft is decoupled from the LIVE schemas (which moved to the full-sequence multi-block unit
# at the M3a rebuild). These inlined snapshots reproduce the exact values referenced when 57ecfc93 was minted,
# so `m3a_spec_dev_hash()` stays frozen as a historical record and never tracks the evolving live schema.
_HIST_MARGINAL_SCHEMA = {
    "length_law": "source_conditioned_variable_length_via_order_restriction",
    "class_law": "dirichlet_multinomial_with_hard_structural_zeros",
    "cluster_size_law": "compound_burst_size_distribution",
    "gap_law": "source_positive_gap_distribution",
    "dt0_law": "cluster_size_induced_simultaneity",
    "required_sources": list(REQUIRED_SOURCES),
    "n_classes": ORACLE_ENV_N_CLASSES,
}
_HIST_M0B_POLICY = {
    "cell_support_floor": 500, "pair_denom_floor": 500, "occupancy_cap_length": 5,
    "occupancy_definition": "distinct_classes / C (C=5); L<5 caps occupancy at L/5",
    "statuses": ["SUPPORTED", "SUPPORT_STARVED", "VACUOUS_ORDER"],
    "discipline": "never-silent: any floor breach => SUPPORT_STARVED with reasons; L<=1 => VACUOUS_ORDER",
    "scope": "v2 realism/emission order cores only; restricted cores never reach fixed-L certification",
}
_HIST_A_ID = "b299a77977f251784667d42b31dd4bbf9a3a470c69b801f5089bda7692065e20"
_HIST_D_ID = "43b55944c75571aa4145a3fa147fcae1802b8a10eeb74885b04d891d41af6345"

M3A_SPEC_VERSION = "m3a_spec_draft_dev"     # -> "m3a_spec_frozen_v1" only after Pi rules

# Frozen coarse length bins reused by every length-conditioned statistic (S1/S5/S6). L in [2..8]; L=1 is M0b.
LENGTH_BINS = ((2, 3), (4, 5), (6, 7), (8, 8))
CLUSTER_SIZE_BINS = ((1, 1), (2, 2), (3, 4), (5, 8))     # burst-size bins for S2/S3/S7

# ---- the six marginal checks (already the v1 realism envelope; thresholds reused verbatim) ----
MARGINAL_CHECKS = {
    "length_ks":        {"stat": "KS(length ECDF)",                 "threshold": ORACLE_ENV_KS,   "op": "<="},
    "class_tv":         {"stat": "TV(5-class prior)",               "threshold": ORACLE_ENV_TV,   "op": "<="},
    "count_ks":         {"stat": "KS(per-seq event-count ECDF)",    "threshold": ORACLE_ENV_KS,   "op": "<="},
    "occupancy_abs":    {"stat": "|Δ mean occupancy| (distinct/5)", "threshold": ORACLE_ENV_OCCUPANCY_ABS, "op": "<="},
    "delta_t_zero_abs": {"stat": "|Δ P(Δt=0)|",                     "threshold": ORACLE_ENV_DT0_ABS, "op": "<="},
    "positive_gap_ks":  {"stat": "KS(positive-gap ECDF)",           "threshold": ORACLE_ENV_KS,   "op": "<="},
}

# ---- cross-statistics S1–S7 (DRAFT thresholds flagged for Pi) ----
# Every entry: definition, bins, threshold (op), denominator floor, refusal rule. Thresholds marked
# "pi_ruling": "PROPOSED" are the ones Pi must confirm or revise before the freeze.
_DENOM_FLOOR = ORACLE_ENV_MIN_DENOM         # 500; below => NOT_EVALUABLE, never zero-filled
_REFUSAL = "denom < floor => NOT_EVALUABLE for that cell; coarsen small cells per the coarsening policy"

CROSS_STATISTICS = {
    "S1": {"identifies": "burst_count/length coupling",
           "definition": "E[cluster-count K | length-bin] on LENGTH_BINS + Kendall tau(L,K)",
           "bins": {"length": LENGTH_BINS},
           "threshold": {"abs_E_K_per_bin": 0.25, "abs_tau": 0.05}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
    "S2": {"identifies": "compound/burst-size law",
           "definition": "ECDF of Δt=0 cluster-run sizes, KS",
           "bins": {"cluster_size": CLUSTER_SIZE_BINS},
           "threshold": {"ks": ORACLE_ENV_KS}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
    "S3": {"identifies": "burst-timing coupling",
           "definition": "mean positive gap | preceding-cluster-size bin, and Kendall tau(preceding size, gap)",
           "bins": {"cluster_size": CLUSTER_SIZE_BINS},
           "threshold": {"abs_tau": 0.05}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
    "S4": {"identifies": "mark–burst tie (same-class panels)",
           "definition": "P(same class | same cluster) - P(same class | adjacent clusters)",
           "bins": {}, "threshold": {"abs": ORACLE_ENV_OCCUPANCY_ABS}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
    "S5": {"identifies": "composition–length coupling (occupancy is L-censored; use the M0b capped ceiling)",
           "definition": "E[occupancy | length-bin] vs the M0b occupancy cap min(L,5)/5",
           "bins": {"length": LENGTH_BINS},
           "threshold": {"abs": ORACLE_ENV_OCCUPANCY_ABS}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
    "S6": {"identifies": "length-dependent class mix (MANDATORY per Pi P-B)",
           "definition": "class TV between length terciles",
           "bins": {"length": "terciles"},
           "threshold": {"tv": ORACLE_ENV_TV}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "mandatory": True, "pi_ruling": "PROPOSED"},
    "S7": {"identifies": "class-diversity allocation across burst sizes (added per Pi P-B)",
           "definition": "E[n_distinct_classes / min(C,5) | cluster-size-bin]",
           "bins": {"cluster_size": CLUSTER_SIZE_BINS},
           "threshold": {"abs": ORACLE_ENV_OCCUPANCY_ABS}, "op": "<=",
           "denom_floor": _DENOM_FLOOR, "refusal": _REFUSAL, "pi_ruling": "PROPOSED"},
}

# ---- parameter -> identifying statistic attribution (Pi P-B; also the escalation component->check map) ----
# Each D_copula dependence parameter is identified by exactly the listed statistic(s); marginal laws map to
# the marginal checks + the structure-only cross-stats (S2/S5/S6).
PARAM_TO_STATISTIC = {
    "burst_count_length":        ["S1"],
    "burst_timing":              ["S3"],
    "mark_burst_tie":            ["S4"],
    "cluster_size_mark_diversity": ["S7"],
    # marginal laws (shared by A and D)
    "length_law":                ["length_ks", "S5", "S6"],
    "class_law":                 ["class_tv", "S6"],
    "cluster_size_law":          ["S2", "count_ks"],
    "gap_law":                   ["positive_gap_ks", "S3"],
    "dt0_law":                   ["delta_t_zero_abs", "S2"],
}

# ---- identifiability battery spec (RUN at M3; DECLARED + frozen here) ----
IDENTIFIABILITY_BATTERY = {
    "rule": "n_dependence_params <= n_independent_cross_statistic_DoF (counted AFTER constraints/covariance)",
    "local_rank": {"test": "full-column-rank sensitivity Jacobian d(cross-stats)/d(dependence-params)",
                   "at": "every registered profile", "rank_tol": 1e-6, "pi_ruling": "PROPOSED"},
    "global_recovery": {"test": "grid-recovery of each dependence param on a joint grid with the marginal "
                                "nuisance params varied", "recover_tol": {"abs": 0.05}, "pi_ruling": "PROPOSED"},
    "collision_search": {"test": "no two distinct dependence-param settings map to within-threshold-identical "
                                 "cross-statistic vectors", "pi_ruling": "PROPOSED"},
    "registered_profiles": ["scid_like", "mimic_like", "independent_A_baseline", "copula_zeroed_null"],
}

# ---- power statement per check (DRAFT; Pi to rule) ----
POWER = {
    "target_power": 0.80,
    "misspecified_control_must_fail": True,      # the attribution-mapped mis-specified control fails at >=power
    "n_seeds": 5,
    "seed_aggregation": "median across seeds must satisfy the threshold; the mis-specified control's median "
                        "must breach it (worst-of-both rule)",
    "pi_ruling": "PROPOSED",
}

# ---- source conjunction (Pi P-B) ----
SOURCE_CONJUNCTION = {
    "required_sources": list(REQUIRED_SOURCES),
    "rule": "every check (marginals + S1–S7) must pass for EVERY required source; conjunction over sources "
            "AND over checks; any NOT_EVALUABLE source-check => the whole spec is NOT_EVALUABLE (never a pass)",
}

# ---- escalation discipline + immutable ledger (Pi P-D) ----
ESCALATION = {
    "baseline_identity": _HIST_A_ID,                            # M2 binds A first (historical snapshot)
    "escalation_identity": _HIST_D_ID,                          # controls-driven only (historical snapshot)
    "component_to_check": PARAM_TO_STATISTIC,
    "decision_basis": "known-ground-truth control battery ONLY; TRAIN-target diagnostics are non-decisional",
    "tie_rule": "if >1 component maps to the same failed check, escalate the SMALLEST super-set of components "
                "whose union covers the failed checks; ties broken by the frozen PARAM_TO_STATISTIC order",
    "iteration_cap": 3,
    "ledger": {"immutable": True,
               "entry_fields": ["round", "failed_checks", "escalated_components", "new_identity",
                                "battery_result_hash", "seed_set"],
               "on_escalation": "mint new D identity + append one ledger entry + re-run the FULL battery"},
    "pi_ruling": "PROPOSED",
}

M3A_VERIFICATION_SPEC = {
    "version": M3A_SPEC_VERSION,
    "n_classes": ORACLE_ENV_N_CLASSES,
    "marginal_checks": MARGINAL_CHECKS,
    "cross_statistics": CROSS_STATISTICS,
    "length_bins": LENGTH_BINS,
    "cluster_size_bins": CLUSTER_SIZE_BINS,
    "param_to_statistic": PARAM_TO_STATISTIC,
    "identifiability_battery": IDENTIFIABILITY_BATTERY,
    "power": POWER,
    "source_conjunction": SOURCE_CONJUNCTION,
    "escalation": ESCALATION,
    "marginal_schema_ref": _HIST_MARGINAL_SCHEMA,
    "m0b_support_policy": _HIST_M0B_POLICY,
    "admissible_claim": "matches the declared marginal + cross-statistic envelope; NEVER the joint process",
}


def m3a_spec_dev_hash() -> str:
    """DEVELOPMENT identity of the M3a verification-spec DRAFT. The FINAL frozen identity
    (`m3a_spec_frozen_hash`, version bumped) is minted only after Pi rules on this draft."""
    return canonical_hash(M3A_VERIFICATION_SPEC)


def open_pi_rulings() -> dict[str, Any]:
    """Every spec element still flagged PROPOSED (awaiting Pi's ruling before the freeze)."""
    flagged = {k: v for k, v in CROSS_STATISTICS.items() if v.get("pi_ruling") == "PROPOSED"}
    return {"cross_statistic_thresholds": list(flagged),
            "identifiability_tolerances": ["local_rank.rank_tol", "global_recovery.recover_tol", "collision_search"],
            "power": ["target_power", "n_seeds", "seed_aggregation"],
            "escalation": ["tie_rule", "iteration_cap"]}
