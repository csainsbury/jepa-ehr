"""Oracle realism v2 — identity/schema scaffolding (blueprint M1).

Option D: the frozen order mechanism wrapped in a source-conditioned, variable-length, compound-burst copula
realism envelope. This module is IDENTITY SCAFFOLDING ONLY — it declares and content-hashes the frozen v2
schema (the per-source law structure, the sparse copula descriptor, and the predeclared cross-statistics set)
so downstream work binds a stable identity. It contains NO generator behaviour yet and touches NO existing
identity (mechanism/reference/calibration/base/adapter hashes are unmoved).

Governance: synthetic-only. The v2 realism layer is a specification/realism test, not clinical prediction.
The verification spec (marginals + cross-statistics) is FROZEN before any fitting (blueprint M3a), and any
confirmatory realism claim requires a separate pre-registered locked/external gate (blueprint M4). No
latent-mechanism / transfer / counterfactual / causal / order-certification claim.
"""
from __future__ import annotations

from typing import Any

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
from clinical_jepa.eval.rung2_contract import ORACLE_ENV_N_CLASSES

REALISM_V2_VERSION = "realism_v2_scaffold_dev"      # bumped when the v2 generator behaviour lands (M2)

# Per-source law structure the v2 envelope emits (declared; parameterization frozen at M2/M3a).
V2_LAW_STRUCTURE = {
    "length_law": "source_conditioned_variable_length_via_order_restriction",   # M0: restriction-invariant
    "class_law": "dirichlet_multinomial_with_hard_structural_zeros",
    "cluster_size_law": "compound_burst_size_distribution",
    "gap_law": "source_positive_gap_distribution",
    "dt0_law": "cluster_size_induced_simultaneity",                             # NOT a widened 0.9-clip grid
    "join": "sparse_compound_burst_copula",       # only the couplings the cross-statistics can see
    "order_restriction": "deterministic_restriction_of_canonical_fixed_L_ranking",
    "certification": "fixed_L (recipe predict_latent is context-only, hard-coded to training L)",
    "seam": "post_hoc_adapter_only (marks+timestamps); dedicated adapter RNG; never s_true/future_events/nuisance_u",
}

# Predeclared cross-statistics (S1..S6) — necessary beyond the six marginal checks because aggregates
# constrain marginals but do NOT identify the joint process (Fable review #2). Bins/thresholds/power are
# FROZEN at M3a before fitting; NONE are in the committed extraction contract, so every real cross-statistic
# is a NEW governed field at the M4 locked/external gate.
V2_CROSS_STATISTICS = {
    "S1": "E[cluster_count K | length_bin] + kendall_tau(L,K)",
    "S2": "ECDF of Delta_t=0 cluster-run sizes (KS)",
    "S3": "mean_positive_gap | preceding_cluster_size_class (or tau)",
    "S4": "P(same_class|same_cluster) - P(same_class|adjacent_clusters)",
    "S5": "E[occupancy | length_bin]",
    "S6": "class_TV between length terciles (optional)",
    "identifiability_rule": "n_dependence_params <= n_independent_cross_statistic_dof; grid-recover each param",
    "admissible_claim": "matches declared marginal+cross-statistic envelope; NEVER the joint process",
}

# Escalation discipline (frozen at M3a): decisions are made on the KNOWN-GROUND-TRUTH controls ONLY, never on
# the development-seen TRAIN targets.
V2_ESCALATION = {
    "baseline": "option_A_independent_source_conditioned_marginals",
    "escalate_to": "option_D_compound_burst_copula (only attribution-mapped failed components)",
    "decision_basis": "known_ground_truth_control_battery_only (TRAIN-target comparisons are exploratory)",
    "discipline": "identity_bump + full_battery_rerun + iteration_cap + escalation_ledger",
    "parked": "neural_marked_TPP (until A/D fail controlled tests AND an external target exists)",
}


def realism_v2_schema_hash() -> str:
    """Frozen content hash of the v2 realism schema — the identity future v2 work binds. Additive: it does
    NOT alter mechanism/reference/calibration/base/adapter identities."""
    return canonical_hash({
        "version": REALISM_V2_VERSION, "n_classes": ORACLE_ENV_N_CLASSES,
        "required_sources": list(REQUIRED_SOURCES), "law_structure": V2_LAW_STRUCTURE,
        "cross_statistics": V2_CROSS_STATISTICS, "escalation": V2_ESCALATION,
    })
