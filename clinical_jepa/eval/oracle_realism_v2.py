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

from dataclasses import dataclass
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
from clinical_jepa.eval.oracle_literal_gen import LiteralCell, L_ITEMS
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


# ==================================================================================================
# M0 order-restriction boundary (blueprint step 2; Pi P-A + guard-integration condition).
#
# Two additive pieces, both SYNTHETIC-only and emission-free:
#   * `RestrictedOrderCore` + `restrict_order_core` — the production order-restriction primitive that
#     REPLACES the tautological `_restrict` proof fixture. It restricts a canonical full-L LiteralCell to a
#     subset of item positions, recomputes `future_events`, and materializes NO emission fields (fresh
#     realized-length emission is M2 adapter behaviour under the frozen verifier, NOT M0).
#   * `assert_canonical_certification_cell` — the fail-hard guard that REJECTS any restricted/masked or
#     non-canonical object at a fixed-L certification/reference/verdict entrypoint (never relying on the
#     recipe reshape to fail). No-op on a valid canonical cell, so no valid-path output moves.
# The guard's contract is bound into an ADDITIVE v2 boundary identity (`v2_certification_boundary_hash`),
# separate from `realism_v2_schema_hash` (unchanged) and every frozen v1 identity.
# ==================================================================================================

CANONICAL_CERT_L = L_ITEMS      # 8 — certification/reference/verdict accept ONLY this order length

# Public certification/reference/verdict entrypoints the guard protects (Pi guard-integration condition).
CERTIFICATION_ENTRYPOINTS = (
    "eo1_recipe", "eo1_r0", "eo1_r_nuis", "eo1_r_bayes",
    "eo1_mean_embed_quantized", "eo1_random_codebook", "hidden_null_excluded",
    "verdict._cell_pass",
)

# L-indexed channels a canonical cell must carry at exactly CANONICAL_CERT_L.
_L_INDEXED_CHANNELS = ("true_order", "nuisance_u", "item_features", "future_events")


@dataclass(frozen=True)
class RestrictedOrderCore:
    """M0 order-restriction primitive (Pi P-A). An ADDITIVE, EMISSION-FREE order core obtained by restricting
    a canonical full-L `LiteralCell` to a subset of item positions.

    It carries ONLY the order / nuisance / item core needed to reason about the restricted ranking. It
    deliberately MATERIALIZES NO emission fields — `future_timestamps`, `cluster_ids`, `multiplicity`,
    `future_multiset`, marks are ABSENT: generating them fresh at the realized length is M2 adapter behaviour
    under the frozen verifier, not M0. It is NOT a `LiteralCell` and is fail-hard REJECTED by every
    certification/reference/verdict entrypoint (certification stays fixed-L; restriction is emission-only)."""
    family_id: str
    kappa: float
    nuisance_cell: str
    subset: tuple[int, ...]              # selected full-L item positions, in the given order
    realized_length: int
    s_true_subset: np.ndarray           # (N, k) restricted order-scores (exact column selection)
    item_subset: np.ndarray             # (N, k, D_ITEM) restricted item features
    nuisance_subset: np.ndarray         # (N, k) EXACT column slice — NEVER re-standardized
    future_events_subset: np.ndarray    # (N, k) RECOMPUTED argsort(argsort(s_true_subset)) — never sliced
    context_features: np.ndarray        # (N, D_CTX) unchanged (per-sequence, not L-indexed)
    is_null: np.ndarray                 # (N,) unchanged (eval-only label)
    source_restriction: str = "order_restriction_v2"


def restrict_order_core(cell: LiteralCell, subset) -> RestrictedOrderCore:
    """Restrict a canonical full-L LiteralCell to ``subset`` item positions, producing an emission-free
    `RestrictedOrderCore`. `future_events` is RECOMPUTED from the restricted scores (never sliced); the
    correlated nuisance column is taken as an EXACT slice (never re-standardized); NO emission field is
    sliced or fabricated. L=1 is out of scope here (belongs to M0b)."""
    if not isinstance(cell, LiteralCell):
        raise TypeError(f"restrict_order_core requires a canonical LiteralCell, got {type(cell).__name__}")
    L = int(np.asarray(cell.true_order).shape[1])
    sub = tuple(int(i) for i in subset)
    if len(sub) < 2 or len(set(sub)) != len(sub) or any(i < 0 or i >= L for i in sub):
        raise ValueError(f"invalid subset {sub} for L={L}: need >=2 distinct in-range positions (L=1 is M0b)")
    s = np.asarray(cell.true_order)[:, sub]
    fe = np.argsort(np.argsort(s, axis=1), axis=1)          # recomputed realized-order rank, NOT a slice
    return RestrictedOrderCore(
        family_id=cell.family_id, kappa=float(cell.kappa), nuisance_cell=cell.nuisance_cell,
        subset=sub, realized_length=len(sub),
        s_true_subset=s,
        item_subset=np.asarray(cell.item_features)[:, sub],
        nuisance_subset=np.asarray(cell.nuisance_u)[:, sub],    # exact slice; no re-standardization
        future_events_subset=fe,
        context_features=np.asarray(cell.context_features),
        is_null=np.asarray(cell.is_null),
    )


class CertificationBoundaryError(TypeError):
    """A restricted/masked/variable-length or otherwise non-canonical object reached a fixed-L
    certification/reference/verdict entrypoint. Raised by `assert_canonical_certification_cell`."""


def assert_canonical_certification_cell(obj: Any, *, entrypoint: str) -> None:
    """Fail-hard boundary: certification/reference/verdict accept ONLY a canonical fixed-L=8 `LiteralCell`.

    A `RestrictedOrderCore`, an emission mask, a restricted/variable-length or otherwise non-canonical object
    is REJECTED here — NOT left to fail incidentally on the recipe's `self._L` reshape. This is a NO-OP
    (returns None) for a valid canonical cell, so no valid-path output moves (Pi guard-integration
    condition)."""
    if isinstance(obj, RestrictedOrderCore):
        raise CertificationBoundaryError(
            f"{entrypoint}: a RestrictedOrderCore (emission-only, variable length) is not certifiable; "
            "certification is fixed-L and order-restriction is emission/evaluator-realism-only")
    if not isinstance(obj, LiteralCell):
        raise CertificationBoundaryError(
            f"{entrypoint}: expected a canonical LiteralCell, got {type(obj).__name__}")
    if getattr(obj, "restriction_meta", None) is not None:      # defensive: no restriction metadata allowed
        raise CertificationBoundaryError(f"{entrypoint}: cell carries v2 restriction metadata; not certifiable")
    for name in _L_INDEXED_CHANNELS:
        arr = np.asarray(getattr(obj, name))
        got = arr.shape[1] if arr.ndim >= 2 else None
        if got != CANONICAL_CERT_L:
            raise CertificationBoundaryError(
                f"{entrypoint}: {name} has L={got} != canonical {CANONICAL_CERT_L}; "
                "variable-length cells never reach fixed-L certification")


V2_CERTIFICATION_BOUNDARY = {
    "canonical_L": CANONICAL_CERT_L,
    "entrypoints": list(CERTIFICATION_ENTRYPOINTS),
    "l_indexed_channels": list(_L_INDEXED_CHANNELS),
    "rejects": ["RestrictedOrderCore", "non_LiteralCell", "restriction_meta", "L_ne_canonical"],
    "rationale": "fixed-L certification; variable length is emission-only; reject, never rely on the reshape",
}


def v2_certification_boundary_hash() -> str:
    """Additive v2 boundary identity for the fail-hard certification guard. Separate from
    `realism_v2_schema_hash` (unchanged) and from every frozen v1 identity."""
    return canonical_hash(V2_CERTIFICATION_BOUNDARY)
