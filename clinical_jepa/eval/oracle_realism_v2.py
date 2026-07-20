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
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_N_CLASSES, ORACLE_ENV_MIN_DENOM, ORDER_SUPPORT_FLOOR,
)

REALISM_V2_VERSION = "realism_v2_scaffold_dev"      # bumped when the v2 generator behaviour lands (M2)

# Per-source law structure the v2 envelope emits (declared; parameterization frozen at M2/M3a).
V2_LAW_STRUCTURE = {
    "length_law": "source_conditioned_variable_length_via_order_restriction",   # M0: restriction-invariant
    "class_law": "dirichlet_multinomial_with_hard_structural_zeros",
    "cluster_size_law": "compound_burst_size_distribution",
    "gap_law": "source_positive_gap_distribution",
    "dt0_law": "cluster_size_induced_simultaneity",                             # NOT a widened 0.9-clip grid
    "join": "variant_selected: A_independent baseline (bound first at M2) | D_copula escalation (new identity)",
    "variant_selection": "see V2_VARIANT_A_INDEPENDENT / V2_VARIANT_D_COPULA — A and D are DISTINCT identities",
    "order_restriction": "deterministic_restriction_of_canonical_fixed_L_ranking",
    "certification": "fixed_L, enforced by assert_canonical_certification_cell (NOT the recipe reshape, which "
                     "reads item_features via _design); variable length is emission-only",
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


# ==================================================================================================
# Option-A / Option-D identity split (blueprint step 3; Pi P-D-1).
#
# The v2 envelope shares ONE marginal schema across both options; the OPTIONS differ ONLY in the join.
# M2 binds `A_independent` FIRST as a falsifiable baseline. Entering `D_copula` is a controls-driven M3
# escalation that mints a NEW identity + escalation-ledger entry — never a silent switch. Both are
# DEVELOPMENT identities (behaviour lands at M2; the FINAL identities are minted at the M3a freeze). This
# step adds NO sampling law / parameter fit / target comparison — the adapter is an INTERFACE STUB only.
# ==================================================================================================

# The marginals shared by A and D (the join is the ONLY difference between the options).
V2_MARGINAL_SCHEMA = {
    "length_law": "source_conditioned_variable_length_via_order_restriction",
    "class_law": "dirichlet_multinomial_with_hard_structural_zeros",
    "cluster_size_law": "compound_burst_size_distribution",
    "gap_law": "source_positive_gap_distribution",
    "dt0_law": "cluster_size_induced_simultaneity",
    "required_sources": list(REQUIRED_SOURCES),
    "n_classes": ORACLE_ENV_N_CLASSES,
}

V2_VARIANT_A_INDEPENDENT = {
    "variant": "A_independent",
    "join": "independent_source_conditioned_marginals",   # NO cross-item coupling
    "dependence_params": [],                               # empty by construction — the falsifiable baseline
    "role": "baseline_bound_first_at_M2",
}

V2_VARIANT_D_COPULA = {
    "variant": "D_copula",
    "join": "sparse_compound_burst_copula",               # only the couplings the cross-statistics can see
    "dependence_params": ["burst_count_length", "burst_timing", "mark_burst_tie", "cluster_size_mark_diversity"],
    "role": "controls_driven_escalation_only",
    "escalation_precondition": "attribution-mapped marginal/cross-stat failure on known-ground-truth controls",
    "on_escalation": "mint new identity + escalation-ledger entry; re-run the full battery under the cap",
}

# Adapter INTERFACE stub only — declares the emission seam contract. NO sampling law, fit, or target
# comparison exists until M2 (and only after the M3a freeze). Emission-side only; certification never sees it.
V2_ADAPTER_INTERFACE = {
    "adapter_stub": "realism_v2_adapter_iface_dev",
    "inputs": ["canonical_full_L_literal_cell", "source_profile", "variant_identity", "frozen_verifier_spec"],
    "emits": ["marks", "timestamps", "cluster_metadata", "restriction_mask"],
    "forbidden_pre_m3a": ["sampling_law", "parameter_fit", "target_comparison"],
    "certification_boundary": "emits only; certification receives unmasked fixed-L cells (see the guard)",
}


def v2_marginal_schema_hash() -> str:
    """Identity of the marginal schema SHARED by Option A and Option D."""
    return canonical_hash(V2_MARGINAL_SCHEMA)


def v2_variant_identity(variant: str, *, final: bool = False) -> str:
    """Distinct identity per option. `final=False` (default) is the DEVELOPMENT identity; the FINAL identity
    (`final=True`) is minted only at the M3a freeze. `A_independent` and `D_copula` NEVER share a hash, and
    dev never collides with final (Pi P-D-1)."""
    spec = {"A_independent": V2_VARIANT_A_INDEPENDENT, "D_copula": V2_VARIANT_D_COPULA}.get(variant)
    if spec is None:
        raise KeyError(f"unknown v2 variant {variant!r} (expected 'A_independent' or 'D_copula')")
    return canonical_hash({"marginal_schema": V2_MARGINAL_SCHEMA, "variant_spec": spec,
                           "stage": "final_m3a" if final else "dev_scaffold"})


def v2_adapter_interface_hash() -> str:
    """Identity of the adapter INTERFACE stub (no behaviour). Bumps when the M2 adapter behaviour lands."""
    return canonical_hash(V2_ADAPTER_INTERFACE)


# ==================================================================================================
# M0b — support-floor / min-length accounting for the v2 order core (blueprint step 4; Pi P-A owns L=1).
#
# The v2 realism layer emits VARIABLE realized lengths. Every restricted order core must be classified
# EXPLICITLY (never silently): a vacuous L<=1 order, a support-starved cell / pair, and the structural
# occupancy cap at L<5. This is an ACCOUNTING layer for the v2 realism/emission side ONLY — restricted
# cores never reach fixed-L certification (the guard rejects them). Occupancy is `distinct classes / C`
# (C=5), so a realized length L<5 caps occupancy at L/5 by construction; that is an accounting FLAG, not a
# realism miss (a downstream occupancy check must compare against the capped ceiling, not 1.0).
# ==================================================================================================

M0B_OCCUPANCY_CAP_LENGTH = 5                 # occupancy = distinct/C(=5); L below this caps occupancy at L/5
M0B_PAIR_DENOM_FLOOR = ORACLE_ENV_MIN_DENOM  # per-pair eligible (non-tied) denominator floor (500)
M0B_CELL_SUPPORT_FLOOR = ORDER_SUPPORT_FLOOR # per-cell sequence-support floor (500)
_M0B_TIE_ATOL = 1e-9

SUPPORT_OK = "SUPPORTED"
SUPPORT_STARVED = "SUPPORT_STARVED"
VACUOUS_ORDER = "VACUOUS_ORDER"              # L<=1: order undefined, no adjacency


@dataclass(frozen=True)
class OrderSupportAccounting:
    """Explicit, never-silent M0b classification of a (candidate or realized) v2 order core."""
    realized_length: int
    n_sequences: int
    status: str                              # SUPPORTED | SUPPORT_STARVED | VACUOUS_ORDER
    occupancy_cap: float                     # min(L, 5)/5 — structural occupancy ceiling at this length
    occupancy_capped: bool                   # L < 5
    per_pair_min_denom: int                  # smallest eligible (non-tied) pair denominator (n if no pairs)
    reasons: tuple[str, ...]                 # human-readable accounting reasons (always populated when off-nominal)


def account_order_support(realized_length: int, n_sequences: int, per_pair_min_denom: int, *,
                          cell_floor: int = M0B_CELL_SUPPORT_FLOOR,
                          pair_floor: int = M0B_PAIR_DENOM_FLOOR) -> OrderSupportAccounting:
    """Classify a v2 order core by realized length + support counts. L<=1 is VACUOUS (undefined order);
    below the per-cell or per-pair floor is SUPPORT_STARVED; L<5 flags the structural occupancy cap L/5
    (an accounting flag, not a starvation). NEVER returns a silent SUPPORTED when a floor is breached."""
    L = int(realized_length)
    occ_cap = min(max(L, 0), M0B_OCCUPANCY_CAP_LENGTH) / float(M0B_OCCUPANCY_CAP_LENGTH)
    occ_capped = L < M0B_OCCUPANCY_CAP_LENGTH
    if L <= 1:
        return OrderSupportAccounting(L, int(n_sequences), VACUOUS_ORDER, occ_cap, occ_capped,
                                      int(per_pair_min_denom),
                                      ("L<=1: vacuous order, no adjacency — NOT evaluable for order",))
    reasons: list[str] = []
    if int(n_sequences) < cell_floor:
        reasons.append(f"per-cell support {int(n_sequences)} < floor {cell_floor}")
    if int(per_pair_min_denom) < pair_floor:
        reasons.append(f"per-pair min eligible denom {int(per_pair_min_denom)} < floor {pair_floor}")
    status = SUPPORT_STARVED if reasons else SUPPORT_OK
    if occ_capped:                            # accounting flag regardless of starvation status
        reasons.append(f"L={L}<{M0B_OCCUPANCY_CAP_LENGTH}: occupancy structurally capped at {L}/5")
    return OrderSupportAccounting(L, int(n_sequences), status, occ_cap, occ_capped,
                                  int(per_pair_min_denom), tuple(reasons))


def restricted_core_support(core: "RestrictedOrderCore", *, cell_floor: int = M0B_CELL_SUPPORT_FLOOR,
                            pair_floor: int = M0B_PAIR_DENOM_FLOOR) -> OrderSupportAccounting:
    """M0b accounting computed from an actual `RestrictedOrderCore`: the per-pair eligible denominator is the
    count of sequences whose pair is non-tied, minimised over surviving pairs (continuous scores ⇒ typically
    n, but computed honestly)."""
    if not isinstance(core, RestrictedOrderCore):
        raise TypeError(f"restricted_core_support requires a RestrictedOrderCore, got {type(core).__name__}")
    s = np.asarray(core.s_true_subset)
    n, k = s.shape
    min_denom = n
    for a in range(k):
        for b in range(a + 1, k):
            elig = int(np.count_nonzero(np.abs(s[:, a] - s[:, b]) > _M0B_TIE_ATOL))
            min_denom = min(min_denom, elig)
    return account_order_support(core.realized_length, n, min_denom, cell_floor=cell_floor, pair_floor=pair_floor)


M0B_SUPPORT_POLICY = {
    "cell_support_floor": M0B_CELL_SUPPORT_FLOOR,
    "pair_denom_floor": M0B_PAIR_DENOM_FLOOR,
    "occupancy_cap_length": M0B_OCCUPANCY_CAP_LENGTH,
    "occupancy_definition": "distinct_classes / C (C=5); L<5 caps occupancy at L/5",
    "statuses": [SUPPORT_OK, SUPPORT_STARVED, VACUOUS_ORDER],
    "discipline": "never-silent: any floor breach => SUPPORT_STARVED with reasons; L<=1 => VACUOUS_ORDER",
    "scope": "v2 realism/emission order cores only; restricted cores never reach fixed-L certification",
}


def m0b_support_policy_hash() -> str:
    """Additive identity of the M0b support/min-length policy (folded into the M3a freeze later)."""
    return canonical_hash(M0B_SUPPORT_POLICY)
