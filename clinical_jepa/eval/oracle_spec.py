"""FROZEN semi-synthetic oracle specification (Pi consolidated #5/#6).

This module is the *specification*, frozen and hashable, that the generator, evaluator, and
calibration read. It is pure data + a deterministic mechanism hash — NO RNG, NO generation, NO real
data. Freezing the families/grids/sample-counts and the calibration mechanics BEFORE any generation
or aggregate read is the anti-tailoring commitment: the mechanism hash computed here is exactly the
`oracle_mechanism_hash` the trusted committed policy pins (see ``oracle_policy``).

Scope (all safe-public / synthetic):
  * ≥3 TRAIN structural meta-families + ≥2 HELD-OUT families, including an exact NO-h family and an
    OFF-GRID family whose mechanism is unseen in train (Pi #5).
  * For every family: the hidden-state law, the context-generation map, the order mechanism, the
    marked-TPP timing mechanism, the Σ-orthogonal vs correlated-leak nuisance construction, the
    null-mixture weight, the camouflaged-null procedure, the coupling grid, and per-split N (Pi #5).
  * The calibration mechanics: exact aggregate fields, governance class, fitting objective/optimizer,
    parameter ranges, deterministic tie-breaking, tolerances, failure behavior, and its own hash;
    calibration may NOT alter family definitions, seeds, metrics, or the recipe registry (Pi #6).

Nothing here reads governed data. The reference bracket (R_bayes / R0 / R_nuis) and the evaluator
metrics live in separate modules; this file only fixes WHAT is generated and HOW it is calibrated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

SPEC_VERSION = "clinical-jepa-oracle-spec-v3"

# ----------------------------------------------------------------------------------------------
# Coupling grid. kappa = coupling strength between the hidden state h and the ORDER channel.
# kappa=0 => order carries no context-predictable signal (the mechanistic null); larger kappa =>
# more context-conditional order information. The TRAIN grid is on-grid; held-out families probe
# OFF-grid kappa (ORACLE_OFFGRID_KAPPA in the contract) so a certifier cannot overfit a grid point.
# ----------------------------------------------------------------------------------------------
KAPPA_TRAIN_GRID: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50, 0.75)
KAPPA_OFFGRID: tuple[float, ...] = (0.22, 0.63)          # inside ORACLE_OFFGRID_KAPPA=(0.15,0.6)-ish band

# Null-mixture weight: fraction of sequences in a cell drawn from the mechanistic null (order drawn
# from the content-prior pi0 ONLY, independent of h). Positives and nulls share identical marginal
# token/count/timing statistics — the null is CAMOUFLAGED; only the context-conditional order
# residual differs. R0 (content-prior floor) must PASS nulls and FAIL positives.
NULL_MIXTURE_WEIGHT: float = 0.5

# Nuisance construction. "orthogonal": the nuisance channel u is drawn independently of the order
# label (Sigma block-diagonal) => R_nuis must LOSE incremental order-skill here. "correlated_leak":
# u is an order-correlated proxy with bounded mutual information (a monotone-noised copy) => R_nuis
# SHOULD capture that leakage but must not exceed R_bayes.
NUISANCE_LEAK_RHO: float = 0.6                            # proxy correlation in correlated-leak cells
NUISANCE_MI_BITS_CAP: float = 0.5                         # leak is bounded, never order-sufficient


@dataclass(frozen=True)
class StructuralFamily:
    """One generative meta-family. Fields are a frozen mathematical contract, not free parameters."""
    family_id: str
    split: str                       # "train" | "held_out"
    has_h: bool                      # whether a hidden common cause h drives BOTH context and future
    hidden_state: str                # law of h (or the exogenous driver when has_h is False)
    context_map: str                 # x_context = g(.) : how observed context is produced
    order_mechanism: str             # how the TRUE order label is generated (the certified property)
    timing_mechanism: str            # marked temporal point process for event times / multiplicity
    nuisance_cells: tuple[str, ...]  # which Sigma constructions this family instantiates
    null_mixture_weight: float       # fraction of sequences that are mechanistic nulls
    camouflaged_null: str            # how nulls are made marginally indistinguishable from positives
    kappa_cells: tuple[float, ...]   # coupling grid points this family is generated at
    n_sequences: int                 # sequences per (kappa cell) for this family/split
    notes: str = ""


# ----------------------------------------------------------------------------------------------
# TRAIN families (>=3). Diverse hidden-state laws so a certifier can't assume one mechanism.
# ----------------------------------------------------------------------------------------------
_TRAIN_FAMILIES: tuple[StructuralFamily, ...] = (
    StructuralFamily(
        family_id="T_hmm_markov",
        split="train", has_h=True,
        hidden_state="h_t discrete K=6 exogenous Markov chain, fixed transition matrix P (row-stochastic, seeded)",
        context_map="x_context = emission B[h_t] (categorical token emissions) + independent nuisance channel u",
        order_mechanism="true precedence kappa-mixes h-conditioned precedence kernel Q[h] with content-prior pi0: "
                        "P(order)=(1-kappa)*pi0 + kappa*softmax(Q[h])",
        timing_mechanism="marked TPP: inter-event gaps ~ Exp(rate=exp(a0 + a1*state_occupancy)); Delta t=0 clusters = multiplicity",
        nuisance_cells=("orthogonal", "correlated_leak"),
        null_mixture_weight=NULL_MIXTURE_WEIGHT,
        camouflaged_null="nulls draw order from pi0 but keep identical token multiset, count, and timing marginals",
        kappa_cells=KAPPA_TRAIN_GRID,
        n_sequences=4000,
        notes="canonical hidden-common-cause family; h drives both context emissions and order.",
    ),
    StructuralFamily(
        family_id="T_realized_history",
        split="train", has_h=True,
        hidden_state="h = deterministic running summary of the REALIZED token prefix (autoregressive state)",
        context_map="x_context = the realized prefix tokens themselves + nuisance channel u",
        order_mechanism="next-item precedence depends on realized prefix via a fixed scoring net phi(prefix); "
                        "kappa-mixed with pi0 as above",
        timing_mechanism="marked TPP with history-dependent Hawkes-like excitation (bounded, stationary)",
        nuisance_cells=("orthogonal", "correlated_leak"),
        null_mixture_weight=NULL_MIXTURE_WEIGHT,
        camouflaged_null="nulls freeze the prefix-summary influence on order to zero while preserving prefix marginals",
        kappa_cells=KAPPA_TRAIN_GRID,
        n_sequences=4000,
        notes="order depends on realized history rather than an exogenous latent; distinguishes h-projection shortcuts.",
    ),
    StructuralFamily(
        family_id="T_latent_factor",
        split="train", has_h=True,
        hidden_state="h ~ N(0, I_d), d=8 continuous latent factor (static per sequence)",
        context_map="x_context = W_ctx h + eps_ctx (linear-Gaussian) discretized to tokens; nuisance channel u",
        order_mechanism="order-score s = w_order . h ; true order = argsort(s + kappa^{-1}-scaled Gumbel); "
                        "kappa scales how sharply h determines order",
        timing_mechanism="marked TPP: intensity lambda(t)=exp(b0 + b1*(w_time . h)); multiplicity via zero-gap clusters",
        nuisance_cells=("orthogonal", "correlated_leak"),
        null_mixture_weight=NULL_MIXTURE_WEIGHT,
        camouflaged_null="nulls set w_order=0 so order is pi0 while context still reflects h (context marginals preserved)",
        kappa_cells=KAPPA_TRAIN_GRID,
        n_sequences=4000,
        notes="continuous linear-Gaussian common cause; smooth kappa control of order determinism.",
    ),
)

# ----------------------------------------------------------------------------------------------
# HELD-OUT families (>=2). One is EXACTLY no-h (exogenous history, no hidden common cause) so an
# h-projection shortcut MUST fail; one is OFF-grid with an unseen mechanism.
# ----------------------------------------------------------------------------------------------
_HELDOUT_FAMILIES: tuple[StructuralFamily, ...] = (
    StructuralFamily(
        family_id="E_no_h_exogenous",
        split="held_out", has_h=False,
        hidden_state="NONE. Order and context are driven by an EXOGENOUS observable process z_ex "
                     "(a fixed clock/covariate stream); there is NO shared hidden state h.",
        context_map="x_context = deterministic featurization of z_ex + nuisance channel u",
        order_mechanism="true order = fixed exogenous rule r(z_ex); kappa-mixed with pi0. No latent common cause.",
        timing_mechanism="marked TPP driven by z_ex only (renewal process with z_ex-modulated rate)",
        nuisance_cells=("orthogonal", "correlated_leak"),
        null_mixture_weight=NULL_MIXTURE_WEIGHT,
        camouflaged_null="nulls replace r(z_ex) with pi0 while keeping z_ex-derived context/timing marginals",
        kappa_cells=KAPPA_TRAIN_GRID,
        n_sequences=3000,
        notes="H-PROJECTION SHORTCUT PROBE: any method that certifies via reconstructing a hidden h "
              "must FAIL here because no h exists (ORACLE_SHORTCUT_MAX_SKILL applies).",
    ),
    StructuralFamily(
        family_id="E_offgrid_nonlinear",
        split="held_out", has_h=True,
        hidden_state="h ~ heavier-tailed Student-t(nu=4) latent (unseen in train), static per sequence",
        context_map="x_context = nonlinear map MLP_ctx(h) discretized; nuisance channel u",
        order_mechanism="order-score s = nonlinear g_order(h) (unseen link); kappa in KAPPA_OFFGRID (off the train grid)",
        timing_mechanism="marked TPP with off-grid dispersion (unseen intensity family)",
        nuisance_cells=("orthogonal", "correlated_leak"),
        null_mixture_weight=NULL_MIXTURE_WEIGHT,
        camouflaged_null="nulls null-out g_order while preserving nonlinear context marginals",
        kappa_cells=KAPPA_OFFGRID,
        n_sequences=3000,
        notes="OFF-GRID GENERALIZATION PROBE: unseen kappa AND unseen link/timing families.",
    ),
)

STRUCTURAL_FAMILIES: tuple[StructuralFamily, ...] = _TRAIN_FAMILIES + _HELDOUT_FAMILIES


def train_families() -> tuple[StructuralFamily, ...]:
    return _TRAIN_FAMILIES


def heldout_families() -> tuple[StructuralFamily, ...]:
    return _HELDOUT_FAMILIES


def no_h_families() -> tuple[StructuralFamily, ...]:
    """Families with NO hidden common cause — the h-projection shortcut must fail on these."""
    return tuple(f for f in STRUCTURAL_FAMILIES if not f.has_h)


def get_family(family_id: str) -> StructuralFamily:
    for f in STRUCTURAL_FAMILIES:
        if f.family_id == family_id:
            return f
    raise KeyError(f"unknown structural family: {family_id!r}")


# ----------------------------------------------------------------------------------------------
# Calibration mechanics (Pi #6). Frozen BEFORE the aggregate read. Calibration may ONLY tune the
# listed nuisance/scale knobs to match a small set of AGGREGATE (non-governed-detail) real marginals
# so the synthetic realism envelope is defensible; it may NOT touch family definitions, seeds,
# metrics, or the recipe registry. It carries its own hash, separate from the mechanism hash.
# ----------------------------------------------------------------------------------------------
CALIBRATION_SPEC: dict[str, Any] = {
    "spec_version": SPEC_VERSION,
    "governance_class": "aggregate_only_safe_distilled",   # never per-patient; aggregate marginals only
    "aggregate_fields": (
        "sequence_length_quantiles(0.1,0.5,0.9)",
        "events_per_sequence_quantiles(0.1,0.5,0.9)",
        "delta_t_zero_fraction",                            # simultaneity / multiplicity rate
        "positive_inter_event_gap_quantiles(0.5,0.9)",
        "top_token_frequency_decile_curve",
    ),
    "fitting_objective": "minimize_weighted_L1_between_synthetic_and_real_aggregate_quantiles",
    "optimizer": "deterministic_grid_then_Nelder_Mead",    # deterministic; seeded start; no randomness
    "tunable_params": {                                     # ONLY these knobs; ranges are hard bounds
        "timing_rate_scale": (0.5, 2.0),
        "gap_dispersion": (0.5, 2.0),
        "zero_gap_bias": (0.0, 0.9),                       # matches Delta t=0 fraction
        "token_freq_temperature": (0.5, 2.0),
    },
    "frozen_against_calibration": (                        # calibration MUST NOT change any of these
        "structural_family_definitions", "kappa_grid", "null_mixture_weight",
        "order_mechanism", "seeds", "evaluator_metrics", "recipe_registry",
    ),
    "deterministic_tie_breaking": "lexicographic_by_param_name_then_lower_bound",
    "convergence_tol": 1e-4,
    "max_iters": 200,
    "acceptance": {                                        # realism envelope acceptance
        "max_weighted_L1": 0.10,
        "per_field_max_abs_quantile_err": 0.15,
    },
    "failure_behavior": "REFUSE_and_report: if acceptance not met, realism_envelope.within_envelope=False "
                        "and NO certification proceeds (fail-closed).",
}


def _canonical(obj: Any) -> Any:
    """Deterministic canonicalization: dataclasses -> dict, tuples -> lists, sorted keys downstream."""
    if hasattr(obj, "__dataclass_fields__"):
        return _canonical(asdict(obj))
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (tuple, list)):
        return [_canonical(v) for v in obj]
    return obj


def _hash_of(payload: Any) -> str:
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def oracle_mechanism_hash() -> str:
    """Deterministic hash of the FROZEN generative mechanism (families + grids + nuisance + null
    construction + spec version). This is the value the trusted committed policy pins as
    `oracle_mechanism_hash`; regenerating with any changed family/grid/null yields a different hash,
    so a post-hoc mechanism edit cannot silently reuse an approval."""
    payload = {
        "spec_version": SPEC_VERSION,
        "families": [asdict(f) for f in STRUCTURAL_FAMILIES],
        "kappa_train_grid": KAPPA_TRAIN_GRID,
        "kappa_offgrid": KAPPA_OFFGRID,
        "null_mixture_weight": NULL_MIXTURE_WEIGHT,
        "nuisance_leak_rho": NUISANCE_LEAK_RHO,
        "nuisance_mi_bits_cap": NUISANCE_MI_BITS_CAP,
    }
    return _hash_of(payload)


def calibration_hash() -> str:
    """Hash of the calibration spec — SEPARATE from the mechanism hash (Pi #6: calibration output has
    its own hash and cannot alter the mechanism)."""
    return _hash_of(CALIBRATION_SPEC)


def spec_summary() -> dict[str, Any]:
    """Safe-public summary for the implementation-gate artifact (no data, no seeds materialized)."""
    return {
        "spec_version": SPEC_VERSION,
        "n_train_families": len(_TRAIN_FAMILIES),
        "n_heldout_families": len(_HELDOUT_FAMILIES),
        "n_no_h_families": len(no_h_families()),
        "family_ids": [f.family_id for f in STRUCTURAL_FAMILIES],
        "kappa_train_grid": list(KAPPA_TRAIN_GRID),
        "kappa_offgrid": list(KAPPA_OFFGRID),
        "null_mixture_weight": NULL_MIXTURE_WEIGHT,
        "oracle_mechanism_hash": oracle_mechanism_hash(),
        "calibration_hash": calibration_hash(),
    }
