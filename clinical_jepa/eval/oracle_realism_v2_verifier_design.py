"""Oracle realism v2 — executable-verifier DESIGN FREEZE (M3a rebuild step 2; Pi CONFIRM-WITH-REORDER).

The frozen DESIGN that rebuild step 3 will implement: the typed full-sequence input schema, the exact S1–S8
algorithms + candidate tolerances + bins/coarsening/floors/weighting/ties, the exact numeric synthetic control
profiles, the deterministic seed list + per-source sample sizes, the INDEPENDENT control-fixture generator
identity, and the identifiability + power + escalation designs. This is a DEV design (`m3a_design_dev_hash`),
routed to Pi for confirmation BEFORE coding; the FINAL frozen verifier identity is minted only after the
implemented verifier passes Pi's M3a final review (rebuild step 5).

Synthetic-only. No sampling law / fitting / target comparison / governed access. The profiles are closed-form
SYNTHETIC controls (shape-anchored to the cleared aggregate length quantiles), NEVER fitted from M2 output.
The verifier scores CANDIDATE − REFERENCE; the admissible claim is "matches the declared marginal +
cross-statistic envelope," never the joint process.
"""
from __future__ import annotations

from math import log

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_calibration import REQUIRED_SOURCES
from clinical_jepa.eval.oracle_realism_v2 import (
    V2_FROZEN_BINS, V2_BLOCK_COMPOSITION, V2_D_COMPONENT_MENU, m0b_support_policy_hash,
)
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS,
    ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES,
)

M3A_DESIGN_VERSION = "m3a_verifier_design_dev"     # -> "m3a_verifier_impl_frozen_v1" after Pi final review

# ============================ 1. typed full-sequence input schema ============================
# A candidate/reference SAMPLE is N independent full sequences. Each sequence is the emission-side object; only
# complete 8-item blocks are certifiable (handled elsewhere by the guard) — the verifier operates on emission.
INPUT_SCHEMA = {
    "sample": "list[SequenceRecord] of length N (independent sequences, equal-weighted)",
    "SequenceRecord": {
        "source": "one of REQUIRED_SOURCES",
        "L_total": "int >= 1 (full content-token length = 8*B + R)",
        "block_count_B": "int >= 0",
        "residual_R": "int in [0,7]",
        "class_ids": "int array (L_total,), values in [0, C=5)",
        "timestamps": "float array (L_total,), nondecreasing",
        "cluster_ids": "int array (L_total,), Δt=0 runs; clusters MAY span block boundaries",
        "position": "normalized position in [0,1] per item (index/(L_total-1); single-item => 0.0)",
    },
    "malformed_refusal": "any schema violation (nonmonotone ts, R not in [0,7], L_total != 8B+R, C out of "
                         "range, empty) => raise; never silently coerce",
}

# ============================ 2. S1–S8 exact algorithms (candidate − reference) ============================
# Each statistic: per-sequence reduction first (equal-weight sequences), binned on the frozen bins, scored as
# candidate−reference with the stated aggregation, against the candidate tolerance. `floor_unit` names what the
# denominator counts; below the floor after permitted coarsening => NOT_EVALUABLE (never zero-filled).
_KS, _TV, _OCC = ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS
S_ALGORITHMS = {
    # six marginals (candidate vs reference)
    "length_ks":        {"reduce": "per-seq L_total", "bin": None, "agg": "KS", "floor_unit": "sequences",
                         "threshold": _KS, "op": "<="},
    "class_tv":         {"reduce": "pooled 5-class frequency (per-seq normalized then averaged)", "bin": None,
                         "agg": "TV", "floor_unit": "sequences", "threshold": _TV, "op": "<="},
    "count_ks":         {"reduce": "per-seq CLUSTER count K", "bin": None, "agg": "KS",
                         "floor_unit": "sequences", "threshold": _KS, "op": "<="},
    "occupancy_abs":    {"reduce": "per-seq occupancy = distinct classes / C", "bin": None, "agg": "abs-mean-diff",
                         "floor_unit": "sequences", "threshold": _OCC, "op": "<="},
    "delta_t_zero_abs": {"reduce": "per-seq fraction of adjacencies with Δt=0", "bin": None, "agg": "abs-mean-diff",
                         "floor_unit": "adjacencies", "threshold": ORACLE_ENV_DT0_ABS, "op": "<="},
    "positive_gap_ks":  {"reduce": "pooled positive inter-cluster gaps", "bin": None, "agg": "KS",
                         "floor_unit": "positive_gaps", "threshold": _KS, "op": "<="},
    # cross-statistics S1–S8
    "S1": {"reduce": "per-seq K/L_total", "bin": "length", "agg": "max-bin abs-mean-diff + tau-b(L,K) diff",
           "floor_unit": "sequences", "threshold": {"density_maxbin": _OCC, "tau_b": 0.05}, "op": "<="},
    "S2": {"reduce": "Δt=0 cluster-run sizes (pooled, overflow-supported)", "bin": "cluster_size", "agg": "KS",
           "floor_unit": "clusters", "threshold": {"ks": _KS}, "op": "<="},
    "S3": {"reduce": "gap by preceding-cluster-size bin", "bin": "cluster_size",
           "agg": "tau-b(prev size,gap) diff + max-bin |log mean-ratio|", "floor_unit": "adjacent_cluster_pairs",
           "threshold": {"tau_b": 0.05, "log_mean_ratio": log(1.10)}, "op": "<="},
    "S4": {"reduce": "P(same class|same cluster) - P(same class|adjacent clusters)", "bin": None,
           "agg": "abs-diff", "floor_unit": "same_and_adjacent_cluster_pairs", "threshold": {"abs": _OCC},
           "op": "<=", "weighting": "each ELIGIBLE pair weighted equally within a sequence, then avg sequences"},
    "S5": {"reduce": "per-seq occupancy by length-bin (vs reference, NOT the M0b cap)", "bin": "length",
           "agg": "max-bin abs-mean-diff", "floor_unit": "sequences", "threshold": {"abs": _OCC}, "op": "<="},
    "S6": {"reduce": "length-dependent 5-class mix per length-bin", "bin": "length",
           "agg": "max-bin TV(candidate-reference)", "floor_unit": "sequences", "threshold": {"tv": _TV},
           "op": "<=", "mandatory": True},
    "S7": {"reduce": "E[n_distinct/min(C,5)] by cluster-size bin", "bin": "cluster_size",
           "agg": "max-bin abs-diff", "floor_unit": "clusters", "threshold": {"abs": _OCC}, "op": "<=",
           "weighting": "large clusters weighted by cluster COUNT, not item count"},
    "S8": {"reduce": "cluster density + 5-class mix by normalized-position quartile", "bin": "position_quartile",
           "agg": "max-bin abs-diff (density) + max-bin TV (class)", "floor_unit": "items_per_quartile",
           "threshold": {"density_abs": _OCC, "class_tv": _TV}, "op": "<="},
}

CROSS_AGGREGATION = "each check's multi-bin vector is reduced by MAX-NORM (worst bin); the whole-spec decision " \
                    "is a CONJUNCTION over checks AND over sources"
SAMPLING_UNIT = "per-sequence statistic computed first, then sequences equal-weighted; long sequences must NOT " \
                "dominate via raw pair/cluster pooling"
BINS = dict(V2_FROZEN_BINS)
COARSENING = "deterministic adjacent-bin merge order (rightmost sparse bin merges LEFT into its neighbour " \
             "first, repeat); the SAME post-coarsening map is applied to reference and candidate; if a floor " \
             "still fails after all permitted merges => NOT_EVALUABLE"
DENOM_FLOOR = ORACLE_ENV_MIN_DENOM   # 500

# ============================ 3. numeric synthetic control profiles (closed-form; frozen) ============================
# Shape-anchored to the cleared aggregate length quantiles (SCID median 350 / MIMIC median 99). Length is a
# discretized lognormal on L_total>=1. Class prior is a 5-vector with an optional hard structural zero. These
# are SYNTHETIC controls only; NEVER fitted from M2. `dependence` gives the D-component strengths (0 => A).
def _prof(length_mu, length_sigma, class_prior, struct_zero, cluster_size_geom_p, gap_lognorm,
          dt0_rate, dependence):
    return {"length": {"family": "discretized_lognormal", "mu": length_mu, "sigma": length_sigma, "min": 1},
            "class_prior": class_prior, "structural_zero_classes": struct_zero,
            "cluster_size": {"family": "geometric", "p": cluster_size_geom_p},
            "gap": {"family": "lognormal", "mu": gap_lognorm[0], "sigma": gap_lognorm[1]},
            "dt0_rate": dt0_rate, "dependence": dependence}

_NO_DEP = {k: 0.0 for k in V2_D_COMPONENT_MENU}
PROFILES = {
    # known/self references (shape-anchored)
    "scid_like":  _prof(log(350), 0.90, [0.55, 0.20, 0.15, 0.07, 0.03], [], 0.55, (log(1.0), 0.8), 0.35, _NO_DEP),
    "mimic_like": _prof(log(99), 1.00, [0.10, 0.15, 0.20, 0.25, 0.30], [], 0.45, (log(1.5), 0.9), 0.35, _NO_DEP),
    # interior dependence profiles (for identifiability rank at low/mid/high coupling)
    "interior_low":  _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), 0.35,
                           {**_NO_DEP, "burst_count_length": 0.10, "burst_timing": 0.10, "mark_burst_tie": 0.10,
                            "cluster_size_mark_diversity": 0.10, "length_class_mix": 0.10}),
    "interior_mid":  _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), 0.35,
                           {k: 0.35 for k in V2_D_COMPONENT_MENU}),
    "interior_high": _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), 0.35,
                           {k: 0.60 for k in V2_D_COMPONENT_MENU}),
    # negative controls
    "null_independent": _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), 0.35,
                              _NO_DEP),
    "boundary_short":   _prof(log(9), 0.30, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), 0.35,
                              _NO_DEP),   # near the L<5/occupancy-cap + support-floor edges
    "source_swap":      "scid_like MARGINALS emitted under mimic_like context (must fail on a NON-degenerate "
                        "check, not only class-TV; SCID zero-state makes class-TV swap trivial)",
    # minimally-misspecified controls: each perturbs ONE D component off its attribution-mapped check target
    "ablation": {c: f"scid_like + only {c} set to 0.5 (must FAIL exactly its attribution-mapped check, pass "
                    f"non-attributed checks at specificity)" for c in V2_D_COMPONENT_MENU},
}

# ============================ 4. deterministic seeds + sample sizes ============================
SIMULATION = {
    "n_seeds": 25,
    "seed_list": list(range(1000, 1025)),            # deterministic, frozen
    "per_source_sample_size": 4000,                   # sequences per source per seed (>= all floors after bins)
    "power": {"self_known_pass_min": 24, "misspecified_fail_min": 20, "of_seeds": 25,
              "report": "empirical pass/fail RATES + binomial CIs; median is secondary only",
              "specificity": "each ablation passes NON-attributed checks at >= the predeclared specificity rate"},
}

# ============================ 5. INDEPENDENT control-fixture generator identity (Pi) ============================
# The reference/control constructor is CLOSED-FORM and INDEPENDENT of the future M2 candidate implementation —
# otherwise evaluator and candidate could share a bug and self-certify. It exists ONLY to exercise the verifier;
# it is NOT an alternative realism candidate and uses NO TRAIN comparisons.
FIXTURE_GENERATOR = {
    "name": "realism_v2_reference_constructor_dev",
    "construction": "sample each SequenceRecord field directly from the closed-form PROFILES parameters "
                    "(length lognormal, class multinomial with structural zeros, geometric cluster sizes, "
                    "lognormal gaps, Bernoulli Δt=0), composing whole-sequence timestamps/cluster ids per "
                    "V2_BLOCK_COMPOSITION; dependence injected by the declared copula couplings",
    "independence_rule": "shares NO code path with the M2 candidate adapter; frozen BEFORE M2 is implemented",
    "not_a_candidate": "used solely to exercise the verifier; never scored/certified as realism; no TRAIN data",
}

# ============================ 6. identifiability design ============================
IDENTIFIABILITY = {
    "param_ranges": {c: [0.0, 0.6] for c in V2_D_COMPONENT_MENU},
    "transform": "logit-scaled to the range; standardized before the Jacobian",
    "grid": "3^k joint grid over active components (low/mid/high at 0.1/0.35/0.6) with marginal nuisance varied",
    "estimator": "the frozen S-algorithms on FIXTURE_GENERATOR samples at per_source_sample_size",
    "finite_difference": "central, step = 0.02 in raw param units, CRN seeds shared across +/- evaluations",
    "rank_criterion": "standardized/whitened Jacobian sigma_min/sigma_max >= 1e-3 (abs tol secondary guard)",
    "recovery_tol": "<= 0.05 of each param's range AND <= half a grid step",
    "collision": "two settings beyond recovery tol COLLIDE iff ALL standardized cross-stat diffs stay within "
                 "their acceptance tolerances",
    "null_profile_rank": "at null_independent use the ACTIVE-subset / one-sided boundary rank (do NOT require "
                         "full rank for inactive params)",
    "profiles_for_rank": ["interior_low", "interior_mid", "interior_high"],
}

# ============================ 7. escalation design (Pi: separate map, active-set identity) ============================
# CHECK -> D components (NOT PARAM_TO_STATISTIC): which D components can repair each failed check. Includes
# length_class_mix for the S5/S6 length->composition coupling.
CHECK_TO_D_COMPONENTS = {
    "S1": ["burst_count_length"],
    "S2": ["cluster_size_mark_diversity"],       # burst-size law shape
    "S3": ["burst_timing"],
    "S4": ["mark_burst_tie"],
    "S5": ["length_class_mix"],
    "S6": ["length_class_mix"],
    "S7": ["cluster_size_mark_diversity"],
    "S8": ["burst_count_length", "length_class_mix"],   # position nonstationarity of density + class mix
}
ESCALATION = {
    "baseline": "A_independent (bound first at M2)",
    "map": CHECK_TO_D_COMPONENTS,
    "selection": "smallest super-set of components covering the failed checks; ties broken by the frozen "
                 "component order in V2_D_COMPONENT_MENU",
    "monotone": "active set only EXPANDS across rounds; never remove a component; no repeated component/identity",
    "identity": "v2_active_d_identity over the exact active set (never the generic all-components identity)",
    "iteration_cap": 3,
    "on_cap_exhausted": "terminal FAIL / park (never an informal redesign)",
    "ledger_fields": ["parent_identity", "m3a_spec_hash", "control_profile", "failed_statistics",
                      "active_component_set", "decision_hash", "result_hash", "seed_set"],
    "decision_basis": "known-ground-truth control battery ONLY; TRAIN-target diagnostics non-decisional",
}

M3A_VERIFIER_DESIGN = {
    "version": M3A_DESIGN_VERSION,
    "n_classes": ORACLE_ENV_N_CLASSES,
    "required_sources": list(REQUIRED_SOURCES),
    "input_schema": INPUT_SCHEMA,
    "s_algorithms": S_ALGORITHMS,
    "cross_aggregation": CROSS_AGGREGATION,
    "sampling_unit": SAMPLING_UNIT,
    "bins": BINS,
    "coarsening": COARSENING,
    "denom_floor": DENOM_FLOOR,
    "block_composition": V2_BLOCK_COMPOSITION,
    "profiles": PROFILES,
    "simulation": SIMULATION,
    "fixture_generator": FIXTURE_GENERATOR,
    "identifiability": IDENTIFIABILITY,
    "escalation": ESCALATION,
    "m0b_support_policy_hash": m0b_support_policy_hash(),
    "admissible_claim": "matches the declared marginal + cross-statistic envelope; NEVER the joint process",
}


def m3a_design_dev_hash() -> str:
    """DEVELOPMENT identity of the executable-verifier design. The FINAL frozen identity is minted only after
    the implemented verifier passes Pi's M3a final review (rebuild step 5)."""
    return canonical_hash(M3A_VERIFIER_DESIGN)
