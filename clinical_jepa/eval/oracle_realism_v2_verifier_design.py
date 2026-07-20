"""Oracle realism v2 — executable-verifier DESIGN FREEZE, rev-2 (M3a rebuild step 2; Pi REVISE folded).

The frozen DESIGN rebuild step 3 will implement, codifying Pi's rulings. DEV identity `m3a_design_dev_hash`,
re-routed to Pi for confirmation BEFORE coding; the FINAL frozen verifier identity is minted only after the
implemented verifier passes Pi's M3a final review (rebuild step 5).

Rev-2 folds the design-gate REVISE (thread thr-20260720T143304Z):
  * TWO separate routes: the six REGISTERED marginals keep their EXACT v1 estimands (AggregateStats route,
    development-seen match = EXPLORATORY only); S1–S9 are synthetic sequence-level recovery (SequenceSample
    route). Passing both is NOT a real joint-envelope match.
  * canonical maximal-run fixture law (cluster-size law, K, and Δt=0 are LINKED — not independently sampled);
  * S9 block-seam invisibility guard (terminal adequacy, no D route);
  * reference-only coarsening; derive-not-trust input schema; subcheck granularity;
  * S8 declared TERMINAL/out-of-model (no D route); S2 removed from the D map;
  * operational marginal-preserving coupling laws + an ablation expected-outcome matrix.

Synthetic-only. No sampling/fitting/governed access. The fixture performs NO governed read and uses only
previously cleared development-seen aggregate CONSTANTS (length scale). Admissible claim: "matches the declared
marginal + cross-statistic envelope," never the joint process.
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

M3A_DESIGN_VERSION = "m3a_verifier_design_dev_rev2"   # -> "..._impl_frozen_v1" after Pi final review
_KS, _TV, _OCC, _DT0 = ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS
DENOM_FLOOR = ORACLE_ENV_MIN_DENOM   # 500

# ============================ 0. TWO SEPARATE ROUTES + admissible claims ============================
ROUTES = {
    "marginal_route": {
        "io": "AggregateStats <-> AggregateStats",
        "checks": ["length_ks", "class_tv", "count_ks", "occupancy_abs", "delta_t_zero_abs", "positive_gap_ks"],
        "semantics": "EXACT registered v1 estimands (see REGISTERED_MARGINALS); the per-sequence SAMPLING_UNIT "
                     "rule does NOT apply here",
        "claim": "development-seen aggregate-marginal match — EXPLORATORY ONLY",
    },
    "sequence_route": {
        "io": "SequenceSample <-> SequenceSample",
        "checks": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
        "semantics": "synthetic per-sequence recovery vs an INDEPENDENT reference constructor",
        "claim": "synthetic known-profile cross-statistic recovery",
    },
    "separation_rule": "a candidate that passes synthetic S-recovery AND separately matches development-seen "
                       "TRAIN marginals is NOT described as matching a real joint envelope; the two claims are "
                       "reported separately and neither is a joint-process claim",
}

# ============================ 1. registered six marginals — EXACT v1 estimands (Pi defect 1) ============================
REGISTERED_MARGINALS = {
    "length_ks":        {"estimand": "ECDF of per-sequence full content length", "agg": "KS", "threshold": _KS},
    "class_tv":         {"estimand": "POOLED C=5 event-count proportions (class_counts / n_events)", "agg": "TV",
                         "threshold": _TV},
    "count_ks":         {"estimand": "ECDF of per-sequence cluster count K", "agg": "KS", "threshold": _KS},
    "occupancy_abs":    {"estimand": "equal-sequence mean(distinct classes / C)", "agg": "abs-diff",
                         "threshold": _OCC},
    "delta_t_zero_abs": {"estimand": "POOLED zero adjacencies / POOLED adjacencies", "agg": "abs-diff",
                         "threshold": _DT0},
    "positive_gap_ks":  {"estimand": "POOLED positive inter-cluster gap ECDF (8dp, right-continuous)",
                         "agg": "KS", "threshold": _KS},
}

# ============================ 2. derive-not-trust input schema (Pi §1) ============================
INPUT_SCHEMA = {
    "trusted_fields": ["source", "class_ids", "timestamps"],
    "derived": ["L_total", "cluster_ids (Δt=0 runs on 8dp timestamps)", "K", "residual_R", "block_count_B",
                "normalized position (index/(L_total-1); L_total==1 => 0.0)"],
    "reconciliation": "if redundant L_total/B/R/cluster_ids are supplied they must EXACTLY reconcile with the "
                      "derived values, else raise",
    "validation": ["finite timestamps (reject NaN/Inf)", "nondecreasing timestamps", "8dp tie convention "
                   "(equal rounded ts => same Δt=0 cluster)", "canonical contiguous run IDs if IDs remain",
                   "exact source in REQUIRED_SOURCES", "all vector lengths equal", "reject booleans-as-integers"],
    "malformed_refusal": "any violation raises; never silently coerce",
}

# ============================ 3. S1–S9 exact algorithms + subchecks + floors (Pi §1) ============================
# Per-sequence reduction first, equal-weight ELIGIBLE sequences; candidate-reference; conditional bins use the
# REFERENCE-derived coarsening map. Undefined per-sequence summaries (e.g. undefined tau) => that sequence is
# ineligible; if the eligible floor fails => NOT_EVALUABLE (never zero-filled).
S_ALGORITHMS = {
    "S1": {"def": "per-seq K/L_total by length-bin; max-bin abs-diff of equal-seq means + tau-b(L,K) diff",
           "subchecks": {"S1_density": {"abs": _OCC}, "S1_tau": {"tau_b": 0.05}},
           "floors": ["eligible_sequences>=500 per retained length-bin"],
           "notes": "tau-b with exact tie handling; undefined tau => NOT_EVALUABLE"},
    "S2": {"def": "sequence-equal-weighted ECDF of ALL maximal run sizes (incl. singletons); KS",
           "subchecks": {"S2_ks": {"ks": _KS}},
           "floors": ["eligible_sequences>=500", "clusters>=500"],
           "notes": "ECDF over unbounded integer support — NOT binned into cluster-size overflow bins"},
    "S3": {"def": "gap by preceding-cluster-size bin; per-seq bin summaries then equal-weight eligible seqs",
           "subchecks": {"S3_tau": {"tau_b": 0.05}, "S3_loggap": {"abs_E_log_gap": log(1.10)}},
           "floors": ["eligible_sequences>=500 per retained cluster-size bin", "adjacent_cluster_pairs>=500"],
           "notes": "abs(E[log gap]_cand - E[log gap]_ref) (NOT log of arithmetic-mean ratio); per-seq tau-b "
                    "then averaged; undefined => NOT_EVALUABLE"},
    "S4": {"def": "P(same|same cluster) - P(same|adjacent clusters) via class-count combinatorics (unordered "
                  "same-cluster pairs + Cartesian adjacent-cluster pairs, avoid O(L^2)); equal-weight seqs",
           "subchecks": {"S4_abs": {"abs": _OCC}},
           "floors": ["eligible_sequences>=500", "same_cluster_pairs>=500", "adjacent_cluster_pairs>=500"]},
    "S5": {"def": "per-seq occupancy by length-bin, candidate vs reference; max-bin abs-diff of equal-seq means",
           "subchecks": {"S5_abs": {"abs": _OCC}},
           "floors": ["eligible_sequences>=500 per retained length-bin"],
           "notes": "compared to the REFERENCE, not the M0b min(L,5)/5 cap (that is a separate feasibility flag)"},
    "S6": {"def": "length-dependent 5-class mix per length-bin; max-bin TV(cand-ref) of equal-seq means",
           "subchecks": {"S6_tv": {"tv": _TV}},
           "floors": ["eligible_sequences>=500 per retained length-bin"], "mandatory": True},
    "S7": {"def": "E[n_distinct/min(C,5) | cluster-size bin]: within-seq average clusters in bin, then average "
                  "eligible seqs; max-bin abs-diff",
           "subchecks": {"S7_abs": {"abs": _OCC}},
           "floors": ["eligible_sequences>=500 per retained cluster-size bin", "clusters>=500"],
           "notes": "equal-SEQUENCE weighting (not raw cluster-count weighting)"},
    "S8": {"def": "by canonical half-open position quartile: cluster density = cluster-starts/items; and 5-class "
                  "mix; per-seq then equal-weight eligible seqs",
           "subchecks": {"S8_density": {"abs": _OCC}, "S8_class": {"tv": _TV}},
           "floors": ["items>=500 per quartile", "eligible_sequences>=500 per quartile"],
           "escalation": "TERMINAL / out-of-model — NOT a D route (Pi §5)",
           "notes": "position quartiles are NEVER coarsened; missing phase support => NOT_EVALUABLE"},
    "S9": {"def": "block-seam invisibility: candidate-reference seam-vs-nonseam contrasts at 8-item seams",
           "subchecks": {"S9_zero": {"abs": _OCC}, "S9_class": {"abs": _OCC}, "S9_gap": {"ks": _KS}},
           "floors": ["seam_adjacencies>=500", "nonseam_adjacencies>=500"],
           "escalation": "TERMINAL implementation/adequacy guard — NOT a D route; failure blocks until the "
                         "composition implementation is corrected",
           "notes": "the maximal-run/timing process is generated over the WHOLE sequence independently of block "
                    "boundaries (V2_BLOCK_COMPOSITION amended); seams must be statistically invisible"},
}

CONDITIONAL_COARSENING = {
    "rule": "derive the adjacent-bin merge map from the REFERENCE ONLY, then apply it UNCHANGED to the candidate",
    "candidate_floor_fail_under_ref_map": "NOT_EVALUABLE (candidate-driven merging could hide candidate tail "
                                          "collapse)",
    "min_retained": {"length_bins": 3, "cluster_size_bins": 3},
    "merge_direction": "rightmost sparse bin merges LEFT into its neighbour first; repeat; emit the selected "
                       "map in results",
    "position_quartiles": "NEVER coarsened (S8 requires all four)",
}
SAMPLING_UNIT = "S-statistics only: per-sequence statistic first, equal-weight eligible sequences (the "
SAMPLING_UNIT += "REGISTERED_MARGINALS keep their exact pooled/seq estimands and are NOT rewritten by this rule)"
BINS = dict(V2_FROZEN_BINS)

# ============================ 4. canonical fixture law (Pi defect 2) ============================
# One canonical construction; cluster-size law, K, and Δt=0 are DERIVED from maximal runs (NOT independently
# sampled). The whole-sequence timing process is generated INDEPENDENTLY of block boundaries (Pi defect 3).
FIXTURE_LAW = {
    "steps": [
        "1. sample L_total from the length law",
        "2. sample maximal cluster (run) sizes from the cluster-size law until they sum to EXACTLY L_total, "
        "using the frozen terminal-truncation rule (last run truncated to hit L_total; if truncation would be "
        "0, drop it)",
        "3. DERIVE K = number of runs, cluster_ids, and dt0 = (L_total - K)/(L_total - 1) from the runs",
        "4. sample ONE strictly-positive inter-cluster gap per boundary from the gap law",
        "5. derive timestamps (cumulative; Δt=0 within a run) over the WHOLE sequence, independent of 8-item "
        "block boundaries",
        "6. sample class_ids from the class law (with structural zeros); apply the active D coupling(s)",
    ],
    "dt0_rate": "DERIVED diagnostic only (= (L_total-K)/(L_total-1)); NOT an independent Bernoulli parameter",
    "singletons_in_S2": True,
    "block_seam": "timing/marks generated whole-sequence; the 8-item block grid is a certification overlay only "
                  "and must be statistically invisible (guarded by S9)",
}

# ============================ 5. numeric profiles + operational couplings (Pi §2) ============================
# Only the LENGTH SCALE is anchored to cleared development-seen constants; class/timing/burst values are
# SYNTHETIC and are NOT source estimates. Renamed accordingly.
def _prof(length_mu, length_sigma, class_prior, struct_zero, cluster_size_geom_p, gap_lognorm, dependence):
    return {"length": {"family": "discretized_lognormal", "mu": length_mu, "sigma": length_sigma, "min": 1},
            "class_prior": class_prior, "structural_zero_classes": struct_zero,
            "cluster_size": {"family": "geometric", "p": cluster_size_geom_p},
            "gap": {"family": "lognormal", "mu": gap_lognorm[0], "sigma": gap_lognorm[1]},
            "dependence": dependence}

_NO_DEP = {k: 0.0 for k in V2_D_COMPONENT_MENU}
PROFILES = {
    "scid_scale_control":  _prof(log(350), 0.90, [0.55, 0.20, 0.15, 0.07, 0.03], [], 0.55, (log(1.0), 0.8), _NO_DEP),
    "mimic_scale_control": _prof(log(99), 1.00, [0.10, 0.15, 0.20, 0.25, 0.30], [], 0.45, (log(1.5), 0.9), _NO_DEP),
    "structural_zero_control": _prof(log(150), 0.95, [0.40, 0.35, 0.25, 0.0, 0.0], [3, 4], 0.5, (log(1.2), 0.85),
                                     _NO_DEP),   # explicit hard structural zeros on classes 3,4
    "interior_low":  _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85),
                           {k: 0.10 for k in V2_D_COMPONENT_MENU}),
    "interior_mid":  _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85),
                           {k: 0.35 for k in V2_D_COMPONENT_MENU}),
    "interior_high": _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85),
                           {k: 0.55 for k in V2_D_COMPONENT_MENU}),   # 0.55 (not 0.60) to keep central FD in range
    "null_independent": _prof(log(150), 0.95, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), _NO_DEP),
    "boundary_short":   _prof(log(9), 0.30, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), _NO_DEP),
}

# Operational, MARGINAL-PRESERVING coupling laws (strength s in [0,0.6]); every transform is applied AFTER base
# sampling with a dedicated coupling RNG, preserves the six registered marginals BY CONSTRUCTION, breaks ties by
# item index, and clips s to the range.
COUPLING_LAWS = {
    "burst_count_length": "pool-neutral boundary reallocation by length-bin: for a fraction s of high-L-bin "
                          "sequences MERGE a randomly chosen adjacent run pair; for a MATCHED fraction of "
                          "low-L-bin sequences SPLIT a run — chosen so the POOLED cluster-size distribution and "
                          "K totals are unchanged (moves S1 only)",
    "burst_timing": "rank copula on positive gaps within each sequence: reorder the sequence's gaps so a "
                    "fraction s follow the rank of the preceding cluster size; the MULTISET of gaps (hence the "
                    "pooled positive-gap ECDF) is unchanged (moves S3 only)",
    "mark_burst_tie": "within-sequence class relabelling that raises same-cluster same-class probability by a "
                      "fraction s while HOLDING each sequence's per-class counts fixed (pooled class_tv "
                      "preserved) (moves S4 only)",
    "cluster_size_mark_diversity": "within-sequence, class-count-preserving relabelling that concentrates "
                                   "(large clusters) / diversifies (small clusters) class diversity by a "
                                   "fraction s; pooled class proportions and cluster-size marginal unchanged "
                                   "(moves S7 only)",
    "length_class_mix": "length-bin-dependent, POOL-BALANCED class relabelling: shift conditional class mix by "
                        "length bin by a fraction s while keeping the POOLED class proportions fixed (moves "
                        "S5/S6 only; explicitly does NOT move pooled class_tv)",
}

# source_swap is now operational; each ablation is an explicit expected-outcome row.
SOURCE_SWAP = "emit scid_scale_control CLASS/timing/burst marginals under mimic_scale_control LENGTH scale; " \
              "must FAIL a NON-degenerate check (e.g. count_ks/positive_gap_ks/S1), not only class_tv"
# expected-outcome matrix: each ablation sets exactly ONE component to 0.5 on null_independent and must fail its
# mapped subcheck(s) while passing all non-attributed checks at specificity.
ABLATION_MATRIX = {
    "burst_count_length":          {"fails": ["S1_density"], "passes": "all others incl. six marginals"},
    "burst_timing":                {"fails": ["S3_tau", "S3_loggap"], "passes": "all others"},
    "mark_burst_tie":              {"fails": ["S4_abs"], "passes": "all others"},
    "cluster_size_mark_diversity": {"fails": ["S7_abs"], "passes": "all others"},
    "length_class_mix":            {"fails": ["S5_abs", "S6_tv"], "passes": "all others incl. pooled class_tv"},
}

# ============================ 6. simulation (provisional; step-4 must demonstrate) ============================
SIMULATION = {
    "n_seeds": 25,
    "seed_list": list(range(1000, 1025)),
    "per_source_sample_size_candidate": 4000,      # PROVISIONAL — step 4 must demonstrate every floor is met
    "step4_must_demonstrate": ["reference-derived coarsening map", ">=3 retained length + >=3 cluster bins",
                               "all four unmerged position quartiles", "every check-specific "
                               "sequence/cluster/pair/seam floor", "the rate criteria",
                               "a runtime/memory estimate BEFORE the full grid (event volume, not sequence "
                               "count, dominates cost)"],
    "if_underpowered": "increase N ONLY; never change thresholds/definitions after seeing behaviour",
    "power": {"self_known_pass_min": 24, "misspecified_fail_min": 20, "of_seeds": 25,
              "non_attributed_specificity_min": 24,
              "report": "empirical pass/fail RATES + binomial CIs; median secondary only"},
}

# ============================ 7. independent fixture generator identity (Pi) ============================
FIXTURE_GENERATOR = {
    "name": "realism_v2_reference_constructor_dev",
    "law": FIXTURE_LAW,
    "independence_rule": "shares NO code path with the future M2 candidate adapter; frozen BEFORE M2",
    "data_rule": "performs NO governed read; uses ONLY previously cleared development-seen aggregate CONSTANTS "
                 "(length scale). No TRAIN comparisons.",
    "not_a_candidate": "used solely to exercise the verifier; never scored/certified as realism",
}

# ============================ 8. identifiability (Pi §4) ============================
IDENTIFIABILITY = {
    "param_ranges": {c: [0.0, 0.6] for c in V2_D_COMPONENT_MENU},
    "standardization": "AFFINE range standardization (no logit — undefined at endpoints); epsilon clip 1e-6",
    "statistic_vector": "the frozen S1–S7 subcheck scalars (S8/S9 are terminal guards, excluded from the "
                        "identifiability vector)",
    "whitening_reference": "covariance estimated at null_independent under CRN; ridge-regularized (lambda 1e-3)",
    "grid": "3^k joint grid over active components at 0.10/0.35/0.55, with the marginal nuisance grid varied",
    "finite_difference": "central CRN at interior points; FORWARD one-sided active-subset at 0.0; BACKWARD "
                         "one-sided at the top profile (0.55) so no evaluation leaves [0,0.6]; step 0.02",
    "rank_criterion": "standardized Jacobian sigma_min/sigma_max >= 1e-3 (abs tol secondary)",
    "recovery": "inverse via nearest-grid + local least squares on the whitened vector; recover_tol <= 0.05 of "
                "range AND <= half a grid step; grid tie-break by lowest L2 then lowest component index",
    "collision": "two settings beyond recover_tol COLLIDE iff ALL whitened cross-stat diffs stay within "
                 "acceptance tol",
    "compute_budget": "staged runtime gate: run k<=2 active-component grids first, estimate cost, and only "
                      "expand to higher k within the frozen budget (never silently reduce after results)",
}

# ============================ 9. escalation (Pi §5) ============================
# CHECK/subcheck -> D components. S2 (cluster-size marginal) and the six marginals + source-swap NEVER trigger
# D. S8 and S9 are TERMINAL adequacy guards with NO D route.
CHECK_TO_D_COMPONENTS = {
    "S1_density": {"components": ["burst_count_length"], "semantics": "single"},
    "S1_tau":     {"components": ["burst_count_length"], "semantics": "single"},
    "S3_tau":     {"components": ["burst_timing"], "semantics": "single"},
    "S3_loggap":  {"components": ["burst_timing"], "semantics": "single"},
    "S4_abs":     {"components": ["mark_burst_tie"], "semantics": "single"},
    "S5_abs":     {"components": ["length_class_mix"], "semantics": "single"},
    "S6_tv":      {"components": ["length_class_mix"], "semantics": "single"},
    "S7_abs":     {"components": ["cluster_size_mark_diversity"], "semantics": "single"},
}
TERMINAL_CHECKS = ["S2_ks", "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap",
                   "length_ks", "class_tv", "count_ks", "occupancy_abs", "delta_t_zero_abs", "positive_gap_ks"]
ESCALATION = {
    "baseline": "A_independent (bound first at M2)",
    "map": CHECK_TO_D_COMPONENTS,
    "terminal_no_D": TERMINAL_CHECKS,
    "terminal_note": "S2/marginal failures are Option-A / implementation failures (not a D reason); S8/S9 "
                     "failures are terminal adequacy failures; source-swap failures never trigger D",
    "selection": "smallest super-set of components covering the failed D-eligible SUBCHECKS; ties broken by the "
                 "frozen component order in V2_D_COMPONENT_MENU",
    "monotone": "active set only EXPANDS; never remove; no repeated component/identity",
    "identity": "v2_active_d_identity over the exact active set",
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
    "routes": ROUTES,
    "registered_marginals": REGISTERED_MARGINALS,
    "input_schema": INPUT_SCHEMA,
    "s_algorithms": S_ALGORITHMS,
    "conditional_coarsening": CONDITIONAL_COARSENING,
    "sampling_unit": SAMPLING_UNIT,
    "bins": BINS,
    "denom_floor": DENOM_FLOOR,
    "block_composition": V2_BLOCK_COMPOSITION,
    "fixture_law": FIXTURE_LAW,
    "profiles": PROFILES,
    "coupling_laws": COUPLING_LAWS,
    "source_swap": SOURCE_SWAP,
    "ablation_matrix": ABLATION_MATRIX,
    "simulation": SIMULATION,
    "fixture_generator": FIXTURE_GENERATOR,
    "identifiability": IDENTIFIABILITY,
    "escalation": ESCALATION,
    "m0b_support_policy_hash": m0b_support_policy_hash(),
    "admissible_claim": "matches the declared marginal + cross-statistic envelope; NEVER the joint process",
}


def m3a_design_dev_hash() -> str:
    """DEVELOPMENT identity of the rev-2 executable-verifier design. FINAL identity minted only after Pi's M3a
    final review of the IMPLEMENTED verifier (rebuild step 5)."""
    return canonical_hash(M3A_VERIFIER_DESIGN)
