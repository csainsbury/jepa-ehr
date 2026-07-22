"""Oracle realism v2 — executable-verifier DESIGN FREEZE, rev-3 (M3a rebuild step 2; Pi NARROW REVISE folded).

The frozen DESIGN rebuild step 3 will implement, codifying Pi's rulings. DEV identity `m3a_design_dev_hash`,
re-routed to Pi for confirmation BEFORE coding; the FINAL frozen verifier identity is minted only after the
implemented verifier passes Pi's M3a final review (rebuild step 5).

Rev-2 established: two separate routes (registered marginals vs synthetic S1–S9), canonical maximal-run fixture
law, whole-sequence timing + S9 seam guard, reference-only coarsening, terminal S8. Rev-3 folds the NARROW
REVISE (thread thr-20260720T143304Z):
  * EXACT registered timestamp/cluster semantics (dt==0 raw equality; only positive-gap ECDF support 8dp);
    L_total==1 dt0 undefined/excluded;
  * executable S9 conjunction + a 6-step reference-only coarsening algorithm; S1_tau = source-level; S2 =
    mean-of-per-sequence-ECDFs; S3 adjacent-pair floor per retained bin;
  * honest EXACT finite-pool coupling constructions (comonotone-cycle burst_count_length); marginal
    preservation EMPIRICALLY REQUIRED (tested >=24/25) not asserted by construction; recorded S4<->S7
    cross-loading; concrete SOURCE_SWAP pair; oriented ablation matrix;
  * identifiability: central FD at all interior incl 0.55, no logit/epsilon, deterministic nearest-grid
    recovery, explicit covariance ridge + named nuisance profiles + hard compute cap;
  * two DISTINCT admissible claims (never recombined) + explicit NO-joint/NO-confirmatory negatives.

Synthetic-only. No sampling/fitting/governed access. The fixture performs NO governed read and uses only
previously cleared development-seen aggregate CONSTANTS (length scale).
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

M3A_DESIGN_VERSION = "m3a_verifier_design_dev_rev3"   # -> "..._impl_frozen_v1" after Pi final review
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
    "derived": ["L_total", "cluster_ids = maximal runs under EXACT raw timestamp equality (dt==0)", "K",
                "residual_R", "block_count_B", "normalized position (index/(L_total-1); L_total==1 => 0.0)"],
    "cluster_semantics": "EXACT registered extractor semantics: dt=np.diff(t); zero=(dt==0); boundary=(dt>0). "
                         "Clusters are maximal runs under EXACT float equality — NOT 8dp-rounded timestamps. "
                         "Only the positive-gap ECDF SUPPORT is rounded to 8dp (right-continuous).",
    "reconciliation": "if redundant L_total/B/R/cluster_ids are supplied they must EXACTLY reconcile with the "
                      "derived values, else raise",
    "validation": ["finite timestamps (reject NaN/Inf)", "nondecreasing timestamps",
                   "every positive adjacency satisfies t[i+1] > t[i] AFTER materialization",
                   "a generated positive gap lost under cumulative float addition is REJECTED or "
                   "deterministically nudged BEFORE record issuance",
                   "canonical contiguous run IDs if IDs remain", "exact source in REQUIRED_SOURCES",
                   "all vector lengths equal", "reject booleans-as-integers"],
    "L1_rule": "for L_total==1 there are ZERO adjacencies: dt0 is UNDEFINED — exclude the sequence from dt0 "
               "estimation and from the adjacency denominator; never evaluate (L-K)/(L-1)",
    "malformed_refusal": "any violation raises; never silently coerce",
}

# ============================ 3. S1–S9 exact algorithms + subchecks + floors (Pi §1) ============================
# Per-sequence reduction first, equal-weight ELIGIBLE sequences; candidate-reference; conditional bins use the
# REFERENCE-derived coarsening map. Undefined per-sequence summaries (e.g. undefined tau) => that sequence is
# ineligible; if the eligible floor fails => NOT_EVALUABLE (never zero-filled).
S_ALGORITHMS = {
    "S1": {"def": "per-seq K/L_total density by length-bin (max-bin abs-diff of equal-seq means) + ONE "
                  "SOURCE-LEVEL tau-b(L,K) over independent sequences",
           "subchecks": {"S1_density": {"abs": _OCC}, "S1_tau": {"tau_b": 0.05}},
           "floors": ["eligible_sequences>=500 per retained length-bin"],
           "notes": "S1_tau is a single source-level tau-b (with ties) over independent sequences — NOT a "
                    "per-sequence tau (contrast S3); undefined tau => NOT_EVALUABLE"},
    "S2": {"def": "sequence-equal CDF F(x)=mean_i F_i(x) over eligible sequences (each F_i the sequence's own "
                  "run-size ECDF, incl. singletons), then KS(F_cand, F_ref)",
           "subchecks": {"S2_ks": {"ks": _KS}},
           "floors": ["eligible_sequences>=500", "clusters>=500"],
           "notes": "unbounded integer support — NOT binned into cluster-size overflow bins"},
    "S3": {"def": "gap by preceding-cluster-size bin; per-seq bin summaries then equal-weight eligible seqs",
           "subchecks": {"S3_tau": {"tau_b": 0.05}, "S3_loggap": {"abs_E_log_gap": log(1.10)}},
           "floors": ["eligible_sequences>=500 per retained cluster-size bin",
                      "adjacent_cluster_pairs>=500 PER retained cluster-size bin"],
           "notes": "abs(E[log gap]_cand - E[log gap]_ref) (NOT log of arithmetic-mean ratio); S3_tau is "
                    "per-sequence tau-b then averaged over eligible seqs; undefined => NOT_EVALUABLE"},
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
    "S8": {"def": "WITHIN-SEQUENCE CENTERED phase profile (Pi F3): per canonical half-open position quartile, "
                  "phase_class = quartile 5-class vector - whole-sequence 5-class vector; phase_density = "
                  "quartile cluster-starts/items - whole-sequence K/L; equal-weight eligible sequences; "
                  "S8_class = max_q 0.5*||mean phase_class diff||_1; S8_density = max_q |mean phase_density diff|",
           "subchecks": {"S8_density": {"abs": _OCC}, "S8_class": {"tv": _TV}},
           "floors": ["items>=500 per quartile", "eligible_sequences>=500 per quartile"],
           "escalation": "TERMINAL / out-of-model — NOT a D route (Pi §5)",
           "notes": "position quartiles are NEVER coarsened; missing phase support => NOT_EVALUABLE; centering "
                    "removes stationary global/length-conditioned class movement so S8 measures PHASE"},
    "S9": {"def": "block-seam invisibility at 8-item seams; per-eligible-sequence seam/nonseam probabilities "
                  "then equal-weight sequences",
           "subchecks": {
               "S9_zero": "|[(P0_seam - P0_non)_cand - (P0_seam - P0_non)_ref]| <= 0.03",
               "S9_class": "|[(Psame_seam - Psame_non)_cand - (Psame_seam - Psame_non)_ref]| <= 0.03",
               "S9_gap": "KS(F_cand_seam+, F_cand_nonseam+)<=0.05 AND KS(F_ref_seam+, F_ref_nonseam+)<=0.05 AND "
                         "max(KS(cand_seam+,ref_seam+), KS(cand_nonseam+,ref_nonseam+))<=0.05"},
           "floors": ["eligible_sequences>=500", "seam_adjacencies>=500", "nonseam_adjacencies>=500",
                      "positive_seam_gaps>=500", "positive_nonseam_gaps>=500"],
           "escalation": "TERMINAL implementation/adequacy guard — NOT a D route; failure blocks until the "
                         "composition implementation is corrected",
           "notes": "the maximal-run/timing process is generated over the WHOLE sequence independently of block "
                    "boundaries (V2_BLOCK_COMPOSITION amended); seams must be statistically invisible"},
}

CONDITIONAL_COARSENING = {
    "rule": "derive the adjacent-bin merge map from the REFERENCE ONLY, then apply it UNCHANGED to the candidate",
    "algorithm": [
        "1. determine sparse bins from the REFERENCE denominators FOR THAT CHECK ONLY",
        "2. let j = the highest-index sparse bin",
        "3. if j>0 merge bin j INTO bin j-1; if j==0 merge bin 0 INTO bin 1",
        "4. recompute reference floors and repeat from (1)",
        "5. REFUSE (NOT_EVALUABLE) if fewer than three bins remain",
        "6. apply the final map UNCHANGED to the candidate; any candidate floor failure under it => NOT_EVALUABLE",
    ],
    "candidate_floor_fail_under_ref_map": "NOT_EVALUABLE (candidate-driven merging could hide candidate tail "
                                          "collapse)",
    "min_retained": {"length_bins": 3, "cluster_size_bins": 3},
    "position_quartiles": "NEVER coarsened (S8 requires all four)",
    "emit": "the selected per-check map is emitted in results",
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
        "3. DERIVE K = number of runs and cluster_ids from the runs; dt0 = zero-adjacencies/adjacencies "
        "(UNDEFINED for L_total==1 — excluded, per INPUT_SCHEMA.L1_rule)",
        "4. sample ONE strictly-positive inter-cluster gap per boundary from the gap law",
        "5. derive timestamps by cumulative addition (EXACT Δt==0 within a run) over the WHOLE sequence, "
        "independent of 8-item block boundaries; if a positive gap collapses to Δt==0 under float addition, "
        "reject/deterministically nudge before issuance so cluster identity is exact",
        "6. sample class_ids from the class law (with structural zeros); apply the active D coupling(s)",
    ],
    "dt0_rate": "DERIVED diagnostic only (zero-adjacencies/adjacencies; UNDEFINED and excluded at L_total==1); "
                "NOT an independent Bernoulli parameter",
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
    # boundary-short is the CANONICAL bounded-support control (Pi re-gate #3/#4): STRUCTURAL bound L in [1,7]
    # (uniform_int) so no 8-item block forms and S9 refusal is guaranteed. The battery imports THIS profile.
    "boundary_short":   {**_prof(log(9), 0.30, [0.3, 0.25, 0.2, 0.15, 0.1], [], 0.5, (log(1.2), 0.85), _NO_DEP),
                         "length": {"family": "uniform_int", "min": 1, "max": 7}},
}

# Coupling laws: EXACT finite-pool constructions with a dedicated coupling RNG (seed derived from the profile
# identity + component + seed), applied in V2_D_COMPONENT_MENU order. Strength s in [0,0.6] converts to an
# INTEGER number of transformed units. Preservation of the six registered marginals is EMPIRICALLY REQUIRED
# (tested >=24/25 in the ablation battery), NOT asserted "by construction" unless the construction proves it;
# any preservation failure => DESIGN FAIL / re-gate, never threshold tuning.
COUPLING_PROTOCOL = {
    "per_component_freeze": ["exact pre-state and post-state", "integer conversion of s -> #transformed units",
                             "stable tie-breaks + infeasibility/refusal", "dedicated RNG seed derivation + "
                             "draw order", "component composition order = V2_D_COMPONENT_MENU",
                             "whether preservation is EXACT or EMPIRICALLY-REQUIRED",
                             "behaviour with structural-zero classes and short sequences"],
    "preservation_discipline": "intended marginal preservation is TESTED at >=24/25 seeds; any failure returns "
                               "DESIGN FAIL / re-gate — never post-hoc threshold tuning",
}
COUPLING_LAWS = {
    # ACTIVE laws — exactly the V2_D_COMPONENT_MENU (Pi re-gate #4). Rejected explored-space descriptors live in
    # REJECTED_COUPLING_LAWS below and are NEVER part of the active contract.
    "burst_timing": {
        "construction": "rank copula on the sequence's positive gaps: a fraction s of gaps are reassigned to "
                        "follow the rank of the preceding cluster size, holding the sequence's gap MULTISET "
                        "fixed (pooled positive-gap ECDF exact)",
        "preserves_exactly": ["per-sequence gap multiset => pooled positive-gap ECDF"],
        "preservation_required_tested": ["the six registered marginals"],
        "moves": ["S3_tau", "S3_loggap"]},
    "mark_burst_tie": {
        "construction": "within-sequence class relabelling holding per-class counts fixed; a fraction s of "
                        "eligible same-cluster adjacencies are made same-class by count-preserving swaps",
        "preserves_exactly": ["per-sequence class counts => pooled class_tv"],
        "preservation_required_tested": ["S7 (cross-loading recorded, see below)"],
        "moves": ["S4_abs"]},
    "cluster_size_mark_diversity": {
        "construction": "within-sequence, class-count-preserving relabelling concentrating (large clusters) / "
                        "diversifying (small clusters) by a fraction s",
        "preserves_exactly": ["per-sequence class counts => pooled class_tv", "cluster-size multiset"],
        "preservation_required_tested": ["S4 (cross-loading recorded, see below)"],
        "moves": ["S7_abs"]},
}
# Rejected explored-space coupling descriptors (Pi rulings). Retained as provenance ONLY; NEVER dispatched and
# never part of the active contract. Their private implementations may remain in the coupling module.
REJECTED_COUPLING_LAWS = {
    "burst_count_length": {
        "construction": "comonotone K<->L cycle activation preserving the L and K multisets exactly",
        "moves": ["S1_density", "S1_tau"],
        "rejected": "Pi F1 — S1 is structural under the maximal-run law (baseline tau(L,K)~0.92); not a "
                    "separable dependence."},
    "length_class_mix": {
        "construction": "length-bin-dependent POOL-BALANCED class relabelling holding pooled class proportions fixed",
        "moves": ["S5_abs", "S6_tv"],
        "rejected": "Pi step-4 result gate — it drives the TERMINAL S5 occupancy check (candidate-A S5_abs ~19 "
                    "FAIL/25 on MIMIC; real cross-loading, worse at higher N). S5 must not be exempted/weakened, "
                    "so the component is dropped and S6 made terminal/no-D."},
}
# S4<->S7 cross-loading is REAL (diversity-by-cluster-size and within-cluster homogeneity are related). We do
# NOT claim orthogonality; the predeclared cross-loading is recorded and the Jacobian/collision tests decide
# identifiability.
CROSS_LOADING = {"mark_burst_tie<->cluster_size_mark_diversity": "S4 and S7 share within-cluster class "
                 "structure; predeclared, not orthogonalized; identifiability decided by Jacobian + collision"}

# SOURCE_SWAP is a concrete pair; it can NEVER trigger D.
SOURCE_SWAP = {
    "reference": "mimic_scale_control",
    "candidate": "mimic length law PLUS the exact named scid_scale_control class/run/gap laws",
    "expected_failures": ["class_tv", "count_ks", "positive_gap_ks", "S1_density"],
    "never_triggers_D": True,
}

# Ablation orientation (Pi): reference has ONE component at 0.5; candidate A (null-independent) must FAIL the
# mapped row; candidate D-recovery (independent impl at 0.5) must PASS the full row. All non-attributed checks +
# six marginals + S2 + S8 + S9 pass >=24/25 unless listed.
ABLATION_MATRIX = {
    "burst_timing":                {"primary_fail": ["S3_tau", "S3_loggap"], "allowed_sensitive": []},
    "mark_burst_tie":              {"primary_fail": ["S4_abs"], "allowed_sensitive": ["S7_abs"]},
    "cluster_size_mark_diversity": {"primary_fail": ["S7_abs"], "allowed_sensitive": ["S4_abs"]},
    # length_class_mix DROPPED at the step-4 result gate (Pi): its candidate-A violates terminal S5; S6 now
    # terminal/no-D (see REJECTED_D_COMPONENTS).
}
ABLATION_ORIENTATION = {
    "reference": "independent fixture with exactly one component at 0.5",
    "candidate_A": "matched null_independent fixture — MUST FAIL the primary_fail row",
    "candidate_D_recovery": "independent implementation at component 0.5 — MUST PASS the full row",
    "specificity": "all non-attributed checks + six marginals + S2 + S8 + S9 pass >=24/25 (allowed_sensitive "
                   "checks are exempt from the specificity requirement for that component only)",
}

# ============================ 6. simulation (provisional; step-4 must demonstrate) ============================
SIMULATION = {
    "n_seeds": 25,
    "seed_list": list(range(1000, 1025)),
    "per_source_sample_size_candidate": 8000,      # Pi re-gate: one-step N-only escalation 4000->8000 (floors met)
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
# Identifiability vector = only the D-parameter-sensitive scalars (Pi F1/F2). S1/S2/S5/S8/S9 stay mandatory
# adequacy checks but are EXCLUDED from the D Jacobian/recovery objective.
IDENTIFIABILITY_VECTOR = ["S3_tau", "S3_loggap", "S4_abs", "S7_abs"]   # S6_tv dropped with length_class_mix (Pi re-gate)
IDENTIFIABILITY = {
    "param_ranges": {c: [0.0, 0.6] for c in V2_D_COMPONENT_MENU},
    "standardization": "AFFINE range standardization (no logit); NO epsilon clipping (no denominator needs it)",
    "statistic_vector": IDENTIFIABILITY_VECTOR,   # S3_tau/S3_loggap/S4_abs/S7_abs only (Pi re-gate); S1/S2/S5/
                        # S6/S8/S9 remain mandatory adequacy checks but are excluded from the D Jacobian/recovery",
    "whitening_reference": "Sigma estimated at null_independent under CRN (seeds = SIMULATION.seed_list); ridge "
                           "Sigma_lambda = Sigma + 1e-3 * trace(Sigma)/d * I",
    "grid": "3^k joint grid over active components at 0.10/0.35/0.55; nuisance profiles = the exact named list "
            "NUISANCE_PROFILES",
    "finite_difference": "CENTRAL CRN at all interior grid points (incl. 0.55: 0.53/0.57 stay in [0,0.6]); "
                         "FORWARD one-sided active-subset at 0.0; BACKWARD one-sided ONLY at the 0.60 boundary; "
                         "step 0.02",
    "rank_criterion": "standardized Jacobian sigma_min/sigma_max >= 1e-3 (abs tol secondary)",
    "recovery": "DETERMINISTIC nearest-grid recovery on the exact registered grid under a whitened-L2 objective; "
                "lexicographic menu-order tie-break; recover_tol <= 0.05 of range AND <= half a grid step "
                "(no local least-squares solver)",
    "collision": "two settings beyond recover_tol COLLIDE iff ALL whitened cross-stat diffs stay within "
                 "acceptance tol",
    "compute_budget": "explicit cap: ONE worker, <= 8 WALL-CLOCK hours, <= 32 GB RAM; staged (k<=2 grids "
                      "first, then expand within the cap); on cap exceed => terminal PARTIAL result, NO "
                      "adaptive grid reduction after results",
    "nuisance": "IDENTIFIABILITY_NUISANCE (SCID/MIMIC/structural-zero); boundary-short is a terminal support "
                "control, NOT a Jacobian/grid nuisance (its statistics can be NOT_EVALUABLE) — Pi §5",
    "collision_tolerance_vector": "fixed, aligned to (S3_tau,S3_loggap,S4_abs,S7_abs) = "
                                  "[0.05, log(1.10), 0.03, 0.03]",
}
# Jacobian/grid nuisance for identifiability (Pi §5): three profiles; boundary-short excluded (terminal support).
IDENTIFIABILITY_NUISANCE = ["scid_scale_control", "mimic_scale_control", "structural_zero_control"]
NUISANCE_PROFILES = ["scid_scale_control", "mimic_scale_control", "structural_zero_control", "boundary_short"]

# ============================ 9. escalation (Pi §5) ============================
# CHECK/subcheck -> D components. S2 (cluster-size marginal) and the six marginals + source-swap NEVER trigger
# D. S8 and S9 are TERMINAL adequacy guards with NO D route.
# Revised D-eligible map (Pi F1/F2 rulings): S1 (density+tau) and S5 are TERMINAL/structural (no D route);
# burst_count_length dropped. Every remaining component maps 1:1 from a subcheck.
CHECK_TO_D_COMPONENTS = {
    "S3_tau":     {"components": ["burst_timing"], "semantics": "single"},
    "S3_loggap":  {"components": ["burst_timing"], "semantics": "single"},
    "S4_abs":     {"components": ["mark_burst_tie"], "semantics": "single"},
    "S7_abs":     {"components": ["cluster_size_mark_diversity"], "semantics": "single"},
    # S6_tv removed with length_class_mix (Pi step-4 result gate); S6 is now terminal/no-D alongside S5.
}
TERMINAL_CHECKS = ["S1_density", "S1_tau", "S2_ks", "S5_abs", "S6_tv", "S8_density", "S8_class",
                   "S9_zero", "S9_class", "S9_gap",
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
    "coupling_protocol": COUPLING_PROTOCOL,
    "coupling_laws": COUPLING_LAWS,
    "rejected_coupling_laws": REJECTED_COUPLING_LAWS,
    "cross_loading": CROSS_LOADING,
    "source_swap": SOURCE_SWAP,
    "ablation_matrix": ABLATION_MATRIX,
    "ablation_orientation": ABLATION_ORIENTATION,
    "simulation": SIMULATION,
    "fixture_generator": FIXTURE_GENERATOR,
    "identifiability": IDENTIFIABILITY,
    "nuisance_profiles": NUISANCE_PROFILES,
    "escalation": ESCALATION,
    "m0b_support_policy_hash": m0b_support_policy_hash(),
    # two DISTINCT claims (never recombined into a joint-envelope claim), per ROUTES:
    "admissible_claims": {
        "marginal_route": "development-seen aggregate-marginal match — EXPLORATORY ONLY",
        "sequence_route": "synthetic known-profile cross-statistic recovery",
        "explicit_negatives": "NO real joint-envelope claim; NO confirmatory realism claim; NO joint-process "
                              "claim. The independent fixture is synthetic-recovery infrastructure, NOT evidence "
                              "that development-seen TRAIN has these dependencies.",
    },
}


def m3a_design_dev_hash() -> str:
    """DEVELOPMENT identity of the rev-3 executable-verifier design (Pi ACCEPT-to-implement). FINAL identity
    minted only after Pi's M3a final review of the IMPLEMENTED verifier (rebuild step 5)."""
    return canonical_hash(M3A_VERIFIER_DESIGN)
