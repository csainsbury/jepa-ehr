"""Rung-2 SEPARATED-blueprint CONTRACT — the frozen numeric pre-registration (Pi v2 re-gate).

Pi GO-WITH-CHANGES TO BUILD: scaffolding + synthetic tests + sub-gates 1/2 + frozen T1-T3 +
continuous-time/multiplicity head. NOT authorized: governed exports, real-data training/dev
eval, or T4 learned-target work (which needs the separately-gated semi-synthetic oracle).

Every categorical gate below has concrete units / margins / adequacy floors / missing-data
behaviour, content-hashed via `config_hash`. Pure-python + numpy (no torch) so the contract is
cheap to import and fail-hard test. Inherits the Rung-1 spine.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from clinical_jepa.eval.rung1_contract import (  # inherited spine
    N_BOOT, SEED, SWAP_SEED, SENSITIVITY_HORIZONS, deterministic_derangement,
    is_primary_cell, matched_head_hidden,
)

CONTRACT_VERSION = "rung2-contract-v1-pi-v2-regate"

# ============================ cross-cutting ============================
NOMINATION_ONLY = True          # ALL dev decisions are nomination-only (Pi cross-cutting #1)
TEST_ACCESS = False             # test sealed
# shared status vocabulary
NOT_EVALUABLE = "NOT_EVALUABLE"
INCONCLUSIVE = "INCONCLUSIVE"
NEGATIVE = "NEGATIVE"
EFFECT_RULED_OUT = "EFFECT_RULED_OUT"

SUBGATES = ("sg1_rollout", "sg2_count", "sg3_order", "sg4_time")

# ============================ sub-gate 1: rollout diagnosis ============================
# Two estimands: DIRECT-horizon (horizon-conditioned; NO recursion/exposure-gap) and
# RECURSIVE-transition (fixed-width NON-OVERLAPPING δ increments). The recursive path is
# NOT_EVALUABLE unless the checkpoint's metadata proves it was trained on fixed-width
# non-overlapping transition states (Pi #2 — likely absent for horizon-count-1 checkpoints).
RECURSIVE_DELTA = {"SCID": 30.0, "MIMIC": 0.25}          # source-specific fixed increment width
REQUIRE_TRANSITION_TRAINED_CHECKPOINT = True
TRANSITION_META_KEY = "fixed_width_transition_trained"    # checkpoint manifest flag that must be True
ROLLOUT_CLUSTER_FLOOR = 500

# perturbation ensemble (ε chosen from train/synthetic ONLY, before dev)
PERT_EPS_GRID = (0.01, 0.03, 0.1, 0.3)
PERT_EPS = 0.1                                            # frozen operating point (plateau)
PERT_ENSEMBLE_N = 16
PERT_EPS_SELECTION = "train_or_synthetic_only"
JACOBIAN_METHOD = "power_iteration_top_singular_value"    # report top singular value, not a scalar norm

# frozen drift statistic + margins (so signature LABELS may be emitted; else continuous-only)
DRIFT_STAT = "d_self_over_ambient_true_nn"                # d_self normalised by ambient true-true NN distance
EXPOSURE_GAP_MARGIN = 0.02                                # practical margin for g_t
DRIFT_SLOPE_TAU = 0.02
SIG_COLLAPSE_DSELF_OVER_DNN = 0.90                        # d_self/d_NN ≥ this ⇒ no better than nearest wrong instance
ROLLOUT_CI = 0.95
# ρ_t is DESCRIPTIVE ONLY (Pi #1 — never load-bearing / never a signature discriminator)
RHO_T_ROLE = "descriptive_only"
SIGNATURES = ("HEALTHY", "DRIFT_DOMINANT", "COLLAPSE_DOMINANT", "GENUINE_UNIMODALITY",
              "EMA_NONSTATIONARITY_DEFERRED")

# ============================ sub-gate 2: count interface ============================
# ONE matched family (Pi #3): A and B share the same frozen hurdle count-distribution family +
# proper score, varying ONLY where count enters the interface/objective.
COUNT_FAMILY = "hurdle_count"                            # zero-hurdle + count distribution
COUNT_PRIMARY_SCORE = "ranked_probability_score_skill"   # proper distribution score (RPS skill)
COUNT_NOMINATE_MARGIN = 0.02                             # paired margin in RPS-skill units
COUNT_CLUSTER_FLOOR = 500
COUNT_HEAD_PARAM_FORMULA = "matched_head_hidden"         # A/B share the matched param/FLOP budget
COUNT_B_POINT_ESTIMATE_FALLBACK = "NOMINATE_FACTORIZED_STRUCTURAL"  # if B is a point estimate it cannot win a calibrated score
NOMINATE_FACTORIZED = "NOMINATE_FACTORIZED"
NOMINATE_CONCAT = "NOMINATE_CONCAT"
NEITHER_ADEQUATE = "NEITHER_ADEQUATE"

# ============================ sub-gate 3: order (frozen T1-T3; T4 barred) ============================
ORDER_SUPPORT_FLOOR = 500                                # ≥500 eligible exact-fixed-multiset clusters / primary cell
ORDER_L_MAX = 16
ORDER_NO_SILENT_TRUNCATION = True
ORDER_TIE_RULE = "frozen_tie_aware_token_sequence"       # reuse rung1 tie_aware_exact_order_hits
ORDER_TEMPERATURE_SELECTION = "train_internal_only"
# single primary bounded order-skill metric + the two non-redundant gates (G3.1/G3.3 de-duplicated)
PRECEDENCE_SKILL_GATE = 0.10                             # content-prior-adjusted precedence-skill lower-CI
ORDER_SWAP_EXCESS_GATE = 0.10                            # fixed-multiset predictor-swap excess lower-CI
ORDER_T0_IMPROVEMENT_GATE = 0.10                         # improvement over T0 lower-CI (multiplicity-corrected)
ORDER_MULTIPLICITY = "bonferroni_over_frozen_ladder"
# bit accounting + T2 matched-bit ceiling
BIT_ACCOUNTING = "seq_dim32_or_L_log2K"                  # frozen-E: L·D·32 ; VQ: L·log2 K
T2_MATCHED_BIT_QUANTIZER = "uniform_scalar"
T2_CEILING_RETENTION_TOL = 0.05                          # quantised T2 ceiling must stay within tol of unquantised
DECOMP_METRIC = "bounded_order_skill"                    # 3-way decomp is on the SAME bounded score, telescoping
FROZEN_TARGETS = ("T1_pooled_ordinal", "T2_seq_of_latents", "T3_ordinal_tagged_seq")
T4_TARGET = "T4_vq_ordered_codes"                        # BARRED on governed data until the oracle
ORDER_TARGET_FAMILIES = FROZEN_TARGETS + (T4_TARGET,)

# ============================ sub-gate 4: continuous-time / multiplicity head ============================
# NOT a marked TPP (marks are not gated in this blueprint). Timestamp-cluster factorization.
CT_HEAD_NAME = "continuous_time_multiplicity_head"
# 4A — zero/simultaneity (multiplicity), numeric
GATE_4A_MULTIPLICITY_SKILL = 0.05                        # Brier/CRPS skill over context-stratified/rate-only baseline, lower-CI
GATE_4A_SWAP_SKILL = 0.05                                # wrong-context swap skill lower-CI
GATE_4A_ECE = 0.05                                       # calibration ECE upper-CI
GATE_4A_CLUSTER_FLOOR = 500                              # ≥500 clusters, support reported by multiplicity class
# p0 reliability alone cannot pass — multiplicity skill must also pass
# 4B — positive tail, numeric
GATE_4B_KS = 0.05                                        # positive-tail KS upper-CI
GATE_4B_CRPS_SKILL = 0.05                                # over the context-observable stratified marginal, lower-CI
GATE_4B_RATE_HEAD_IMPROVEMENT = 0.05                     # over the rate-only head, lower-CI
GATE_4B_SWAP = 0.05                                      # rate/occupancy-matched wrong-context swap, lower-CI
GATE_4B_CLUSTER_FLOOR = 500                              # positive-tail clusters
GATE_4B_INTERVAL_FLOOR = 1000                            # positive intervals + precision sim
# stratification: context-observable ONLY, bins frozen from train; observed-future = oracle-assisted
CONTEXT_STRATA = ("occupancy_bin", "context_rate_quantile", "horizon")
OBSERVED_FUTURE_STRATA = ("future_occupancy", "future_rate")   # oracle-assisted, non-primary
# 4A/4B are SEPARATE + CONJUNCTIVE; a joint zero-aware proper score is SECONDARY only
TIMING_JOINT_SCORE_ROLE = "secondary_descriptive"


def requires_oracle(target: str) -> bool:
    """T4 (learned VQ target) needs the separately-gated semi-synthetic oracle before ANY
    governed real-data training/evaluation (Pi #7 / Q4)."""
    return target == T4_TARGET


def is_oracle_assisted_stratum(var: str) -> bool:
    """Observed-future strata may never be the operational primary baseline (Pi #5)."""
    return var in OBSERVED_FUTURE_STRATA


def is_fixed_width_transition_training(*, autoregression_mode: str | None, horizon_count: Any,
                                       horizon_stride_tokens: Any, max_target_tokens: Any,
                                       encode_empty: bool = False) -> bool:
    """DERIVE whether a training configuration produces fixed-width NON-OVERLAPPING transition states.

    This is the single source of truth for the `fixed_width_transition_trained` flag, kept next to the gate
    that consumes it so the writer and the reader cannot drift apart. It is DERIVED from the training config,
    never caller-asserted.

    Requires ALL of:
      * `autoregression_mode == "recursive"` — horizon-conditioned heads are a different estimand entirely;
      * `horizon_count >= 2` — one window is a single prediction, not a transition sequence, so there is
        nothing recursive to diagnose (this is why every existing v0B / encode-empty checkpoint fails: they
        train at the default horizon_count=1, and `--encode-empty` pins it to 1);
      * `stride == max_target_tokens` — equal-width windows tiled contiguously. A stride below the width
        OVERLAPS (states share events, so drift is not attributable to the transition); a stride above it
        leaves GAPS (the states are not a contiguous delta-tiling).
    """
    if encode_empty:
        return False                                   # encode-empty pins horizon_count to 1 by construction
    if autoregression_mode != "recursive":
        return False
    try:
        k = int(horizon_count); stride = int(horizon_stride_tokens); width = int(max_target_tokens)
    except (TypeError, ValueError):
        return False
    if isinstance(horizon_count, bool) or isinstance(horizon_stride_tokens, bool):
        return False
    return k >= 2 and width > 0 and stride == width


def recursive_path_evaluable(checkpoint_meta: dict[str, Any] | None) -> bool:
    """The recursive-transition path is NOT_EVALUABLE unless the checkpoint's metadata proves it
    was trained on fixed-width non-overlapping transition states (Pi #2). No pseudo-rollouts.

    Accepts the flag if present; otherwise DERIVES it from the training config the checkpoint already
    records, so a checkpoint written before the flag existed is judged on its actual regime rather than on
    the absence of a field. A checkpoint carrying neither the flag nor the config stays NOT_EVALUABLE."""
    if not REQUIRE_TRANSITION_TRAINED_CHECKPOINT:
        return True
    meta = checkpoint_meta or {}
    if TRANSITION_META_KEY in meta:
        return bool(meta[TRANSITION_META_KEY])
    if "autoregression_mode" not in meta:
        return False                                   # nothing to derive from -> fail closed
    return is_fixed_width_transition_training(
        autoregression_mode=meta.get("autoregression_mode"),
        horizon_count=meta.get("horizon_count_trained", 0),
        horizon_stride_tokens=meta.get("horizon_stride_tokens", 0),
        max_target_tokens=meta.get("max_target_tokens", meta.get("horizon_stride_tokens", 0)),
        encode_empty=bool(meta.get("encode_empty", False)))


# ============================ fail-hard independence + stop-line (Pi #6) ============================
NOMINATE_DIRECTION = "NOMINATE_DIRECTION"
RETAIN_INCUMBENT = "RETAIN_INCUMBENT"
ESCALATE_REDESIGN = "ESCALATE_REDESIGN"
_NOMINATION_LABELS = frozenset({
    NOMINATE_FACTORIZED, NOMINATE_CONCAT, NEITHER_ADEQUATE, NOMINATE_DIRECTION,
    RETAIN_INCUMBENT, ESCALATE_REDESIGN, NOT_EVALUABLE, INCONCLUSIVE, NEGATIVE, EFFECT_RULED_OUT,
})
# metrics that ONLY the recursive-transition path may emit (direct-horizon rows must not)
RECURSIVE_ONLY_METRICS = ("exposure_gap", "d_self_free", "d_self_tf", "recursive_step_k")


def is_nomination_only_decision(label: str) -> bool:
    """No dev decision may be an ADOPT/certify (Pi cross-cutting #1)."""
    return label in _NOMINATION_LABELS and not str(label).upper().startswith("ADOPT")


def assert_gates_independent(provenance_by_gate: dict[str, dict[str, Any]]) -> bool:
    """Fail-hard if any TRAINED artifact (checkpoint/optimizer/run_id) or a per-gate config_hash is
    shared across sub-gates. A common FROZEN context encoder is allowed and is NOT counted."""
    seen: dict[Any, str] = {}
    for gate, prov in provenance_by_gate.items():
        for k in ("optimizer", "run_id", "trained_checkpoint", "config_hash"):
            v = prov.get(k)
            if v is None:
                continue
            if v in seen and seen[v] != gate:
                raise AssertionError(f"gate-independence breach: {k}={v!r} shared by {seen[v]} and {gate}")
            seen[v] = gate
    return True


def validate_direct_path_row(row: dict[str, Any]) -> bool:
    """Direct cumulative-horizon rows must never carry recursive/exposure-gap metrics (Pi #6)."""
    bad = [k for k in RECURSIVE_ONLY_METRICS if k in row]
    if bad:
        raise AssertionError(f"direct-horizon path must not emit recursive metrics {bad}")
    return True


# ============ ORACLE (external-to-encoder synthetic SPECIFICATION TEST) frozen OC (Pi v3 #8) ============
# The oracle is a recipe FALSIFIER, not a real-EHR certificate. These are concrete pre-registered
# numbers (no "will be frozen"). Order units = order-skill (beyond-content-prior log-loss skill).
ORACLE_SCHEMA_VERSION = "clinical-jepa-oracle-order-authorization-v3"
ORACLE_EVALUATOR_IDENTITY = "oracle_meta_eval_v5"   # pinned evaluator schema id (Pi #3 identity check);
                                                    # v5 = regime-aware references (Pi C=5 R0-defect ruling)
ORACLE_R_BAYES_MARGIN = 0.05            # R_bayes must beat R0 by this lower-CI margin (else HIDDEN NULL)
ORACLE_R_BAYES_MC_TOL = 0.01           # Monte-Carlo error tolerance for the R_bayes posterior estimate
ORACLE_CTRL_REF_KAPPA = 0.30           # DEV reference κ the U6 control quantizers are frozen from — MUST
                                       # be an allowed TRAIN-grid κ so no control is fitted on an OC-only
                                       # cell (Pi hardening #4; κ_mid=0.35 is OC-only and is NOT allowed)
ORACLE_R_BAYES_MC_SEEDS = 8            # independent MC seeds for R_bayes
ORACLE_NUISANCE_MARGIN = 0.05          # recipe INCREMENTAL skill over R_nuis (lower-CI)
ORACLE_NULL_ALPHA = 0.05               # per-property null FPR (upper-CI)
ORACLE_POWER_FLOOR = 0.80              # power at kappa_mid
ORACLE_PRECISION_N_STUDIES = 70        # frozen precision-sim study count (enough to bound type-I ≤ α
                                       # under 0 observed false positives: rule-of-three 3/70 < 0.05)
ORACLE_PRECISION_MC_TOL = 0.05         # precision-sim resolution tolerance (its own, NOT the tight
                                       # R_bayes posterior MC tol) — refuse if the CI is coarser
ORACLE_POWER_KAPPA_MID = 0.35          # the SINGLE power/MDE coupling point — DISJOINT from the train
                                       # grid and from the two frozen off-grid endpoint probes (Pi 2nd-pass)
ORACLE_MONO_SPEARMAN = 0.90            # monotonicity Spearman lower-CI
ORACLE_CALIB_SLOPE_BAND = (0.8, 1.2)   # recovery calibration slope band
ORACLE_CALIB_INTERCEPT_TOL = 0.05
# calibration realism envelope — CONJUNCTIVE per source block (Pi 2nd-pass #3, exact thresholds)
ORACLE_ENV_DT0_ABS = 0.02              # |Δt=0 fraction difference|
ORACLE_ENV_TV = 0.05                   # six-class token/event distribution total variation
ORACLE_ENV_KS = 0.05                   # length / positive-gap / per-seq-count distribution KS
ORACLE_ENV_OCCUPANCY_ABS = 0.03        # mean class-occupancy fraction (distinct classes / C, C=5)
# The FIVE natural clinical content families of the corrected-v1 tokenised substrate, in token-ID order
# (Pi option-C ruling). Structural families are EXCLUDED from every clinical class / count / timing
# denominator: special [0,4) and dataset_context [1048,1050) — i.e. the masked [BOS] + DATASET:X source
# prefix. n_events = sum(class_counts) AFTER that exclusion; occupancy = distinct represented / 5.
ORACLE_ENV_N_CLASSES = 5
ORACLE_ENV_CLASS_FAMILIES = (        # calibration class -> vocab family -> token-ID range [start, end)
    ("demographic", 4, 51), ("diagnosis", 51, 91), ("lab", 91, 951),
    ("medication", 951, 1032), ("state", 1032, 1048),
)
ORACLE_ENV_STRUCTURAL_RANGES = (("special", 0, 4), ("dataset_context", 1048, 1050))
ORACLE_ENV_MIN_DENOM = 500             # minimum denominator floor; below => NOT_EVALUABLE (never zero-fill)
ORACLE_N_NULL_SEEDS = 200
ORACLE_N_POS_SEEDS = 100
ORACLE_N_HELDOUT_FAMILIES = 2          # >=2 DISTINCT held-out structural meta-families
ORACLE_CLUSTER_UNIT = "sequence"       # FPR aggregated per sequence, NOT per precedence-pair
ORACLE_MULTIPLE_TESTING = "bonferroni_over_evaluable_cells"
ORACLE_OFFGRID_KAPPA = (0.15, 0.6)     # the TWO frozen held-out endpoint probes (NOT a band, NOT in the
                                       # train grid). Preserved exactly per Pi's 2nd-pass residual correction.
ORACLE_MDE_DEF = "smallest_kappa_with_power_ge_0.80"
# Frozen OC grids (Pi #5): the monotonicity grid (U3, over the held-out family's own κ) and the reference
# power/MDE grid. ORACLE_MDE_KAPPAS is GENUINELY DISJOINT from the train grid ∪ held-out endpoints ∪ κmid
# (= {0.0,0.10,0.30,0.50,0.75} ∪ {0.15,0.60} ∪ {0.35}); it characterizes evaluator power, not a probe of
# any scored cell. Both are hashed into the ledger identity so a silent grid change breaks certification.
ORACLE_MONO_KAPPAS = (0.15, 0.35, 0.60)
ORACLE_MDE_KAPPAS = (0.05, 0.08, 0.12, 0.20, 0.25, 0.40)
ORACLE_HIDDEN_NULL_RULE = "R_bayes_within_margin_of_R0_excluded_from_positive"
ORACLE_SHORTCUT_MAX_SKILL = 0.10       # an h-projection-only shortcut method must NOT exceed this
# ---- decision margins (Pi consolidated #3 — practical effect, not merely statistical positivity) ----
ORACLE_EO1_SKILL_GATE = 0.10           # E-O1 recipe order-skill lower-CI (PRACTICAL floor, not >0)
ORACLE_U6_BANDWIDTH_MARGIN = 0.05      # recipe must beat EACH matched-bit control by this lower-CI
ORACLE_R0_POSITIVE_FAIL_MAX = 0.10     # R0 order-skill upper-CI must be < this on positive cells
ORACLE_NUIS_ORTHO_FAIL_MAX = 0.05      # R_nuis skill upper-CI must be < this in Σ-orthogonal cells
ORACLE_PRECISION_COVERAGE = 0.95       # evaluator precision-sim nominal coverage floor
ORACLE_PRECISION_POWER = 0.80          # evaluator precision-sim power floor
ORACLE_FAMILY_CONJUNCTION = "all_held_out_families_must_pass"   # family-level conjunction
# ---- sequence-level null statistic (Pi consolidated #4 — one decision per sequence, not per pair) ----
ORACLE_NULL_SEQ_STATISTIC = "mean_beyond_prior_pair_skill_per_sequence"  # aggregate pairs -> 1 per seq
ORACLE_NULL_MIN_PAIRS = 3              # a sequence needs >=this many eligible precedence pairs to count
ORACLE_NULL_TIE_HANDLING = "exclude_same_class_ties"           # tied-class pairs carry no order info
ORACLE_NULL_BOOTSTRAP_UNIT = "sequence"                        # FPR bootstrapped over sequences
ORACLE_NULL_FIRE_RULE = "sequence_skill_lower_CI_gt_0"         # a firing null study = a false positive
ORACLE_N_BOOT = 1000                  # bootstrap replicates for a study's skill CI (DISTINCT from the
                                      # ORACLE_N_NULL_SEEDS independent null STUDIES — Pi #4 conflated them)
ORACLE_NULL_STUDY_SEQS = 200          # sequences per independent null study (frozen study size)
ORACLE_NULL_FIRE_ALPHA = 0.01         # one-sided level of the per-study fire test (stricter than the
                                      # 0.05 unlock so the FPR UPPER CI clears 0.05 with margin)
ORACLE_FPR_UPPER_CI_MAX = 0.05        # gate: one-sided 95% Clopper-Pearson UPPER bound on null FPR

# the ORDER-T4 unlock is PROPERTY-SPECIFIC (Pi oracle #5): only order-relevant checks; timing
# certification is a SEPARATE manifest and must NOT veto an order-only target.
ORDER_UNLOCK_CHECKS = ("U1_order_recovery", "U2_null", "U3_monotone", "U4_nuisance_incremental",
                       "U6_bandwidth_fair")
# every field below is MANDATORY for a governed T4 authorization; each is checked for exact
# value/format/membership (not merely non-empty) — Pi v3 guard defect.
_REQUIRED_MANIFEST_FIELDS = ("oracle_mechanism_hash", "evaluator_commit", "certified_recipe_hash",
                             "recipe_registry_id", "sealed_cert_run_id", "gate_event_ref", "blueprint_hash")


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def _distinct_nonempty_strs(x: Any, n: int) -> bool:
    """Fail-closed list validation: a list of ≥n DISTINCT non-empty strings (malformed -> False)."""
    if not isinstance(x, list) or not all(isinstance(e, str) and e for e in x):
        return False
    return len(set(x)) >= n


def _dget(m: Any, key: str) -> Any:
    """Fail-closed nested read: a non-dict at `key` (or missing) yields None, never raises."""
    return m.get(key) if isinstance(m, dict) else None


def _finite_float(x: Any) -> float | None:
    """Return x as a finite float, or None for non-numeric / bool / NaN / inf (fail-closed)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    xf = float(x)
    return xf if xf == xf and xf not in (float("inf"), float("-inf")) else None


def t4_governed_allowed(inputs_are_governed: bool, oracle_authorization: dict[str, Any] | None,
                        *, presented_recipe_hash: str | None = None) -> bool:
    """Governed T4 (a learned VQ ORDER target) is refused until a frozen, Pi-gated oracle manifest
    CERTIFIES the EXACT recipe presented AND matches the TRUSTED COMMITTED oracle policy. Synthetic/
    safe-public scaffolding is always allowed.

    FAIL-CLOSED. There is NO caller-injectable trust root: the blueprint/gate/mechanism/schema/
    evaluator/registry/sealed-run anchors ALL come from the committed ``oracle_policy`` (Pi #1 — a
    ``policy=`` argument here would let the run operator supply a matching ad-hoc policy and pass).
    The presented RECIPE hash is the ONLY run-supplied identity. The schema is the frozen constant,
    not a caller argument. Any malformed manifest (wrong nested type, non-numeric alpha, …) REFUSES
    rather than raising. Omitting the recipe hash, an empty/mismatched committed policy, or any failing
    check is a REFUSAL. Only ``synthetic_recovery_CERTIFIED`` is accepted."""
    if not inputs_are_governed:
        return True
    from clinical_jepa.eval.oracle_policy import manifest_matches_policy  # committed trust anchor
    if not isinstance(oracle_authorization, dict):
        return False
    m = oracle_authorization
    # the presented recipe hash is the ONLY run-supplied identity (names the actual governed recipe).
    if not _nonempty_str(presented_recipe_hash):
        return False
    if m.get("schema_version") != ORACLE_SCHEMA_VERSION:                  # frozen constant, not caller arg
        return False
    if not (m.get("oracle_frozen") is True and m.get("pi_gate") == "PASS"):
        return False
    if m.get("verdict") != "synthetic_recovery_CERTIFIED":               # legacy "CERTIFIED" removed
        return False
    if any(not _nonempty_str(m.get(k)) for k in _REQUIRED_MANIFEST_FIELDS):
        return False
    if m.get("certified_recipe_hash") != presented_recipe_hash:          # exact recipe binding
        return False
    # TRUSTED anchor: blueprint/gate/mechanism/schema/evaluator/registry/sealed-run vs the COMMITTED
    # policy ONLY — no alternate trust-root parameter exists. Empty committed policy => refuse.
    if not manifest_matches_policy(m):
        return False
    if not _distinct_nonempty_strs(m.get("held_out_family_ids"), ORACLE_N_HELDOUT_FAMILIES):
        return False
    if not (m.get("codebook_postdates_oracle") is True and m.get("labels_eval_only_verified") is True):
        return False
    if m.get("governed_t4_real_output_ceiling") != "NOMINATE" or not _nonempty_str(m.get("transfer_caveat")):
        return False
    # PROPERTY-SPECIFIC order unlock: only the order checks, all PASS; timing is a separate manifest.
    # All nested reads are type-guarded so a list/str/None in place of a dict REFUSES, not raises.
    checks = m.get("unlock_checks")
    if not (isinstance(checks, dict) and all(checks.get(c) == "PASS" for c in ORDER_UNLOCK_CHECKS)):
        return False
    if _dget(m.get("precision_sim"), "adequate") is not True:
        return False
    if _dget(m.get("realism_envelope"), "within_envelope") is not True:
        return False
    rb = m.get("reference_bounds")
    if not isinstance(rb, dict):
        return False
    alpha = _finite_float(rb.get("evaluator_realized_alpha"))
    if not (rb.get("R_bayes_beats_R0") is True and rb.get("R0_null_pass") is True
            and rb.get("R0_positive_fail") is True and rb.get("nuisance_incremental_margin_ok") is True
            and alpha is not None and alpha <= ORACLE_NULL_ALPHA):
        return False
    return True


def order_support_status(n_eligible_clusters: int) -> str:
    """Below the exact-fixed-multiset support floor => NOT_EVALUABLE (no silent relaxation, Pi #4)."""
    return "SUPPORTED" if int(n_eligible_clusters) >= ORDER_SUPPORT_FLOOR else NOT_EVALUABLE


def frozen_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "cross_cutting": {"nomination_only": NOMINATION_ONLY, "test_access": TEST_ACCESS,
                          "subgates": list(SUBGATES)},
        "sg1_rollout": {"recursive_delta": RECURSIVE_DELTA,
                        "require_transition_trained_checkpoint": REQUIRE_TRANSITION_TRAINED_CHECKPOINT,
                        "cluster_floor": ROLLOUT_CLUSTER_FLOOR, "pert_eps_grid": list(PERT_EPS_GRID),
                        "pert_eps": PERT_EPS, "pert_ensemble_n": PERT_ENSEMBLE_N,
                        "drift_stat": DRIFT_STAT, "exposure_gap_margin": EXPOSURE_GAP_MARGIN,
                        "drift_slope_tau": DRIFT_SLOPE_TAU,
                        "sig_collapse_dself_over_dnn": SIG_COLLAPSE_DSELF_OVER_DNN,
                        "rho_t_role": RHO_T_ROLE, "signatures": list(SIGNATURES)},
        "sg2_count": {"family": COUNT_FAMILY, "primary_score": COUNT_PRIMARY_SCORE,
                      "nominate_margin": COUNT_NOMINATE_MARGIN, "cluster_floor": COUNT_CLUSTER_FLOOR,
                      "b_point_estimate_fallback": COUNT_B_POINT_ESTIMATE_FALLBACK},
        "sg3_order": {"support_floor": ORDER_SUPPORT_FLOOR, "l_max": ORDER_L_MAX,
                      "no_silent_truncation": ORDER_NO_SILENT_TRUNCATION, "tie_rule": ORDER_TIE_RULE,
                      "precedence_skill_gate": PRECEDENCE_SKILL_GATE,
                      "swap_excess_gate": ORDER_SWAP_EXCESS_GATE,
                      "t0_improvement_gate": ORDER_T0_IMPROVEMENT_GATE, "multiplicity": ORDER_MULTIPLICITY,
                      "bit_accounting": BIT_ACCOUNTING, "t2_quantizer": T2_MATCHED_BIT_QUANTIZER,
                      "t2_ceiling_retention_tol": T2_CEILING_RETENTION_TOL, "decomp_metric": DECOMP_METRIC,
                      "frozen_targets": list(FROZEN_TARGETS), "t4_target": T4_TARGET},
        "sg4_time": {"head": CT_HEAD_NAME,
                     "gate_4a": {"multiplicity_skill": GATE_4A_MULTIPLICITY_SKILL,
                                 "swap_skill": GATE_4A_SWAP_SKILL, "ece": GATE_4A_ECE,
                                 "cluster_floor": GATE_4A_CLUSTER_FLOOR},
                     "gate_4b": {"ks": GATE_4B_KS, "crps_skill": GATE_4B_CRPS_SKILL,
                                 "rate_head_improvement": GATE_4B_RATE_HEAD_IMPROVEMENT,
                                 "swap": GATE_4B_SWAP, "cluster_floor": GATE_4B_CLUSTER_FLOOR,
                                 "interval_floor": GATE_4B_INTERVAL_FLOOR},
                     "context_strata": list(CONTEXT_STRATA),
                     "observed_future_strata_oracle_assisted": list(OBSERVED_FUTURE_STRATA),
                     "joint_score_role": TIMING_JOINT_SCORE_ROLE},
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "swap_seed": SWAP_SEED},
    }


def config_hash(run_config: dict[str, Any] | None = None) -> str:
    payload = {"contract": frozen_contract(), "run_config": run_config or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
