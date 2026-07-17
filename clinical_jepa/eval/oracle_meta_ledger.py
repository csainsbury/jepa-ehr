"""Frozen hypothesis ledger for the whole-pass certification (Pi keystone #4/#5).

Enumerates EVERY load-bearing hypothesis (held-out family × endpoint κ × required endpoint) BEFORE any
recipe is scored, and gives each CI-based endpoint a Bonferroni-corrected α over the full evaluable
family. Support-based exclusions are decided before scoring and cannot shrink the ledger; a recipe
cannot reduce multiplicity by causing a cell to fail (the denominator is fixed here).
"""
from __future__ import annotations

from dataclasses import dataclass

from clinical_jepa.eval.oracle_meta_gen import HELDOUT_FAMILIES, KAPPA_HELDOUT_ENDPOINTS
from clinical_jepa.eval.rung2_contract import (
    ORACLE_FPR_UPPER_CI_MAX, ORACLE_MDE_KAPPAS, ORACLE_MONO_KAPPAS, ORACLE_MONO_SPEARMAN, ORACLE_N_BOOT,
    ORACLE_N_NULL_SEEDS, ORACLE_N_POS_SEEDS, ORACLE_NULL_ALPHA, ORACLE_NULL_FIRE_ALPHA,
    ORACLE_NULL_STUDY_SEQS, ORACLE_POWER_FLOOR,
)

FAMILY_ALPHA = ORACLE_NULL_ALPHA        # 0.05 family-wise error budget

# The sampling-based OC decisions (U2 null FPR, U3 monotonicity, U1/MDE power) are NOT Bonferroni CI
# endpoints — they carry their own frozen grids / study counts / bootstrap counts / α-families. Pi #5:
# these were previously stored as α=0.0 and left the ledger hash blind to the sampling design. Bind them.
OC_CONFIG = {
    "mono_kappas": [round(float(k), 9) for k in ORACLE_MONO_KAPPAS],
    "mde_kappas": [round(float(k), 9) for k in ORACLE_MDE_KAPPAS],
    "n_null_studies": ORACLE_N_NULL_SEEDS, "n_pos_seeds": ORACLE_N_POS_SEEDS,
    "null_study_seqs": ORACLE_NULL_STUDY_SEQS, "n_boot": ORACLE_N_BOOT,
    "alpha_families": {
        "u2_null_fire": round(float(ORACLE_NULL_FIRE_ALPHA), 9),
        "u2_fpr_upper_ci_max": round(float(ORACLE_FPR_UPPER_CI_MAX), 9),
        "u3_mono_spearman_lo": round(float(ORACLE_MONO_SPEARMAN), 9),
        "u1_power_floor": round(float(ORACLE_POWER_FLOOR), 9)},
}

# CI-based endpoints evaluated per (family, endpoint κ) — these carry the Bonferroni correction.
CI_ENDPOINTS = (
    "recipe_minus_R0",              # U1 practical order skill over content prior
    "recipe_minus_Rnuis_orth",      # U4 incremental over nuisance in the Σ-orthogonal cell
    "recipe_minus_meanembed",       # U6 control 1
    "recipe_minus_randomcodebook",  # U6 control 2
    "Rbayes_minus_R0",              # hidden-null / reference-bound endpoint
)
# per-(family, κ) non-CI checks with their own frozen thresholds (listed for completeness + counting).
PER_CELL_CHECKS = ("e_o2_calibration", "u6_randomcodebook_null_pass", "u6_randomcodebook_positive_fail",
                   "u4_Rnuis_orth_upper_ci", "u4_leak_diagnostic")
# per-family checks (not per-κ).
PER_FAMILY_CHECKS = ("u3_monotone", "u2_recipe_null_fpr", "u1_kmid_power")


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    family_id: str
    kappa: float | None
    kind: str
    alpha: float                    # corrected α for CI-based endpoints; frozen-threshold checks carry 0.0


@dataclass(frozen=True)
class HypothesisLedger:
    hypotheses: tuple[Hypothesis, ...]
    n_ci: int
    bonferroni_alpha: float

    def ci_alpha(self) -> float:
        return self.bonferroni_alpha

    def cardinality(self) -> int:
        return len(self.hypotheses)


def ledger_hash() -> str:
    """Content hash of the frozen hypothesis ledger — pinned in the UnlockEvaluation identities and
    re-verified by the pure verdict (Pi #6: a fabricated ledger must not certify)."""
    from clinical_jepa.eval.oracle_contracts import canonical_hash
    from clinical_jepa.eval.oracle_meta_refs import _REGIMES
    led = build_ledger()
    return canonical_hash({"n_ci": led.n_ci, "alpha": round(led.bonferroni_alpha, 9),
                           "hyps": [(h.hid, h.family_id, h.kappa, h.kind, round(h.alpha, 9))
                                    for h in led.hypotheses],
                           "oc_config": OC_CONFIG,        # Pi #5: bind the sampling grids/counts/α-families
                           # Pi C=5 R0-defect: bind the regime-aware reference contract, so the corrected
                           # U6 null semantics (null rows vs R0(0)) move the ledger/reference identity.
                           "reference_regime_contract": {
                               "regimes": list(_REGIMES),
                               "positive": "R0(kappa_cell) on ~is_null rows",
                               "null": "R0(0) on is_null rows",
                               "unlabelled_mixture": "p_null*R0(0)+(1-p_null)*R0(kappa_cell), all rows, explicit p_null",
                               "u6_null_pass": "random-codebook skill vs R0(0) on null rows",
                               "e_o2": "positive rows vs pi_star(kappa_cell)"}})


def build_ledger(families: tuple[str, ...] = HELDOUT_FAMILIES,
                 endpoints: tuple[float, ...] = KAPPA_HELDOUT_ENDPOINTS) -> HypothesisLedger:
    n_ci = len(families) * len(endpoints) * len(CI_ENDPOINTS)
    alpha = FAMILY_ALPHA / max(1, n_ci)
    hyps: list[Hypothesis] = []
    for fam in families:
        for kap in endpoints:
            for ep in CI_ENDPOINTS:
                hyps.append(Hypothesis(f"{fam}|{kap}|{ep}", fam, kap, ep, alpha))
            for ck in PER_CELL_CHECKS:
                hyps.append(Hypothesis(f"{fam}|{kap}|{ck}", fam, kap, ck, 0.0))
        for fk in PER_FAMILY_CHECKS:
            hyps.append(Hypothesis(f"{fam}|*|{fk}", fam, None, fk, 0.0))
    return HypothesisLedger(tuple(hyps), n_ci, alpha)
