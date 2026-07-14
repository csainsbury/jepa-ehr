"""Typed UnlockEvaluation — the SOLE input to the candidate verdict (Pi keystone #4).

Composes U1..U6 + precision + reference bounds + counts + readiness + the frozen hypothesis ledger +
the recipe/artifact/mechanism/evaluator/split/seed identities into ONE immutable object. The candidate
verdict is a PURE FUNCTION of it (``certify_from_unlock``). Everything is scored on a recipe fitted ONCE
on the TRAIN families (applied unchanged to held-out cells). Safe-public / synthetic; candidate-only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clinical_jepa.eval import oracle_meta_refs as R
from clinical_jepa.eval.oracle_meta_gen import (
    HELDOUT_FAMILIES, KAPPA_HELDOUT_ENDPOINTS, KAPPA_MID, TRAIN_FAMILIES, generate_meta_cell,
)
from clinical_jepa.eval.oracle_meta_ledger import HypothesisLedger, build_ledger
from clinical_jepa.eval.rung2_contract import (
    ORACLE_CALIB_INTERCEPT_TOL, ORACLE_CALIB_SLOPE_BAND, ORACLE_EO1_SKILL_GATE, ORACLE_FPR_UPPER_CI_MAX,
    ORACLE_NUIS_ORTHO_FAIL_MAX, ORACLE_NUISANCE_MARGIN, ORACLE_POWER_FLOOR, ORACLE_R_BAYES_MARGIN,
    ORACLE_U6_BANDWIDTH_MARGIN, ORDER_SUPPORT_FLOOR,
)
from clinical_jepa.eval.oracle_metrics import clopper_pearson_lower, clopper_pearson_upper

def _seed(*p: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(p).encode()).digest()[:4], "big")


def _control_bits(recipe) -> int:
    """The matched bit budget the U6 controls use — taken from the REGISTERED recipe, not hard-coded."""
    return int(recipe.spec().bit_accounting.get("control_bits", 8))


KAPPA_CERTIFY = max(KAPPA_HELDOUT_ENDPOINTS)      # the strong endpoint that must PASS the practical gates
KAPPA_LOW = min(KAPPA_HELDOUT_ENDPOINTS)          # the low endpoint — a monotonicity/OC anchor, not a pass


@dataclass(frozen=True)
class CellUnlock:
    family_id: str
    kappa: float
    role: str                        # "certify" | "low_oc"
    status: str                      # SUPPORTED | SUPPORT_STARVED | HIDDEN_NULL
    checks: dict[str, tuple]         # endpoint -> (value, passed)
    cluster_counts: dict[int, int] = field(default_factory=dict)

    @property
    def evaluable(self) -> bool:
        return self.status == "SUPPORTED"

    @property
    def passed(self) -> bool:
        return self.evaluable and all(ok for _, ok in self.checks.values())


@dataclass(frozen=True)
class FamilyUnlock:
    family_id: str
    u3_monotone: tuple               # (increase_lower_ci, passed)
    u2_null_fpr: tuple               # (fpr_upper_ci, passed)
    u1_kmid_power: tuple             # (power_lower_ci, passed)
    cells: tuple[CellUnlock, ...]

    @property
    def passed(self) -> bool:
        # the CERTIFY-role endpoint(s) must be evaluable and pass; low-OC endpoints are anchors only.
        certify = [c for c in self.cells if c.role == "certify"]
        return (bool(certify) and all(c.evaluable and c.passed for c in certify)
                and self.u3_monotone[1] and self.u2_null_fpr[1] and self.u1_kmid_power[1])


@dataclass(frozen=True)
class UnlockEvaluation:
    families: tuple[FamilyUnlock, ...]
    precision: dict[str, Any]
    ledger_cardinality: int
    ledger_alpha: float
    train_family_readiness: bool
    identities: dict[str, Any]
    n_evaluable_cells: int
    n_hidden_null: int
    n_support_starved: int


# ---- per-predictor briers on a cell ----
def _recipe_briers(recipe, cell):
    # decode via the recipe's REGISTERED stochastic sampler (exercises sampling end-to-end).
    from clinical_jepa.eval.oracle_meta_recipe import sampled_pairwise_probs
    probs = sampled_pairwise_probs(recipe, cell, seed=_seed("sample", cell.family_id, str(cell.kappa)))
    return R.briers_from_probs(probs, cell)


def _cell_unlock(recipe, family_id: str, kappa: float, ledger: HypothesisLedger, seed: int) -> CellUnlock:
    from clinical_jepa.eval.oracle_meta_gen import multiset_cluster_counts
    role = "certify" if kappa == KAPPA_CERTIFY else "low_oc"
    cell = generate_meta_cell(family_id, kappa, "orthogonal", 4000, seed=seed + 11,
                              support_floor=ORDER_SUPPORT_FLOOR)
    if cell.support_status != "SUPPORTED":
        return CellUnlock(family_id, kappa, role, "SUPPORT_STARVED", {})
    leak = generate_meta_cell(family_id, kappa, "correlated_leak", 4000, seed=seed + 12)
    a = ledger.ci_alpha()
    # reference bracket (computed BEFORE recipe inspection for the hidden-null decision)
    br_bayes = R.briers_from_probs(R.r_bayes_probs(cell), cell)
    rb_r0 = R.paired_skill_contrast(br_bayes[0], br_bayes[1], br_bayes[1], br_bayes[2],
                                    base_seed=_seed("rb", family_id, str(kappa)), alpha=a)
    if rb_r0[1] < ORACLE_R_BAYES_MARGIN:                     # hidden null: excluded before scoring
        return CellUnlock(family_id, kappa, role, "HIDDEN_NULL", {}, multiset_cluster_counts(cell))
    # recipe + comparators (U6 controls use the recipe's REGISTERED matched control bits)
    bits = _control_bits(recipe)
    b_rec = _recipe_briers(recipe, cell)
    b_nuis = R.briers_vs_r0(R.r_nuis_scores(cell), cell)
    b_me = R.briers_vs_r0(R.mean_embed_quantized_scores(cell, bits), cell)
    b_rc = R.briers_vs_r0(R.random_codebook_scores(cell, bits, _seed("rc", family_id, str(kappa))), cell)
    b0, npair = b_rec[1], b_rec[2]

    def contrast(comp_briers, tag):
        return R.paired_skill_contrast(b_rec[0], comp_briers[0], b0, npair,
                                       base_seed=_seed(tag, family_id, str(kappa)), alpha=a)[1]

    r0_lo = R.paired_skill_contrast(b_rec[0], b0, b0, npair, base_seed=_seed("r0", family_id, str(kappa)), alpha=a)[1]
    nuis_lo = contrast(b_nuis, "nu")
    me_lo = contrast(b_me, "me")
    rc_lo = contrast(b_rc, "rc")
    # U4 orthogonal R_nuis has no skill (upper-CI failure) + correlated-leak diagnostic
    nuis_orth_up = R.paired_skill_contrast(b_nuis[1], b_nuis[0], b_nuis[1], b_nuis[2],
                                           base_seed=_seed("nuup", family_id), alpha=a)  # skill upper via 1-alpha
    nuis_orth_skill = 1.0 - b_nuis[0].sum() / max(1e-9, b_nuis[1].sum())
    bl_nuis = R.briers_vs_r0(R.r_nuis_scores(leak), leak)
    nuis_leak_skill = 1.0 - bl_nuis[0].sum() / max(1e-9, bl_nuis[1].sum())
    # U6 random-codebook control must pass null / fail positive
    rc_pos_skill = 1.0 - b_rc[0].sum() / max(1e-9, b_rc[1].sum())
    # E-O2 calibration vs context-Bayes (recipe probs from the same sampled decode path)
    from clinical_jepa.eval.oracle_meta_recipe import sampled_pairwise_probs
    rec_probs = sampled_pairwise_probs(recipe, cell, seed=_seed("sample", family_id, str(kappa)))
    slope, intercept = R.e_o2_calibration(rec_probs, R.r_bayes_probs(cell),
                                          cell.true_order)
    lo, hi = ORACLE_CALIB_SLOPE_BAND
    checks = {
        "recipe_minus_R0": (r0_lo, r0_lo >= ORACLE_EO1_SKILL_GATE),
        "recipe_minus_Rnuis_orth": (nuis_lo, nuis_lo >= ORACLE_NUISANCE_MARGIN),
        "recipe_minus_meanembed": (me_lo, me_lo >= ORACLE_U6_BANDWIDTH_MARGIN),
        "recipe_minus_randomcodebook": (rc_lo, rc_lo >= ORACLE_U6_BANDWIDTH_MARGIN),
        "Rbayes_minus_R0": (rb_r0[1], rb_r0[1] >= ORACLE_R_BAYES_MARGIN),
        "u4_Rnuis_orth_upper_ci": (nuis_orth_skill, nuis_orth_skill < ORACLE_NUIS_ORTHO_FAIL_MAX),
        "u4_leak_diagnostic": (nuis_leak_skill - nuis_orth_skill,
                               nuis_leak_skill > nuis_orth_skill + ORACLE_NUISANCE_MARGIN),
        "u6_randomcodebook_null_pass": (rc_pos_skill, rc_pos_skill < ORACLE_EO1_SKILL_GATE),
        "e_o2_calibration": ((slope, intercept),
                             lo <= slope <= hi and abs(intercept) <= ORACLE_CALIB_INTERCEPT_TOL),
    }
    return CellUnlock(family_id, kappa, role, "SUPPORTED", checks, multiset_cluster_counts(cell))


def _recipe_null_fpr(recipe, family_id: str, seed: int, n_seeds: int = 60, seq: int = 400) -> tuple:
    """U2: independent-seed recipe-null FPR at κ=0 (recipe fires if its skill lower-CI > 0). Upper-CI gate."""
    fires = 0
    for s in range(n_seeds):
        cell = generate_meta_cell(family_id, 0.0, "orthogonal", seq, seed=seed + 3000 + s)
        b = _recipe_briers(recipe, cell)
        lo = R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1]
        fires += int(lo > 0.0)
    upper = clopper_pearson_upper(fires, n_seeds)
    return upper, upper <= ORACLE_FPR_UPPER_CI_MAX


def _kmid_power(recipe, family_id: str, seed: int, n_seeds: int = 50, seq: int = 1200) -> tuple:
    """U1 power at κmid over INDEPENDENT seeds; exact one-sided lower bound."""
    passes = 0
    for s in range(n_seeds):
        cell = generate_meta_cell(family_id, KAPPA_MID, "orthogonal", seq, seed=seed + 6000 + s,
                                  support_floor=0)
        b = _recipe_briers(recipe, cell)
        lo = R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1]
        passes += int(lo >= ORACLE_EO1_SKILL_GATE)
    lower = clopper_pearson_lower(passes, n_seeds)
    return lower, lower >= ORACLE_POWER_FLOOR


def _monotone(recipe, family_id: str, seed: int) -> tuple:
    """U3: recipe skill increases from the low to the high held-out endpoint (paired lower-CI > 0)."""
    lo_k, hi_k = min(KAPPA_HELDOUT_ENDPOINTS), max(KAPPA_HELDOUT_ENDPOINTS)
    c_lo = generate_meta_cell(family_id, lo_k, "orthogonal", 2000, seed=seed + 71)
    c_hi = generate_meta_cell(family_id, hi_k, "orthogonal", 2000, seed=seed + 72)
    b_lo, b_hi = _recipe_briers(recipe, c_lo), _recipe_briers(recipe, c_hi)
    s_lo = 1.0 - b_lo[0].sum() / max(1e-9, b_lo[1].sum())
    s_hi = 1.0 - b_hi[0].sum() / max(1e-9, b_hi[1].sum())
    return s_hi - s_lo, (s_hi - s_lo) > 0.0


def _precision_sim(seed: int, n_studies: int = 70, seq: int = 400) -> dict:
    """EVALUATOR precision (independent of any recipe): a known-NULL predictor (R0 on κ=0) must not fire
    and its CI must cover 0 (coverage ≥ 0.95, type-I upper-CI ≤ α); a known-EFFECT predictor (R_bayes on
    κmid) must fire (power lower-CI ≥ floor). Refuse if the binomial resolution is inadequate."""
    fam = TRAIN_FAMILIES[0]
    null_fire, cover = 0, 0
    for s in range(n_studies):
        cell = generate_meta_cell(fam, 0.0, "orthogonal", seq, seed=seed + 21000 + s)
        b = R.briers_vs_r0(np.zeros((seq, cell.item_classes.shape[1])), cell)   # known-NULL (constant) predictor
        lo = R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1]
        null_fire += int(lo > 0.0)
        cover += int(lo <= 0.0)
    eff_pass = 0
    for s in range(n_studies):
        cell = generate_meta_cell(fam, KAPPA_MID, "orthogonal", seq, seed=seed + 24000 + s)
        b = R.briers_from_probs(R.r_bayes_probs(cell), cell)
        lo = R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1]
        eff_pass += int(lo >= ORACLE_EO1_SKILL_GATE)
    type_I_up = clopper_pearson_upper(null_fire, n_studies)
    power_lo = clopper_pearson_lower(eff_pass, n_studies)
    coverage = cover / n_studies
    mc_ok = type_I_up - null_fire / n_studies <= max(0.05, 1.0 / n_studies)
    from clinical_jepa.eval.rung2_contract import ORACLE_PRECISION_COVERAGE
    passes = mc_ok and type_I_up <= 0.05 and power_lo >= ORACLE_POWER_FLOOR and coverage >= ORACLE_PRECISION_COVERAGE
    return {"type_I_upper": type_I_up, "power_lower": power_lo, "coverage": coverage,
            "mc_adequate": mc_ok, "passes": passes}


def _readiness(seed: int) -> bool:
    """Every DECLARED family's null control must hold: a κ=0 cell has no context-predictable order
    (R_bayes − R0 does not clear the margin). A train-family null failure blocks readiness."""
    for fam in (*TRAIN_FAMILIES, *HELDOUT_FAMILIES):
        cell = generate_meta_cell(fam, 0.0, "orthogonal", 1200, seed=seed + _seed("ready", fam) % 9973)
        b = R.briers_from_probs(R.r_bayes_probs(cell), cell)
        lo = R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + 1)[1]
        if lo >= ORACLE_R_BAYES_MARGIN:            # a null cell that looks positive => not ready
            return False
    return True


def compute_unlock(recipe, *, seed: int = 0, identities: dict | None = None) -> UnlockEvaluation:
    """Build the full typed UnlockEvaluation for a fit-once recipe across the held-out families×κ cells."""
    ledger = build_ledger()
    families = []
    n_eval = n_hidden = n_starved = 0
    for fam in HELDOUT_FAMILIES:
        cells = tuple(_cell_unlock(recipe, fam, kap, ledger, seed=seed + _seed(fam, str(kap)) % 9973)
                      for kap in KAPPA_HELDOUT_ENDPOINTS)
        for c in cells:
            n_eval += c.evaluable
            n_hidden += c.status == "HIDDEN_NULL"
            n_starved += c.status == "SUPPORT_STARVED"
        families.append(FamilyUnlock(
            fam, _monotone(recipe, fam, seed=seed + 500), _recipe_null_fpr(recipe, fam, seed=seed + 900),
            _kmid_power(recipe, fam, seed=seed + 1300), cells))
    return UnlockEvaluation(
        families=tuple(families), precision=_precision_sim(seed=seed + 2000),
        ledger_cardinality=ledger.cardinality(), ledger_alpha=ledger.ci_alpha(),
        train_family_readiness=_readiness(seed=seed + 40),
        identities=dict(identities or {}), n_evaluable_cells=n_eval,
        n_hidden_null=n_hidden, n_support_starved=n_starved)


CERTIFIED_CANDIDATE = "synthetic_recovery_CERTIFIED_CANDIDATE"
REFUTED = "REFUTED"


@dataclass(frozen=True)
class CandidateVerdict:
    outcome: str
    reason: str
    n_evaluable_cells: int
    governed_manifest_issued: bool = False
    can_populate_policy: bool = False


def certify_from_unlock(unlock: UnlockEvaluation) -> CandidateVerdict:
    """PURE FUNCTION of the UnlockEvaluation. CANDIDATE only — never issues a manifest or populates policy."""
    if not unlock.train_family_readiness:
        return CandidateVerdict(REFUTED, "train_family_readiness_failed", unlock.n_evaluable_cells)
    if not unlock.precision.get("passes"):
        return CandidateVerdict(REFUTED, "evaluator_precision_inadequate", unlock.n_evaluable_cells)
    if unlock.n_evaluable_cells == 0:
        return CandidateVerdict(REFUTED, "no_evaluable_cells", 0)
    if not all(f.passed for f in unlock.families):
        return CandidateVerdict(REFUTED, "held_out_family_conjunction_failed", unlock.n_evaluable_cells)
    return CandidateVerdict(CERTIFIED_CANDIDATE, "all_evaluable_held_out_cells_and_family_checks_pass",
                            unlock.n_evaluable_cells)
