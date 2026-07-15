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
    ORACLE_MDE_KAPPAS, ORACLE_MONO_KAPPAS, ORACLE_MONO_SPEARMAN, ORACLE_N_NULL_SEEDS, ORACLE_N_POS_SEEDS,
    ORACLE_NUIS_ORTHO_FAIL_MAX, ORACLE_NUISANCE_MARGIN, ORACLE_POWER_FLOOR, ORACLE_PRECISION_COVERAGE,
    ORACLE_PRECISION_N_STUDIES, ORACLE_R_BAYES_MARGIN, ORACLE_U6_BANDWIDTH_MARGIN, ORDER_SUPPORT_FLOOR,
)

# frozen monotonicity grid (U3, held-out family's own κ) + genuinely-disjoint reference MDE grid (#5);
# both are defined in the contract and hashed into the ledger identity.
MONO_KAPPAS = ORACLE_MONO_KAPPAS
MDE_KAPPAS = ORACLE_MDE_KAPPAS


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float); yr = np.argsort(np.argsort(y)).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    d = np.sqrt((xr ** 2).sum() * (yr ** 2).sum())
    return float((xr @ yr) / d) if d > 0 else 0.0
from clinical_jepa.eval.oracle_metrics import clopper_pearson_lower, clopper_pearson_upper

def _seed(*p: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(p).encode()).digest()[:4], "big")


def _rnd_tuple(t) -> list:
    """Round a (value, passed) result tuple for stable payload hashing (value → 9dp, flag → bool)."""
    return [round(float(t[0]), 9), bool(t[1])]


def _control_bits(recipe) -> int:
    """The matched bit budget the U6 controls use — taken from the REGISTERED recipe, not hard-coded."""
    return int(recipe.spec().bit_accounting.get("control_bits", 8))


from functools import lru_cache


@lru_cache(maxsize=8)
def _frozen_control_quants(bits: int):
    """Freeze the U6 control quantizer endpoints from a fixed TRAIN/DEV reference cell (Pi #2), so the
    controls are FIXED order-blind maps applied unchanged to every held-out/null cell — no per-cell
    bounds adaptation. Cached per bit budget."""
    dev = generate_meta_cell(TRAIN_FAMILIES[0], KAPPA_MID, "orthogonal", 1200, seed=_seed("ctrl_ref"))
    me_q = R.fit_frozen_quantizer(R.mean_embed_raw_scores(dev), bits)
    rc_q = R.fit_frozen_quantizer(R.random_codebook_raw_scores(dev, bits), bits)
    return me_q, rc_q


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
    power_mde: dict[str, Any]          # reference power curve + MDE (#5)
    ledger_cardinality: int
    ledger_alpha: float
    ledger_hash: str                   # content hash of the frozen hypothesis ledger (#6)
    train_family_readiness: bool
    identities: dict[str, Any]
    n_evaluable_cells: int
    n_hidden_null: int
    n_support_starved: int
    eval_access_trace: dict[str, Any] = field(default_factory=dict)   # held-out cells scored (Pi #4)

    def payload_hash(self) -> str:
        """Content hash of the COMPLETE evaluation payload (Pi #3) — bound to the registry outcome so a
        mutated field cannot be laundered through the pure verdict."""
        from clinical_jepa.eval.oracle_contracts import canonical_hash

        def _rnd(x):                                     # a check value[0] may be a scalar or a tuple
            if isinstance(x, (tuple, list)):
                return [round(float(e), 9) for e in x]
            return round(float(x), 9)

        return canonical_hash({
            "families": [(f.family_id, _rnd_tuple(f.u3_monotone), _rnd_tuple(f.u2_null_fpr),
                          _rnd_tuple(f.u1_kmid_power),
                          [(c.family_id, c.kappa, c.role, c.status,
                            {k: [_rnd(v[0]), bool(v[1])] for k, v in c.checks.items()})
                           for c in f.cells]) for f in self.families],
            "precision": {k: (round(v, 9) if isinstance(v, float) else v) for k, v in self.precision.items()},
            "power_mde": {"mde": self.power_mde.get("mde"),
                          "curve": {str(k): round(float(v), 9) for k, v in self.power_mde.get("power_curve", {}).items()}},
            "ledger": [self.ledger_cardinality, round(self.ledger_alpha, 12), self.ledger_hash],
            "readiness": self.train_family_readiness, "identities": self.identities,
            "counts": [self.n_evaluable_cells, self.n_hidden_null, self.n_support_starved],
            "eval_access_trace": self.eval_access_trace})


# ---- per-predictor briers on a cell ----
def _recipe_briers(recipe, cell):
    # decode via the recipe's REGISTERED stochastic sampler (exercises sampling end-to-end).
    from clinical_jepa.eval.oracle_meta_recipe import sampled_pairwise_probs
    probs = sampled_pairwise_probs(recipe, cell)
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
    # recipe + comparators (U6 controls use the recipe's REGISTERED matched control bits, quantized with
    # bounds FROZEN from a TRAIN/DEV reference — not adapted to this held-out cell, Pi #2).
    bits = _control_bits(recipe)
    me_q, rc_q = _frozen_control_quants(bits)
    b_rec = _recipe_briers(recipe, cell)
    b_nuis = R.briers_vs_r0(R.r_nuis_scores(cell), cell)
    b_me = R.briers_vs_r0(R.mean_embed_quantized_scores(cell, me_q), cell)
    rc_scores = R.random_codebook_scores(cell, rc_q)
    b_rc = R.briers_vs_r0(rc_scores, cell)                              # positive-restricted
    b_rc_all = R.briers_vs_r0(rc_scores, cell, positives_only=False)    # incl nulls, for the U6 null-pass
    b0, npair = b_rec[1], b_rec[2]

    def contrast(comp_briers, tag):
        return R.paired_skill_contrast(b_rec[0], comp_briers[0], b0, npair,
                                       base_seed=_seed(tag, family_id, str(kappa)), alpha=a)[1]

    r0_lo = R.paired_skill_contrast(b_rec[0], b0, b0, npair, base_seed=_seed("r0", family_id, str(kappa)), alpha=a)[1]
    nuis_lo = contrast(b_nuis, "nu")
    me_lo = contrast(b_me, "me")
    rc_lo = contrast(b_rc, "rc")

    def skill_upper(b, npair_use, tag):                     # UPPER CI of (predictor skill over R0)
        return -R.paired_skill_contrast(b[1], b[0], b[1], npair_use,
                                        base_seed=_seed(tag, family_id, str(kappa)), alpha=a)[1]

    # U4: orthogonal R_nuis skill UPPER-CI must be below the fail bound (not the point estimate) +
    # the correlated-leak diagnostic (R_nuis DOES capture leak in the leak cell).
    nuis_orth_upper = skill_upper(b_nuis, b_nuis[2], "nuup")
    bl_nuis = R.briers_vs_r0(R.r_nuis_scores(leak), leak)
    nuis_leak_skill = 1.0 - bl_nuis[0].sum() / max(1e-9, bl_nuis[1].sum())
    nuis_orth_skill = 1.0 - b_nuis[0].sum() / max(1e-9, b_nuis[1].sum())
    # U6: the random-codebook control must PASS NULL (its skill on the NULL sequences ~ 0, upper-CI
    # below the gate) AND FAIL POSITIVE (its skill on positives upper-CI below the gate).
    rc_null_np = np.where(cell.is_null, b_rc_all[2], 0)
    rc_null_upper = skill_upper(b_rc_all, rc_null_np, "rcnull")
    rc_pos_upper = skill_upper(b_rc, b_rc[2], "rcpos")
    # E-O2 calibration vs the EXACT context-Bayes π* (same sampled decode path)
    from clinical_jepa.eval.oracle_meta_recipe import sampled_pairwise_probs
    rec_probs = sampled_pairwise_probs(recipe, cell)
    slope, intercept = R.e_o2_calibration(rec_probs, R.r_bayes_probs(cell), cell.true_order)
    lo, hi = ORACLE_CALIB_SLOPE_BAND
    checks = {
        "recipe_minus_R0": (r0_lo, r0_lo >= ORACLE_EO1_SKILL_GATE),
        "recipe_minus_Rnuis_orth": (nuis_lo, nuis_lo >= ORACLE_NUISANCE_MARGIN),
        "recipe_minus_meanembed": (me_lo, me_lo >= ORACLE_U6_BANDWIDTH_MARGIN),
        "recipe_minus_randomcodebook": (rc_lo, rc_lo >= ORACLE_U6_BANDWIDTH_MARGIN),
        "Rbayes_minus_R0": (rb_r0[1], rb_r0[1] >= ORACLE_R_BAYES_MARGIN),
        "u4_Rnuis_orth_upper_ci": (nuis_orth_upper, nuis_orth_upper < ORACLE_NUIS_ORTHO_FAIL_MAX),
        "u4_leak_diagnostic": (nuis_leak_skill - nuis_orth_skill,
                               nuis_leak_skill > nuis_orth_skill + ORACLE_NUISANCE_MARGIN),
        "u6_randomcodebook_null_pass": (rc_null_upper, rc_null_upper < ORACLE_EO1_SKILL_GATE),
        "u6_randomcodebook_positive_fail": (rc_pos_upper, rc_pos_upper < ORACLE_EO1_SKILL_GATE),
        "e_o2_calibration": ((slope, intercept),
                             lo <= slope <= hi and abs(intercept) <= ORACLE_CALIB_INTERCEPT_TOL),
    }
    return CellUnlock(family_id, kappa, role, "SUPPORTED", checks, multiset_cluster_counts(cell))


def _recipe_null_fpr(recipe, family_id: str, seed: int, n_seeds: int = ORACLE_N_NULL_SEEDS,
                     seq: int = 400) -> tuple:
    """U2: recipe-null FPR at κ=0 over ORACLE_N_NULL_SEEDS INDEPENDENT seeds (recipe fires if skill
    lower-CI > 0); gate the one-sided 95% UPPER bound."""
    fires = 0
    for s in range(n_seeds):
        cell = generate_meta_cell(family_id, 0.0, "orthogonal", seq, seed=seed + 3000 + s)
        b = _recipe_briers(recipe, cell)
        fires += int(R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1] > 0.0)
    upper = clopper_pearson_upper(fires, n_seeds)
    return upper, upper <= ORACLE_FPR_UPPER_CI_MAX


def _kmid_power(recipe, family_id: str, seed: int, n_seeds: int = ORACLE_N_POS_SEEDS,
                seq: int = 1200) -> tuple:
    """U1 power at κmid over ORACLE_N_POS_SEEDS INDEPENDENT seeds; exact one-sided lower bound."""
    passes = 0
    for s in range(n_seeds):
        cell = generate_meta_cell(family_id, KAPPA_MID, "orthogonal", seq, seed=seed + 6000 + s)
        b = _recipe_briers(recipe, cell)
        passes += int(R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1] >= ORACLE_EO1_SKILL_GATE)
    lower = clopper_pearson_lower(passes, n_seeds)
    return lower, lower >= ORACLE_POWER_FLOOR


def _monotone(recipe, family_id: str, seed: int, n_boot: int = 1000) -> tuple:
    """U3: SPEARMAN rank correlation between κ and recipe skill over the frozen monotonicity grid, with a
    sequence-clustered bootstrap LOWER CI >= ORACLE_MONO_SPEARMAN (not a two-point point difference)."""
    per_k = []
    for k in MONO_KAPPAS:
        c = generate_meta_cell(family_id, k, "orthogonal", 2000, seed=seed + int(round(k * 1000)) + 71)
        b = _recipe_briers(recipe, c)
        keep = b[2] > 0
        per_k.append((b[0][keep], b[1][keep]))
    ks = np.array(MONO_KAPPAS, float)
    rng = np.random.default_rng(seed + 7)
    sp = np.empty(n_boot)
    for t in range(n_boot):
        skills = []
        for br, b0 in per_k:
            idx = rng.integers(0, br.shape[0], size=br.shape[0])
            skills.append(1.0 - br[idx].sum() / max(1e-9, b0[idx].sum()))
        sp[t] = _spearman(ks, np.array(skills))
    lower = float(np.quantile(sp, 0.05))
    return lower, lower >= ORACLE_MONO_SPEARMAN


def _precision_sim(seed: int, n_studies: int = ORACLE_PRECISION_N_STUDIES, seq: int = 400) -> dict:
    """EVALUATOR precision (recipe-independent): the KNOWN-NULL method is the EXACT R0 (true skill 0);
    the KNOWN-EFFECT method is the EXACT π* (true skill θ* estimated on a large sample). Per study a
    TWO-SIDED CI is formed; coverage = fraction covering the known truth; type-I = fraction the null CI
    lower bound exceeds 0; power = fraction the effect CI lower bound clears the gate. Refuse if the MC
    resolution is inadequate."""
    fam = TRAIN_FAMILIES[0]
    # θ*: large-sample true skill of π* over R0 at κmid.
    big = generate_meta_cell(fam, KAPPA_MID, "orthogonal", 12000, seed=seed + 999)
    bb = R.briers_from_probs(R.r_bayes_probs(big), big)
    keep = bb[2] > 0                                                    # POSITIVES only (consistent w/ studies)
    theta_star = 1.0 - bb[0][keep].sum() / max(1e-9, bb[1][keep].sum())
    null_fire = null_cover = eff_pass = eff_cover = 0
    for s in range(n_studies):
        cN = generate_meta_cell(fam, 0.0, "orthogonal", seq, seed=seed + 21000 + s)
        r0 = R.r0_pairwise(cN.family_id, cN.kappa, cN.item_classes)     # known-NULL method = exact R0
        bN = R.briers_from_probs(r0, cN)
        _, loN, upN, _ = R.pooled_eo1_skill(bN[0], bN[1], np.where(~cN.is_null, bN[2], 0), base_seed=seed + s)
        null_fire += int(loN > 0.0); null_cover += int(loN <= 0.0 <= upN)
        cE = generate_meta_cell(fam, KAPPA_MID, "orthogonal", seq, seed=seed + 24000 + s)
        bE = R.briers_from_probs(R.r_bayes_probs(cE), cE)              # known-EFFECT method = exact π*
        _, loE, upE, _ = R.pooled_eo1_skill(bE[0], bE[1], np.where(~cE.is_null, bE[2], 0), base_seed=seed + s)
        eff_pass += int(loE >= ORACLE_EO1_SKILL_GATE); eff_cover += int(loE <= theta_star <= upE)
    type_I_up = clopper_pearson_upper(null_fire, n_studies)
    power_lo = clopper_pearson_lower(eff_pass, n_studies)
    coverage = min(null_cover, eff_cover) / n_studies                  # worst-case two-sided coverage
    mc_ok = type_I_up - null_fire / n_studies <= max(0.05, 1.0 / n_studies)
    passes = (mc_ok and type_I_up <= 0.05 and power_lo >= ORACLE_POWER_FLOOR
              and coverage >= ORACLE_PRECISION_COVERAGE)
    return {"type_I_upper": type_I_up, "power_lower": power_lo, "coverage": coverage,
            "theta_star": theta_star, "mc_adequate": mc_ok, "passes": passes}


def _reference_power_curve_mde(seed: int, n_seeds: int = ORACLE_N_POS_SEEDS, seq: int = 800) -> dict:
    """Reference (π*) power curve over the frozen DISJOINT MDE grid + the MDE (smallest grid κ whose π*
    power lower-bound clears the floor). Evaluator characterization, recipe-independent (#5). Uses the
    full ORACLE_N_POS_SEEDS positive seeds, matching every other positive power point in the contract."""
    fam = TRAIN_FAMILIES[0]
    curve = {}
    for k in MDE_KAPPAS:
        p = 0
        for s in range(n_seeds):
            c = generate_meta_cell(fam, k, "orthogonal", seq, seed=seed + int(round(k * 1000)) * 97 + s)
            b = R.briers_from_probs(R.r_bayes_probs(c), c)
            p += int(R.paired_skill_contrast(b[0], b[1], b[1], b[2], base_seed=seed + s)[1] >= ORACLE_EO1_SKILL_GATE)
        curve[k] = clopper_pearson_lower(p, n_seeds)
    mde = next((k for k in sorted(MDE_KAPPAS) if curve[k] >= ORACLE_POWER_FLOOR), None)
    return {"power_curve": curve, "mde": mde}


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


def compute_unlock(recipe, *, evaluator_assignment, identities: dict | None = None) -> UnlockEvaluation:
    """Build the full typed UnlockEvaluation for a fit-once recipe across the REGISTRY-ASSIGNED held-out
    families × endpoint κ (Pi #4). The evaluator does NOT pick its own inventory — it consumes the
    evaluator assignment, derives every eval/OC RNG seed from its seed IDs, and records the held-out
    cells it scored as an external eval access trace (reconciled by the verdict)."""
    from clinical_jepa.eval.oracle_meta_recipe import registry_seed
    from clinical_jepa.eval.oracle_meta_ledger import ledger_hash
    ea = evaluator_assignment
    base = registry_seed(ea.seed_ids, "eval")
    ledger = build_ledger()
    families, scored = [], []
    n_eval = n_hidden = n_starved = 0
    for fam in ea.held_out_families:
        cells = tuple(_cell_unlock(recipe, fam, kap, ledger, seed=registry_seed(ea.seed_ids, "cell", fam, kap))
                      for kap in ea.endpoints)
        for c in cells:
            n_eval += c.evaluable
            n_hidden += c.status == "HIDDEN_NULL"
            n_starved += c.status == "SUPPORT_STARVED"
            scored.append([fam, float(c.kappa)])
        families.append(FamilyUnlock(
            fam, _monotone(recipe, fam, seed=base + 500), _recipe_null_fpr(recipe, fam, seed=base + 900),
            _kmid_power(recipe, fam, seed=base + 1300), cells))
    eval_trace = {"scored_cells": sorted(scored), "held_out_families": list(ea.held_out_families),
                  "endpoints": [float(k) for k in ea.endpoints],
                  "evaluator_assignment_hash": ea.assignment_hash()}
    return UnlockEvaluation(
        families=tuple(families), precision=_precision_sim(seed=base + 2000),
        power_mde=_reference_power_curve_mde(seed=base + 5000),
        ledger_cardinality=ledger.cardinality(), ledger_alpha=ledger.ci_alpha(),
        ledger_hash=ledger_hash(), train_family_readiness=_readiness(seed=base + 40),
        identities=dict(identities or {}), n_evaluable_cells=n_eval,
        n_hidden_null=n_hidden, n_support_starved=n_starved, eval_access_trace=eval_trace)


CERTIFIED_CANDIDATE = "synthetic_recovery_CERTIFIED_CANDIDATE"
REFUTED = "REFUTED"

REQUIRED_IDENTITY_KEYS = frozenset({
    "recipe_hash", "artifact_hash", "mechanism_hash", "evaluator_identity", "sampler_fingerprint",
    "bit_accounting", "split_assignment_hash", "seed_ids", "access_trace"})
EXPECTED_CELL_CHECKS = frozenset({
    "recipe_minus_R0", "recipe_minus_Rnuis_orth", "recipe_minus_meanembed", "recipe_minus_randomcodebook",
    "Rbayes_minus_R0", "u4_Rnuis_orth_upper_ci", "u4_leak_diagnostic", "u6_randomcodebook_null_pass",
    "u6_randomcodebook_positive_fail", "e_o2_calibration"})


@dataclass(frozen=True)
class CandidateVerdict:
    outcome: str
    reason: str
    n_evaluable_cells: int
    governed_manifest_issued: bool = False
    can_populate_policy: bool = False


def _identity_mismatch(ids: dict) -> str | None:
    """Pin every registry identity to its FROZEN expected value (Pi #3): non-emptiness is not enough —
    a plausible WRONG hash / contaminated trace must be refused, not laundered through the pure verdict."""
    from clinical_jepa.eval.oracle_meta_gen import invariant_hash
    from clinical_jepa.eval.oracle_meta_recipe import (
        RECIPE_CONTROL_BITS, RECIPE_TARGET_BITS, expected_sampler_fingerprint,
    )
    from clinical_jepa.eval.oracle_meta_gen import KAPPA_TRAIN_GRID
    from clinical_jepa.eval.rung2_contract import ORACLE_EVALUATOR_IDENTITY
    if not REQUIRED_IDENTITY_KEYS <= set(ids) or any(not ids.get(k) for k in REQUIRED_IDENTITY_KEYS):
        return "missing_identity_fields"
    if ids["mechanism_hash"] != invariant_hash():
        return "mechanism_hash_mismatch"
    if ids["evaluator_identity"] != ORACLE_EVALUATOR_IDENTITY:
        return "evaluator_identity_mismatch"
    if ids["sampler_fingerprint"] != expected_sampler_fingerprint():
        return "sampler_fingerprint_mismatch"
    ba = ids["bit_accounting"]
    if not (isinstance(ba, dict) and ba.get("target_bits") == RECIPE_TARGET_BITS
            and ba.get("control_bits") == RECIPE_CONTROL_BITS):
        return "bit_accounting_mismatch"
    seed_ids = ids["seed_ids"]
    if not (isinstance(seed_ids, (list, tuple)) and seed_ids and len(set(seed_ids)) == len(seed_ids)):
        return "seed_ids_malformed"
    # training access trace: train families only, no held-out / OC family, only train κ.
    tr = ids["access_trace"]
    fams, kaps = set(tr.get("families", [])), set(tr.get("kappas", []))
    if not fams or not fams <= set(TRAIN_FAMILIES) or fams & set(HELDOUT_FAMILIES):
        return "train_access_trace_contaminated"
    if not kaps <= set(KAPPA_TRAIN_GRID):
        return "train_access_trace_offgrid_kappa"
    if tr.get("split_assignment_hash") != ids["split_assignment_hash"]:
        return "split_assignment_hash_inconsistent"
    return None


def _eval_trace_mismatch(unlock: UnlockEvaluation) -> str | None:
    """The eval access trace must equal EXACTLY the canonical held-out family × endpoint inventory —
    the evaluator did not secretly score an unassigned family/κ (Pi #4)."""
    et = unlock.eval_access_trace
    if not et:
        return "eval_access_trace_missing"
    if set(et.get("held_out_families", [])) != set(HELDOUT_FAMILIES):
        return "eval_trace_family_inventory_mismatch"
    if set(et.get("endpoints", [])) != set(KAPPA_HELDOUT_ENDPOINTS):
        return "eval_trace_endpoint_inventory_mismatch"
    want = {(f, float(k)) for f in HELDOUT_FAMILIES for k in KAPPA_HELDOUT_ENDPOINTS}
    got = {(f, float(k)) for f, k in et.get("scored_cells", [])}
    if got != want:
        return "eval_trace_scored_cells_mismatch"
    return None


def _power_mde_mismatch(power_mde: dict) -> str | None:
    """Validate the reference power/MDE payload against the frozen grid + the MDE definition (Pi #3)."""
    curve = power_mde.get("power_curve", {})
    if set(curve) != set(MDE_KAPPAS):
        return "power_curve_grid_mismatch"
    passing = sorted(k for k in MDE_KAPPAS if curve[k] >= ORACLE_POWER_FLOOR)
    expected_mde = passing[0] if passing else None
    if power_mde.get("mde") != expected_mde:
        return "mde_not_smallest_passing_point"
    if expected_mde is None:
        return "evaluator_has_no_detectable_MDE"
    return None


def certify_from_unlock(unlock: UnlockEvaluation) -> CandidateVerdict:
    """PURE FUNCTION of the UnlockEvaluation. ENFORCES the frozen ledger, the PINNED registry identities
    (exact values, not mere non-emptiness), the eval access trace, the reconciled counts/inventory, and
    the reference power/MDE — before any pass. A fabricated or mutated UnlockEvaluation cannot certify
    (Pi #3/#6). CANDIDATE only; never issues a governed manifest or populates policy."""
    from clinical_jepa.eval.oracle_meta_ledger import build_ledger, ledger_hash
    led = build_ledger()
    n = unlock.n_evaluable_cells
    # (1) frozen ledger identity/cardinality/alpha
    if (unlock.ledger_hash != ledger_hash() or unlock.ledger_cardinality != led.cardinality()
            or abs(unlock.ledger_alpha - led.ci_alpha()) > 1e-12):
        return CandidateVerdict(REFUTED, "ledger_identity_mismatch", n)
    # (2) PINNED registry identities (exact expected values) + eval trace
    idm = _identity_mismatch(unlock.identities)
    if idm is not None:
        return CandidateVerdict(REFUTED, idm, n)
    etm = _eval_trace_mismatch(unlock)
    if etm is not None:
        return CandidateVerdict(REFUTED, etm, n)
    # (3) exact held-out family inventory + per-cell check inventory (no unexpected/missing)
    if {f.family_id for f in unlock.families} != set(HELDOUT_FAMILIES) or \
            len(unlock.families) != len(HELDOUT_FAMILIES):
        return CandidateVerdict(REFUTED, "family_inventory_mismatch", n)
    r_eval = r_hidden = r_starved = 0
    for f in unlock.families:
        if {c.kappa for c in f.cells} != set(KAPPA_HELDOUT_ENDPOINTS):
            return CandidateVerdict(REFUTED, "cell_inventory_mismatch", n)
        for c in f.cells:
            r_eval += c.evaluable
            r_hidden += c.status == "HIDDEN_NULL"
            r_starved += c.status == "SUPPORT_STARVED"
            if c.evaluable and set(c.checks) != EXPECTED_CELL_CHECKS:
                return CandidateVerdict(REFUTED, "check_inventory_mismatch", n)
    # (3b) reconcile the advertised counts against the ACTUAL nested cell records (Pi #3)
    if (r_eval, r_hidden, r_starved) != (unlock.n_evaluable_cells, unlock.n_hidden_null,
                                         unlock.n_support_starved):
        return CandidateVerdict(REFUTED, "count_reconciliation_mismatch", n)
    # (4) reference power/MDE + readiness + precision + family conjunction
    pm = _power_mde_mismatch(unlock.power_mde)
    if pm is not None:
        return CandidateVerdict(REFUTED, pm, n)
    if not unlock.train_family_readiness:
        return CandidateVerdict(REFUTED, "train_family_readiness_failed", n)
    if not unlock.precision.get("passes"):
        return CandidateVerdict(REFUTED, "evaluator_precision_inadequate", n)
    if n == 0:
        return CandidateVerdict(REFUTED, "no_evaluable_cells", 0)
    if not all(f.passed for f in unlock.families):
        return CandidateVerdict(REFUTED, "held_out_family_conjunction_failed", n)
    return CandidateVerdict(CERTIFIED_CANDIDATE, "all_evaluable_held_out_cells_and_family_checks_pass", n)
