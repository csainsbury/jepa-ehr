"""Oracle EVALUATOR + reference bracket (Pi #3/#7).

All certification-side prediction is CONTEXT-ONLY: predictors read x_ctx (and item features f), never
the label s_true. The reference bracket frames the achievable range:
  * R_bayes  — Bayes-optimal CONTEXT predictor E[s | x_ctx, f]; the FAIR ceiling.
  * R0       — content-prior floor (no context); expected order-skill 0.
  * R_nuis   — nuisance-only predictor; must LOSE incremental skill in Σ-orthogonal cells and only
               partially capture leakage in correlated-leak cells (never exceeding R_bayes).

Negative-control predictors (for the Pi #7 end-to-end tests):
  * context-blind — a perfect target-side coupling but with the context ZEROED => must fail.
  * shortcut      — reads the h-leak channel only => must fail the no-h family.

Decision margins (ORACLE_EO1_SKILL_GATE, ORACLE_R0_POSITIVE_FAIL_MAX, ORACLE_NUIS_ORTHO_FAIL_MAX,
ORACLE_SHORTCUT_MAX_SKILL) turn skills into PASS/FAIL; a statistically nonzero but negligible effect
does not authorize anything (Pi #3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.oracle_generator import GeneratedCell, mechanism_matrices
from clinical_jepa.eval.oracle_metrics import SkillResult, sequence_skill
from clinical_jepa.eval.rung2_contract import (
    ORACLE_EO1_SKILL_GATE, ORACLE_NUIS_ORTHO_FAIL_MAX, ORACLE_R0_POSITIVE_FAIL_MAX,
    ORACLE_SHORTCUT_MAX_SKILL,
)


def _order_score(driver_like: np.ndarray, M: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-item order-score from a driver estimate and item features: ŝ_k = driver^T M f_k."""
    return np.einsum("nd,de,nle->nl", driver_like, M, f)


def _split(pred: np.ndarray, cell: GeneratedCell, *, want_null: bool) -> tuple[list, list]:
    mask = cell.is_null if want_null else ~cell.is_null
    idx = np.nonzero(mask)[0]
    return [pred[i] for i in idx], [cell.s_true[i] for i in idx]


def _skill(pred: np.ndarray, cell: GeneratedCell, *, want_null: bool, seed: int) -> SkillResult:
    p, t = _split(pred, cell, want_null=want_null)
    if not p:
        return SkillResult(0.0, 0.0, 0.0, 0, 0, False)
    return sequence_skill(p, t, base_seed=seed)


# ---- predictors (all read ONLY the declared channels; none read s_true) ----
def predict_context(cell: GeneratedCell) -> np.ndarray:
    """Bayes-optimal CONTEXT predictor: reconstruct the driver from x_ctx, score items with f."""
    mech = mechanism_matrices(cell.family_id)
    driver_hat = cell.x_ctx @ mech.A.T
    return _order_score(driver_hat, mech.M, cell.f)


def predict_content_prior(cell: GeneratedCell) -> np.ndarray:
    """R0: no context — constant score => all pairs predicted-tie => tau 0."""
    return np.zeros((cell.x_ctx.shape[0], cell.f.shape[1]))


def predict_nuisance(cell: GeneratedCell) -> np.ndarray:
    """R_nuis: rank items by the nuisance channel only."""
    return cell.u.copy()


def predict_shortcut(cell: GeneratedCell) -> np.ndarray:
    """h-projection shortcut: use the h-leak channel as the driver estimate (no proper context posterior)."""
    mech = mechanism_matrices(cell.family_id)
    return _order_score(cell.x_hleak, mech.M, cell.f)


def predict_context_blind(cell: GeneratedCell) -> np.ndarray:
    """Perfect coupling but CONTEXT-BLIND: driver estimated from a ZEROED context => ~0 => tau 0."""
    mech = mechanism_matrices(cell.family_id)
    driver_hat = np.zeros((cell.x_ctx.shape[0], mech.A.shape[0]))
    return _order_score(driver_hat, mech.M, cell.f)


def _stable_seed(*parts: str) -> int:
    """Deterministic seed from a cryptographic hash — reproducible across processes (unlike Python's
    randomized ``hash()``, which varies with PYTHONHASHSEED; Pi #7 reproduced that non-determinism)."""
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


def predict_context_bandwidth_control(cell: GeneratedCell) -> np.ndarray:
    """Bandwidth-matched control (U6): same context bit budget, but a RANDOM orthogonal posterior map
    that destroys the driver->context alignment. Skill here would mean order is being read from raw
    context bandwidth rather than correct context decoding; the proper predictor must beat it."""
    mech = mechanism_matrices(cell.family_id)
    rng = np.random.default_rng(_stable_seed("u6_bandwidth_control", cell.family_id))
    q, _ = np.linalg.qr(rng.standard_normal((mech.A.shape[1], mech.A.shape[1])))
    driver_hat = (cell.x_ctx @ q) @ mech.A.T                  # rotate context before decoding
    return _order_score(driver_hat, mech.M, cell.f)


@dataclass(frozen=True)
class ReferenceBracket:
    R_bayes_pos: SkillResult
    R_bayes_null: SkillResult
    R0_pos: SkillResult
    R0_null: SkillResult
    R_nuis_pos: SkillResult
    R_nuis_null: SkillResult

    # ---- decision predicates (Pi #3) ----
    @property
    def R_bayes_beats_R0(self) -> bool:
        return self.R_bayes_pos.lower_ci >= ORACLE_EO1_SKILL_GATE and \
            self.R_bayes_pos.lower_ci > self.R0_pos.upper_ci

    @property
    def R0_null_pass(self) -> bool:                        # R0 does not spuriously fire on nulls
        return self.R0_null.upper_ci < ORACLE_R0_POSITIVE_FAIL_MAX

    @property
    def R0_positive_fail(self) -> bool:                    # R0 alone cannot explain positives
        return self.R0_pos.upper_ci < ORACLE_R0_POSITIVE_FAIL_MAX


def reference_bracket(cell: GeneratedCell, *, seed: int = 0) -> ReferenceBracket:
    ctx, r0, nu = predict_context(cell), predict_content_prior(cell), predict_nuisance(cell)
    return ReferenceBracket(
        R_bayes_pos=_skill(ctx, cell, want_null=False, seed=seed),
        R_bayes_null=_skill(ctx, cell, want_null=True, seed=seed + 1),
        R0_pos=_skill(r0, cell, want_null=False, seed=seed + 2),
        R0_null=_skill(r0, cell, want_null=True, seed=seed + 3),
        R_nuis_pos=_skill(nu, cell, want_null=False, seed=seed + 4),
        R_nuis_null=_skill(nu, cell, want_null=True, seed=seed + 5),
    )


def nuisance_incremental_ok(orthogonal_cell: GeneratedCell, leak_cell: GeneratedCell,
                            *, seed: int = 0) -> bool:
    """R_nuis must LOSE incremental order-skill in Σ-orthogonal cells (< ORACLE_NUIS_ORTHO_FAIL_MAX)
    while capturing SOME (bounded) leakage in the correlated-leak cell."""
    orth = _skill(predict_nuisance(orthogonal_cell), orthogonal_cell, want_null=False, seed=seed)
    leak = _skill(predict_nuisance(leak_cell), leak_cell, want_null=False, seed=seed + 10)
    return orth.upper_ci < ORACLE_NUIS_ORTHO_FAIL_MAX and leak.lower_ci > 0.0


def shortcut_fails_no_h(no_h_cell: GeneratedCell, *, seed: int = 0) -> bool:
    """The h-projection shortcut must NOT exceed ORACLE_SHORTCUT_MAX_SKILL on a NO-h family."""
    sc = _skill(predict_shortcut(no_h_cell), no_h_cell, want_null=False, seed=seed)
    return sc.upper_ci < ORACLE_SHORTCUT_MAX_SKILL
