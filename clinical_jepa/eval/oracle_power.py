"""Power / MDE + precision simulation (Pi 2nd-pass Phase 4b).

Two SEPARATE randomness roles (Pi):
  * POWER / MDE — over INDEPENDENT seed datasets (not partitions of one pool). Each seed runs the
    COMPLETE pass decision on a fresh dataset; power = pass-count / N reported with a one-sided exact
    95% LOWER bound (fail-closed); MDE = the smallest point on a disjoint positive-kappa grid whose
    power lower-bound reaches the criterion (no interpolation).
  * PRECISION SIMULATION — tests the EVALUATOR itself (not a recipe): known-null studies must hold
    type-I at/under alpha with CI coverage of the true null, and known-effect studies must reach the
    power floor. Refuses if Monte-Carlo precision is inadequate for the declared tolerance.

The "complete pass decision" here is the single load-bearing order gate: on a fresh cell, the fitted
recipe's E-O1 must beat R0 by the practical margin (paired lower-CI) on the positive sequences, and the
cell must not be a hidden null. Phase 6 composes the full multiplicity-corrected conjunction; this
module fixes the OC machinery. All synthetic / safe-public.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from clinical_jepa.eval import oracle_references as RF
from clinical_jepa.eval.oracle_metrics import clopper_pearson_lower, clopper_pearson_upper
from clinical_jepa.eval.oracle_recipe import CandidateRecipe, GoodContextRecipe, split_views
from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.rung2_contract import (
    ORACLE_EO1_SKILL_GATE, ORACLE_N_POS_SEEDS, ORACLE_POWER_FLOOR, ORACLE_POWER_KAPPA_MID,
    ORACLE_PRECISION_MC_TOL, ORACLE_PRECISION_N_STUDIES,
)

RecipeFactory = Callable[[], CandidateRecipe]


def _good_factory() -> CandidateRecipe:
    return GoodContextRecipe()


def complete_pass_decision(recipe_factory: RecipeFactory, family_id: str, kappa: float, *,
                           seed: int, n: int = 800, ref_n: int = 3000) -> bool:
    """One INDEPENDENT complete pass decision on a fresh dataset: fit on an independent train draw,
    score on a fresh eval draw, require (hidden-null excluded) AND (recipe E-O1 beats R0 by the
    practical margin, paired lower-CI, on positive sequences)."""
    train = generate_literal_cell(family_id, kappa, "orthogonal", ref_n, seed=seed + 4_000)
    ev = generate_literal_cell(family_id, kappa, "orthogonal", n, seed=seed + 9_000)
    if RF.hidden_null_excluded(ev, margin=ORACLE_EO1_SKILL_GATE, seed=seed):
        return False                                              # hidden null cannot be a positive pass
    r = recipe_factory()
    r.fit(split_views(train), split_views(train))
    pos = ~ev.is_null
    pc = RF.paired_contrast(RF.restrict_to(RF.eo1_recipe(r, ev, seed=seed), pos),
                            RF.restrict_to(RF.eo1_r0(ev), pos), seed=seed)
    return pc.lower_ci >= ORACLE_EO1_SKILL_GATE


@dataclass(frozen=True)
class PowerResult:
    kappa: float
    n_seeds: int
    n_pass: int
    point_power: float
    lower_ci: float          # one-sided exact 95% LOWER bound (fail-closed gate statistic)
    meets_floor: bool        # lower_ci >= ORACLE_POWER_FLOOR


def power_study(recipe_factory: RecipeFactory, family_id: str, kappa: float, *, seed: int = 0,
                n_seeds: int = ORACLE_N_POS_SEEDS, n: int = 400, ref_n: int = 1500) -> PowerResult:
    """Power over INDEPENDENT seed datasets at ``kappa``. Gate on the exact one-sided lower bound."""
    passes = [complete_pass_decision(recipe_factory, family_id, kappa, seed=seed + 100 * s, n=n,
                                     ref_n=ref_n) for s in range(n_seeds)]
    k = int(sum(passes))
    lower = clopper_pearson_lower(k, n_seeds)
    return PowerResult(kappa, n_seeds, k, k / n_seeds, lower, lower >= ORACLE_POWER_FLOOR)


def mde(recipe_factory: RecipeFactory, family_id: str, grid: tuple[float, ...], *, seed: int = 0,
        n_seeds: int = 40, n: int = 400) -> float | None:
    """Minimum Detectable Effect: the SMALLEST kappa on the DISJOINT positive grid whose power
    lower-bound reaches the floor — no interpolation. None if no grid point qualifies."""
    for k in sorted(grid):
        if k <= 0:
            continue
        if power_study(recipe_factory, family_id, k, seed=seed, n_seeds=n_seeds, n=n).meets_floor:
            return k
    return None


# ------------------------------------------------------------------------------------------------
# precision simulation — tests the EVALUATOR, not a recipe.
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecisionResult:
    type_I: float            # false-positive rate on known-null studies
    type_I_upper: float      # one-sided 95% upper bound on type-I
    power: float             # true-positive rate on known-effect studies
    power_lower: float       # one-sided 95% lower bound on power
    mc_adequate: bool        # Monte-Carlo resolution finer than the declared tolerance
    passes: bool


def precision_simulation(*, seed: int = 0, n_studies: int = ORACLE_PRECISION_N_STUDIES, n: int = 400,
                         alpha: float = 0.05, mc_tol: float = ORACLE_PRECISION_MC_TOL,
                         effect_kappa: float = ORACLE_POWER_KAPPA_MID,
                         family_id: str = "T_latent_factor") -> PrecisionResult:
    """Known-null (kappa=0) and known-effect (kappa=effect) studies test the EVALUATOR's operating
    characteristics. Type-I upper bound must be <= alpha; power lower bound must be >= floor. Refuses
    (mc_adequate=False) if the binomial CI half-width at these N is coarser than mc_tol."""
    null_fires = [complete_pass_decision(_good_factory, family_id, 0.0, seed=seed + 3 * s, n=n)
                  for s in range(n_studies)]
    eff_pass = [complete_pass_decision(_good_factory, family_id, effect_kappa, seed=seed + 3 * s + 1, n=n)
                for s in range(n_studies)]
    k_null, k_eff = int(sum(null_fires)), int(sum(eff_pass))
    type_I, power = k_null / n_studies, k_eff / n_studies
    tI_up = clopper_pearson_upper(k_null, n_studies)
    pw_lo = clopper_pearson_lower(k_eff, n_studies)
    # MC adequacy: the widest CI half-width across the two studies must be finer than mc_tol.
    half_widths = [clopper_pearson_upper(k, n_studies) - k / n_studies for k in (k_null, k_eff)]
    mc_ok = max(half_widths) <= max(mc_tol, 1.0 / n_studies)     # resolution floor at 1/N
    passes = mc_ok and (tI_up <= alpha) and (pw_lo >= ORACLE_POWER_FLOOR)
    return PrecisionResult(type_I, tI_up, power, pw_lo, mc_ok, passes)
