"""Whole-pass certification verdict (Pi 2nd-pass Phase 6).

Composes the boundary guards + per-cell paired contrasts (E-O1 over R0, incremental over R_nuis, and
both U6 controls) with Bonferroni over the evaluable held-out cells and family-level conjunction. The
output is a CANDIDATE aggregate result only — it can NEVER populate the committed policy or issue a
governed manifest; those are separate, explicitly-gated stages. Safe-public / synthetic.

Order of operations (Pi):
  1. Boundary: predict is context-only and fit reads no labels — else REFUTED.
  2. Support: a support-starved cell is NOT_EVALUABLE (fixed BEFORE recipe scoring; cannot alter
     multiplicity).
  3. Hidden-null: decided from R_bayes − R0 BEFORE the recipe is inspected (excluded, not a failure).
  4. Per evaluable cell (Bonferroni α over the evaluable family): recipe−R0 ≥ EO1 gate,
     recipe−R_nuis ≥ nuisance margin, recipe−mean-embed and recipe−random-codebook ≥ U6 margin, and
     the random-codebook control itself passes null / fails positive (else U6 not-evaluable).
  5. Family conjunction: every evaluable held-out family must pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from clinical_jepa.eval import oracle_references as RF
from clinical_jepa.eval.oracle_recipe import CandidateRecipe, split_views
from clinical_jepa.eval.oracle_literal_gen import generate_literal_cell
from clinical_jepa.eval.oracle_spec import heldout_families, train_families
from clinical_jepa.eval.rung2_contract import (
    ORACLE_EO1_SKILL_GATE, ORACLE_NUISANCE_MARGIN, ORACLE_U6_BANDWIDTH_MARGIN,
)

RecipeFactory = Callable[[], CandidateRecipe]

CERTIFIED_CANDIDATE = "synthetic_recovery_CERTIFIED_CANDIDATE"
REFUTED = "REFUTED"
NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True)
class CellVerdict:
    family_id: str
    status: str                          # "PASS" | "FAIL" | "NOT_EVALUABLE" | "HIDDEN_NULL"
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeVerdict:
    outcome: str                         # CERTIFIED_CANDIDATE | REFUTED
    reason: str
    cells: tuple[CellVerdict, ...]
    governed_manifest_issued: bool = False    # ALWAYS False here
    can_populate_policy: bool = False         # ALWAYS False here


def _cell_pass(recipe_factory: RecipeFactory, fam_id: str, kappa: float, seed: int, n: int,
               ref_n: int, bonf: int) -> CellVerdict:
    # the recipe is fit on this family's OWN independent train split (each held-out family has a
    # DISTINCT frozen mechanism, so a single fitted map cannot transfer — certification adapts per cell).
    train = generate_literal_cell(fam_id, kappa, "orthogonal", ref_n, seed=seed + 6000)
    recipe = recipe_factory()
    recipe.fit(split_views(train), split_views(train))
    cell = generate_literal_cell(fam_id, kappa, "orthogonal", n, seed=seed + 1)
    if cell.support_status != "SUPPORTED":
        return CellVerdict(fam_id, NOT_EVALUABLE, {"support": cell.support_status})
    if RF.hidden_null_excluded(cell, margin=ORACLE_EO1_SKILL_GATE, seed=seed):
        return CellVerdict(fam_id, "HIDDEN_NULL", {})           # excluded, not a failure
    leak = generate_literal_cell(fam_id, kappa, "correlated_leak", n, seed=seed + 2)
    pos, pos_l = ~cell.is_null, ~leak.is_null
    alpha = 0.05 / max(1, bonf)                                # Bonferroni over evaluable cells

    recipe_v = RF.eo1_recipe(recipe, cell, seed=seed)
    c_r0 = RF.paired_contrast(RF.restrict_to(recipe_v, pos), RF.restrict_to(RF.eo1_r0(cell), pos),
                              seed=seed, alpha=alpha)
    # INCREMENTAL over nuisance in the Σ-ORTHOGONAL cell (R_nuis carries no order info there): the
    # recipe's skill must not be nuisance-attributable. The LEAK cell is a separate diagnostic that
    # R_nuis genuinely captures the planted leak (so the orthogonal null is meaningful).
    c_nu = RF.paired_contrast(RF.restrict_to(recipe_v, pos), RF.restrict_to(RF.eo1_r_nuis(cell), pos),
                              seed=seed, alpha=alpha)
    rnuis_leak_skill = float(np.nanmean(RF.eo1_r_nuis(leak)[pos_l]))
    rnuis_orth_skill = float(np.nanmean(RF.eo1_r_nuis(cell)[pos]))
    rnuis_captures_leak = rnuis_leak_skill > (rnuis_orth_skill + ORACLE_NUISANCE_MARGIN)
    # U6: recipe beats BOTH bandwidth-matched controls; the random codebook must pass-null/fail-positive.
    c_me = RF.paired_contrast(RF.restrict_to(recipe_v, pos),
                              RF.restrict_to(RF.eo1_mean_embed_quantized(cell), pos), seed=seed, alpha=alpha)
    c_rc = RF.paired_contrast(RF.restrict_to(recipe_v, pos),
                              RF.restrict_to(RF.eo1_random_codebook(cell, seed=seed), pos), seed=seed, alpha=alpha)
    rc_positive = float(np.nanmean(RF.eo1_random_codebook(cell, seed=seed)[pos]))
    u6_evaluable = rc_positive < ORACLE_EO1_SKILL_GATE         # control must NOT itself predict positives
    checks = {
        "recipe_minus_R0": (c_r0.lower_ci, c_r0.lower_ci >= ORACLE_EO1_SKILL_GATE),
        "recipe_minus_Rnuis_orthogonal": (c_nu.lower_ci, c_nu.lower_ci >= ORACLE_NUISANCE_MARGIN),
        "Rnuis_captures_leak": (rnuis_leak_skill - rnuis_orth_skill, rnuis_captures_leak),
        "recipe_minus_meanembed": (c_me.lower_ci, c_me.lower_ci >= ORACLE_U6_BANDWIDTH_MARGIN),
        "recipe_minus_randomcodebook": (c_rc.lower_ci, c_rc.lower_ci >= ORACLE_U6_BANDWIDTH_MARGIN),
        "u6_control_evaluable": (rc_positive, u6_evaluable),
    }
    status = "PASS" if all(ok for _, ok in checks.values()) else "FAIL"
    return CellVerdict(fam_id, status, {k: v for k, v in checks.items()})


def certify_recipe_candidate(recipe_factory: RecipeFactory, *, kappa: float = 0.60, seed: int = 0,
                             n: int = 2500, ref_n: int = 4000) -> RecipeVerdict:
    """Whole-pass CANDIDATE verdict on the held-out families. Never issues a governed manifest / never
    populates policy. A boundary-guard failure short-circuits to REFUTED."""
    fams = [f.family_id for f in heldout_families()]
    train = generate_literal_cell(fams[0], kappa, "orthogonal", ref_n, seed=seed + 7000)
    probe = generate_literal_cell(fams[0], kappa, "orthogonal", n, seed=seed + 8000)
    fitted = recipe_factory()
    fitted.fit(split_views(train), split_views(probe))
    if not _boundary_ok(recipe_factory, fitted, probe, train):
        return RecipeVerdict(REFUTED, "boundary_guard_failed", ())

    # evaluability determined BEFORE scoring; Bonferroni factor = number of evaluable cells.
    prelim = []
    for fam in fams:
        c = generate_literal_cell(fam, kappa, "orthogonal", n, seed=seed + 1)
        starved = c.support_status != "SUPPORTED"
        hidden = (not starved) and RF.hidden_null_excluded(c, margin=ORACLE_EO1_SKILL_GATE, seed=seed)
        prelim.append((fam, starved, hidden))
    bonf = sum(1 for _, s, h in prelim if not s and not h)
    cells = []
    for fam, starved, hidden in prelim:
        if starved:
            cells.append(CellVerdict(fam, NOT_EVALUABLE, {"support": "SUPPORT_STARVED"}))
        elif hidden:
            cells.append(CellVerdict(fam, "HIDDEN_NULL", {}))
        else:
            cells.append(_cell_pass(recipe_factory, fam, kappa, seed, n, ref_n, bonf))
    scored = [c for c in cells if c.status in ("PASS", "FAIL")]
    outcome = (CERTIFIED_CANDIDATE if scored and all(c.status == "PASS" for c in scored) else REFUTED)
    reason = "all_evaluable_held_out_families_pass" if outcome == CERTIFIED_CANDIDATE else "a_family_failed_or_none_evaluable"
    return RecipeVerdict(outcome, reason, tuple(cells))


def _boundary_ok(recipe_factory: RecipeFactory, fitted: CandidateRecipe, probe, train) -> bool:
    from clinical_jepa.eval.oracle_recipe import assert_labels_eval_only, assert_predictor_context_only
    return assert_predictor_context_only(fitted, probe) and \
        assert_labels_eval_only(recipe_factory(), train, probe)


def evaluator_ready(*, seed: int = 0, n: int = 700) -> bool:
    """A declared TRAIN family failing its null control blocks evaluator readiness even if the held-outs
    happen to pass (Pi acceptance matrix). Checks each declared family's null control at kappa=0."""
    for f in [*train_families(), *heldout_families()]:
        cell = generate_literal_cell(f.family_id, 0.0, "orthogonal", n, seed=seed + hash(f.family_id) % 97)
        # a true kappa=0 cell must be flagged hidden-null (no positive signal) — the null control holds.
        if not RF.hidden_null_excluded(cell, margin=ORACLE_EO1_SKILL_GATE, seed=seed):
            return False
    return True
