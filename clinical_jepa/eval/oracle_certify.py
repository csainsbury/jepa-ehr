"""Evaluator-plumbing SMOKE runner (Pi #3/#4/#7/#8) — the ORDER_UNLOCK_CHECKS U1..U6.

Runs the frozen families through the generator + context-only evaluator and turns the reference
bracket into the property-specific order-unlock checks. This exercises the EVALUATOR/plumbing on a
KNOWN synthetic order signal: it recovers the signal, passes camouflaged nulls (with an upper-CI-bounded
false-positive OC study over INDEPENDENT null seeds), scales monotonically with coupling, ignores
orthogonal nuisance, and is not a raw-bandwidth artifact.

It does NOT certify a T4 recipe. The predictors here are hand-coded reference predictors, not a
registered candidate recipe's sampled ``D(zhat(context))`` — so the runner is deliberately
NON-CERTIFYING (Pi #8): it returns ``evaluator_plumbing_smoke_ok`` / ``certifies_recipe=False``, never
``synthetic_recovery_certified=True``. It issues NO governed authorization manifest and unlocks NO
governed T4. All computation is safe-public / fully synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval import oracle_evaluator as EV
from clinical_jepa.eval.oracle_generator import generate_cell
from clinical_jepa.eval.oracle_metrics import NullOCStudy, null_oc_from_fires, sequence_skill
from clinical_jepa.eval.oracle_spec import (
    KAPPA_TRAIN_GRID, get_family, heldout_families, train_families,
)
from clinical_jepa.eval.rung2_contract import (
    ORACLE_MONO_SPEARMAN, ORACLE_N_NULL_SEEDS, ORACLE_NULL_STUDY_SEQS,
    ORACLE_U6_BANDWIDTH_MARGIN, ORDER_UNLOCK_CHECKS,
)

N_CERT = 1200    # sequences per cell — sized so the Σ-orthogonal nuisance upper-CI collapses toward
                 # its true zero, comfortably under ORACLE_NUIS_ORTHO_FAIL_MAX (a 600-seq CI straddled it).


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    denom = np.sqrt((xr ** 2).sum() * (yr ** 2).sum())
    return float((xr @ yr) / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class FamilyCertification:
    family_id: str
    checks: dict[str, str]           # U-check -> "PASS" | "FAIL"
    passed: bool

    @property
    def all_pass(self) -> bool:
        return all(v == "PASS" for v in self.checks.values())


def _p(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def certify_family(family_id: str, *, seed: int = 0) -> FamilyCertification:
    """Run U1..U6 for one family at ITS OWN strongest coupling cell. Held-out families use their
    off-grid kappa_cells (Pi #5 — never silently re-evaluated on the train grid)."""
    fam = get_family(family_id)
    cells = fam.kappa_cells                          # the family's OWN grid (off-grid for held-out)
    kap = max(cells)
    pos = generate_cell(family_id, kap, "orthogonal", N_CERT, seed=seed)
    leak = generate_cell(family_id, kap, "correlated_leak", N_CERT, seed=seed + 1)
    rb = EV.reference_bracket(pos, seed=seed)

    # U1: context predictor recovers order above the PRACTICAL gate; beats R0.
    u1 = rb.R_bayes_beats_R0
    # U2: nulls do not fire (context predictor + R0). Certification-level FPR upper-CI is the OC study
    # in certify_evaluator_smoke; here U2 is the per-family diagnostic (surfaced, not silently dropped).
    u2 = (not rb.R_bayes_null.fires) and rb.R0_null_pass
    # U3: order-skill increases monotonically with coupling over the family's OWN cells.
    curve = np.array([EV.reference_bracket(generate_cell(family_id, k, "orthogonal", N_CERT, seed=seed + 7),
                                           seed=seed).R_bayes_pos.mean_skill for k in cells])
    u3 = _spearman(np.asarray(cells, dtype=float), curve) >= ORACLE_MONO_SPEARMAN
    # U4: nuisance loses incremental skill in the orthogonal cell, captures bounded leak in the leak cell.
    u4 = EV.nuisance_incremental_ok(pos, leak, seed=seed + 3)
    # U6: proper context predictor beats the bandwidth-matched control by the frozen margin.
    proper = EV._skill(EV.predict_context(pos), pos, want_null=False, seed=seed + 5)
    ctrl = EV._skill(EV.predict_context_bandwidth_control(pos), pos, want_null=False, seed=seed + 6)
    u6 = (proper.lower_ci - ctrl.upper_ci) >= ORACLE_U6_BANDWIDTH_MARGIN

    checks = {"U1_order_recovery": _p(u1), "U2_null": _p(u2), "U3_monotone": _p(u3),
              "U4_nuisance_incremental": _p(u4), "U6_bandwidth_fair": _p(u6)}
    assert set(checks) == set(ORDER_UNLOCK_CHECKS), "U-check set must match the contract exactly"
    return FamilyCertification(family_id, checks, passed=all(v == "PASS" for v in checks.values()))


def null_oc_study(*, seed: int = 0, n_seeds: int = ORACLE_N_NULL_SEEDS,
                  seqs_per_study: int = ORACLE_NULL_STUDY_SEQS) -> NullOCStudy:
    """False-positive OC study over INDEPENDENT null seeds (Pi #4). Each study is a freshly-seeded
    all-null population (kappa=0 => order carries NO context signal); the context predictor's
    sequence-level fire rule is applied per study. Returns the point FPR AND a one-sided 95% UPPER
    confidence bound — the gate is the UPPER bound, not the point rate. Studies rotate across families
    so the calibration is not specific to one mechanism instance."""
    fams = [*train_families(), *heldout_families()]
    fires: list[bool] = []
    for s in range(n_seeds):
        fam = fams[s % len(fams)]
        cell = generate_cell(fam.family_id, 0.0, "orthogonal", seqs_per_study, seed=seed + 10_000 + s)
        pred = EV.predict_context(cell)
        preds = [pred[i] for i in range(pred.shape[0])]
        trues = [cell.s_true[i] for i in range(cell.s_true.shape[0])]
        fires.append(sequence_skill(preds, trues, base_seed=seed + 20_000 + s).fires)
    return null_oc_from_fires(fires)


def certify_evaluator_smoke(*, seed: int = 0) -> dict:
    """Exercise the evaluator/plumbing across all frozen families. NON-CERTIFYING (Pi #8): reports
    whether the machinery recovers a known synthetic signal and stays calibrated on nulls. The
    held-out conjunction (Pi #3) and a GLOBAL null-control requirement (every declared family's U2 must
    pass — a train-family null failure is diagnostic, never silently dropped) gate the smoke result."""
    train = {f.family_id: certify_family(f.family_id, seed=seed) for f in train_families()}
    held = {f.family_id: certify_family(f.family_id, seed=seed + 50) for f in heldout_families()}
    all_fams = {**train, **held}
    heldout_conjunction = all(fc.passed for fc in held.values())
    # a check passes overall only if it passes in EVERY held-out family (Pi #3 conjunction)...
    unlock = {c: _p(all(fc.checks[c] == "PASS" for fc in held.values())) for c in ORDER_UNLOCK_CHECKS}
    # ...and NO declared family (train OR held-out) may fail the null control (Pi #4).
    null_control_global = all(fc.checks["U2_null"] == "PASS" for fc in all_fams.values())
    oc = null_oc_study(seed=seed)
    smoke_ok = (heldout_conjunction and all(v == "PASS" for v in unlock.values())
                and null_control_global and oc.passes)
    return {
        "evaluator_plumbing_smoke_ok": smoke_ok,
        "certifies_recipe": False,      # hand-coded reference predictor, NOT a registered T4 recipe
        "family_conjunction": "all_held_out_families_must_pass",
        "null_control": "all_declared_families_U2_must_pass",
        "null_control_global_pass": null_control_global,
        "train_families": {k: fc.checks for k, fc in train.items()},
        "held_out_families": {k: fc.checks for k, fc in held.items()},
        "unlock_checks": unlock,
        "null_oc_study": {"n_studies": oc.n_studies, "n_fired": oc.n_fired,
                          "point_fpr": oc.point_fpr, "upper_ci": oc.upper_ci, "passes": oc.passes},
        "governed_manifest_issued": False,      # NEVER here — needs the committed policy + Pi gate
        "note": "evaluator/plumbing smoke on hand-coded reference predictors; does NOT certify a T4 "
                "recipe; NOMINATE ceiling; no governed T4 unlocked.",
    }
