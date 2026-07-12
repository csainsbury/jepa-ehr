"""Synthetic-recovery CERTIFICATION runner (Pi #3/#7/#8) — the ORDER_UNLOCK_CHECKS U1..U6.

Runs the frozen families through the generator + context-only evaluator and turns the reference
bracket into the property-specific order-unlock checks. This certifies SYNTHETIC RECOVERY only: it
proves the certification machinery recovers a KNOWN synthetic order signal, passes camouflaged nulls,
scales monotonically with coupling, ignores orthogonal nuisance, and is not a raw-bandwidth artifact.

It issues NO governed authorization manifest and unlocks NO governed T4. The trusted committed policy
(``oracle_policy``) plus a mandatory Pi implementation gate remain required before any governed work.
All computation is safe-public / fully synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval import oracle_evaluator as EV
from clinical_jepa.eval.oracle_generator import generate_cell
from clinical_jepa.eval.oracle_metrics import realized_alpha, sequence_skill
from clinical_jepa.eval.oracle_spec import (
    KAPPA_TRAIN_GRID, heldout_families, train_families,
)
from clinical_jepa.eval.rung2_contract import (
    ORACLE_MONO_SPEARMAN, ORACLE_NULL_ALPHA, ORACLE_U6_BANDWIDTH_MARGIN, ORDER_UNLOCK_CHECKS,
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
    realized_alpha: float
    passed: bool

    @property
    def all_pass(self) -> bool:
        return all(v == "PASS" for v in self.checks.values())


def _p(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def certify_family(family_id: str, *, top_kappa: float | None = None, seed: int = 0) -> FamilyCertification:
    """Run U1..U6 for one family at its strongest on-grid (or family-specific) coupling."""
    grid = KAPPA_TRAIN_GRID
    kap = top_kappa if top_kappa is not None else max(grid)
    pos = generate_cell(family_id, kap, "orthogonal", N_CERT, seed=seed)
    leak = generate_cell(family_id, kap, "correlated_leak", N_CERT, seed=seed + 1)
    rb = EV.reference_bracket(pos, seed=seed)

    # U1: context predictor recovers order above the PRACTICAL gate; beats R0.
    u1 = rb.R_bayes_beats_R0
    # U2: nulls do not fire (context predictor + R0), realized alpha within the frozen bound.
    npred, ntrue = EV._split(EV.predict_context(pos), pos, want_null=True)
    ralpha = realized_alpha(npred, ntrue, base_seed=seed + 100)
    u2 = (not rb.R_bayes_null.fires) and rb.R0_null_pass and (ralpha <= ORACLE_NULL_ALPHA)
    # U3: order-skill increases monotonically with coupling kappa.
    curve = np.array([EV.reference_bracket(generate_cell(family_id, k, "orthogonal", N_CERT, seed=seed + 7),
                                           seed=seed).R_bayes_pos.mean_skill for k in grid])
    u3 = _spearman(np.asarray(grid, dtype=float), curve) >= ORACLE_MONO_SPEARMAN
    # U4: nuisance loses incremental skill in the orthogonal cell, captures bounded leak in the leak cell.
    u4 = EV.nuisance_incremental_ok(pos, leak, seed=seed + 3)
    # U6: proper context predictor beats the bandwidth-matched control by the frozen margin.
    proper = EV._skill(EV.predict_context(pos), pos, want_null=False, seed=seed + 5)
    ctrl = EV._skill(EV.predict_context_bandwidth_control(pos), pos, want_null=False, seed=seed + 6)
    u6 = (proper.lower_ci - ctrl.upper_ci) >= ORACLE_U6_BANDWIDTH_MARGIN

    checks = {"U1_order_recovery": _p(u1), "U2_null": _p(u2), "U3_monotone": _p(u3),
              "U4_nuisance_incremental": _p(u4), "U6_bandwidth_fair": _p(u6)}
    assert set(checks) == set(ORDER_UNLOCK_CHECKS), "U-check set must match the contract exactly"
    fc = FamilyCertification(family_id, checks, ralpha, passed=all(v == "PASS" for v in checks.values()))
    return fc


def estimate_realized_alpha(*, seed: int = 0, n_groups: int = 80) -> float:
    """Pooled, STABLE evaluator false-positive rate: pool the camouflaged-null sequences across ALL
    frozen families and run the context predictor's sequence-level fire rule over `n_groups`
    independent sub-studies. Pooling (vs a coarse per-family max) concentrates the estimate near the
    true well-calibrated FPR (~0.03) instead of straddling the gate on 20 groups."""
    preds: list = []
    trues: list = []
    for i, f in enumerate([*train_families(), *heldout_families()]):
        cell = generate_cell(f.family_id, max(KAPPA_TRAIN_GRID), "orthogonal", N_CERT, seed=seed + 200 + i)
        p, t = EV._split(EV.predict_context(cell), cell, want_null=True)
        preds.extend(p); trues.extend(t)
    return realized_alpha(preds, trues, n_groups=n_groups, base_seed=seed + 900)


def certify_synthetic_recovery(*, seed: int = 0) -> dict:
    """Certify across all frozen families. Family-level CONJUNCTION over HELD-OUT families (Pi #3):
    every held-out family must pass. Returns a safe-public summary + a CANDIDATE reference-bounds /
    unlock-checks block. This does NOT issue a governed manifest and does NOT set any policy anchor."""
    train = {f.family_id: certify_family(f.family_id, seed=seed) for f in train_families()}
    held = {f.family_id: certify_family(f.family_id, seed=seed + 50) for f in heldout_families()}
    heldout_conjunction = all(fc.passed for fc in held.values())
    # aggregate unlock_checks: a check PASSES overall only if it passes in EVERY held-out family.
    unlock = {c: _p(all(fc.checks[c] == "PASS" for fc in held.values())) for c in ORDER_UNLOCK_CHECKS}
    pooled_alpha = estimate_realized_alpha(seed=seed)
    return {
        "synthetic_recovery_certified": heldout_conjunction and all(v == "PASS" for v in unlock.values()),
        "family_conjunction": "all_held_out_families_must_pass",
        "train_families": {k: fc.checks for k, fc in train.items()},
        "held_out_families": {k: fc.checks for k, fc in held.items()},
        "unlock_checks": unlock,
        "reference_bounds_candidate": {
            "R_bayes_beats_R0": all(certify_family(f.family_id, seed=seed).checks["U1_order_recovery"] == "PASS"
                                    for f in heldout_families()),
            "evaluator_realized_alpha": pooled_alpha,
        },
        "governed_manifest_issued": False,      # NEVER here — needs the committed policy + Pi gate
        "note": "synthetic-recovery only; NOMINATE ceiling; no governed T4 unlocked.",
    }
