"""Rung-2 sub-gate 4 continuous-time/multiplicity head scoring (Pi v2: authorized to build).

Two SEPARATE, CONJUNCTIVE, NUMERIC gates (a joint zero-aware score is secondary):
  * 4A — zero/simultaneity (multiplicity): skill over a context-stratified/rate-only baseline
    (>=GATE_4A_MULTIPLICITY_SKILL), a rate-matched wrong-context swap skill (>=GATE_4A_SWAP_SKILL),
    calibration ECE (<=GATE_4A_ECE). p0 reliability ALONE cannot pass.
  * 4B — positive tail (Δt>0): positive-tail KS upper-CI (<=GATE_4B_KS), CRPS skill over the
    CONTEXT-OBSERVABLE stratified marginal (>=GATE_4B_CRPS_SKILL), improvement over a rate-only
    head (>=GATE_4B_RATE_HEAD_IMPROVEMENT), rate/occupancy-matched wrong-context swap
    (>=GATE_4B_SWAP).
Stratification variables must be context-observable; observed-future strata are oracle-assisted and
never the operational primary baseline. numpy-only; reuses the Rung-1 hurdle/PIT/CRPS machinery.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_probes import ks_d_upper_ci
from clinical_jepa.eval.rung2_contract import (
    CONTEXT_STRATA, GATE_4A_ECE, GATE_4A_MULTIPLICITY_SKILL, GATE_4A_SWAP_SKILL, GATE_4B_CRPS_SKILL,
    GATE_4B_KS, GATE_4B_RATE_HEAD_IMPROVEMENT, GATE_4B_SWAP, NOT_EVALUABLE, is_oracle_assisted_stratum,
)


def assert_context_observable_strata(strata: list[str]) -> bool:
    """Fail-hard if any stratification variable is observed-future (oracle-assisted; Pi #5)."""
    bad = [s for s in strata if is_oracle_assisted_stratum(s)]
    if bad:
        raise AssertionError(f"stratification vars {bad} are observed-future (oracle-assisted) — not a primary baseline")
    return True


def expected_calibration_error(pred_prob: Any, outcome: Any, n_bins: int = 10) -> float:
    """ECE of a probability (e.g. multiplicity/zero probability) against the binary outcome."""
    p = np.asarray(pred_prob, dtype=np.float64); y = np.asarray(outcome, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < n_bins - 1 else p <= edges[b + 1])
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def gate_4a(*, multiplicity_skill_lo: float, swap_skill_lo: float, ece_hi: float,
            evaluable: bool) -> dict[str, Any]:
    """Zero/simultaneity gate — multiplicity skill (not p0 reliability), rate-matched swap, ECE."""
    if not evaluable:
        return {"gate_4a": NOT_EVALUABLE}
    passed = bool(multiplicity_skill_lo >= GATE_4A_MULTIPLICITY_SKILL
                  and swap_skill_lo >= GATE_4A_SWAP_SKILL and ece_hi <= GATE_4A_ECE)
    return {"gate_4a": "PASS" if passed else "FAIL", "multiplicity_skill_lo": multiplicity_skill_lo,
            "swap_skill_lo": swap_skill_lo, "ece_hi": ece_hi}


def gate_4b(*, ks_upper_ci: float, crps_skill_lo: float, rate_head_improvement_lo: float,
            swap_skill_lo: float, evaluable: bool) -> dict[str, Any]:
    """Positive-tail gate — KS upper-CI, CRPS-skill over the stratified marginal, improvement over
    the rate-only head, and a rate/occupancy-matched wrong-context swap (all non-compensatory)."""
    if not evaluable:
        return {"gate_4b": NOT_EVALUABLE}
    passed = bool(ks_upper_ci <= GATE_4B_KS and crps_skill_lo >= GATE_4B_CRPS_SKILL
                  and rate_head_improvement_lo >= GATE_4B_RATE_HEAD_IMPROVEMENT
                  and swap_skill_lo >= GATE_4B_SWAP)
    return {"gate_4b": "PASS" if passed else "FAIL", "ks_upper_ci": ks_upper_ci,
            "crps_skill_lo": crps_skill_lo, "rate_head_improvement_lo": rate_head_improvement_lo,
            "swap_skill_lo": swap_skill_lo}


def timing_verdict(g4a: dict[str, Any], g4b: dict[str, Any]) -> str:
    """4A ∧ 4B, separate + conjunctive (Pi Q3). NOT_EVALUABLE if either is; else PASS only when
    both PASS."""
    a, b = g4a.get("gate_4a"), g4b.get("gate_4b")
    if a == NOT_EVALUABLE or b == NOT_EVALUABLE:
        return NOT_EVALUABLE
    return "PASS" if (a == "PASS" and b == "PASS") else "FAIL"
