"""Calibration + realism envelope (Pi 2nd-pass Phase 5).

Fits ONLY the predeclared scale/nuisance knobs so the synthetic marginals fall inside a frozen
CONJUNCTIVE realism envelope. It CANNOT touch the mechanism, grids, seeds, evaluator, or registry —
its output is a set of knob values plus its own hashes. Eligibility is the CONJUNCTION of six per-
source-block checks (NOT a generic weighted-L1). Missing / under-supported aggregates REFUSE
(NOT_EVALUABLE), never zero-filled. Tests use SYNTHETIC aggregate fixtures ONLY; a separate micro-gate
is required before the first aggregate-real read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_spec import CALIBRATION_SPEC, calibration_hash, oracle_mechanism_hash
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_DT0_ABS, ORACLE_ENV_KS, ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES,
    ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_TV,
)

# the ONLY knobs calibration may move (Pi: predeclared scale/nuisance knobs; hard bounds from the spec)
TUNABLE_KNOBS = tuple(CALIBRATION_SPEC["tunable_params"].keys())
NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True)
class AggregateStats:
    """Source-stratified aggregate marginals — the ONLY thing calibration reads. No patient rows."""
    source: str
    n_sequences: int
    n_events: int
    n_clusters: int
    n_positive_gaps: int
    class_counts: tuple[int, ...]              # length ORACLE_ENV_N_CLASSES
    delta_t_zero_fraction: float
    length_ecdf: tuple[tuple[float, float], ...]      # (support_point, cdf) on a frozen support
    positive_gap_ecdf: tuple[tuple[float, float], ...]
    count_ecdf: tuple[tuple[float, float], ...]
    mean_occupancy_fraction: float

    def class_probs(self) -> np.ndarray:
        c = np.asarray(self.class_counts, float)
        return c / c.sum() if c.sum() > 0 else c


def validate_aggregate_input(agg: Any) -> tuple[bool, str]:
    """Fail-closed schema/type/finiteness/denominator validation. Returns (ok, reason)."""
    if not isinstance(agg, AggregateStats):
        return False, "not an AggregateStats"
    if len(agg.class_counts) != ORACLE_ENV_N_CLASSES:
        return False, f"class_counts must have {ORACLE_ENV_N_CLASSES} entries"
    denoms = (agg.n_sequences, agg.n_events, agg.n_clusters, agg.n_positive_gaps)
    if any((not isinstance(d, int)) or d < 0 for d in denoms):
        return False, "denominators must be non-negative ints"
    if min(agg.n_sequences, agg.n_clusters) < ORACLE_ENV_MIN_DENOM:
        return False, NOT_EVALUABLE                    # under-supported -> refuse, never zero-fill
    for x in (agg.delta_t_zero_fraction, agg.mean_occupancy_fraction):
        if not np.isfinite(x) or not (0.0 <= x <= 1.0):
            return False, "fraction out of [0,1] or non-finite"
    for name, ecdf in (("length", agg.length_ecdf), ("gap", agg.positive_gap_ecdf),
                       ("count", agg.count_ecdf)):
        if not ecdf or any((not np.isfinite(s)) or (not 0.0 <= c <= 1.0) for s, c in ecdf):
            return False, f"{name}_ecdf malformed"
    return True, "ok"


def _ks(a: tuple, b: tuple) -> float:
    """KS distance between two ECDFs sampled on a common frozen support (max |cdf_a - cdf_b|)."""
    da, db = dict(a), dict(b)
    support = sorted(set(da) | set(db))

    def _interp(d, s):
        keys = sorted(d)
        prev = 0.0
        for k in keys:
            if k > s:
                break
            prev = d[k]
        return prev
    return max(abs(_interp(da, s) - _interp(db, s)) for s in support) if support else 1.0


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


@dataclass(frozen=True)
class EnvelopeResult:
    within_envelope: bool
    checks: dict[str, tuple[float, bool]]      # stat_name -> (value, passed)
    reason: str = "ok"


def realism_envelope(synthetic: AggregateStats, target: AggregateStats) -> EnvelopeResult:
    """CONJUNCTIVE six-check acceptance for one source block. Any failing check => within_envelope=False.
    Refuses if either aggregate is invalid (fail-closed)."""
    for a in (synthetic, target):
        ok, reason = validate_aggregate_input(a)
        if not ok:
            return EnvelopeResult(False, {}, reason)
    checks = {
        "delta_t_zero_abs": (abs(synthetic.delta_t_zero_fraction - target.delta_t_zero_fraction),
                             None),
        "class_tv": (_tv(synthetic.class_probs(), target.class_probs()), None),
        "length_ks": (_ks(synthetic.length_ecdf, target.length_ecdf), None),
        "positive_gap_ks": (_ks(synthetic.positive_gap_ecdf, target.positive_gap_ecdf), None),
        "count_ks": (_ks(synthetic.count_ecdf, target.count_ecdf), None),
        "occupancy_abs": (abs(synthetic.mean_occupancy_fraction - target.mean_occupancy_fraction),
                          None),
    }
    thresh = {"delta_t_zero_abs": ORACLE_ENV_DT0_ABS, "class_tv": ORACLE_ENV_TV,
              "length_ks": ORACLE_ENV_KS, "positive_gap_ks": ORACLE_ENV_KS, "count_ks": ORACLE_ENV_KS,
              "occupancy_abs": ORACLE_ENV_OCCUPANCY_ABS}
    resolved = {k: (v, v <= thresh[k]) for k, (v, _) in checks.items()}
    return EnvelopeResult(all(passed for _, passed in resolved.values()), resolved)


@dataclass(frozen=True)
class CalibrationResult:
    fitted_knobs: dict[str, float]
    within_envelope: bool
    input_hash: str
    spec_hash: str                 # the calibration SPEC hash (frozen)
    fitted_param_hash: str         # SEPARATE hash of the fitted knobs
    mechanism_hash: str            # unchanged — proof calibration did not touch the mechanism
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _forward_aggregate(base: AggregateStats, knobs: dict[str, float]) -> AggregateStats:
    """Deterministic knob->aggregate forward model over ONLY the tunable knobs (no mechanism change):
    zero_gap_bias sets Δt=0 fraction; token_freq_temperature tempers the class distribution toward
    uniform; the others are pass-through here (their effect on gap/length ECDFs is applied at
    generation time). Structural fields (denominators, ECDF supports) are untouched."""
    zgb = float(np.clip(knobs.get("zero_gap_bias", base.delta_t_zero_fraction), 0.0, 0.9))
    temp = float(np.clip(knobs.get("token_freq_temperature", 1.0), 0.5, 2.0))
    p = base.class_probs() ** (1.0 / temp)                  # temper the base class SHAPE (peakedness)
    p = p / p.sum() if p.sum() > 0 else p
    counts = tuple(int(round(x)) for x in p * max(1, base.n_events))
    # occupancy / length / gap / count ECDFs are STRUCTURAL — these knobs do not move them (pass-through).
    return AggregateStats(base.source, base.n_sequences, base.n_events, base.n_clusters,
                          base.n_positive_gaps, counts, zgb, base.length_ecdf, base.positive_gap_ecdf,
                          base.count_ecdf, base.mean_occupancy_fraction)


def fit_calibration(target: AggregateStats, base: AggregateStats) -> CalibrationResult:
    """Deterministic fit of the tunable knobs to match ``target`` aggregates, then check the frozen
    conjunctive envelope. Fail-closed: an invalid target refuses. The mechanism hash is returned
    UNCHANGED — calibration provably cannot mutate the mechanism/grids/seeds/evaluator/registry."""
    ok, reason = validate_aggregate_input(target)
    if not ok:
        return CalibrationResult({}, False, "", str(calibration_hash()), "", oracle_mechanism_hash(),
                                 {"refused": reason})
    # deterministic closed-form solve of the two knobs that have a monotone forward map, others fixed.
    knobs = {k: float(np.mean(CALIBRATION_SPEC["tunable_params"][k])) for k in TUNABLE_KNOBS}
    knobs["zero_gap_bias"] = float(np.clip(target.delta_t_zero_fraction, 0.0, 0.9))
    # grid-then-refine token temperature to minimise class TV (deterministic)
    best_t, best_tv = 1.0, 1e9
    for t in np.linspace(0.5, 2.0, 61):
        cand = _forward_aggregate(base, {**knobs, "token_freq_temperature": float(t)})
        tv = _tv(cand.class_probs(), target.class_probs())
        if tv < best_tv - 1e-12:
            best_tv, best_t = tv, float(t)
    knobs["token_freq_temperature"] = best_t
    fitted = _forward_aggregate(base, knobs)
    env = realism_envelope(fitted, target)
    fitted_knobs = {k: round(v, 6) for k, v in knobs.items()}
    return CalibrationResult(
        fitted_knobs=fitted_knobs, within_envelope=env.within_envelope,
        input_hash=canonical_hash({"target": _agg_payload(target), "base": _agg_payload(base)}),
        spec_hash=str(calibration_hash()), fitted_param_hash=canonical_hash(fitted_knobs),
        mechanism_hash=oracle_mechanism_hash(),
        diagnostics={"envelope": {k: v for k, (v, _) in env.checks.items()}, "class_tv": best_tv},
    )


def _agg_payload(a: AggregateStats) -> dict[str, Any]:
    return {"source": a.source, "class_counts": list(a.class_counts),
            "dt0": a.delta_t_zero_fraction, "occ": a.mean_occupancy_fraction,
            "n_sequences": a.n_sequences, "n_clusters": a.n_clusters}
