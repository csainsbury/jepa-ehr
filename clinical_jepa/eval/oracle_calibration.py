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
from clinical_jepa.eval.oracle_spec import CALIBRATION_SPEC
from clinical_jepa.eval.oracle_meta_gen import invariant_hash
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_DT0_ABS, ORACLE_ENV_KS, ORACLE_ENV_MIN_DENOM, ORACLE_ENV_N_CLASSES,
    ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_TV,
)

# the ONLY knobs calibration may move (Pi: predeclared scale/nuisance knobs; hard bounds from the spec)
TUNABLE_KNOBS = tuple(CALIBRATION_SPEC["tunable_params"].keys())
NOT_EVALUABLE = "NOT_EVALUABLE"
# the FROZEN required source set — calibrate_sources must cover EXACTLY these (Pi #10)
REQUIRED_SOURCES = ("SCID", "MIMIC")
# the aggregate fields calibration binds (Pi #10: the calibration schema hash must bind AggregateStats).
_AGG_FIELDS = ("source", "n_sequences", "n_events", "n_clusters", "n_positive_gaps", "class_counts",
               "delta_t_zero_fraction", "length_ecdf", "positive_gap_ecdf", "count_ecdf",
               "mean_occupancy_fraction")


def calibration_schema_hash() -> str:
    """Hash the EXACT calibration schema (Pi #10): aggregate fields, tunable knobs + bounds, the frozen
    conjunctive-envelope constants, and the required source set. Distinct from the mechanism hash."""
    return canonical_hash({
        "agg_fields": _AGG_FIELDS, "tunable": {k: list(v) for k, v in CALIBRATION_SPEC["tunable_params"].items()},
        "envelope": {"dt0": ORACLE_ENV_DT0_ABS, "ks": ORACLE_ENV_KS, "tv": ORACLE_ENV_TV,
                     "occ": ORACLE_ENV_OCCUPANCY_ABS, "n_classes": ORACLE_ENV_N_CLASSES,
                     "min_denom": ORACLE_ENV_MIN_DENOM},
        "required_sources": REQUIRED_SOURCES})


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


def _ecdf_valid(ecdf) -> bool:
    """A well-formed ECDF: finite strictly-increasing support, non-decreasing cdf in [0,1] reaching
    ~1 at the last support point."""
    if not ecdf or len(ecdf) < 2:
        return False
    supp = [s for s, _ in ecdf]; cdf = [c for _, c in ecdf]
    if any(not np.isfinite(s) for s in supp) or any(not 0.0 <= c <= 1.0 for c in cdf):
        return False
    if any(supp[i + 1] <= supp[i] for i in range(len(supp) - 1)):        # strictly increasing support
        return False
    if any(cdf[i + 1] < cdf[i] - 1e-9 for i in range(len(cdf) - 1)):     # non-decreasing cdf
        return False
    return abs(cdf[-1] - 1.0) <= 1e-6                                    # final mass reaches 1


def validate_aggregate_input(agg: Any) -> tuple[bool, str]:
    """Fail-closed schema/type/finiteness/denominator/reconciliation validation. Returns (ok, reason)."""
    if not isinstance(agg, AggregateStats):
        return False, "not an AggregateStats"
    if not isinstance(agg.source, str) or not agg.source:
        return False, "missing source label"
    if len(agg.class_counts) != ORACLE_ENV_N_CLASSES:
        return False, f"class_counts must have {ORACLE_ENV_N_CLASSES} entries"
    if any((not isinstance(c, int)) or c < 0 for c in agg.class_counts):
        return False, "class_counts must be non-negative ints"
    denoms = (agg.n_sequences, agg.n_events, agg.n_clusters, agg.n_positive_gaps)
    if any((not isinstance(d, int)) or d < 0 for d in denoms):
        return False, "denominators must be non-negative ints"
    # denominator floors on every axis (Pi #8): sequences/clusters/events/positive-gaps/classes.
    if min(agg.n_sequences, agg.n_clusters) < ORACLE_ENV_MIN_DENOM:
        return False, NOT_EVALUABLE
    if agg.n_events < ORACLE_ENV_MIN_DENOM or agg.n_positive_gaps < ORACLE_ENV_MIN_DENOM:
        return False, NOT_EVALUABLE
    if sum(agg.class_counts) != agg.n_events:                            # class counts reconcile with n_events
        return False, "class_counts do not sum to n_events"
    for x in (agg.delta_t_zero_fraction, agg.mean_occupancy_fraction):
        if not np.isfinite(x) or not (0.0 <= x <= 1.0):
            return False, "fraction out of [0,1] or non-finite"
    for name, ecdf in (("length", agg.length_ecdf), ("gap", agg.positive_gap_ecdf),
                       ("count", agg.count_ecdf)):
        if not _ecdf_valid(ecdf):
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
    if synthetic.source != target.source:                    # a block must be matched to its OWN source
        return EnvelopeResult(False, {}, "source_mismatch")
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


def _transform_gap_ecdf(ecdf, rate_scale: float, dispersion: float):
    """Apply the timing knobs to the positive-gap ECDF: ``timing_rate_scale`` shrinks gaps (faster
    events), ``gap_dispersion`` stretches the support around its median. Support stays positive and
    strictly increasing; the cdf is untouched (an ECDF transform of the same probabilities)."""
    supp = np.array([s for s, _ in ecdf], float); cdf = np.array([c for _, c in ecdf], float)
    med = float(np.interp(0.5, cdf, supp))
    supp2 = np.round(np.maximum((med + (supp - med) * dispersion) / max(1e-6, rate_scale), 1e-6), 8)
    for i in range(1, len(supp2)):                            # nudge ONLY actual monotonicity violations
        if supp2[i] <= supp2[i - 1]:
            supp2[i] = round(supp2[i - 1] + 1e-6, 8)
    return tuple((float(s), float(c)) for s, c in zip(supp2, cdf))


def _forward_aggregate(base: AggregateStats, knobs: dict[str, float]) -> AggregateStats:
    """Deterministic knob->aggregate forward model over ONLY the tunable knobs (no mechanism change).
    ALL four tunables are active: ``zero_gap_bias`` sets Δt=0; ``token_freq_temperature`` shapes the
    class distribution; ``timing_rate_scale`` + ``gap_dispersion`` transform the positive-gap ECDF.
    Denominators, length/count ECDF supports, and occupancy are structural (untouched)."""
    zgb = float(np.clip(knobs.get("zero_gap_bias", base.delta_t_zero_fraction), 0.0, 0.9))
    temp = float(np.clip(knobs.get("token_freq_temperature", 1.0), 0.5, 2.0))
    rate = float(np.clip(knobs.get("timing_rate_scale", 1.0), 0.5, 2.0))
    disp = float(np.clip(knobs.get("gap_dispersion", 1.0), 0.5, 2.0))
    p = base.class_probs() ** (1.0 / temp)
    p = p / p.sum() if p.sum() > 0 else p
    raw = np.round(p * base.n_events).astype(int)
    raw[int(np.argmax(raw))] += base.n_events - int(raw.sum())   # reconcile exactly to n_events
    counts = tuple(int(x) for x in raw)
    gap = _transform_gap_ecdf(base.positive_gap_ecdf, rate, disp)
    return AggregateStats(base.source, base.n_sequences, base.n_events, base.n_clusters,
                          base.n_positive_gaps, counts, zgb, base.length_ecdf, gap,
                          base.count_ecdf, base.mean_occupancy_fraction)


def fit_calibration(target: AggregateStats, base: AggregateStats) -> CalibrationResult:
    """Deterministic fit of the tunable knobs to match ``target`` aggregates, then check the frozen
    conjunctive envelope. Fail-closed: an invalid target refuses. The mechanism hash is returned
    UNCHANGED — calibration provably cannot mutate the mechanism/grids/seeds/evaluator/registry."""
    ok, reason = validate_aggregate_input(target)
    if not ok:
        return CalibrationResult({}, False, "", calibration_schema_hash(), "", invariant_hash(),
                                 {"refused": "target_" + reason})
    ok_b, reason_b = validate_aggregate_input(base)              # validate the BASE too (Pi #10)
    if not ok_b:
        return CalibrationResult({}, False, "", calibration_schema_hash(), "", invariant_hash(),
                                 {"refused": "base_" + reason_b})
    if target.source != base.source:
        return CalibrationResult({}, False, "", calibration_schema_hash(), "", invariant_hash(),
                                 {"refused": "source_mismatch"})
    # deterministic grid solve of ALL FOUR tunable knobs to match the target marginals.
    knobs = {k: float(np.mean(CALIBRATION_SPEC["tunable_params"][k])) for k in TUNABLE_KNOBS}
    knobs["zero_gap_bias"] = float(np.clip(target.delta_t_zero_fraction, 0.0, 0.9))
    best_t, best_tv = 1.0, 1e9
    for t in np.linspace(0.5, 2.0, 61):
        cand = _forward_aggregate(base, {**knobs, "token_freq_temperature": float(t)})
        tv = _tv(cand.class_probs(), target.class_probs())
        if tv < best_tv - 1e-12:
            best_tv, best_t = tv, float(t)
    knobs["token_freq_temperature"] = best_t
    best_gap, best_rd = 1e9, (1.0, 1.0)                       # solve timing_rate_scale + gap_dispersion
    for rate in np.linspace(0.5, 2.0, 16):
        for disp in np.linspace(0.5, 2.0, 16):
            cand = _forward_aggregate(base, {**knobs, "timing_rate_scale": float(rate),
                                             "gap_dispersion": float(disp)})
            gks = _ks(cand.positive_gap_ecdf, target.positive_gap_ecdf)
            if gks < best_gap - 1e-12:
                best_gap, best_rd = gks, (float(rate), float(disp))
    knobs["timing_rate_scale"], knobs["gap_dispersion"] = best_rd
    fitted = _forward_aggregate(base, knobs)
    env = realism_envelope(fitted, target)
    fitted_knobs = {k: round(v, 6) for k, v in knobs.items()}
    return CalibrationResult(
        fitted_knobs=fitted_knobs, within_envelope=env.within_envelope,
        input_hash=canonical_hash({"target": _agg_full(target), "base": _agg_full(base)}),  # FULL canonical input
        spec_hash=calibration_schema_hash(), fitted_param_hash=canonical_hash(fitted_knobs),
        mechanism_hash=invariant_hash(),
        diagnostics={"envelope": {k: v for k, (v, _) in env.checks.items()},
                     "class_tv": best_tv, "gap_ks": best_gap},
    )


def _agg_full(a: AggregateStats) -> dict[str, Any]:
    """FULL canonical aggregate payload (Pi #8: input hash must cover every load-bearing field)."""
    return {"source": a.source, "n_sequences": a.n_sequences, "n_events": a.n_events,
            "n_clusters": a.n_clusters, "n_positive_gaps": a.n_positive_gaps,
            "class_counts": list(a.class_counts), "delta_t_zero_fraction": a.delta_t_zero_fraction,
            "mean_occupancy_fraction": a.mean_occupancy_fraction,
            "length_ecdf": [list(x) for x in a.length_ecdf],
            "positive_gap_ecdf": [list(x) for x in a.positive_gap_ecdf],
            "count_ecdf": [list(x) for x in a.count_ecdf]}


@dataclass(frozen=True)
class CollectionCalibrationResult:
    per_source: dict[str, CalibrationResult]
    all_sources_within_envelope: bool        # multi-source CONJUNCTION: every block must pass
    source_coverage_ok: bool                 # covers EXACTLY the frozen required source set
    combined_hash: str
    schema_hash: str
    mechanism_hash: str


def calibrate_sources(targets: dict[str, AggregateStats],
                      bases: dict[str, AggregateStats]) -> CollectionCalibrationResult:
    """Multi-source conjunctive calibration over the FROZEN required source set (Pi #10): the collection
    is accepted ONLY if the targets cover EXACTLY ``REQUIRED_SOURCES`` and EVERY source block falls
    within its own envelope. Unexpected/missing sources fail coverage."""
    coverage_ok = set(targets) == set(REQUIRED_SOURCES)
    per = {}
    for src, tgt in targets.items():
        base = bases.get(src)
        per[src] = (fit_calibration(tgt, base) if base is not None
                    else CalibrationResult({}, False, "", calibration_schema_hash(), "", invariant_hash(),
                                           {"refused": "missing_base_for_source"}))
    ok = coverage_ok and bool(per) and all(r.within_envelope for r in per.values())
    combined = canonical_hash({s: per[s].input_hash + per[s].fitted_param_hash for s in sorted(per)})
    return CollectionCalibrationResult(per, ok, coverage_ok, combined,
                                       calibration_schema_hash(), invariant_hash())
