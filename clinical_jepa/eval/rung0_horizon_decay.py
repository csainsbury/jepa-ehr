"""Rung-0 horizon-decay driver (Pi R5) — orchestrates rank → paired stats → verdict.

Per source, per frozen horizon W, per granularity {coarse, coarse_B, fine, fine_B,
(k1)}: rank via patient-disjoint frozen groups, assemble POPULATED-stratum record
streams, then apply the corrected gate — co-primary level horizons (SCID 90d AND 365d;
MIMIC 0.5d), budget-matched coarse_B confirmation, per-unit-time slope separation, all
with paired patient bootstraps; K=1 harness null; plus externally-supplied raw-count
corroboration / time-shuffle veto / sufficiency flags → the 3-way decision. Aggregate-
only manifest.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung0_retrieval import rung0_rank
from clinical_jepa.eval.rung0_stats import (
    assert_k1_null,
    decision,
    paired_gap_streams,
    paired_slope_streams,
)

GRANULARITIES = ("coarse", "coarse_B", "fine", "fine_B", "k1")


def _rank_cell(cell: dict[str, Any], *, seed: int, max_candidates: int) -> list[dict[str, Any]]:
    res = rung0_rank(cell["queries"], cell["targets"], cell["index"],
                     max_candidates=max_candidates, seed=seed)
    return res["records"]


def _populated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("occupancy") == "populated"]


def evaluate_source(
    source: str,
    per_W: dict[float, dict[str, dict[str, Any]]],
    *,
    level_horizons: list[float],
    k: int = 10,
    n_boot: int = 2000,
    seed: int = 20260523,
    max_candidates: int = 200,
    raw_count_ok: bool = False,
    veto: bool = False,
    sufficiency_ok: bool = False,
    practical_level: float = 0.10,
    practical_widening: float = 0.05,
    adequacy_floor: int = 500,
) -> dict[str, Any]:
    """per_W = {W: {granularity: {queries, targets, index}}}. Returns the source verdict."""
    # 1) rank every cell -> populated record streams keyed by granularity + W.
    recs: dict[str, dict[float, list[dict[str, Any]]]] = {g: {} for g in GRANULARITIES}
    adequacy: dict[str, dict[str, bool]] = {}
    for W, cells in per_W.items():
        for g, cell in cells.items():
            if g not in recs:
                continue
            rr = _populated(_rank_cell(cell, seed=seed, max_candidates=max_candidates))
            recs[g][float(W)] = rr
            adequacy.setdefault(str(W), {})[g] = len(rr) >= adequacy_floor

    # 2) K=1 harness null (coarse ≡ fine ⇒ ~0 gap) if present.
    k1_ok = True
    for W, r in recs.get("k1", {}).items():
        cg = recs.get("coarse", {}).get(W, [])
        if r and cg:
            try:
                assert_k1_null(paired_gap_streams(cg, r, k=k, n_boot=max(200, n_boot // 4), seed=seed), tol=1e-9)
            except AssertionError:
                k1_ok = False

    # 3) level gate at the co-primary horizons (require the WORST to clear).
    per_horizon_level = {}
    for W in level_horizons:
        c, f = recs["coarse"].get(float(W), []), recs["fine"].get(float(W), [])
        if c and f:
            per_horizon_level[W] = paired_gap_streams(c, f, k=k, n_boot=n_boot, seed=seed)
    worst_level = min(per_horizon_level.values(), key=lambda g: g["ci_lo"]) if per_horizon_level else {"gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    per_horizon_cb = {}
    for W in level_horizons:
        c, f = recs["coarse_B"].get(float(W), []), recs["fine_B"].get(float(W), [])
        if c and f:
            per_horizon_cb[W] = paired_gap_streams(c, f, k=k, n_boot=n_boot, seed=seed)
    worst_cb = min(per_horizon_cb.values(), key=lambda g: g["ci_lo"]) if per_horizon_cb else {"gap": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    # 4) slope over all horizons present.
    slope = paired_slope_streams(recs["coarse"], recs["fine"], k=k, n_boot=n_boot, seed=seed) if recs["coarse"] and recs["fine"] else {"ci_lo": float("nan"), "ci_hi": float("nan"), "slope_diff_fine_minus_coarse": float("nan")}

    # 5) adequacy of the decision cells (co-primary level horizons, coarse + fine).
    adequate = all(
        adequacy.get(str(W), {}).get("coarse", False) and adequacy.get(str(W), {}).get("fine", False)
        for W in level_horizons
    ) and bool(level_horizons)

    # EFFECT-RULED-OUT requires EVERY co-primary coarse_B cell to exclude the effect,
    # not just the worst one (verification-found gate-logic fix).
    cb_ruled_out = bool(per_horizon_cb) and all(
        (g.get("ci_hi") is not None and g["ci_hi"] < practical_level) for g in per_horizon_cb.values()
    )
    verdict = decision(
        level_gap=worst_level, coarse_b_gap=worst_cb, slope=slope, coarse_b_ruled_out=cb_ruled_out,
        raw_count_ok=raw_count_ok, veto=(veto or not k1_ok), sufficiency_ok=sufficiency_ok,
        adequate=adequate, practical_level=practical_level, practical_widening=practical_widening,
    )
    return {
        "source": source,
        "level_horizons": [float(W) for W in level_horizons],
        "per_horizon_level_gap": {str(W): g for W, g in per_horizon_level.items()},
        "per_horizon_coarse_b_gap": {str(W): g for W, g in per_horizon_cb.items()},
        "worst_level_gap": worst_level,
        "worst_coarse_b_gap": worst_cb,
        "slope": slope,
        "k1_harness_ok": k1_ok,
        "adequacy": adequacy,
        "adequate": adequate,
        "raw_count_ok": raw_count_ok,
        "sufficiency_ok": sufficiency_ok,
        "veto": veto,
        "decision": verdict["decision"],
        "decision_detail": verdict,
        "n_boot": n_boot,
        "aggregate_only": True,
    }


def build_manifest(source_verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    from clinical_jepa.utils import now_utc
    return {
        "schema": "clinical-jepa-rung0-horizon-decay-v0",
        "created_utc": now_utc(),
        "per_source": {v["source"]: v for v in source_verdicts},
        "decisions": {v["source"]: v["decision"] for v in source_verdicts},
        "aggregate_only": True,
        "notes": "within-source coarse-vs-fine wall-clock horizon-decay; per-source hierarchy verdict.",
    }
