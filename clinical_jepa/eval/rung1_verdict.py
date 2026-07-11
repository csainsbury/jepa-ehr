"""Rung-1 verdict assembly (Pi R7/R8) — per-readout classification -> scoped arm/property
verdict -> worst-primary-cell combination -> 1a incumbent record + 1b nominations.

Consumes pre-computed per-cell metric dicts (the driver runs the probes); pure logic so the
verdict contract is cheap to test. Sensitivity horizons (MIMIC 2 d) are reported but never
gate a primary verdict; only rung-1b arms with a direct/coarse-slot scope may nominate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clinical_jepa.eval.rung1_contract import (
    ARM_COMPARISON_ORDER, ARMS, CONTRACT_VERSION, CRPS_SKILL_GATE, DECODABLE_NONLINEAR,
    DECODABLE_SIMPLE, INCONCLUSIVE, KS_D_GATE, MARGINAL_ONLY, NOT_DECODABLE, NOT_EVALUABLE,
    NOT_EVALUATED, PRIMARY_ARM, PRIOR_MASKED, PROPERTIES, classify_readout, config_hash,
    is_primary_cell, scoped_verdict,
)
from clinical_jepa.utils import now_utc, write_json

# worst-first precedence for combining cells (a property is only as strong as its worst cell)
_PRECEDENCE = [NOT_DECODABLE, PRIOR_MASKED, MARGINAL_ONLY, INCONCLUSIVE, DECODABLE_SIMPLE, DECODABLE_NONLINEAR]


def classify_count_order_cell(metrics: dict[str, Any]) -> str:
    """count/order cell -> base class via the per-readout attribution (Pi R7 #1)."""
    return classify_readout(
        m1_gate_ok=bool(metrics.get("m1_gate_ok", False)), m1_excess_lo=float(metrics.get("m1_excess_lo", -1.0)),
        m2_gate_ok=bool(metrics.get("m2_gate_ok", False)), m2_excess_lo=float(metrics.get("m2_excess_lo", -1.0)),
        m2_copy_ok=bool(metrics.get("m2_copy_ok", True)),
        evaluable=bool(metrics.get("evaluable", False)), precise=bool(metrics.get("precise", False)),
    )


def classify_timing_cell(metrics: dict[str, Any]) -> str:
    """timing cell -> base class. Both non-compensatory gates required: KS-D upper-CI <= gate
    AND normalized-CRPS skill lower-CI >= gate. When KS is calibrated but the skill gate fails,
    distinguish MARGINAL_ONLY (M2's own wrong-instance-swap excess is positive -> it uses the
    latent, just weakly) from PRIOR_MASKED (swap excess <= 0 -> it reads its prior) — Pi Rung-1
    result gate #2.1."""
    if not metrics.get("evaluable", False):
        return NOT_EVALUABLE
    swap_lo = metrics.get("swap_excess_lo")
    if swap_lo is None:                              # fail CLOSED (Pi amended #2): a missing
        return INCONCLUSIVE                          # attribution control cannot claim latent use
    swap_ok = float(swap_lo) > 0.0
    ks_ok = float(metrics.get("ks_upper_ci", 1.0)) <= KS_D_GATE
    skill_ok = float(metrics.get("crps_skill_lo", -1.0)) >= CRPS_SKILL_GATE
    if not ks_ok:
        return NOT_DECODABLE if metrics.get("precise", False) else INCONCLUSIVE
    if not swap_ok:                                  # non-positive swap => reads its prior
        return PRIOR_MASKED
    if skill_ok:
        return DECODABLE_NONLINEAR                   # requires KS + skill + a POSITIVE own-swap
    return MARGINAL_ONLY                             # calibrated + uses the latent, sub-gate skill


def classify_order_cell(metrics: dict[str, Any]) -> str:
    """order cell -> base class. Order-blind arms: the frozen unconditional metric is
    NOT_EVALUATED (a labelled oracle probe is reported but never gates). temporal_slot: the real
    slot-fidelity metric via the per-readout attribution."""
    if metrics.get("unconditional_order") == "NOT_EVALUATED":
        return NOT_EVALUATED
    if not metrics.get("evaluable", False):
        return NOT_EVALUABLE
    return classify_readout(
        m1_gate_ok=bool(metrics.get("m1_gate_ok", False)), m1_excess_lo=float(metrics.get("m1_excess_lo", -1.0)),
        m2_gate_ok=bool(metrics.get("m2_gate_ok", False)), m2_excess_lo=float(metrics.get("m2_excess_lo", -1.0)),
        m2_copy_ok=bool(metrics.get("m2_copy_ok", True)), evaluable=True, precise=bool(metrics.get("precise", False)))


def _combine(cell_classes: list[str]) -> str:
    """Worst-first combination over PRIMARY cells. NOT_EVALUABLE / NOT_EVALUATED are excluded
    from the conjunction; if nothing else remains, return NOT_EVALUATED when every cell was
    deliberately not-evaluated, else NOT_EVALUABLE (below floor)."""
    excluded = (NOT_EVALUABLE, NOT_EVALUATED)
    evaluable = [c for c in cell_classes if c not in excluded]
    if not evaluable:
        return NOT_EVALUATED if all(c == NOT_EVALUATED for c in cell_classes) else NOT_EVALUABLE
    for c in _PRECEDENCE:
        if c in evaluable:
            return c
    return NOT_EVALUABLE


def evaluate_property(arm: str, prop: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """cells = [{source, window_days, base_class}]. Combines PRIMARY cells into the scoped
    arm/property verdict; sensitivity cells are recorded but never gate."""
    primary, sensitivity = [], []
    for c in cells:
        (primary if is_primary_cell(c["source"], c["window_days"]) else sensitivity).append(c)
    combined_base = _combine([c["base_class"] for c in primary])
    scoped = scoped_verdict(arm, prop, combined_base, rung=ARMS[arm]["rung"])
    return {
        "arm": arm, "property": prop, "rung": ARMS[arm]["rung"],
        "combined_base_class": combined_base,
        "verdict": scoped["verdict"], "information_scope": scoped["information_scope"],
        "can_nominate": scoped["can_nominate"],
        "per_primary_cell": [{"source": c["source"], "window_days": c["window_days"],
                              "base_class": c["base_class"]} for c in primary],
        "per_sensitivity_cell": [{"source": c["source"], "window_days": c["window_days"],
                                  "base_class": c["base_class"]} for c in sensitivity],
    }


def build_rung1_manifest(property_evals: list[dict[str, Any]], *, run_config: dict[str, Any] | None = None,
                         evaluator_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the 1a incumbent record + 1b nominations. property_evals = outputs of
    evaluate_property for every (arm, property). ``evaluator_provenance`` (Pi amended #5) records
    the evaluator code commit / run id beyond the frozen scalar config_hash."""
    rung1a: dict[str, Any] = {}
    rung1b: dict[str, dict[str, Any]] = {}
    for ev in property_evals:
        block = {k: ev[k] for k in ("verdict", "information_scope", "can_nominate",
                                    "combined_base_class", "per_primary_cell", "per_sensitivity_cell")}
        if ev["arm"] == PRIMARY_ARM:
            rung1a.setdefault("arm", PRIMARY_ARM)
            rung1a.setdefault("properties", {})[ev["property"]] = block
        else:
            rung1b.setdefault(ev["arm"], {}).setdefault("properties", {})[ev["property"]] = block

    # nominations in the FROZEN comparison order (Pi R7 #6 multiplicity control)
    nominations = []
    for arm in ARM_COMPARISON_ORDER:
        props = rung1b.get(arm, {}).get("properties", {})
        for prop in PROPERTIES:
            ev = props.get(prop)
            if ev and ev["can_nominate"]:
                nominations.append({"arm": arm, "property": prop, "verdict": ev["verdict"],
                                    "information_scope": ev["information_scope"],
                                    "decision": "NOMINATE_TARGET_FOR_RUNG2"})
    return {
        "schema": "clinical-jepa-rung1-ceiling-v1",
        "created_utc": now_utc(),
        "contract_version": CONTRACT_VERSION,
        "config_hash": config_hash(run_config),
        "evaluator_provenance": evaluator_provenance or {},   # Pi amended #5: code commit / run id
        "aggregate_only": True,
        "test_access": False,
        "rung1a": rung1a,
        "rung1b": rung1b,
        "nominations": nominations,
        "notes": "frozen-decode ceiling; 1a incumbent mean_embed verdict is independent; "
                 "1b arms NOMINATE targets for Rung 2 only (no SWITCH); test held out.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the Rung-1 ceiling verdict from per-cell metrics")
    ap.add_argument("--cell-metrics", required=True, help="JSON: [{arm, property, source, window_days, ...metrics}]")
    ap.add_argument("--run-config", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.cell_metrics).read_text())
    run_cfg = json.loads(Path(args.run_config).read_text()) if args.run_config else None
    # base-classify each cell then group by (arm, property)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        prop = row["property"]
        base = (classify_timing_cell(row) if prop == "timing"
                else classify_order_cell(row) if prop == "order"
                else classify_count_order_cell(row))
        grouped.setdefault((row["arm"], prop), []).append(
            {"source": row["source"], "window_days": float(row["window_days"]), "base_class": base})
    evals = [evaluate_property(arm, prop, cells) for (arm, prop), cells in grouped.items()]
    manifest = build_rung1_manifest(evals, run_config=run_cfg)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "rung1-ceiling-manifest.json", manifest)
    print(json.dumps({"output": str(out / "rung1-ceiling-manifest.json"),
                      "nominations": manifest["nominations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
