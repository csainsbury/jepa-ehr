"""Wall-clock readiness — per source x horizon feasibility with the empty/censored
split (Pi R4 / R3). Committed sibling of the local feasibility grid.

For each source and candidate wall-clock horizon W (days), classify every feasible
window's target (gap=0, mirroring extract_blocks._t0_wall_clock_block):

  - nonempty  : has >= 1 event in [t_query, t_query+W);
  - empty     : zero events AND the full window is observed (genuine silence);
  - censored  : zero events AND the window runs past observed time (Pi R4 Q7 —
                absence unverifiable, ineligible, NOT silence);
  - saturated : the target captured the ENTIRE remaining sequence (undiscriminating).

A horizon is "adequate" (frozen rule, Pi R3) when valid_matched >= floor AND empty,
saturated and censored rates are all <= 0.5 AND the non-empty median occupancy >= 2
(no median-1-only). Horizons that are empty/censored-dominated are flagged
conditional/incomplete (Pi item 3). Aggregate-only: counts/rates/quantiles, no ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.eval.retrieval import length_bin
from clinical_jepa.splits.readiness_manifest import _context_rate_bin
from clinical_jepa.targets.extract_blocks import _observed_end_day, is_monotone_nondecreasing
from clinical_jepa.utils import ensure_dir, load_yaml, now_utc, write_json
from clinical_jepa.validation import validate_artifact

DEFAULT_HORIZONS_DAYS = [0.25, 0.5, 1.0, 2.0, 3.0, 7.0, 14.0, 30.0, 90.0, 365.0, 730.0]
FLOOR = 500
MIN_MATCHED_CANDIDATES = 2
MAX_DEGENERATE_RATE = 0.50
MIN_NONEMPTY_MEDIAN_OCC = 2


def _per_source(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((dataset_cfg.get("sources", {}) or {}).get("primary", {}) or {}).get("source_datasets", []))


def _min_context_for(arms_cfg: dict[str, Any], source: str) -> int:
    t0 = ((arms_cfg.get("common", {}) or {}).get("target_blocks", {}) or {}).get("T0", {}) or {}
    ps = (t0.get("per_source", {}) or {}).get(source, {}) or {}
    return int(ps.get("min_context", t0.get("min_context", 8)))


def _median(vals: list[int]) -> float:
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def readiness_for_source(h5_path: str, source: str, min_context: int, source_prefix_len: int,
                         horizons: list[float], floor: int) -> dict[str, Any]:
    import h5py

    feasible = {W: 0 for W in horizons}
    nonempty = {W: 0 for W in horizons}
    empty = {W: 0 for W in horizons}
    censored = {W: 0 for W in horizons}
    saturated = {W: 0 for W in horizons}
    occ_ne: dict[float, list[int]] = {W: [] for W in horizons}
    keys_ne: dict[float, list[tuple]] = {W: [] for W in horizons}
    n_seq = non_monotone = infeasible = 0

    with h5py.File(h5_path, "r") as h5:
        for group in h5.keys():
            grp = h5.get(group)
            if grp is None or "token_ids" not in grp or "cumulative_days" not in grp:
                continue
            seq_len = int(grp["token_ids"].shape[0])
            if seq_len <= 0:
                continue
            n_seq += 1
            days = grp["cumulative_days"][:]
            L = min(seq_len, len(days))
            if not is_monotone_nondecreasing([days[i] for i in range(L)]):
                non_monotone += 1
                continue
            low_ce = source_prefix_len + min_context - 1
            max_ce = L - 2
            if not (low_ce <= max_ce):
                infeasible += 1
                continue
            context_end = min(max(low_ce, (source_prefix_len + L) // 2), max_ce)
            t_query = float(days[context_end])
            span = float(days[L - 1]) - float(days[0])
            lb, rb = length_bin(seq_len), _context_rate_bin(seq_len, span)
            observed_end = _observed_end_day(days, L, context_end, segment_ids=None)
            tail = [float(days[i]) for i in range(context_end + 1, L)]
            n_tail = len(tail)
            lo_idx = bisect_left(tail, t_query)
            for W in horizons:
                feasible[W] += 1
                hi_idx = bisect_left(tail, t_query + W)
                n_target = hi_idx - lo_idx
                if n_target > 0:
                    nonempty[W] += 1
                    occ_ne[W].append(n_target)
                    keys_ne[W].append((lb, rb))
                    if hi_idx >= n_tail:
                        saturated[W] += 1
                else:
                    fully_observed = observed_end is not None and (t_query + W) <= observed_end + 1e-9
                    if fully_observed:
                        empty[W] += 1
                    else:
                        censored[W] += 1

    out: dict[str, Any] = {"source": source, "n_sequences": n_seq, "non_monotone": non_monotone,
                           "infeasible": infeasible, "min_context": min_context, "per_horizon": {}}
    for W in horizons:
        f, ne = feasible[W], nonempty[W]
        kc = Counter(keys_ne[W])
        matched = sum(1 for k in keys_ne[W] if kc[k] >= MIN_MATCHED_CANDIDATES)
        empty_rate = float(empty[W] / f) if f else 0.0
        censored_rate = float(censored[W] / f) if f else 0.0
        saturated_rate = float(saturated[W] / ne) if ne else 0.0
        ne_med = _median(occ_ne[W])
        non_degenerate = (empty_rate <= MAX_DEGENERATE_RATE and saturated_rate <= MAX_DEGENERATE_RATE
                          and censored_rate <= MAX_DEGENERATE_RATE and ne_med >= MIN_NONEMPTY_MEDIAN_OCC)
        out["per_horizon"][str(W)] = {
            "window_days": W, "feasible_windows": f, "nonempty_target_windows": ne,
            "empty_target_windows": empty[W], "censored_target_windows": censored[W],
            "valid_matched_windows": matched,
            "empty_target_rate": empty_rate, "censored_target_rate": censored_rate,
            "saturated_target_rate": saturated_rate, "nonempty_median_occupancy": ne_med,
            "meets_floor": matched >= floor, "non_degenerate": non_degenerate,
            "adequate": (matched >= floor) and non_degenerate,
            # Pi item 3: empty/censored-dominated horizons are conditional/incomplete.
            "conditional_incomplete": (empty_rate + censored_rate) > MAX_DEGENERATE_RATE,
        }
    return out


def build_wallclock_readiness(dataset_cfg: dict[str, Any], arms_cfg: dict[str, Any], split: str,
                              horizons: list[float], *, floor: int = FLOOR) -> dict[str, Any]:
    source_prefix_len = max(0, int((dataset_cfg.get("mask", {}) or {}).get("source_prefix_len", 0)))
    per_source: dict[str, Any] = {}
    for entry in _per_source(dataset_cfg):
        source = str(entry["name"])
        h5_path = str((entry.get("h5_paths", {}) or {})[split])
        per_source[source] = readiness_for_source(
            h5_path, source, _min_context_for(arms_cfg, source), source_prefix_len, horizons, floor
        )
    sources = list(per_source.keys())
    adequate_by_source = {
        s: [float(W) for W in horizons if per_source[s]["per_horizon"][str(W)]["adequate"]] for s in sources
    }
    manifest = {
        "schema_version": "clinical-jepa-wallclock-readiness-v0",
        "created_utc": now_utc(), "split": split, "horizons_days": horizons, "floor": floor,
        "source_prefix_len": source_prefix_len, "gap_days": 0.0,
        "per_source": per_source, "adequate_horizons_by_source": adequate_by_source,
        "aggregate_only": True,
        "notes": "per source x horizon wall-clock readiness with empty/censored split; aggregate-only.",
    }
    validate_artifact("wallclock-readiness", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wall-clock readiness (per source x horizon; empty/censored split)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--arms-config", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--horizons-days", type=float, nargs="+", default=DEFAULT_HORIZONS_DAYS)
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    manifest = build_wallclock_readiness(
        load_yaml(args.dataset_config), load_yaml(args.arms_config), args.split,
        [float(w) for w in args.horizons_days], floor=args.floor,
    )
    outdir = ensure_dir(args.output_dir)
    write_json(outdir / "wallclock-readiness-manifest.json", manifest)
    print(json.dumps({"output": str(outdir / "wallclock-readiness-manifest.json"),
                      "split": args.split, "adequate_horizons_by_source": manifest["adequate_horizons_by_source"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
