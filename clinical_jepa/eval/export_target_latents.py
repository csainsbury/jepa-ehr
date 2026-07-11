"""Rung-1 governed export bridge (Pi R7/R8) — build per-arm z+ bundles + target props from
real blocks, DEV-ONLY (test blocks are never loaded here; Pi R8 #8).

For each (arm × source×horizon cell × split) it emits the z+ matrix and the aligned target
properties (count, ordered target token ids, inter-event Δt, occupancy, patient cluster) the
driver needs. Censored blocks are excluded upstream (extract_blocks); empties route to the
frozen z_empty dimensionally. Sidecars are LOCAL GOVERNED artifacts (gitignored) — only
aggregate metrics ever leave the boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clinical_jepa.targets.block_spans import is_censored, read_target_span
from clinical_jepa.targets.target_reps import ARM_NAMES, build_target_rep, temporal_slot_token_sets
from clinical_jepa.eval.rung1_contract import D_TIME, M_PRIMARY

ALLOWED_SPLITS = ("train", "dev")           # TEST is never loaded in this run (Pi R8 #8)


def block_props(block: dict[str, Any], token_ids: Any, cumulative_days: Any) -> dict[str, Any]:
    ids, is_empty = read_target_span(block, token_ids)
    times, _ = read_target_span(block, cumulative_days)
    times = np.asarray(times, dtype=np.float64)
    count = 0 if is_empty else int(block.get("n_target_events", len(ids)))
    dt = np.diff(times) if len(times) >= 2 else np.asarray([], dtype=np.float64)
    return {"count": int(count), "ordered_ids": np.asarray(ids, dtype=np.int64), "dt": dt,
            "is_empty": bool(is_empty), "patient": str(block.get("patient_hash")),
            "source": block.get("source_dataset"), "window_days": float(block.get("window_days")),
            "split": block.get("split")}


def build_bundles(blocks: list[dict[str, Any]], seqs: dict[str, dict[str, Any]], *, model: Any,
                  arms: list[str] = list(ARM_NAMES), d_time: int = D_TIME, slots: int = M_PRIMARY,
                  splits: tuple[str, ...] = ALLOWED_SPLITS) -> dict[tuple[str, str, float], dict[str, Any]]:
    """seqs[sequence_id] = {'token_ids': arr, 'cumulative_days': arr}. Returns
    bundles[(arm, source, W)] = {split: {z, counts, patients, dt_lists, ordered_ids}}."""
    from clinical_jepa.targets.target_reps import _embedding_matrix, _z_empty_vec
    E = _embedding_matrix(model)                     # derive once — no per-block torch
    z_empty = _z_empty_vec(model)
    out: dict[tuple[str, str, float], dict[str, Any]] = {}
    for arm in arms:
        for blk in blocks:
            split = blk.get("split")
            if split not in splits:                          # test sealed
                continue
            if is_censored(blk):                             # silence-only; censored excluded
                continue
            seq = seqs.get(blk.get("sequence_id"))
            if seq is None:
                continue
            tok = np.asarray(seq["token_ids"]); cum = np.asarray(seq["cumulative_days"])
            props = block_props(blk, tok, cum)
            z = build_target_rep(arm, blk, tok, cum, E=E, z_empty=z_empty, d_time=d_time, slots=slots)
            key = (arm, props["source"], props["window_days"])
            cell = out.setdefault(key, {})
            b = cell.setdefault(split, {"z": [], "counts": [], "patients": [], "dt_lists": [], "ordered_ids": []})
            b["z"].append(z); b["counts"].append(props["count"]); b["patients"].append(props["patient"])
            b["dt_lists"].append(props["dt"]); b["ordered_ids"].append(props["ordered_ids"])
            if arm == "temporal_slot":                    # true per-slot token sets for the slot metric
                b.setdefault("slot_sets", []).append(temporal_slot_token_sets(blk, tok, cum, slots=slots))
    for cell in out.values():
        for split, b in cell.items():
            b["z"] = np.asarray(b["z"], dtype=np.float32)
            b["counts"] = np.asarray(b["counts"], dtype=np.int64)
            b["patients"] = np.asarray(b["patients"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-1 governed z+ export (DEV-ONLY; test never loaded)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", choices=list(ALLOWED_SPLITS), required=True)  # 'test' is not accepted
    ap.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    if args.split not in ALLOWED_SPLITS:
        raise SystemExit("Rung-1 is dev-only: test split is not exportable in this run.")
    # The governed runbook wires checkpoint + block/sequence loading (local gitignored config);
    # kept out of the committed tree. build_bundles() is the tested transform.
    print(json.dumps({"note": "invoke build_bundles() from the governed runbook",
                      "arms": args.arms, "split": args.split, "source": args.source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
