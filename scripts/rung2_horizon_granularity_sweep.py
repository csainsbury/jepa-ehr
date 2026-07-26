#!/usr/bin/env python3
"""Map the horizon x granularity envelope — the two levers that survived every re-run.

Everything else was ruled out: predictor capacity leaves 0.02-0.03 of headroom, order-aware targets are
worse, and a full token-histogram context adds nothing. What did move the number, repeatedly and by a lot,
was WHERE the target sits and HOW COARSE it is:

    latent_next8   1.308      an 8-event window is too fine to predict
    latent_next32  0.824
    latent_next128 0.855
    latent_far32   1.282      offset 128 — predictability decays sharply with distance

Two points on each curve is a direction, not an envelope. This sweeps the grid so the operating region can
be read off rather than inferred.

Design, following the constructions that survived:
  * CEILING, not model — the train-fitted ridge (fitted on TRAIN with more rows than the model saw, scored
    on DEV). It answers "what is achievable here", which is what an envelope should be built from, and it
    is comparable across cells in a way a fixed-stride model's own predictions are not.
  * LINEAR ridge as the sweep estimator. In the validated train-fitted run RFF beat linear by 0.001-0.004
    (0.780 vs 0.781 MIMIC, 0.788 vs 0.792 SCID), so linear is within noise of the best family and is far
    cheaper across a grid. Stated rather than assumed.
  * MATCHED ELIGIBILITY across the whole grid — every cell uses the SAME rows, requiring max(offset+window)
    future tokens. Row eligibility is a selection on how much sequence remains and it dominates results, so
    letting each cell choose its own sample would make the grid unreadable.
  * Guards carried: shared-component degeneracy per cell, and persistence/chance as references.

DEV for evaluation, TRAIN for fitting, TEST untouched, aggregate-only.
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_horizon_granularity_sweep.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist
from scripts.rung2_train_fitted_ceiling import _chunked_ridge, _norm, _read_rows

WINDOWS = (4, 8, 16, 32, 64, 128)
OFFSETS = (0, 32, 128, 256)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--train-rows", type=int, default=40000)
    ap.add_argument("--dev-rows", type=int, default=6000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    need = max(o + w for o in OFFSETS for w in WINDOWS)      # ONE eligibility rule for the whole grid
    tr = _read_rows(args.target_blocks, "train", need=need, max_ctx=args.max_context_tokens,
                    max_blocks=args.train_rows, seed=args.seed)
    de = _read_rows(args.target_blocks, "dev", need=need, max_ctx=args.max_context_tokens,
                    max_blocks=args.dev_rows, seed=args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)

    def ctx(items):
        return _norm(np.asarray([E[c].mean(axis=0) if len(c) else np.zeros(E.shape[1]) for c, _ in items]))

    def tgt(items, off, win):
        return _norm(np.asarray([E[f[off:off + win]].mean(axis=0) if len(f[off:off + win])
                                 else np.zeros(E.shape[1]) for _, f in items]))

    lin = lambda X: np.hstack([X, np.ones((len(X), 1))])
    out = {"design": "train-fitted LINEAR ridge ceiling; one eligibility rule for the whole grid",
           "eligibility_future_tokens": need, "windows": list(WINDOWS), "offsets": list(OFFSETS),
           "aggregate_only": True, "per_source": {}}

    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        if len(itr) < 2000 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue
        Ctr, Cde = ctx(itr), ctx(ide)
        grid, rng = {}, np.random.default_rng(args.seed)
        for off in OFFSETS:
            for win in WINDOWS:
                Ttr, Tde = tgt(itr, off, win), tgt(ide, off, win)
                amb = ambient_true_nn_distance(Tde, np.arange(len(Tde)))
                pred = _chunked_ridge(lin, Ctr, Ttr, args.ridge, Cde)
                r = float(np.mean(cos_dist(pred, Tde))) / max(amb, 1e-9)
                pers = float(np.mean(cos_dist(Cde, Tde))) / max(amb, 1e-9)
                chance = float(np.mean(cos_dist(Tde[rng.permutation(len(Tde))], Tde))) / max(amb, 1e-9)
                shared = float(np.linalg.norm(Tde.mean(axis=0)))
                grid[f"off{off}_win{win}"] = {
                    "offset": off, "window": win, "ceiling": round(r, 4),
                    "persistence": round(pers, 4), "chance": round(chance, 4),
                    "ambient_nn": round(float(amb), 4), "shared_component_norm": round(shared, 4),
                    "degenerate": bool(shared > 0.80 or (chance < 2.0 and amb < 0.20)),
                    "clears_bar": bool(r < 1.0)}
        healthy = {k: v for k, v in grid.items() if not v["degenerate"]}
        best = min(healthy, key=lambda k: healthy[k]["ceiling"]) if healthy else None
        out["per_source"][src] = {
            "n_train": len(itr), "n_dev": len(ide), "grid": grid,
            "best_cell": best, "best_ceiling": healthy[best]["ceiling"] if best else None,
            "n_cells_clearing_bar": sum(1 for v in healthy.values() if v["clears_bar"]),
            "n_cells_healthy": len(healthy), "n_cells_degenerate": len(grid) - len(healthy)}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
