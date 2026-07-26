#!/usr/bin/env python3
"""WALL-CLOCK targets — turning the event-count envelope into a clinically meaningful one.

Everything measured so far used EVENT-COUNT windows, but the clinical question is always temporal ("what
happens in the next 30 days"), and 64 events means something entirely different inside a MIMIC admission
than across a year of SCI-D care. Rung 0 already established there is no common cross-source wall-clock
horizon, so the grid here is PER SOURCE, taking its horizons from the frozen Rung-0 choices.

Two wall-clock-specific hazards, both handled rather than assumed away:

  * SATURATION. The feasibility grid shows MIMIC saturates brutally — 63% of windows at W=3 days and >=90%
    at W>=7 are "the entire remaining sequence". A saturated window is NOT a horizon test: the target is
    simply everything left. MIMIC windows are therefore sub-day, matching Rung 0.
  * CENSORING. A window whose end lies past the last recorded event is not fully observed. Scoring it would
    measure "how much record remains" rather than predictability. Rows are required to have a FULLY OBSERVED
    window, and empty/censored exclusions are counted and reported per cell.

Span semantics mirror production `_wall_clock_target_span`: half-open [t_query + offset, t_query + offset +
window), and only events strictly AFTER context_end are eligible so the context-end event cannot leak in.

The eligibility rule (one fully-observed window at the grid's most demanding cell) is applied to EVERY cell,
because eligibility is a selection on how much record remains and it has repeatedly dominated results — the
event-count version moved SCID's usable horizon by 3-4x on its own.

Ceiling is the validated train-fitted linear ridge: fitted on TRAIN with more rows than the model saw,
scored on DEV. TEST untouched. Aggregate-only.

Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_wallclock_target_ceiling.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist
from scripts.rung2_train_fitted_ceiling import _chunked_ridge, _norm

# per-source wall-clock grids, taken from the frozen Rung-0 horizons (no common cross-source horizon exists)
GRIDS = {
    "SCID":  {"windows": (30.0, 90.0, 365.0), "offsets": (0.0, 30.0, 90.0, 365.0)},
    "MIMIC": {"windows": (0.25, 0.5, 1.0),    "offsets": (0.0, 0.25, 0.5, 1.0)},
}


def _rows(target_blocks, split, *, max_ctx, max_blocks, seed, need_days):
    """Read (context tokens, cumulative_days, token_ids, t_query, context_end) keeping only rows whose most
    demanding window is FULLY OBSERVED."""
    import h5py
    blocks = [b for b in json.loads(Path(target_blocks).read_text()).get("blocks", [])
              if str(b.get("split")) == split and b.get("sequence_file") and b.get("sequence_group")
              and int(b.get("target_start_ref", -1)) >= 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    cache, per_src, excl = {}, {}, {}
    try:
        for b in blocks:
            src = str(b.get("source_dataset"))
            if src not in GRIDS:
                continue
            if len(per_src.get(src, ())) >= max_blocks:
                if all(len(per_src.get(s, ())) >= max_blocks for s in GRIDS):
                    break
                continue
            try:
                p = str(b["sequence_file"])
                if p not in cache:
                    cache[p] = h5py.File(p, "r")
                g = cache[p][str(b["sequence_group"])]
                ids = g["token_ids"][:]
                days = np.asarray(g["cumulative_days"][:], dtype=np.float64)
                c0, c1 = max(0, int(b.get("context_start_ref", 0))), int(b["context_end_ref"])
                if c1 < c0 or c1 >= len(ids):
                    continue
                t_query = float(days[c1])
                if not np.isfinite(t_query):
                    continue
                # FULLY OBSERVED at the most demanding cell, else the window is censored
                if float(days[-1]) < t_query + need_days[src]:
                    excl[src] = excl.get(src, 0) + 1
                    continue
                ctx = np.asarray(ids[c0:c1 + 1][-max_ctx:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                per_src.setdefault(src, []).append((ctx, ids, days, t_query, c1))
            except Exception:
                continue
    finally:
        for h in cache.values():
            h.close()
    return per_src, excl


def _span_tokens(ids, days, t_query, c1, off, win):
    """Half-open [t_query+off, t_query+off+win), events strictly after context_end (production semantics)."""
    lo, hi = t_query + off, t_query + off + win
    sel = np.nonzero((days > -np.inf) & (days >= lo) & (days < hi))[0]
    sel = sel[sel > c1]
    return ids[sel]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--train-rows", type=int, default=30000)
    ap.add_argument("--dev-rows", type=int, default=5000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    need = {s: max(o + w for o in g["offsets"] for w in g["windows"]) for s, g in GRIDS.items()}
    tr, extr = _rows(args.target_blocks, "train", max_ctx=args.max_context_tokens,
                     max_blocks=args.train_rows, seed=args.seed, need_days=need)
    de, exde = _rows(args.target_blocks, "dev", max_ctx=args.max_context_tokens,
                     max_blocks=args.dev_rows, seed=args.seed, need_days=need)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)
    lin = lambda X: np.hstack([X, np.ones((len(X), 1))])

    out = {"design": ("PER-SOURCE wall-clock grid from the frozen Rung-0 horizons; fully-observed windows "
                      "only; train-fitted linear ridge ceiling"),
           "eligibility_days_required": need, "grids": {s: {k: list(v) for k, v in g.items()}
                                                        for s, g in GRIDS.items()},
           "censored_excluded": {"train": extr, "dev": exde}, "aggregate_only": True, "per_source": {}}

    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        if len(itr) < 1500 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue
        Ctr = _norm(np.asarray([E[c].mean(axis=0) for c, *_ in itr]))
        Cde = _norm(np.asarray([E[c].mean(axis=0) for c, *_ in ide]))
        grid, rng = {}, np.random.default_rng(args.seed)
        for off in GRIDS[src]["offsets"]:
            for win in GRIDS[src]["windows"]:
                def rep(items):
                    toks = [_span_tokens(i, d, t, c, off, win) for _, i, d, t, c in items]
                    keep = np.array([len(x) > 0 for x in toks])
                    reps = np.asarray([E[x].mean(axis=0) if len(x) else np.zeros(E.shape[1]) for x in toks])
                    return _norm(reps), keep, np.array([len(x) for x in toks])
                Ttr, ktr, _ = rep(itr)
                Tde, kde, ntok = rep(ide)
                if kde.sum() < 200 or ktr.sum() < 1000:
                    grid[f"off{off}_win{win}"] = {"offset": off, "window": win,
                                                  "n_dev_nonempty": int(kde.sum()),
                                                  "note": "too few non-empty targets"}
                    continue
                Tt, Ct = Ttr[ktr], Ctr[ktr]
                Td, Cd = Tde[kde], Cde[kde]
                amb = ambient_true_nn_distance(Td, np.arange(len(Td)))
                pred = _chunked_ridge(lin, Ct, Tt, args.ridge, Cd)
                r = float(np.mean(cos_dist(pred, Td))) / max(amb, 1e-9)
                pers = float(np.mean(cos_dist(Cd, Td))) / max(amb, 1e-9)
                chance = float(np.mean(cos_dist(Td[rng.permutation(len(Td))], Td))) / max(amb, 1e-9)
                shared = float(np.linalg.norm(Td.mean(axis=0)))
                grid[f"off{off}_win{win}"] = {
                    "offset": off, "window": win, "n_dev_nonempty": int(kde.sum()),
                    "empty_rate_dev": round(1.0 - float(kde.mean()), 4),
                    "median_events_in_window": float(np.median(ntok[kde])),
                    "ceiling": round(r, 4), "persistence": round(pers, 4), "chance": round(chance, 4),
                    "ambient_nn": round(float(amb), 4), "shared_component_norm": round(shared, 4),
                    "degenerate": bool(shared > 0.80 or (chance < 2.0 and amb < 0.20)),
                    "clears_bar": bool(r < 1.0)}
        scored = {k: v for k, v in grid.items() if "ceiling" in v and not v["degenerate"]}
        best = min(scored, key=lambda k: scored[k]["ceiling"]) if scored else None
        out["per_source"][src] = {
            "n_train": len(itr), "n_dev": len(ide), "grid": grid, "best_cell": best,
            "best_ceiling": scored[best]["ceiling"] if best else None,
            "n_cells_clearing_bar": sum(1 for v in scored.values() if v["clears_bar"]),
            "n_cells_scored": len(scored), "n_cells_skipped": len(grid) - len(scored)}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
