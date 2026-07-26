#!/usr/bin/env python3
"""The CUT-POINT head — the one component the AR decomposition still needs.

The decomposition is established: predicting the next K events is achievable, those K events already CONTAIN
the wall-clock window (32 events cover SCID's 30-day window 99.6% of the time), and given the true block the
target is recoverable (MIMIC 0.385, SCID 0.944) provided per-event timing is retained. What no probe has yet
supplied is the CUT: which of the next K events fall inside [t_query, t_query + H).

This builds it and isolates its quality. The head predicts, from CONTEXT ONLY, the cumulative days to each of
the next K events (ridge on log1p, fitted on TRAIN, closed form — the same construction as the validated
ceilings, so no training loop and no new failure surface). The cut is then `predicted_cum_days[j] < H`.

Three arms, so the cut is measured rather than inferred:

  TRUE CUT      — the true event block cut at the true boundary. The upper bound already measured.
  PREDICTED CUT — the true event block cut where the HEAD says the horizon falls. The quantity of interest:
                  content is granted, so any shortfall is purely the cut.
  RATE-ONLY CUT — the same, but timing comes from a context-blind global rate. The contract requires this
                  baseline for sub-gate 4: a head must beat rate-only, or it has learned nothing about
                  timing specifically.

Reported both as cut accuracy (index error) and, more importantly, as the composed ceiling ratio — because a
cut can be off by an event or two and still recover the windowed content, or be nearly right and miss a
dense burst. Only the composed metric says whether the decomposition works end to end.

DEV for evaluation, TRAIN for fitting, TEST untouched, aggregate-only.
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_cut_point_head.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist
from scripts.rung2_train_fitted_ceiling import _norm

HORIZONS = {"SCID": 30.0, "MIMIC": 1.0}
K = 32


def _rows(target_blocks, split, *, max_ctx, max_blocks, seed):
    import h5py
    blocks = [b for b in json.loads(Path(target_blocks).read_text()).get("blocks", [])
              if str(b.get("split")) == split and b.get("sequence_file") and b.get("sequence_group")
              and int(b.get("target_start_ref", -1)) >= 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    cache, per_src = {}, {}
    try:
        for b in blocks:
            src = str(b.get("source_dataset"))
            if src not in HORIZONS:
                continue
            if len(per_src.get(src, ())) >= max_blocks:
                if all(len(per_src.get(s, ())) >= max_blocks for s in HORIZONS):
                    break
                continue
            try:
                p = str(b["sequence_file"])
                if p not in cache:
                    cache[p] = h5py.File(p, "r")
                g = cache[p][str(b["sequence_group"])]
                ids, days = g["token_ids"][:], np.asarray(g["cumulative_days"][:], dtype=np.float64)
                c0, c1 = max(0, int(b.get("context_start_ref", 0))), int(b["context_end_ref"])
                if c1 < c0 or c1 + 1 + K > len(ids):
                    continue
                t_q = float(days[c1])
                if not np.isfinite(t_q) or float(days[-1]) < t_q + HORIZONS[src]:
                    continue
                ctx = np.asarray(ids[c0:c1 + 1][-max_ctx:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                fut_ids = np.asarray(ids[c1 + 1:c1 + 1 + K], dtype=np.int64)
                fut_rel = days[c1 + 1:c1 + 1 + K] - t_q            # days since t_query, per future event
                if not np.all(np.isfinite(fut_rel)):
                    continue
                per_src.setdefault(src, []).append((ctx, fut_ids, fut_rel))
            except Exception:
                continue
    finally:
        for h in cache.values():
            h.close()
    return per_src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--train-rows", type=int, default=25000)
    ap.add_argument("--dev-rows", type=int, default=4000)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    tr = _rows(args.target_blocks, "train", max_ctx=args.max_context_tokens,
               max_blocks=args.train_rows, seed=args.seed)
    de = _rows(args.target_blocks, "dev", max_ctx=args.max_context_tokens,
               max_blocks=args.dev_rows, seed=args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)

    out = {"horizons_days": HORIZONS, "k_events": K, "aggregate_only": True, "per_source": {}}
    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        H = HORIZONS[src]
        if len(itr) < 1500 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue

        def ctx_rep(items):
            return _norm(np.asarray([E[c].mean(axis=0) for c, _, _ in items]))

        Ctr, Cde = ctx_rep(itr), ctx_rep(ide)
        Ytr = np.log1p(np.asarray([r for _, _, r in itr]))          # log1p cumulative days, K columns
        Rde = np.asarray([r for _, _, r in ide])

        # the HEAD: ridge from context -> log1p(cumulative days) per future position
        X = np.hstack([Ctr, np.ones((len(Ctr), 1))])
        Xd = np.hstack([Cde, np.ones((len(Cde), 1))])
        Wt = np.linalg.solve(X.T @ X + args.ridge * np.eye(X.shape[1]), X.T @ Ytr)
        pred_days = np.expm1(Xd @ Wt)

        # RATE-ONLY baseline: context-blind per-position median from TRAIN
        rate_days = np.tile(np.expm1(np.median(Ytr, axis=0)), (len(ide), 1))

        def cut_idx(rel):                                            # count of events inside the horizon
            return (rel < H).sum(axis=1)

        true_cut = cut_idx(Rde)
        head_cut = cut_idx(pred_days)
        rate_cut = cut_idx(rate_days)

        def compose(cuts):
            reps = []
            for (_, fi, _), n in zip(ide, cuts):
                n = int(np.clip(n, 0, len(fi)))
                reps.append(E[fi[:n]].mean(axis=0) if n > 0 else np.zeros(E.shape[1]))
            return _norm(np.asarray(reps))

        keep = true_cut > 0                                          # a non-empty true window to compare to
        Ttrue = compose(true_cut)[keep]
        amb = ambient_true_nn_distance(Ttrue, np.arange(int(keep.sum())))
        ratio = lambda R: round(float(np.mean(cos_dist(R[keep], Ttrue))) / max(amb, 1e-9), 4)

        err = np.abs(head_cut - true_cut)[keep]
        rerr = np.abs(rate_cut - true_cut)[keep]
        out["per_source"][src] = {
            "n_train": len(itr), "n_dev": int(keep.sum()), "horizon_days": H, "K": K,
            "ambient_nn": round(float(amb), 4),
            "true_cut_median_events": float(np.median(true_cut[keep])),
            "cut_accuracy": {
                "head_exact_rate": round(float(np.mean(err == 0)), 4),
                "head_within_1": round(float(np.mean(err <= 1)), 4),
                "head_median_abs_err": float(np.median(err)),
                "rate_only_exact_rate": round(float(np.mean(rerr == 0)), 4),
                "rate_only_median_abs_err": float(np.median(rerr))},
            "composed_ceiling": {
                "true_cut": ratio(compose(true_cut)),                # 0 by construction; sanity check
                "head_cut": ratio(compose(head_cut)),
                "rate_only_cut": ratio(compose(rate_cut))},
            "beats_rate_only": bool(ratio(compose(head_cut)) < ratio(compose(rate_cut))),
        }
        r = out["per_source"][src]["composed_ceiling"]
        out["per_source"][src]["verdict"] = (
            f"CUT IS USABLE — head-cut content sits at {r['head_cut']:.3f} vs rate-only "
            f"{r['rate_only_cut']:.3f}; the decomposition composes end to end"
            if r["head_cut"] < 1.0 and r["head_cut"] < r["rate_only_cut"] else
            f"CUT NOT USABLE — head-cut {r['head_cut']:.3f} (rate-only {r['rate_only_cut']:.3f}); the "
            f"timing head does not yet recover the window even with true content")
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
