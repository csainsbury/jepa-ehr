#!/usr/bin/env python3
"""Distributional cut head — per-position P(event j is inside the horizon), with K chosen so the cut matters.

The point-estimate cut head was marginal for SCID (composed 0.974) and unskilled for MIMIC — the latter
because at K=32 a 1-day horizon swallows 30 of 32 events, so 'take almost everything' is already right and
the head had nothing to add. Two corrections here:

  * K PER SOURCE, so the horizon falls INSIDE the block rather than at its edge. MIMIC: 32 events span
    ~0.98d, so a 1-day cut is trivial; 128 events span ~3.98d, putting the cut ~25% in. SCID keeps K=32
    (30d is ~14% of a 210d span).
  * A DISTRIBUTIONAL object. The right prediction is not a boundary index but, for each future position j,
    P(event j lands inside [t_query, t_query + H)). That yields a calibration target (gate 4A's object) and a
    SOFT prefix, which is what a probabilistic cut actually implies.

Arms: soft-weighted content (weight each event by its probability of being inside), hard cut at the expected
count, the true cut (must reproduce the target exactly — a sanity check), and the rate-only baseline the
contract requires. Calibration is reported as ECE, because a cut head that is confidently wrong is worse than
one that is uncertain.

IMPORTANT ARCHITECTURAL CAVEAT, made explicit because every oracle arm here depends on it: cutting a block at
an event boundary requires EVENT-LEVEL (token) predictions. A JEPA that emits a pooled LATENT cannot be cut
that way. All content in these arms is the TRUE token block, so what is validated is a route that presupposes
a token-level generator plus a timing head — not latent rollout followed by a cut. Rung 1 already rejected
frozen per-instance count/order/timing fidelity for the mean-pooled latent, so that generator does not yet
exist here.

DEV for evaluation, TRAIN for fitting, TEST untouched, aggregate-only.
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_distributional_cut_head.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist
from clinical_jepa.eval.rung2_timing import expected_calibration_error
from scripts.rung2_train_fitted_ceiling import _norm

# horizon and the K that puts the cut INSIDE the block (from the measured spans)
CFG = {"SCID": {"H": 30.0, "K": 32}, "MIMIC": {"H": 1.0, "K": 128}}


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
            if src not in CFG:
                continue
            K, H = CFG[src]["K"], CFG[src]["H"]
            if len(per_src.get(src, ())) >= max_blocks:
                if all(len(per_src.get(s, ())) >= max_blocks for s in CFG):
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
                if not np.isfinite(t_q) or float(days[-1]) < t_q + H:
                    continue
                ctx = np.asarray(ids[c0:c1 + 1][-max_ctx:], dtype=np.int64)
                rel = days[c1 + 1:c1 + 1 + K] - t_q
                if len(ctx) == 0 or not np.all(np.isfinite(rel)):
                    continue
                per_src.setdefault(src, []).append(
                    (ctx, np.asarray(ids[c1 + 1:c1 + 1 + K], dtype=np.int64), rel))
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

    out = {"config": CFG, "aggregate_only": True,
           "architectural_caveat": ("all content arms use the TRUE token block; cutting at an event boundary "
                                    "needs EVENT-LEVEL predictions, which a pooled-latent JEPA does not emit. "
                                    "This validates a token-generator + timing-head route, not latent rollout."),
           "per_source": {}}
    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        H, K = CFG[src]["H"], CFG[src]["K"]
        if len(itr) < 1500 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue
        Ctr = _norm(np.asarray([E[c].mean(axis=0) for c, _, _ in itr]))
        Cde = _norm(np.asarray([E[c].mean(axis=0) for c, _, _ in ide]))
        Itr = (np.asarray([r for _, _, r in itr]) < H).astype(np.float64)   # inside-horizon indicator, K cols
        Ide = (np.asarray([r for _, _, r in ide]) < H).astype(np.float64)

        # DISTRIBUTIONAL head: linear probability model per position, clipped to [0,1]
        X = np.hstack([Ctr, np.ones((len(Ctr), 1))])
        Xd = np.hstack([Cde, np.ones((len(Cde), 1))])
        Wt = np.linalg.solve(X.T @ X + args.ridge * np.eye(X.shape[1]), X.T @ Itr)
        P = np.clip(Xd @ Wt, 0.0, 1.0)
        Prate = np.tile(np.clip(Itr.mean(axis=0), 0, 1), (len(ide), 1))     # context-blind per-position rate

        true_cut = Ide.sum(axis=1).astype(int)
        keep = (true_cut > 0) & (true_cut < K)          # exclude trivial all-in / all-out rows
        if keep.sum() < 300:
            out["per_source"][src] = {"n_dev_nontrivial": int(keep.sum()),
                                      "note": "cut is trivial for almost every row at this K"}
            continue

        def soft(Pm):
            w = Pm / np.clip(Pm.sum(axis=1, keepdims=True), 1e-9, None)
            return _norm(np.asarray([w[i] @ E[ide[i][1]] for i in range(len(ide))]))

        def hard(Pm):
            n = np.rint(Pm.sum(axis=1)).astype(int)
            return _norm(np.asarray([E[ide[i][1][:int(np.clip(n[i], 0, K))]].mean(axis=0)
                                     if n[i] > 0 else np.zeros(E.shape[1]) for i in range(len(ide))]))

        Ttrue = hard(Ide)[keep]
        amb = ambient_true_nn_distance(Ttrue, np.arange(int(keep.sum())))
        rt = lambda R: round(float(np.mean(cos_dist(R[keep], Ttrue))) / max(amb, 1e-9), 4)

        ece = expected_calibration_error(P[keep].ravel(), Ide[keep].ravel())
        ece_rate = expected_calibration_error(Prate[keep].ravel(), Ide[keep].ravel())
        cnt_err = np.abs(P.sum(axis=1) - true_cut)[keep]
        rate_err = np.abs(Prate.sum(axis=1) - true_cut)[keep]
        res = {
            "n_train": len(itr), "n_dev_nontrivial": int(keep.sum()), "H_days": H, "K": K,
            "true_cut_median": float(np.median(true_cut[keep])),
            "true_cut_frac_of_K": round(float(np.median(true_cut[keep])) / K, 3),
            "calibration": {"head_ece": round(float(ece), 4), "rate_only_ece": round(float(ece_rate), 4),
                            "head_better_calibrated": bool(ece < ece_rate)},
            "expected_count_abs_err": {"head_median": float(np.median(cnt_err)),
                                       "rate_only_median": float(np.median(rate_err))},
            "composed_ceiling": {"true_cut": rt(hard(Ide)), "head_soft": rt(soft(P)),
                                 "head_hard": rt(hard(P)), "rate_only_soft": rt(soft(Prate)),
                                 "rate_only_hard": rt(hard(Prate))},
            "ambient_nn": round(float(amb), 4)}
        c = res["composed_ceiling"]
        best_head = min(c["head_soft"], c["head_hard"])
        best_rate = min(c["rate_only_soft"], c["rate_only_hard"])
        res["beats_rate_only"] = bool(best_head < best_rate)
        res["verdict"] = (
            f"USABLE — best head arm {best_head:.3f} clears 1.0 and beats rate-only {best_rate:.3f}"
            if best_head < 1.0 and best_head < best_rate else
            f"NOT USABLE — best head arm {best_head:.3f} vs rate-only {best_rate:.3f}")
        out["per_source"][src] = res
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
