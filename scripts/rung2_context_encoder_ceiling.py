#!/usr/bin/env python3
"""Is the CONTEXT encoder the bottleneck? Vary the context representation, hold the target fixed.

Two results narrowed this down. The honest held-out ceiling found SCID has no out-of-sample headroom on
the mean-pooled target (model 1.148, ceiling 1.134) and that an untrained random embedding matches the
trained one. The order-target probe then showed that making the TARGET order-aware does not help either
(T1/T2/T3 all 1.13-1.18, none clearing 1.0). So the limit is not the target's invariance — which points at
the CONTEXT, also mean-pooled, simply not carrying enough to predict any target of this block.

This varies ONLY the context representation and holds the target at the mean-pooled baseline. That is
deliberate: with the target space fixed, every arm shares the SAME ambient nearest-neighbour normaliser, so
the ratios are directly comparable — avoiding the cross-space degeneracy that made the T3 arm unreadable.

Context arms, all parameter-free (no training run required, which is the point):

    mean_pooled        the incumbent: mean of context token embeddings — order- and multiplicity-blind
    last16_mean        recency only: mean of the final 16 tokens
    multiscale         whole (+) last 32 (+) last 8 — coarse and fine together
    mean_plus_std      first and second moments — adds dispersion the mean discards
    token_histogram    L1-normalised COUNT VECTOR over the vocabulary; uses NO embedding at all, and
                       preserves which tokens occurred and how often — precisely what pooling destroys
    histogram_plus_mean  both

If a richer, still training-free context drops the ceiling below 1.0, the information was present all
along and the mean-pooled ENCODER was discarding it. If nothing moves, the limit is upstream of the
representation.

DEV only, TEST untouched, aggregate-only. Reads the governed substrate (approved).
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_context_encoder_ceiling.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rung2_honest_ceiling import held_out_ceiling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-blocks", type=int, default=6000)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--target-window-events", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import h5py
    import torch

    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    blocks = [b for b in json.loads(Path(args.target_blocks).read_text()).get("blocks", [])
              if str(b.get("split")) == args.split and b.get("sequence_file") and b.get("sequence_group")
              and int(b.get("target_start_ref", -1)) >= 0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(blocks)
    cache, per_src = {}, {}
    try:
        for b in blocks:
            if sum(len(v) for v in per_src.values()) >= args.max_blocks:
                break
            try:
                p = str(b["sequence_file"])
                if p not in cache:
                    cache[p] = h5py.File(p, "r")
                ids = cache[p][str(b["sequence_group"])]["token_ids"][:]
                c0, c1 = max(0, int(b.get("context_start_ref", 0))), int(b["context_end_ref"])
                t0 = int(b["target_start_ref"])
                if c1 < c0 or t0 + args.target_window_events > len(ids):
                    continue
                ctx = np.asarray(ids[c0:c1 + 1][-args.max_context_tokens:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                per_src.setdefault(str(b.get("source_dataset")), []).append(
                    (ctx, np.asarray(ids[t0:t0 + args.target_window_events], dtype=np.int64)))
            except Exception:
                continue
    finally:
        for h in cache.values():
            h.close()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    V = int(ck["vocab_size"])
    E = model.embedding.weight.detach().numpy().astype(np.float64)

    def emb_mean(ids, lo=None):
        sel = ids[-lo:] if lo else ids
        return E[sel].mean(axis=0) if len(sel) else np.zeros(E.shape[1])

    def hist(ids):
        h = np.bincount(ids, minlength=V).astype(np.float64)
        return h / max(h.sum(), 1.0)

    def build(ctx_ids, kind):
        rows = []
        for ids in ctx_ids:
            if kind == "mean_pooled":
                r = emb_mean(ids)
            elif kind == "last16_mean":
                r = emb_mean(ids, 16)
            elif kind == "multiscale":
                r = np.concatenate([emb_mean(ids), emb_mean(ids, 32), emb_mean(ids, 8)])
            elif kind == "mean_plus_std":
                e = E[ids]
                r = np.concatenate([e.mean(axis=0), e.std(axis=0)]) if len(ids) else np.zeros(2 * E.shape[1])
            elif kind == "token_histogram":
                r = hist(ids)
            elif kind == "histogram_plus_mean":
                r = np.concatenate([hist(ids), emb_mean(ids)])
            else:
                raise ValueError(kind)
            rows.append(r)
        return np.asarray(rows)

    kinds = ["mean_pooled", "last16_mean", "multiscale", "mean_plus_std",
             "token_histogram", "histogram_plus_mean"]
    out = {"split": args.split, "aggregate_only": True, "checkpoint": Path(args.checkpoint).parent.name,
           "design": ("target held FIXED at the mean-pooled baseline so every arm shares one ambient "
                      "normaliser and the ratios are directly comparable"),
           "per_source": {}}
    for src, items in sorted(per_src.items()):
        if len(items) < 300:
            out["per_source"][src] = {"n": len(items), "note": "too few rows"}
            continue
        ctx_ids = [c for c, _ in items]
        T = build([t for _, t in items], "mean_pooled")          # FIXED target
        arms = {}
        for kind in kinds:
            C = build(ctx_ids, kind)
            r = held_out_ceiling(C, T, np.random.default_rng(args.seed))
            arms[kind] = {"context_dim": int(C.shape[1]), "ceiling_ratio": r["best_fitted_ratio"],
                          "best_family": r["best_fitted_family"], "ratios": r["ratios"],
                          "degenerate_geometry": r["degenerate_geometry"]}
        base = arms["mean_pooled"]["ceiling_ratio"]
        best = min(arms, key=lambda k: arms[k]["ceiling_ratio"])
        out["per_source"][src] = {
            "n": len(items), "ambient_nn": r["ambient_nn"], "arms": arms,
            "baseline_ratio": base, "best_arm": best, "best_ratio": arms[best]["ceiling_ratio"],
            "improvement_over_mean_pooling": round(base - arms[best]["ceiling_ratio"], 4),
            "clears_bar": bool(arms[best]["ceiling_ratio"] < 1.0)}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
