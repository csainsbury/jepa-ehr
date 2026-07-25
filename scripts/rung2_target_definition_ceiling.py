#!/usr/bin/env python3
"""Vary the TARGET DEFINITION — the last axis left open after three negatives.

Predictor capacity, target invariance and context encoding have each been ruled out for SCID: the trained
model sits at its own held-out ceiling, order-aware targets do not move it, and a raw token histogram
context does not either. Every one of those probes varied a MODEL component while holding one target
definition fixed — the latent identity of a 32-event block at a fixed offset. This varies the thing they
all held constant.

Two families of variation, over the SAME rows so the arms are comparable in sample:

  WINDOW / OFFSET (same target TYPE, mean-pooled latent identity)
    next 8 / 32 / 128 events, and a 128-token-offset horizon — is a coarser or more distant slice more
    predictable than a fine adjacent one? SCID is the long-sequence source, so a 32-event window may be
    too fine a cut of a slow trajectory.

  TARGET TYPE (coarser than latent identity, over the next 32 events)
    token_histogram    WHICH tokens occur and how often — distributional, not instance identity
    family_mix         the 7-way vocabulary family composition (lab / medication / state / diagnosis / ...)
                       — a coarse, clinically meaningful summary
    presence_binary    which vocabulary families appear at all

A within-space question is asked of each: does a fitted map clear 1.0 in THAT target's own space? Comparing
predictability ACROSS target spaces is the step that went wrong before, so the degeneracy guard and the
chance arm (which reports the usable dynamic range) travel with every row, and cross-space claims are made
only where the geometry is flagged healthy.

DEV only, TEST untouched, aggregate-only. Reads the governed substrate (approved).
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_target_definition_ceiling.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rung2_honest_ceiling import held_out_ceiling

# vocabulary family boundaries [start, end) — vocab METADATA (safe), from the joint dataset config
FAMILY_RANGES = {"special": (0, 4), "demographic": (4, 51), "diagnosis": (51, 91), "lab": (91, 951),
                 "medication": (951, 1032), "state": (1032, 1048), "dataset_context": (1048, 1050)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-blocks", type=int, default=6000)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--max-window", type=int, default=128, help="longest window required of EVERY row")
    ap.add_argument("--far-offset", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import h5py
    import torch

    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    need = args.far_offset + args.max_window          # every row must support the most demanding arm
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
                if c1 < c0 or t0 + need > len(ids):
                    continue                                   # SAME rows serve every arm
                ctx = np.asarray(ids[c0:c1 + 1][-args.max_context_tokens:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                fut = np.asarray(ids[t0:t0 + need], dtype=np.int64)
                per_src.setdefault(str(b.get("source_dataset")), []).append((ctx, fut))
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

    def mean_emb(ids):
        return E[ids].mean(axis=0) if len(ids) else np.zeros(E.shape[1])

    def histogram(ids):
        h = np.bincount(ids, minlength=V).astype(np.float64)
        return h / max(h.sum(), 1.0)

    def family_mix(ids):
        h = np.bincount(ids, minlength=V).astype(np.float64)
        v = np.array([h[lo:hi].sum() for lo, hi in FAMILY_RANGES.values()])
        return v / max(v.sum(), 1.0)

    def presence(ids):
        h = np.bincount(ids, minlength=V)
        return np.array([float((h[lo:hi] > 0).any()) for lo, hi in FAMILY_RANGES.values()])

    def target(fut, kind):
        if kind == "latent_next8":
            return mean_emb(fut[:8])
        if kind == "latent_next32":
            return mean_emb(fut[:32])
        if kind == "latent_next128":
            return mean_emb(fut[:128])
        if kind == "latent_far32":
            return mean_emb(fut[args.far_offset:args.far_offset + 32])
        if kind == "token_histogram32":
            return histogram(fut[:32])
        if kind == "family_mix32":
            return family_mix(fut[:32])
        if kind == "presence_binary32":
            return presence(fut[:32])
        raise ValueError(kind)

    kinds = ["latent_next8", "latent_next32", "latent_next128", "latent_far32",
             "token_histogram32", "family_mix32", "presence_binary32"]
    out = {"split": args.split, "aggregate_only": True, "checkpoint": Path(args.checkpoint).parent.name,
           "rows_require": need, "context": "mean_pooled (held fixed)", "per_source": {}}
    for src, items in sorted(per_src.items()):
        if len(items) < 300:
            out["per_source"][src] = {"n": len(items), "note": "too few rows"}
            continue
        C = np.asarray([mean_emb(c) for c, _ in items])          # context FIXED
        arms = {}
        for kind in kinds:
            T = np.asarray([target(f, kind) for _, f in items])
            r = held_out_ceiling(C, T, np.random.default_rng(args.seed))
            arms[kind] = {"target_dim": int(T.shape[1]), "ceiling_ratio": r["best_fitted_ratio"],
                          "best_family": r["best_fitted_family"], "ambient_nn": r["ambient_nn"],
                          "chance_ratio": r["chance_ratio"], "persistence_ratio": r["persistence_ratio"],
                          "degenerate_geometry": r["degenerate_geometry"],
                          "shared_component_norm": r["shared_component_norm"]}
        healthy = {k: v for k, v in arms.items() if not v["degenerate_geometry"]}
        best = min(healthy, key=lambda k: healthy[k]["ceiling_ratio"]) if healthy else None
        out["per_source"][src] = {
            "n": len(items), "arms": arms,
            "best_healthy_arm": best,
            "best_healthy_ratio": (healthy[best]["ceiling_ratio"] if best else None),
            "any_arm_clears_bar": bool(best and healthy[best]["ceiling_ratio"] < 1.0),
            "degenerate_arms": [k for k, v in arms.items() if v["degenerate_geometry"]]}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
