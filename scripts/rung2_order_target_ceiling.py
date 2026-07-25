#!/usr/bin/env python3
"""Do the ORDER targets move the held-out ceiling that mean-pooling could not clear?

The honest held-out ceiling found that for SCID no fitted map beats the nearest-wrong-instance bar on the
mean-pooled target (best 1.134, trained model already at 1.148), and that an UNTRAINED random embedding
reproduces the trained representation almost exactly. That is a target/representation limit, not a capacity
one — so the lever is the TARGET, not a bigger predictor.

`clinical_jepa.targets.order_targets` already ships three frozen, parameter-free alternatives, built for
Rung-2 sub-gate 3 and shown earlier to separate permutations that mean-pooling cannot distinguish at all:

    T1_pooled_ordinal      mean E  (+) mean psi(rank) (+) proj(mean(E (x) psi(rank)))
    T2_seq_of_latents      ordered stack [E(id_1) .. E(id_L)]        (order = stack index)
    T3_ordinal_tagged_seq  [E(id_i) (+) psi(rank_i)]_{i=1..L}

This asks the one question that follows: does predicting one of THOSE from the same context clear the bar
that mean-pooling could not? The comparison is legitimate across target spaces because the metric is
self-normalising — d_self divided by that space's OWN ambient nearest-neighbour distance — so it always
asks "is the prediction closer to its own target than to the nearest other target, in this space".

Same discipline as the honest ceiling, and for the same reason: every fit is HELD OUT (k-fold), and the
persistence, random-embedding and chance controls come along, because an in-sample number here would be as
misleading as it was there.

DEV only, TEST untouched, aggregate-only. Reads the governed substrate (approved).
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_order_target_ceiling.py --help

CONFOUNDER — READ BEFORE REUSING THE 'ORDER TARGETS DO NOT MOVE THE CEILING' READING (T1/T2/T3 ALL 1.13-1.18 FOR SCID) (added after the target-definition probe).

Every result here was computed with an eligibility rule of "the row must have >= 32 future tokens". That
rule is a SELECTION on how much sequence remains, and it dominates the outcome. A later control varied only
that rule, holding the target definition, model, and metric identical:

    rows needing >=  32 future tokens : SCID n=1722, latent_next32 ceiling 1.134  (fails the <1.0 bar)
    rows needing >= 256 future tokens : SCID n=3775, latent_next32 ceiling 0.824  (clears it)

The ambient normaliser FELL (0.212 -> 0.165) across those samples, so the denominator got harder and the
ratio still improved — the gain is real, not a normalisation artefact.

Consequence: this script's internal logic is sound (it varies one component while holding the rest fixed),
and its WITHIN-RUN comparisons remain valid, because all arms share one sample. But its absolute SCID
numbers, and any "hypothesis ruled out for SCID" reading built on them, are SAMPLE-SPECIFIC — they describe
the short-future subset, where prediction is genuinely harder. They are NOT evidence that the component
under test is irrelevant in general.

Re-running under a matched, longer-future eligibility rule is the proper fix and HAS NOT been done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.targets.order_targets import ORDER_TARGET_NAMES, build_order_target
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
    V, D = int(ck["vocab_size"]), int(ck["embedding_dim"])
    E_trained = model.embedding.weight.detach().numpy().astype(np.float64)
    E_random = np.random.default_rng(7).normal(size=(V, D)) / np.sqrt(D)
    z_empty = np.zeros(D, dtype=np.float32)

    def pad(seqs):
        L = max(len(s) for s in seqs)
        t = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            t[i, :len(s)] = torch.as_tensor(s)
        return t

    def ctx_rep(ids_list, E):
        out = []
        for ids in ids_list:
            e = E[ids]
            out.append(e.mean(axis=0) if len(ids) else np.zeros(E.shape[1]))
        return np.asarray(out)

    def mean_target(ids_list, E):
        return ctx_rep(ids_list, E)

    def order_target(ids_list, E, name):
        return np.asarray([build_order_target(name, ids, E=E, z_empty=z_empty)[0] for ids in ids_list])

    out = {"split": args.split, "aggregate_only": True, "checkpoint": Path(args.checkpoint).parent.name,
           "note": ("ratios are vs each target space's OWN ambient nearest-neighbour distance, so they are "
                    "comparable across target representations; <1.0 clears the nearest-wrong-instance bar"),
           "per_source": {}}
    for src, items in sorted(per_src.items()):
        if len(items) < 300:
            out["per_source"][src] = {"n": len(items), "note": "too few rows"}
            continue
        ctx_ids = [c for c, _ in items]
        tgt_ids = [t for _, t in items]
        arms = {}
        for label, E in (("trained_embedding", E_trained), ("random_embedding_control", E_random)):
            C = ctx_rep(ctx_ids, E)
            reps = {"mean_pooled_baseline": mean_target(tgt_ids, E)}
            for nm in ORDER_TARGET_NAMES:
                reps[nm] = order_target(tgt_ids, E, nm)
            arms[label] = {}
            # same-space persistence: build the SAME target representation from the context's trailing
            # tokens, i.e. "the next ordered block looks like the last one". Undefined across spaces, so it
            # must be constructed per representation rather than reused from the mean-pooled case.
            tail = [c[-args.target_window_events:] for c in ctx_ids]
            persist = {"mean_pooled_baseline": mean_target(tail, E)}
            for nm2 in ORDER_TARGET_NAMES:
                persist[nm2] = order_target(tail, E, nm2)
            for nm, T in reps.items():
                r = held_out_ceiling(C, T, np.random.default_rng(args.seed),
                                     persistence_pred=persist[nm])
                arms[label][nm] = {"dim": int(T.shape[1]), "ambient_nn": r["ambient_nn"],
                                   "best_fitted_ratio": r["best_fitted_ratio"],
                                   "best_family": r["best_fitted_family"],
                                   "persistence_ratio": r["persistence_ratio"],
                                   "shared_component_norm": r["shared_component_norm"],
                                   "degenerate_geometry": r["degenerate_geometry"],
                                   "degeneracy_note": r["degeneracy_note"],
                                   "ratios": r["ratios"]}
        out["per_source"][src] = {"n": len(items), "arms": arms}

    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
