#!/usr/bin/env python3
"""HELD-OUT ceiling for the Rung-2 collapse question — and the controls that decide whether it means anything.

An IN-SAMPLE fit bounds in-sample error, but the capacity-vs-target question is about what a predictor can
achieve on data it has not seen. So the honest ceiling is HELD-OUT. That distinction is not academic here:
the in-sample RFF ceiling read 0.879-0.913 for SCID while its held-out counterpart is 1.127-1.198, i.e. the
nonlinear family fits noise. The earlier 0.708 headline came from a LINEAR in-sample fit whose held-out
counterpart was never computed at all.

Every family is therefore fitted on one half and scored on the other, and three controls decide whether a
sub-1.0 ceiling means anything:

  * PERSISTENCE — predict the target latent with the CONTEXT latent, unchanged. Patients resemble
    themselves, so a context->target map can look predictive while only exploiting autocorrelation. If
    persistence alone clears the bar, the "recoverable information" is not future-specific.
  * RANDOM EMBEDDING — recompute both latents from an UNTRAINED random embedding table. If the ceiling
    survives, the predictability is intrinsic to token statistics and owes nothing to the trained
    representation; if it collapses, the ceiling is a property of what training happened to learn.
  * CHANCE — a shuffled pairing, the floor.

DEV only, TEST untouched, aggregate-only. Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_honest_ceiling.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _ridge(X, Y, lam, Xe):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return _norm(Xe @ np.linalg.solve(A, X.T @ Y))


def held_out_ceiling(ctx, tgt, rng, *, rff_dim=256, lam=1e-3, folds=4, persistence_pred=None):
    """K-fold held-out fit for every family, plus persistence and chance. All ratios are vs ambient NN.

    `persistence_pred` must live in the TARGET space. When context and target share a space (mean pooling)
    the context itself is the natural persistence prediction and is used by default. When they do not — the
    order targets are 280/4096/4224-dim against a 256-dim context — persistence is undefined across spaces
    and the caller must supply a same-space analogue (e.g. the order target built from the context's
    trailing tokens), or it is reported as not-applicable rather than silently skipped."""
    c, t = _norm(ctx), _norm(tgt)
    n, d = c.shape
    amb = ambient_true_nn_distance(tgt, np.arange(n))
    rff_dim = int(min(rff_dim, max(8, n // 8)))
    W0 = rng.normal(size=(d, rff_dim)) / np.sqrt(d)
    b0 = rng.uniform(0, 2 * np.pi, size=rff_dim)
    Phi = np.hstack([np.sqrt(2.0 / rff_dim) * np.cos(c @ W0 + b0), np.ones((n, 1))])
    X = np.hstack([c, np.ones((n, 1))])

    idx = rng.permutation(n)
    cuts = np.array_split(idx, folds)
    acc = {"linear_held_out": [], "rff_held_out": [], "knn5_held_out": []}
    for f in range(folds):
        te = cuts[f]; tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
        acc["linear_held_out"].append(cos_dist(_ridge(X[tr], t[tr], lam, X[te]), t[te]))
        acc["rff_held_out"].append(cos_dist(_ridge(Phi[tr], t[tr], lam, Phi[te]), t[te]))
        sims = c[te] @ c[tr].T                                   # neighbours drawn only from TRAIN
        nb = np.argsort(-sims, axis=1)[:, :5]
        acc["knn5_held_out"].append(cos_dist(_norm(t[tr][nb].mean(axis=1)), t[te]))
    out = {k: float(np.mean(np.concatenate(v))) for k, v in acc.items()}
    if persistence_pred is not None:
        out["persistence_no_fit"] = float(np.mean(cos_dist(_norm(persistence_pred), t)))
    elif c.shape[1] == t.shape[1]:
        out["persistence_no_fit"] = float(np.mean(cos_dist(c, t)))   # same space: target = context
    # else: omitted — undefined across differing spaces; surfaced as persistence_ratio=None below
    out["chance_shuffled"] = float(np.mean(cos_dist(t[rng.permutation(n)], t)))
    # DEGENERACY GUARD. The ratio is only comparable across target spaces if no space is dominated by a
    # component shared by every instance. T3 concatenates a DETERMINISTIC rank code identical for all
    # sequences; with near-orthogonal random token embeddings that constant dominates the cosine geometry,
    # the ambient NN distance collapses (0.65 -> 0.13) and the ratio stops measuring predictability. Two
    # symptoms are recorded so a degenerate space cannot be read as a good one.
    shared = float(np.linalg.norm(t.mean(axis=0)))               # 0 = no common direction, 1 = all identical
    ratios = {k: round(v / max(amb, 1e-9), 4) for k, v in out.items()}
    chance_r = ratios["chance_shuffled"]
    degenerate = bool(shared > 0.80 or chance_r < 2.0 and amb < 0.20)
    best = min(("linear_held_out", "rff_held_out", "knn5_held_out"), key=lambda k: ratios[k])
    return {"ambient_nn": round(float(amb), 4), "n": int(n), "rff_features": int(rff_dim),
            "ratios": ratios, "best_fitted_family": best, "best_fitted_ratio": ratios[best],
            "persistence_ratio": ratios.get("persistence_no_fit"),
            "persistence_applicable": bool("persistence_no_fit" in ratios),
            "chance_ratio": ratios["chance_shuffled"],
            "shared_component_norm": round(shared, 4),
            "degenerate_geometry": degenerate,
            "degeneracy_note": ("OK" if not degenerate else
                                "DEGENERATE — a component shared by all instances dominates this target "
                                "space (ambient NN collapsed and/or chance range collapsed); the ratio is "
                                "NOT comparable with other representations and must not be read as skill")}


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
    rand_E = torch.tensor(np.random.default_rng(7).normal(size=(V, D)) / np.sqrt(D), dtype=torch.float32)

    def pad(seqs):
        L = max(len(s) for s in seqs)
        t = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            t[i, :len(s)] = torch.as_tensor(s)
        return t

    def rand_mean(ids):
        m = (ids != 0).float().unsqueeze(-1)
        e = torch.nn.functional.embedding(ids, rand_E) * m
        return (e.sum(1) / m.sum(1).clamp_min(1.0)).numpy()

    out = {"split": args.split, "aggregate_only": True, "checkpoint": Path(args.checkpoint).parent.name,
           "per_source": {}}
    for src, items in sorted(per_src.items()):
        if len(items) < 300:
            out["per_source"][src] = {"n": len(items), "note": "too few rows"}
            continue
        C_ids, T_ids = pad([c for c, _ in items]), pad([t for _, t in items])
        with torch.no_grad():
            C, T = model.mean_embed(C_ids).numpy(), model.mean_embed(T_ids).numpy()
            Pm = model.predict_rollout_from_context_ids(C_ids, 1)[:, 0, :].numpy()
        amb = ambient_true_nn_distance(T, np.arange(len(T)))
        trained = held_out_ceiling(C, T, np.random.default_rng(args.seed))
        random_rep = held_out_ceiling(rand_mean(C_ids), rand_mean(T_ids), np.random.default_rng(args.seed))
        out["per_source"][src] = {
            "n": len(items),
            "model_ratio": round(float(np.mean(cos_dist(Pm, T))) / max(amb, 1e-9), 4),
            "trained_representation": trained,
            "random_embedding_control": random_rep,
        }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
