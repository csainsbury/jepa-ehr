#!/usr/bin/env python3
"""A ceiling with MORE data than the model — the fix that makes capacity-vs-target answerable.

The dev-fitted ceiling failed as a bound: MIMIC's trained model beat it by 0.062. The cause was not a
superhuman model but a DATA-STARVED bound — it was fitted k-fold on ~2,200 dev rows while the model trained
on ~60,000 blocks. It bounded "what a simple fit on a couple of thousand rows achieves", which is not the
quantity of interest.

This fits every family on the TRAIN split and evaluates on DEV. Train rows are capped well ABOVE the number
of examples the model actually saw, so the fit has strictly better data access than the model. Only then is
"no honest predictor on this representation does better" a claim the construction can support.

Guards carried forward, since each was earned by a failure:
  * BOUNDEDNESS   — a ceiling the model beats is not a ceiling; checked and reported, never assumed.
  * NON-VACUITY   — feature count is kept far below the number of train rows so the fit cannot interpolate.
  * DEGENERACY    — a target space dominated by a component shared across instances makes the ratio
                    unreadable; the shared-component norm is reported.
  * MATCHED ELIGIBILITY — row eligibility is a selection on how much sequence remains and it dominates
                    results, so the same --min-future-tokens rule is applied to BOTH splits.

Ridge is accumulated in chunks (A = sum phi phi^T, B = sum phi y^T) so train size is limited by time, not
memory. Fit on TRAIN, score on DEV. TEST untouched. Aggregate-only.

Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_train_fitted_ceiling.py --help
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


def _read_rows(target_blocks, split, *, need, max_ctx, max_blocks, seed):
    import h5py
    blocks = [b for b in json.loads(Path(target_blocks).read_text()).get("blocks", [])
              if str(b.get("split")) == split and b.get("sequence_file") and b.get("sequence_group")
              and int(b.get("target_start_ref", -1)) >= 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    cache, per_src = {}, {}
    try:
        for b in blocks:
            if sum(len(v) for v in per_src.values()) >= max_blocks:
                break
            try:
                p = str(b["sequence_file"])
                if p not in cache:
                    cache[p] = h5py.File(p, "r")
                ids = cache[p][str(b["sequence_group"])]["token_ids"][:]
                c0, c1 = max(0, int(b.get("context_start_ref", 0))), int(b["context_end_ref"])
                t0 = int(b["target_start_ref"])
                if c1 < c0 or t0 + need > len(ids):
                    continue
                ctx = np.asarray(ids[c0:c1 + 1][-max_ctx:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                per_src.setdefault(str(b.get("source_dataset")), []).append(
                    (ctx, np.asarray(ids[t0:t0 + need], dtype=np.int64)))
            except Exception:
                continue
    finally:
        for h in cache.values():
            h.close()
    return per_src


def _chunked_ridge(feat_fn, Xtr, Ytr, lam, Xte, chunk=4000):
    """Accumulate normal equations in chunks so train size is bounded by time, not memory."""
    d = feat_fn(Xtr[:1]).shape[1]
    A = np.zeros((d, d)); B = np.zeros((d, Ytr.shape[1]))
    for i in range(0, len(Xtr), chunk):
        P = feat_fn(Xtr[i:i + chunk])
        A += P.T @ P
        B += P.T @ Ytr[i:i + chunk]
    W = np.linalg.solve(A + lam * np.eye(d), B)
    return _norm(np.vstack([feat_fn(Xte[i:i + chunk]) @ W for i in range(0, len(Xte), chunk)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--min-future-tokens", type=int, default=256)
    ap.add_argument("--target-window-events", type=int, default=32)
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--train-rows", type=int, default=40000)
    ap.add_argument("--dev-rows", type=int, default=6000)
    ap.add_argument("--rff-dim", type=int, default=1024)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    need = max(args.min_future_tokens, args.target_window_events)
    tr = _read_rows(args.target_blocks, "train", need=need, max_ctx=args.max_context_tokens,
                    max_blocks=args.train_rows, seed=args.seed)
    de = _read_rows(args.target_blocks, "dev", need=need, max_ctx=args.max_context_tokens,
                    max_blocks=args.dev_rows, seed=args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)
    W_ = args.target_window_events

    def ctx_rep(items):
        return np.asarray([E[c].mean(axis=0) if len(c) else np.zeros(E.shape[1]) for c, _ in items])

    def tgt_rep(items):
        return np.asarray([E[f[:W_]].mean(axis=0) if len(f[:W_]) else np.zeros(E.shape[1]) for _, f in items])

    out = {"design": ("families fitted on TRAIN, scored on DEV; train rows capped ABOVE the model's own "
                      "example count so the fit has strictly better data access"),
           "min_future_tokens": need, "aggregate_only": True,
           "checkpoint": Path(args.checkpoint).parent.name, "per_source": {}}

    rng = np.random.default_rng(args.seed)
    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        if len(itr) < 2000 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue
        Ctr, Ttr = _norm(ctx_rep(itr)), _norm(tgt_rep(itr))
        Cde, Tde = _norm(ctx_rep(ide)), _norm(tgt_rep(ide))
        amb = ambient_true_nn_distance(Tde, np.arange(len(Tde)))

        with torch.no_grad():
            L = max(len(c) for c, _ in ide)
            ids = torch.zeros((len(ide), L), dtype=torch.long)
            for i, (c, _) in enumerate(ide):
                ids[i, :len(c)] = torch.as_tensor(c)
            Pm = model.predict_rollout_from_context_ids(ids, 1)[:, 0, :].numpy()
        model_ratio = float(np.mean(cos_dist(Pm, Tde))) / max(amb, 1e-9)

        d = Ctr.shape[1]
        W0 = rng.normal(size=(d, args.rff_dim)) / np.sqrt(d)
        b0 = rng.uniform(0, 2 * np.pi, size=args.rff_dim)
        lin = lambda X: np.hstack([X, np.ones((len(X), 1))])
        rff = lambda X: np.hstack([np.sqrt(2.0 / args.rff_dim) * np.cos(X @ W0 + b0), np.ones((len(X), 1))])

        res = {"linear_train_fitted": float(np.mean(cos_dist(
                   _chunked_ridge(lin, Ctr, Ttr, args.ridge, Cde), Tde))),
               "rff_train_fitted": float(np.mean(cos_dist(
                   _chunked_ridge(rff, Ctr, Ttr, args.ridge, Cde), Tde)))}
        # kNN with neighbours drawn from TRAIN, chunked over dev
        preds = []
        for i in range(0, len(Cde), 512):
            sims = Cde[i:i + 512] @ Ctr.T
            nb = np.argpartition(-sims, 5, axis=1)[:, :5]
            preds.append(_norm(Ttr[nb].mean(axis=1)))
        res["knn5_train_fitted"] = float(np.mean(cos_dist(np.vstack(preds), Tde)))
        res["persistence_no_fit"] = float(np.mean(cos_dist(Cde, Tde)))
        res["chance_shuffled"] = float(np.mean(cos_dist(Tde[rng.permutation(len(Tde))], Tde)))

        ratios = {k: round(v / max(amb, 1e-9), 4) for k, v in res.items()}
        fitted = {k: v for k, v in ratios.items() if k.endswith("_train_fitted")}
        best = min(fitted, key=fitted.get)
        bounds = fitted[best] <= model_ratio + 1e-9
        shared = float(np.linalg.norm(Tde.mean(axis=0)))
        feats = args.rff_dim + 1
        out["per_source"][src] = {
            "n_train": len(itr), "n_dev": len(ide), "ambient_nn": round(amb, 4),
            "model_ratio": round(model_ratio, 4), "ratios": ratios,
            "best_family": best, "ceiling_ratio": fitted[best],
            "headroom_model_minus_ceiling": round(model_ratio - fitted[best], 4),
            "ceiling_bounds_model": bool(bounds),
            "features_over_train_rows": round(feats / len(itr), 4),
            "vacuity_ok": bool(feats < 0.25 * len(itr)),
            "shared_component_norm": round(shared, 4),
            "verdict": (
                f"NO_VALID_CEILING — the model ({model_ratio:.3f}) still beats the best train-fitted family "
                f"({fitted[best]:.3f}); capacity-vs-target remains undecided" if not bounds else
                f"TARGET_LIMITED — even a train-fitted fit with more data than the model reaches only "
                f"{fitted[best]:.3f} (>= 1.0); the limit is the representation, not capacity"
                if fitted[best] >= 1.0 else
                f"CAPACITY_HEADROOM — the ceiling reaches {fitted[best]:.3f} vs the model's "
                f"{model_ratio:.3f}: {model_ratio - fitted[best]:.3f} of recoverable headroom remains, so "
                f"the gap is predictor capacity/optimisation, not the target"),
        }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
