#!/usr/bin/env python3
"""SUPERSEDED — its capacity/target verdicts are NOT reliable. Use `rung2_honest_ceiling.py`.

RETRACTION. This script produced two capacity-vs-target verdicts and BOTH were wrong, for two different
reasons. It is kept only because the retraction is instructive and because its ladder measurements (the
per-checkpoint model ratios) remain valid.

  1st verdict, TARGET_BOTTLENECK — rested on a 1-NN "ceiling" that is not a ceiling. The trained model beat
     it in all eight arms, because borrowing a single neighbour's target is high-variance while a predictor
     smooths. A bound the model beats bounds nothing.
  2nd verdict, CAPACITY_BOTTLENECK — rested on an IN-SAMPLE fit. In-sample error bounds in-sample error, but
     the capacity-vs-target question is about held-out generalisation. The headline "ceiling 0.708" came from
     a linear in-sample fit whose held-out counterpart was never computed. When it was, the honest held-out
     ceiling for SCID is 1.134 and the trained model is already AT it — the claimed 0.44 of headroom was
     purely overfitting.

The generalisable lesson, now enforced in the replacement: a ceiling claim needs (a) to actually bound every
trained arm, (b) to be fitted OUT OF SAMPLE, and (c) controls — persistence, and an untrained random
embedding — before it can support any verdict about capacity or targets.

The MODEL RATIOS this script reports are unaffected and still usable. The verdict/ceiling fields are not.
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


def nn_ceiling(ctx: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """LEAVE-ONE-OUT 1-NN regression in context space: predict row i's target with the target of the row
    whose CONTEXT is nearest to i's. Parameter-free, so it bounds any smooth context->target map."""
    c, t = _norm(ctx), _norm(tgt)
    sims = c @ c.T
    np.fill_diagonal(sims, -np.inf)                     # leave-one-out
    nn = np.argmax(sims, axis=1)
    return cos_dist(t[nn], t)                           # distance of the borrowed target to the true one


def fitted_ceiling(ctx: np.ndarray, tgt: np.ndarray, rng, *, rff_dim: int = 256, ridge: float = 1e-3) -> dict:
    """A GENUINE ceiling on what any context -> target map can achieve on this representation.

    The 1-NN arm failed as a bound because a trained model smooths and beats it. The valid construction
    uses the fact that IN-SAMPLE fit lower-bounds out-of-sample error: fit an optimistic, high-capacity map
    on the very rows it is scored on, and no honest predictor can do better. Reported alongside a HELD-OUT
    fit of the same family, so an optimistic bound and a realistic one are both visible.

    Two guards, both learned the hard way:
      * VACUITY — with as many free parameters as rows, an in-sample fit interpolates and the 'ceiling'
        collapses to 0, which says nothing. Feature count is kept well below n and the ratio is reported.
      * BOUNDEDNESS — a ceiling that the trained model beats is not a ceiling. Checked by the caller.

    Families: k-NN over several k (variance-reduced version of the failed arm), linear ridge, and random
    Fourier feature ridge (a nonlinear family, still closed-form).
    """
    c, t = _norm(ctx), _norm(tgt)
    n, d = c.shape
    out: dict[str, float] = {}

    sims = c @ c.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)
    for k in (1, 5, 20, 50):
        if k >= n:
            continue
        pred = _norm(t[order[:, :k]].mean(axis=1))
        out[f"knn_k{k}"] = float(np.mean(cos_dist(pred, t)))

    def _ridge_fit(X, Y, lam, Xe):
        A = X.T @ X + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, X.T @ Y)
        return _norm(Xe @ W)

    X = np.hstack([c, np.ones((n, 1))])
    out["linear_ridge_in_sample"] = float(np.mean(cos_dist(_ridge_fit(X, t, ridge, X), t)))

    # nonlinear family; rff_dim is capped so the fit cannot interpolate
    rff_dim = int(min(rff_dim, max(8, n // 4)))
    W0 = rng.normal(size=(d, rff_dim)) / np.sqrt(d)
    b0 = rng.uniform(0, 2 * np.pi, size=rff_dim)
    Phi = np.hstack([np.sqrt(2.0 / rff_dim) * np.cos(c @ W0 + b0), np.ones((n, 1))])
    out["rff_ridge_in_sample"] = float(np.mean(cos_dist(_ridge_fit(Phi, t, ridge, Phi), t)))

    # held-out version of the same nonlinear family (honest generalisation, not a bound)
    idx = rng.permutation(n); half = n // 2
    tr, te = idx[:half], idx[half:]
    pred_te = _ridge_fit(Phi[tr], t[tr], ridge, Phi[te])
    out["rff_ridge_held_out"] = float(np.mean(cos_dist(pred_te, t[te])))

    best = min(out, key=out.get)
    return {"per_method_d_self": {k: round(v, 4) for k, v in out.items()},
            "best_method": best, "best_d_self": round(out[best], 4),
            "n_rows": int(n), "rff_features": int(rff_dim),
            "features_over_n": round((rff_dim + 1) / max(n, 1), 3),
            "vacuity_guard": ("OK — features well below n" if (rff_dim + 1) < 0.5 * n else
                              "WARNING — feature count approaches n; the in-sample fit may interpolate and "
                              "the bound would be vacuous")}



def target_space_diagnostics(tgt: np.ndarray, patients) -> dict:
    t = _norm(tgt)
    s = np.linalg.svd(t - t.mean(0, keepdims=True), compute_uv=False)
    var = s ** 2
    eff_rank = float((var.sum() ** 2) / max(float((var ** 2).sum()), 1e-12))
    return {"ambient_nn_distance": round(float(ambient_true_nn_distance(tgt, patients)), 4),
            "mean_pairwise_cos_dist": round(float(1.0 - (t @ t.T)[np.triu_indices(len(t), 1)].mean()), 4),
            "effective_rank": round(eff_rank, 2), "dim": int(t.shape[1])}


def analyse(ctx: np.ndarray, tgt: np.ndarray, pred: np.ndarray, label: str, rng) -> dict:
    n = len(tgt)
    pats = np.arange(n)                                 # <=1 block per sequence => row == sequence
    amb = ambient_true_nn_distance(tgt, pats)
    d_model = cos_dist(pred, tgt)
    d_ceiling = nn_ceiling(ctx, tgt)
    shuffled = tgt[rng.permutation(n)]                  # chance arm: context->target link destroyed
    d_chance = cos_dist(shuffled, tgt)
    fc = fitted_ceiling(ctx, tgt, rng)
    r = lambda d: round(float(np.mean(d)) / max(amb, 1e-9), 4)
    rv = lambda v: round(float(v) / max(amb, 1e-9), 4)
    return {"arm": label, "n": int(n), "ambient_nn": round(float(amb), 4),
            "model_ratio": r(d_model), "nn_ceiling_ratio": r(d_ceiling), "chance_ratio": r(d_chance),
            "fitted_ceiling": {**fc, "best_ratio": rv(fc["best_d_self"]),
                               "per_method_ratio": {k: rv(v) for k, v in fc["per_method_d_self"].items()}},
            "model_d_self": round(float(np.mean(d_model)), 4),
            "nn_ceiling_d_self": round(float(np.mean(d_ceiling)), 4),
            "chance_d_self": round(float(np.mean(d_chance)), 4)}


def verdict(rows: list[dict]) -> dict:
    """Decide CAPACITY vs TARGET from the ladder and the VALIDATED fitted ceiling.

    A ceiling only counts if (a) it actually bounds every trained arm — a bound the model beats is not a
    bound — and (b) it is not vacuous through interpolation. Both are checked before it can carry a verdict.
    The 1-NN arm is retained for reporting only; it failed (a) and is barred from the decision.
    """
    ratios = [r["model_ratio"] for r in rows]
    best_model, span = min(ratios), max(ratios) - min(ratios)
    chance = float(np.median([r["chance_ratio"] for r in rows]))
    knn = float(np.median([r["nn_ceiling_ratio"] for r in rows]))
    ceils = [r["fitted_ceiling"]["best_ratio"] for r in rows]
    ceil = float(np.min(ceils))                                  # most optimistic bound across arms
    bounds_model = ceil <= best_model + 1e-9
    vacuous = any("WARNING" in r["fitted_ceiling"]["vacuity_guard"] for r in rows)
    usable = bounds_model and not vacuous

    if not usable:
        why = ("the fitted ceiling does not bound the trained model" if not bounds_model
               else "the fitted ceiling may be interpolating (vacuous)")
        v = f"NO_VALID_CEILING — {why}; no capacity/target verdict can be drawn from this run"
    elif ceil >= 1.0:
        v = (f"TARGET_BOTTLENECK — even an optimistic IN-SAMPLE high-capacity fit only reaches {ceil:.3f} "
             f"(>= 1.0). In-sample error lower-bounds out-of-sample error, so NO honest predictor on this "
             f"representation can beat the nearest wrong instance: the limit is the target, not capacity")
    elif best_model < 1.0:
        v = (f"CAPACITY_BOTTLENECK — the model reaches {best_model:.3f} (< 1.0) as capacity/steps rise; the "
             f"ceiling at {ceil:.3f} shows further headroom")
    else:
        v = (f"CAPACITY_BOTTLENECK (unrealised) — the ceiling reaches {ceil:.3f} (< 1.0) so the information "
             f"IS present and recoverable, but the trained ladder plateaus at {best_model:.3f}; the gap is "
             f"predictor capacity/optimisation, NOT the target")
    return {"verdict": v, "ladder_ratio_span": round(span, 4), "best_model_ratio": round(best_model, 4),
            "fitted_ceiling_ratio": round(ceil, 4), "ceiling_usable": bool(usable),
            "ceiling_bounds_model": bool(bounds_model), "ceiling_vacuous": bool(vacuous),
            "median_knn_arm_ratio": round(knn, 4),
            "knn_arm_note": "reporting only — failed as a bound (the trained model beats it)",
            "median_chance_ratio": round(chance, 4),
            "headroom_model_minus_ceiling": round(best_model - ceil, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint .pt paths (capacity ladder)")
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

    manifest = json.loads(Path(args.target_blocks).read_text())
    blocks = [b for b in manifest.get("blocks", [])
              if str(b.get("split")) == args.split and b.get("sequence_file") and b.get("sequence_group")
              and b.get("target_start_ref") is not None and int(b.get("target_start_ref", -1)) >= 0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(blocks)

    # ONE pass over the substrate: collect context/target token windows per source
    cache: dict[str, object] = {}
    per_src: dict[str, list] = {}
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
                tgt = np.asarray(ids[t0:t0 + args.target_window_events], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                per_src.setdefault(str(b.get("source_dataset")), []).append((ctx, tgt))
            except Exception:
                continue
    finally:
        for h in cache.values():
            h.close()

    out = {"split": args.split, "aggregate_only": True, "per_source": {}}
    for src, items in sorted(per_src.items()):
        if len(items) < 200:
            out["per_source"][src] = {"n": len(items), "note": "too few rows — skipped"}
            continue
        rows = []
        for ckpt_path in args.checkpoints:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = build_mean_token_jepa_from_checkpoint(ck)
            model.eval()
            with torch.no_grad():
                def pad(seqs):
                    L = max(len(s) for s in seqs)
                    t = torch.zeros((len(seqs), L), dtype=torch.long)
                    for i, s in enumerate(seqs):
                        t[i, :len(s)] = torch.as_tensor(s)
                    return t
                C = model.mean_embed(pad([c for c, _ in items])).numpy()
                T = model.mean_embed(pad([t for _, t in items])).numpy()
                P = model.predict_rollout_from_context_ids(pad([c for c, _ in items]), 1)[:, 0, :].numpy()
            tag = f"{Path(ckpt_path).parent.name} (dim={int(ck['embedding_dim'])})"
            rows.append({**analyse(C, T, P, tag, rng),
                         "target_space": target_space_diagnostics(T, np.arange(len(T)))})
        out["per_source"][src] = {"n": len(items), "arms": rows, **verdict(rows)}

    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
