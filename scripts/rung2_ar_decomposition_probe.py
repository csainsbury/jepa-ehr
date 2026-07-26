#!/usr/bin/env python3
"""Does the AR decomposition rescue the wall-clock negative?

Direct wall-clock prediction failed at every horizon (0/12 cells; SCID best 1.284, MIMIC 1.064) while
event-count prediction succeeded comfortably (0.609 / 0.674). But the direct probe demanded CONTENT and
TIMING simultaneously: predict exactly which events land inside [t, t+30d) in one shot.

The AR alternative decomposes it — predict the next K events (which works), then decide where the 30-day
boundary falls and cut. That only needs the boundary, not the whole window, and it is how an autoregressive
model would actually operate. This probe measures whether the decomposition is viable, in three parts:

  1. SPAN — how much wall-clock time do the next K events actually cover, per source? If K events already
     overshoot the horizon, no rollout is needed and the task is purely a CUT-POINT problem. If they fall
     short, the horizon needs multiple AR steps and compounding error becomes the binding constraint.
  2. COVERAGE — what fraction of the true 30-day (or 1-day) window's events lie inside the next-K block? If
     coverage is high the content is available; the only missing piece is where to cut.
  3. ORACLE-CONTENT CEILING — the decisive arm. Fit the wall-clock target from the TRUE next-K event block
     instead of from the context. This grants perfect event-content prediction and asks whether the
     wall-clock target is then recoverable. If it is, the wall-clock failure is a TIMING failure and the
     decomposition is sound. If it is not, knowing the events does not determine the windowed content and
     the decomposition does not help.

Arm 3 is an upper bound on the decomposition, not a system: it uses the true future block, which no
predictor has. That is the point — it isolates whether timing is the gap.

DEV for evaluation, TRAIN for fitting, TEST untouched, aggregate-only.
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_ar_decomposition_probe.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_jepa.eval.rung2_rollout_diag import ambient_true_nn_distance, cos_dist
from scripts.rung2_train_fitted_ceiling import _chunked_ridge, _norm

HORIZONS = {"SCID": 30.0, "MIMIC": 1.0}          # the clinically-natural horizon per source (Rung-0 frozen)
K_EVENTS = (32, 64, 128)


def _rows(target_blocks, split, *, max_ctx, max_blocks, seed, k_max):
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
                ids = g["token_ids"][:]
                days = np.asarray(g["cumulative_days"][:], dtype=np.float64)
                c0, c1 = max(0, int(b.get("context_start_ref", 0))), int(b["context_end_ref"])
                if c1 < c0 or c1 + 1 + k_max > len(ids):
                    continue                                  # need K events of real future
                t_q = float(days[c1])
                if not np.isfinite(t_q) or float(days[-1]) < t_q + HORIZONS[src]:
                    continue                                  # horizon must be fully observed
                ctx = np.asarray(ids[c0:c1 + 1][-max_ctx:], dtype=np.int64)
                if len(ctx) == 0:
                    continue
                per_src.setdefault(src, []).append((ctx, ids, days, t_q, c1))
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
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    tr = _rows(args.target_blocks, "train", max_ctx=args.max_context_tokens,
               max_blocks=args.train_rows, seed=args.seed, k_max=max(K_EVENTS))
    de = _rows(args.target_blocks, "dev", max_ctx=args.max_context_tokens,
               max_blocks=args.dev_rows, seed=args.seed, k_max=max(K_EVENTS))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)
    lin = lambda X: np.hstack([X, np.ones((len(X), 1))])
    out = {"horizons_days": HORIZONS, "k_events": list(K_EVENTS), "aggregate_only": True, "per_source": {}}

    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        H = HORIZONS[src]
        if len(itr) < 1500 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue

        def wc_tokens(i, d, t, c):                    # the true wall-clock window content
            sel = np.nonzero((d >= t) & (d < t + H))[0]
            return i[sel[sel > c]]

        def k_tokens(i, d, t, c, K):                  # the next K events
            return i[c + 1: c + 1 + K]

        # 1. SPAN + 2. COVERAGE, measured on dev
        spans, cover = {}, {}
        for K in K_EVENTS:
            sp, cv = [], []
            for _, i, d, t, c in ide:
                kd = d[c + 1: c + 1 + K]
                sp.append(float(kd[-1] - t) if len(kd) else 0.0)
                w = set(np.nonzero((d >= t) & (d < t + H))[0].tolist()) - set(range(0, c + 1))
                kk = set(range(c + 1, min(c + 1 + K, len(i))))
                cv.append(len(w & kk) / max(len(w), 1) if w else 1.0)
            spans[K] = {"median_days_spanned": round(float(np.median(sp)), 3),
                        "p25": round(float(np.percentile(sp, 25)), 3),
                        "p75": round(float(np.percentile(sp, 75)), 3),
                        "frac_reaching_horizon": round(float(np.mean(np.asarray(sp) >= H)), 4)}
            cover[K] = {"median_coverage_of_window": round(float(np.median(cv)), 4),
                        "mean_coverage_of_window": round(float(np.mean(cv)), 4),
                        "frac_fully_covered": round(float(np.mean(np.asarray(cv) >= 0.999)), 4)}

        # 3. ORACLE-CONTENT CEILING: predict the wall-clock target from the TRUE next-K block
        Ttr = _norm(np.asarray([E[wc_tokens(i, d, t, c)].mean(axis=0)
                                if len(wc_tokens(i, d, t, c)) else np.zeros(E.shape[1])
                                for _, i, d, t, c in itr]))
        Tde_raw = [wc_tokens(i, d, t, c) for _, i, d, t, c in ide]
        keep = np.array([len(x) > 0 for x in Tde_raw])
        Tde = _norm(np.asarray([E[x].mean(axis=0) if len(x) else np.zeros(E.shape[1]) for x in Tde_raw]))
        ktr = np.array([len(wc_tokens(i, d, t, c)) > 0 for _, i, d, t, c in itr])

        Ctr = _norm(np.asarray([E[c_].mean(axis=0) for c_, *_ in itr]))
        Cde = _norm(np.asarray([E[c_].mean(axis=0) for c_, *_ in ide]))
        amb = ambient_true_nn_distance(Tde[keep], np.arange(int(keep.sum())))
        res = {"context_only_ceiling": round(float(np.mean(cos_dist(
                   _chunked_ridge(lin, Ctr[ktr], Ttr[ktr], args.ridge, Cde[keep]), Tde[keep]))) / max(amb, 1e-9), 4)}
        def timed_block(i, d, t, c, K, Lcap=32):
            """Per-event embedding CONCATENATED with days-since-t_query, ordered, padded to Lcap.

            The mean-pooled oracle cannot express 'the first 14% of these events', which is precisely the
            selection the wall-clock cut requires — so a mean-pooled oracle understates the decomposition.
            This arm preserves per-event identity AND relative time, i.e. exactly what a cut needs."""
            sl = slice(c + 1, c + 1 + K)
            ids_, dd = i[sl], d[sl] - t
            L = min(len(ids_), Lcap)
            rows = [np.concatenate([E[ids_[j]], [dd[j]]]) for j in range(L)]
            pad = Lcap - L
            if pad > 0:
                rows += [np.zeros(E.shape[1] + 1)] * pad
            return np.concatenate(rows)

        for K in K_EVENTS:
            Otr = _norm(np.asarray([E[k_tokens(i, d, t, c, K)].mean(axis=0) for _, i, d, t, c in itr]))
            Ode = _norm(np.asarray([E[k_tokens(i, d, t, c, K)].mean(axis=0) for _, i, d, t, c in ide]))
            pred = _chunked_ridge(lin, Otr[ktr], Ttr[ktr], args.ridge, Ode[keep])
            res[f"oracle_next{K}_ceiling"] = round(
                float(np.mean(cos_dist(pred, Tde[keep]))) / max(amb, 1e-9), 4)
            # TIMING-PRESERVING oracle: same events, but per-event identity + relative time retained
            Ptr = _norm(np.asarray([timed_block(i, d, t, c, K) for _, i, d, t, c in itr]))
            Pde = _norm(np.asarray([timed_block(i, d, t, c, K) for _, i, d, t, c in ide]))
            predt = _chunked_ridge(lin, Ptr[ktr], Ttr[ktr], args.ridge, Pde[keep])
            res[f"oracle_next{K}_TIMED_ceiling"] = round(
                float(np.mean(cos_dist(predt, Tde[keep]))) / max(amb, 1e-9), 4)
        out["per_source"][src] = {
            "n_train": int(ktr.sum()), "n_dev": int(keep.sum()), "horizon_days": H,
            "ambient_nn": round(float(amb), 4), "span": spans, "coverage": cover, "ceilings": res,
            "verdict": ("TIMING-LIMITED — the wall-clock target IS recoverable from the true event block, so "
                        "the direct failure was joint content+timing; the AR decomposition is sound and the "
                        "remaining gap is the cut point (sub-gate 4 territory)"
                        if min(min(res[f"oracle_next{K}_ceiling"], res[f"oracle_next{K}_TIMED_ceiling"])
                               for K in K_EVENTS) < 1.0 else
                        "NOT RESCUED — even the true event block does not determine the windowed content, so "
                        "decomposing into events-then-cut does not recover the horizon")}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
