#!/usr/bin/env python3
"""Predict the next n events IN ORDER WITH INTERVALS — the probe for the restated goal.

This replaces the time-window framing. There is no horizon, no cut, no selection: the target is the next K
events as an ORDERED sequence of (type, inter-arrival interval) pairs. Two consequences, both deliberate:

  * NO AMBIENT DENOMINATOR ANYWHERE. Every prior arm was scored as
    `mean cos_dist(p, y) / mean_j min_j cos_dist(y_i, y_j)`, and five of the criterion's known failure modes
    (shared-component degeneracy, cross-space incomparability, sample-size dependence of the normaliser,
    ratio-of-means hiding per-instance failure, small-N histogram collisions) are artifacts of that
    normaliser rather than of prediction. Per-position type log-loss and interval log-score are per-instance,
    need no corpus of other targets, and are comparable across arms. The metric problem is dissolved, not
    repaired.

  * TERMINATION IS CONTENT, NOT ELIGIBILITY. Every earlier arm filtered on `--min-future-tokens`, which is a
    selection on how much sequence remains, and that filter alone moved a headline from 1.134 to 0.824. Here a
    position past the end of the sequence gets an explicit END class, so no instance is filtered for being
    short and the model is asked to PREDICT termination instead of being protected from it. `--min-future`
    defaults to 1.

The question is not "is the pooled window recoverable" but "does the frozen context carry ORDER and TIMING".
Three comparisons answer it, and the middle one is the load-bearing one:

  A. type head vs context-free marginal    -> is next-event content predictable at all?
  B. POSITION-DEPENDENT head vs POSITION-POOLED head -> is anything about ORDER predicted, or is the model
     only reproducing a position-independent bag? A position-pooled head that ties the per-position head is
     the signature of a representation that knows WHAT comes next but not WHEN in the sequence.
  C. interval head vs per-position marginal lognormal -> are the INTERVALS predictable?

Comparison C is only interpretable under `--time-features`. The base context representation is a mean of
TIME-FREE token embeddings, so it carries no timing at all, and the first two runs were asking the interval
head to predict inter-arrivals from features that never contained any. Those runs' null interval result
(0.004-0.042 nats, no bucket clearing MATERIAL_INTERVAL_NATS) is therefore uninterpretable as stated. With
`--time-features` the context is augmented with recent-interval summaries, which forks cleanly: intervals
becoming predictable means the ENCODER discards time and the fix is to time-augment per-element features
(the DeepSets result says a time-free mean pool provably cannot express interval queries); intervals staying
flat when handed the timing directly is a much stronger negative about the context itself.

Note the arms are not perfectly isolated: the time features enter the shared feature vector, so they reach
the TYPE head too. That is informative — it says whether timing helps predict order — but it means type-head
numbers differ between the two arms in two ways at once.

Baselines are not decoration. Next-event prediction in an event stream is dominated by type frequency and by
recurrence of types already seen in the context, so a head that beats neither has learned nothing, however
good its absolute accuracy looks. Both baselines are therefore reported, and skill is stated against the
stronger of them.

Fit on TRAIN, score on DEV, TEST untouched, aggregate-only (no sequence ids, no tokens, no per-row output).
Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_next_events_sequence_probe.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rung2_train_fitted_ceiling import _norm

# Position buckets: individual early positions (where order information should be strongest) then widening
# groups. Predicting position 1 and position 30 are different problems and pooling them hides the decay.
BUCKETS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 16), (16, 32))

# Minimum interval log-score gain (nats per interval) that counts as predicting the intervals at all. The
# first K=32 run produced "beats_marginal" at 0.004 nats, which is a sign test on noise, not timing skill.
MATERIAL_INTERVAL_NATS = 0.05


def _read_rows(target_blocks, split, *, K, min_future, max_ctx, max_rows, seed):
    """Context ids + the next K (type, inter-arrival) pairs, with END marks past the sequence end."""
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
            if not src:
                continue
            if len(per_src.get(src, ())) >= max_rows:
                continue
            try:
                p = str(b["sequence_file"])
                if p not in cache:
                    cache[p] = h5py.File(p, "r")
                g = cache[p][str(b["sequence_group"])]
                ids = np.asarray(g["token_ids"][:], dtype=np.int64)
                days = np.asarray(g["cumulative_days"][:], dtype=np.float64)
                c0 = max(0, int(b.get("context_start_ref", 0)))
                c1 = int(b["context_end_ref"])
                if c1 < c0 or c1 + 1 > len(ids) - min_future:
                    continue
                ctx = ids[c0:c1 + 1][-max_ctx:]
                if len(ctx) == 0:
                    continue
                n_avail = min(K, len(ids) - (c1 + 1))
                if n_avail < min_future:
                    continue
                fut = ids[c1 + 1:c1 + 1 + n_avail]
                # inter-arrival intervals, first one measured from the context's last event
                t = days[c1:c1 + 1 + n_avail]
                if not np.all(np.isfinite(t)):
                    continue
                dt = np.diff(t)
                # context arrival times, aligned to `ctx`, for the time-augmented feature arm
                ctx_t = days[c0:c1 + 1][-max_ctx:]
                if not np.all(np.isfinite(ctx_t)):
                    continue
                per_src.setdefault(src, []).append(
                    (ctx, fut, dt.astype(np.float64), int(n_avail), ctx_t.astype(np.float64)))
            except (KeyError, OSError, ValueError, IndexError):
                continue
    finally:
        for f in cache.values():
            try:
                f.close()
            except OSError:
                pass
    return per_src


def _time_features(items):
    """Recent-interval summaries of the CONTEXT.

    The base context representation is a mean of time-free token embeddings, so it contains no timing
    whatsoever — asking the interval arm to predict inter-arrivals from it is asking for timing from features
    that never carried any. Without this arm a null interval result is uninterpretable. It is also the
    concrete form of the DeepSets point from the consult: time-augmented per-element features are what make a
    window/interval query expressible, and time-free mean pooling is provably insufficient for it.
    """
    rows = []
    for _c, _f, _d, _n, ctx_t in items:
        gaps = np.diff(ctx_t) if len(ctx_t) > 1 else np.zeros(1)
        lg = np.log1p(np.clip(gaps, 0.0, None))
        span = float(ctx_t[-1] - ctx_t[0]) if len(ctx_t) > 1 else 0.0
        rows.append([
            float(np.mean(lg)), float(np.median(lg)), float(np.std(lg)),
            float(lg[-1]), float(np.mean(lg[-4:])), float(np.mean(lg[-16:])),
            float(np.max(lg)), np.log1p(max(span, 0.0)),
            np.log1p(len(ctx_t)), np.log1p(max(span, 0.0) / max(len(ctx_t) - 1, 1)),
        ])
    return np.asarray(rows, dtype=np.float64)


def _stack(items, K, end_id):
    """Types (END-padded) and intervals (NaN past the end) as dense K-column arrays."""
    Y = np.full((len(items), K), end_id, dtype=np.int64)
    D = np.full((len(items), K), np.nan, dtype=np.float64)
    for i, (_, fut, dt, n, *_rest) in enumerate(items):
        Y[i, :n] = fut
        D[i, :n] = dt
    return Y, D


def _fit_softmax(X, rows, y, n_classes, *, lam, iters, lr, chunk=8192):
    """Multinomial logistic by full-batch gradient descent with momentum.

    `rows` indexes into X once per training position, so the K-fold-repeated design matrix is never
    materialised — at K=32 and 20k sequences that would be ~4 GB. Chunking also keeps the probability
    matrix (chunk x 1051) small.
    """
    n, d = len(rows), X.shape[1]
    W = np.zeros((d, n_classes), dtype=X.dtype)
    V = np.zeros_like(W)
    trace = []
    for it in range(iters):
        G = np.zeros_like(W)
        obj = 0.0
        for s in range(0, n, chunk):
            Xb, yb = X[rows[s:s + chunk]], y[s:s + chunk]
            Z = Xb @ W
            m = Z.max(axis=1, keepdims=True)
            Z -= m
            P = np.exp(Z)
            ssum = P.sum(axis=1, keepdims=True)
            P /= ssum
            obj += float(np.sum(np.log(ssum[:, 0]) - Z[np.arange(len(yb)), yb]))
            P[np.arange(len(yb)), yb] -= 1.0
            G += Xb.T @ P
        G /= n
        G += lam * W
        V = 0.9 * V - lr * G
        W += V
        trace.append(obj / n)
    # Under-convergence biases the comparison AGAINST the head, which would read as a false negative. Report
    # the trace so that failure mode is visible instead of silent.
    tail = trace[-min(10, len(trace)):]
    diag = {"train_log_loss_first": trace[0] if trace else None,
            "train_log_loss_last": trace[-1] if trace else None,
            "improvement_over_last_10_iters": float(tail[0] - tail[-1]) if len(tail) > 1 else None,
            "iters": iters}
    diag["converged"] = bool(diag["improvement_over_last_10_iters"] is not None
                             and diag["improvement_over_last_10_iters"] < 0.01)
    return W, diag


def _score_softmax(W, X, rows, y, *, chunk=8192):
    """Mean log-loss plus top-1 / top-5 accuracy, accumulated in chunks over the position index."""
    ll = 0.0
    top1 = 0
    top5 = 0
    for s in range(0, len(rows), chunk):
        Xb, yb = X[rows[s:s + chunk]], y[s:s + chunk]
        Z = Xb @ W
        m = Z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))
        ll += float(np.sum(lse - Z[np.arange(len(yb)), yb]))
        top1 += int(np.sum(Z.argmax(axis=1) == yb))
        k = min(5, Z.shape[1])
        idx = np.argpartition(-Z, k - 1, axis=1)[:, :k]
        top5 += int(np.sum(np.any(idx == yb[:, None], axis=1)))
    n = max(len(rows), 1)
    return {"log_loss": ll / n, "top1": top1 / n, "top5": top5 / n}


def _marginal_scores(counts, y_dev, *, n_classes, alpha=0.5):
    """Context-free baseline: the smoothed empirical next-type distribution at these positions."""
    p = (counts + alpha) / float(counts.sum() + alpha * n_classes)
    lp = np.log(p)
    order = np.argsort(-p)
    top5 = set(order[:5].tolist())
    return {"log_loss": float(-np.mean(lp[y_dev])),
            "top1": float(np.mean(y_dev == order[0])),
            "top5": float(np.mean([int(v in top5) for v in y_dev]))}


def _persistence_scores(B, rows, y_dev, marg_p, *, mix=0.5):
    """Recurrence baseline: types already present in the context, mixed with the marginal. In event streams
    this is the baseline that matters — most of what happens next has happened before.

    `rows` indexes B so the repeated bag matrix is never materialised (n_positions x 1051 would be ~0.5 GB).
    """
    ll = 0.0
    top1 = 0
    top5 = 0
    for r, y in zip(rows, y_dev):
        p = mix * B[r] + (1.0 - mix) * marg_p
        p = p / max(p.sum(), 1e-12)
        ll -= float(np.log(max(p[y], 1e-12)))
        order = np.argpartition(-p, 4)[:5]
        top1 += int(p.argmax() == y)
        top5 += int(y in set(order.tolist()))
    n = max(len(y_dev), 1)
    return {"log_loss": ll / n, "top1": top1 / n, "top5": top5 / n}


def _skill(head, base):
    """Fraction of the baseline's log-loss removed. Negative means the head is worse than the baseline."""
    b = base.get("log_loss")
    h = head.get("log_loss")
    if b is None or h is None or not np.isfinite(b) or b <= 0:
        return None
    return float(1.0 - h / b)


def _interval_arm(Ftr, rtr, Dtr, Fde, rde, Dde, *, ridge):
    """Lognormal interval prediction. Ridge on log1p(dt) with a held-out residual scale gives a proper
    log-score; the baseline is the same lognormal with the mean but no context. CRPS is not computed in
    closed form here — the log-score already separates the arms and needs no extra assumption.

    NaN intervals are positions past the sequence end: there is no interval to predict, so they are dropped
    from the timing arm only. Termination itself is scored by the type head's END class, not here.
    """
    mtr = np.isfinite(Dtr)
    mde = np.isfinite(Dde)
    if mtr.sum() < 200 or mde.sum() < 50:
        return {"note": "too few observed intervals", "n_train": int(mtr.sum()), "n_dev": int(mde.sum())}
    ytr = np.log1p(np.clip(Dtr[mtr], 0.0, None))
    yde = np.log1p(np.clip(Dde[mde], 0.0, None))
    itr, ide = rtr[mtr], rde[mde]

    # The predictive scale is fitted on a HELD-OUT slice of train, not on the fitting residuals. An in-sample
    # s2 understates the residual and hands the head a free log-score advantage over the marginal — the same
    # in-sample-bound error this project already made once.
    cut = int(0.8 * len(ytr))
    if cut < 100 or len(ytr) - cut < 50:
        return {"note": "too few observed intervals to hold out a scale slice", "n_train": int(mtr.sum())}

    d = Ftr.shape[1]
    chunk = 8192
    A = ridge * np.eye(d)
    b = np.zeros(d)
    for s in range(0, cut, chunk):
        Xb = Ftr[itr[s:s + chunk]]
        A += Xb.T @ Xb
        b += Xb.T @ ytr[s:s + chunk]
    w = np.linalg.solve(A, b)

    def sq_err(idx, y, pred_fn):
        tot = 0.0
        for s in range(0, len(idx), chunk):
            tot += float(np.sum((y[s:s + chunk] - pred_fn(idx[s:s + chunk])) ** 2))
        return tot / max(len(idx), 1)

    def abs_err(idx, y, pred_fn):
        tot = 0.0
        for s in range(0, len(idx), chunk):
            tot += float(np.sum(np.abs(y[s:s + chunk] - pred_fn(idx[s:s + chunk]))))
        return tot / max(len(idx), 1)

    head_tr = lambda ix: Ftr[ix] @ w
    head_de = lambda ix: Fde[ix] @ w
    b_mu = float(ytr[:cut].mean())
    const = lambda ix: np.full(len(ix), b_mu)

    s2 = max(sq_err(itr[cut:], ytr[cut:], head_tr), 1e-9)
    b_s2 = max(sq_err(itr[cut:], ytr[cut:], const), 1e-9)

    head_ll = 0.5 * np.log(2 * np.pi * s2) + sq_err(ide, yde, head_de) / (2 * s2)
    base_ll = 0.5 * np.log(2 * np.pi * b_s2) + sq_err(ide, yde, const) / (2 * b_s2)

    return {"n_train": int(mtr.sum()), "n_dev": int(mde.sum()),
            "head_neg_log_lik": float(head_ll), "marginal_neg_log_lik": float(base_ll),
            "nats_gained_per_interval": float(base_ll - head_ll),
            "head_mae_log1p_days": abs_err(ide, yde, head_de),
            "marginal_mae_log1p_days": abs_err(ide, yde, const),
            "beats_marginal": bool(head_ll < base_ll),
            # Sign alone is too generous: at 4,000 dev rows a 0.004-nat edge is noise dressed as a result.
            # MATERIAL_INTERVAL_NATS is the threshold the verdict uses.
            "materially_beats_marginal": bool(base_ll - head_ll >= MATERIAL_INTERVAL_NATS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-blocks", required=True)
    ap.add_argument("--events", type=int, default=32, help="n = how many next events to predict")
    ap.add_argument("--min-future", type=int, default=1,
                    help="termination is content: default 1, NOT an eligibility filter on remaining length")
    ap.add_argument("--max-context-tokens", type=int, default=128)
    ap.add_argument("--train-rows", type=int, default=20000)
    ap.add_argument("--dev-rows", type=int, default=4000)
    ap.add_argument("--rff-dim", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--softmax-l2", type=float, default=1e-4)
    ap.add_argument("--softmax-iters", type=int, default=150)
    ap.add_argument("--softmax-lr", type=float, default=2.0)
    ap.add_argument("--time-features", action="store_true",
                    help="augment the context representation with recent-interval summaries; REQUIRED for "
                         "the interval arm to be interpretable, since the base representation is time-free")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from clinical_jepa.arms.v0b.mean_token_model import build_mean_token_jepa_from_checkpoint

    K = args.events
    tr = _read_rows(args.target_blocks, "train", K=K, min_future=args.min_future,
                    max_ctx=args.max_context_tokens, max_rows=args.train_rows, seed=args.seed)
    de = _read_rows(args.target_blocks, "dev", K=K, min_future=args.min_future,
                    max_ctx=args.max_context_tokens, max_rows=args.dev_rows, seed=args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_mean_token_jepa_from_checkpoint(ck)
    model.eval()
    E = model.embedding.weight.detach().numpy().astype(np.float64)
    V = E.shape[0]
    end_id = V                      # termination as an explicit class, not a filter
    n_classes = V + 1

    out = {"goal": ("predict the next n events as an ORDERED sequence of (type, inter-arrival interval) "
                    "pairs; no horizon, no cut, no selection"),
           "events_n": K, "min_future": args.min_future,
           "termination_as_content": True,
           "eligibility_note": ("no --min-future-tokens style filter; positions past the sequence end are the "
                                "END class, so short sequences are predicted rather than excluded"),
           "no_ambient_denominator": True,
           "context_time_features": bool(args.time_features),
           "metric_note": ("per-position type log-loss (nats) and interval log-score; both per-instance, "
                           "neither uses a corpus of other targets, so the ratio criterion's "
                           "normaliser-driven failure modes do not apply"),
           "checkpoint": Path(args.checkpoint).parent.name,
           "aggregate_only": True, "per_source": {}}

    rng = np.random.default_rng(args.seed)
    for src in sorted(set(tr) & set(de)):
        itr, ide = tr[src], de[src]
        if len(itr) < 2000 or len(ide) < 300:
            out["per_source"][src] = {"n_train": len(itr), "n_dev": len(ide), "note": "too few rows"}
            continue

        Ytr, Dtr = _stack(itr, K, end_id)
        Yde, Dde = _stack(ide, K, end_id)

        # frozen context representation, exactly the encoder the project already has
        Ctr = _norm(np.asarray([E[c].mean(axis=0) for c, *_ in itr]))
        Cde = _norm(np.asarray([E[c].mean(axis=0) for c, *_ in ide]))
        d = Ctr.shape[1]
        W0 = rng.normal(size=(d, args.rff_dim)) / np.sqrt(d)
        b0 = rng.uniform(0, 2 * np.pi, size=args.rff_dim)
        feat = lambda X: np.hstack([X, np.sqrt(2.0 / args.rff_dim) * np.cos(X @ W0 + b0),
                                    np.ones((len(X), 1))])
        Ftr, Fde = feat(Ctr), feat(Cde)
        if args.time_features:
            Ttr_f, Tde_f = _time_features(itr), _time_features(ide)
            mu, sd = Ttr_f.mean(axis=0), np.clip(Ttr_f.std(axis=0), 1e-6, None)
            Ftr = np.hstack([Ftr, (Ttr_f - mu) / sd])
            Fde = np.hstack([Fde, (Tde_f - mu) / sd])
        srec_time_features = bool(args.time_features)
        # float32 for the softmax fits (the dominant cost); the interval ridge keeps float64 for its solve
        Ftr32, Fde32 = Ftr.astype(np.float32), Fde.astype(np.float32)

        # context bag-of-types for the recurrence baseline
        def bags(items):
            B = np.zeros((len(items), n_classes))
            for i, (c, *_r) in enumerate(items):
                np.add.at(B[i], c, 1.0)
                B[i] /= max(B[i].sum(), 1e-12)
            return B
        Bde = bags(ide)

        srec = {"n_train": len(itr), "n_dev": len(ide),
                "context_time_features": srec_time_features,
                "median_available_events": float(np.median([it[3] for it in ide])),
                "frac_dev_reaching_n": float(np.mean([it[3] >= K for it in ide])),
                "vocab": V, "buckets": {}}

        # ---- position-POOLED head: same weights for every position. If this ties the per-position head,
        # ---- the representation carries content but not order.
        rows_tr = np.repeat(np.arange(len(itr)), K)
        W_pool, pool_diag = _fit_softmax(Ftr32, rows_tr, Ytr.reshape(-1), n_classes,
                                         lam=args.softmax_l2, iters=args.softmax_iters, lr=args.softmax_lr)
        srec["position_pooled_fit"] = pool_diag

        for lo, hi in BUCKETS:
            if lo >= K:
                continue
            hi = min(hi, K)
            rtr = np.repeat(np.arange(len(itr)), hi - lo)
            rde = np.repeat(np.arange(len(ide)), hi - lo)
            ytr = Ytr[:, lo:hi].reshape(-1)
            yde = Yde[:, lo:hi].reshape(-1)

            W_pos, fit_diag = _fit_softmax(Ftr32, rtr, ytr, n_classes,
                                           lam=args.softmax_l2, iters=args.softmax_iters, lr=args.softmax_lr)
            head = _score_softmax(W_pos, Fde32, rde, yde)
            pooled = _score_softmax(W_pool, Fde32, rde, yde)

            counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
            marg = _marginal_scores(counts, yde, n_classes=n_classes)
            marg_p = (counts + 0.5) / float(counts.sum() + 0.5 * n_classes)
            pers = _persistence_scores(Bde, rde, yde, marg_p)

            strongest = min((marg, pers), key=lambda r: r["log_loss"])
            srec["buckets"][f"pos_{lo + 1}_{hi}"] = {
                "n_dev_positions": int(len(yde)),
                "fit": fit_diag,
                "frac_END": float(np.mean(yde == end_id)),
                "type_head": head,
                "position_pooled_head": pooled,
                "baseline_marginal": marg,
                "baseline_persistence": pers,
                "skill_vs_marginal": _skill(head, marg),
                "skill_vs_persistence": _skill(head, pers),
                "skill_vs_strongest_baseline": _skill(head, strongest),
                "beats_both_baselines": bool(head["log_loss"] < marg["log_loss"]
                                             and head["log_loss"] < pers["log_loss"]),
                "order_information_nats": float(pooled["log_loss"] - head["log_loss"]),
                "order_information_detected": bool(pooled["log_loss"] - head["log_loss"] > 0.01),
                "interval": _interval_arm(Ftr, rtr, Dtr[:, lo:hi].reshape(-1),
                                          Fde, rde, Dde[:, lo:hi].reshape(-1), ridge=args.ridge),
            }

        out["per_source"][src] = srec

    # A verdict per source, stated in the terms of the goal rather than as a ratio.
    out["verdict"] = {}
    for src, rec in out["per_source"].items():
        b = rec.get("buckets") or {}
        if not b:
            out["verdict"][src] = "not evaluable"
            continue
        unconv = [k for k, v in b.items() if not (v.get("fit") or {}).get("converged")]
        content = [k for k, v in b.items() if v.get("beats_both_baselines")]
        order = [k for k, v in b.items() if v.get("order_information_detected")]
        timing = [k for k, v in b.items()
                  if (v.get("interval") or {}).get("materially_beats_marginal")]
        out["verdict"][src] = {
            "content_beats_both_baselines_at": content,
            "order_information_at": order,
            "intervals_materially_beat_marginal_at": timing,
            "all_three_at": sorted((set(content) & set(order) & set(timing)) - set(unconv)),
            "UNCONVERGED_not_interpretable_at": unconv,
            "note": ("content_/order_ lists are raw comparisons; an unconverged fit biases them DOWNWARD "
                     "and they are excluded from all_three_at. Interval arm is a closed-form ridge and is "
                     "unaffected by softmax convergence."),
        }

    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
