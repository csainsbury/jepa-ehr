"""Encode-empty (Option A) training core for v0B — pure, testable functions (Pi R4).

The hybrid silence objective. Authority hierarchy (Pi R4 Q1): the supervised
occupancy scalar is the AUTHORITATIVE empty/non-empty + count-0 owner; the frozen
z_empty prototype is AUXILIARY latent geometry. Collapse-avoidance is structural:

  - empties are targeted to the FROZEN z_empty buffer under ``no_grad`` (it cannot
    chase the predictor);
  - occupancy lives in a scale-free BCE channel the cosine loss cannot represent
    or game;
  - the variance regulariser is computed on NON-EMPTY predictions only, so the one
    shared silence attractor is never a batch-variance sink.

Metrics keep empty and non-empty separate; calibration is reported on the natural
(uncapped) prevalence because training may cap the empty fraction for stability.
"""
from __future__ import annotations

import math
from typing import Any


def occupancy_targets(is_empty: Any, count: Any) -> tuple[Any, Any]:
    """(y_occ, y_logcount): occupancy 1 = non-empty (has events), 0 = silence."""
    import torch

    y_occ = (~is_empty.to(torch.bool)).float()
    y_logcount = torch.log1p(count.clamp_min(0).float())
    return y_occ, y_logcount


def encode_empty_loss(
    model: Any,
    ctx_ids: Any,
    tgt_ids: Any,
    is_empty: Any,
    count: Any,
    *,
    lambda_occ: float = 1.0,
    lambda_count: float = 0.5,
    var_floor: float = 0.05,
    var_weight: float = 0.01,
) -> tuple[Any, dict[str, float]]:
    """Hybrid loss = cosine(pred, target_latent) over ALL rows (empties -> frozen
    z_empty, never a zero vector) + variance reg on NON-EMPTY preds only + BCE
    occupancy + Huber log1p-count on occupied rows. Returns (loss, parts)."""
    import torch
    import torch.nn.functional as F

    ctx_z = model.mean_embed(ctx_ids)                       # context latent (has grad)
    with torch.no_grad():
        tgt_z = model.target_latent(tgt_ids, is_empty)      # stop-grad target; empties -> frozen z_empty
    pred = model.predict_rollout_from_latent(ctx_z, 1)[:, 0, :]

    cos = F.cosine_similarity(pred, tgt_z, dim=-1)
    cos_loss = (1.0 - cos).mean()

    nonempty = (~is_empty.to(torch.bool))
    if bool(nonempty.any()) and int(nonempty.sum()) >= 2:
        var = pred[nonempty].var(dim=0).mean()
        var_reg = var_weight * torch.relu(pred.new_tensor(var_floor) - var)
    else:
        var_reg = pred.sum() * 0.0

    occ_logit, count_pred = model.predict_occupancy_from_latent(ctx_z, 1)
    occ_logit = occ_logit[:, 0, 0]
    count_pred = count_pred[:, 0, 0]
    y_occ, y_logcount = occupancy_targets(is_empty, count)
    # No pos_weight: class balance is handled by the empty-fraction-capped sampler
    # (Pi Q5), and the natural prior is restored via calibration on uncapped data.
    occ_loss = F.binary_cross_entropy_with_logits(occ_logit, y_occ)
    if bool(nonempty.any()):
        count_loss = F.smooth_l1_loss(count_pred[nonempty], y_logcount[nonempty])
    else:
        count_loss = count_pred.sum() * 0.0

    loss = cos_loss + var_reg + lambda_occ * occ_loss + lambda_count * count_loss
    return loss, {
        "cos": float(cos_loss.detach()),
        "occ": float(occ_loss.detach()),
        "count": float(count_loss.detach()),
        "var_reg": float(var_reg.detach()),
        "total": float(loss.detach()),
    }


def binary_auc(scores: Any, labels: Any) -> float:
    """Mann-Whitney AUC of ``scores`` separating label 1 from label 0.

    0.5 when a class is absent (undefined but neutral). Ties counted as 0.5.
    """
    import torch

    s = scores.detach().reshape(-1).float()
    y = labels.detach().reshape(-1).float()
    pos = s[y > 0.5]
    neg = s[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    # rank-sum: fraction of (pos, neg) pairs with pos > neg (+0.5 for ties).
    diff = pos.reshape(-1, 1) - neg.reshape(1, -1)
    wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    return float(wins / (pos.numel() * neg.numel()))


def separation_cosine(model: Any, ctx_ids_nonempty: Any) -> float:
    """|cos(z_empty, mean non-empty TARGET latent)| — the init separation check.

    Compares against the non-empty *target* latents (mean_embed of populated
    targets), per Pi Q2 (check against a sample of non-empty latents, not one
    arbitrary centroid). Reseed z_empty if this is too high.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        z_ne = model.mean_embed(ctx_ids_nonempty)          # sample of populated latents
        centroid = z_ne.mean(dim=0)
        cos = F.cosine_similarity(model.empty_prototype.to(centroid.dtype), centroid, dim=0)
    return float(abs(cos))


def maybe_reseed_prototype(model: Any, ctx_ids_nonempty: Any, *, threshold: float = 0.15,
                           seeds: tuple[int, ...] = (20260706, 7, 13, 101, 2027, 99991)) -> dict[str, Any]:
    """Reseed the frozen prototype until it is well-separated from the non-empty
    sample (|cos| <= threshold), or exhaust the seed list. Returns a report."""
    tried = []
    for seed in seeds:
        model.reseed_empty_prototype(seed)
        sep = separation_cosine(model, ctx_ids_nonempty)
        tried.append({"seed": seed, "abs_cos": sep})
        if sep <= threshold:
            return {"chosen_seed": seed, "abs_cos": sep, "threshold": threshold, "attempts": tried, "ok": True}
    best = min(tried, key=lambda t: t["abs_cos"])
    model.reseed_empty_prototype(int(best["seed"]))
    return {"chosen_seed": int(best["seed"]), "abs_cos": best["abs_cos"], "threshold": threshold, "attempts": tried, "ok": False}


def collapse_diagnostics(model: Any, ctx_ids: Any, is_empty: Any, count: Any) -> dict[str, Any]:
    """Empty/non-empty-separated anti-collapse diagnostics (Pi R4 Q1/Q3).

    - occupancy AUC + Brier vs the marginal (base-rate) Brier;
    - empty-vs-populated cosine-to-z_empty margin (must be positive/high);
    - empty-vs-1-event occupancy AUC (Pi Q3: the hard boundary, not just recall);
    - empty recall + false-positive-rate (recall alone is gameable).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        ctx_z = model.mean_embed(ctx_ids)
        pred = model.predict_rollout_from_latent(ctx_z, 1)[:, 0, :]
        occ_logit, _ = model.predict_occupancy_from_latent(ctx_z, 1)
        occ_prob = torch.sigmoid(occ_logit[:, 0, 0])
        y_occ = (~is_empty.to(torch.bool)).float()
        empty_mask = is_empty.to(torch.bool)
        z_e = model.empty_prototype.to(pred.dtype).view(1, -1).expand_as(pred)
        cos_e = F.cosine_similarity(pred, z_e, dim=-1)

    empty_cos = float(cos_e[empty_mask].mean()) if bool(empty_mask.any()) else float("nan")
    pop_cos = float(cos_e[~empty_mask].mean()) if bool((~empty_mask).any()) else float("nan")
    margin = (empty_cos - pop_cos) if not (math.isnan(empty_cos) or math.isnan(pop_cos)) else float("nan")

    # occupancy classifier metrics (predict occupied = non-empty). Empty prediction
    # = occ_prob < 0.5. Empty recall = of true empties, fraction predicted empty.
    pred_empty = occ_prob < 0.5
    n_empty = int(empty_mask.sum())
    n_pop = int((~empty_mask).sum())
    empty_recall = float((pred_empty & empty_mask).sum() / max(1, n_empty)) if n_empty else float("nan")
    empty_fpr = float((pred_empty & (~empty_mask)).sum() / max(1, n_pop)) if n_pop else float("nan")

    auc = binary_auc(occ_prob, y_occ)
    brier = float(((occ_prob - y_occ) ** 2).mean())
    base_rate = float(y_occ.mean())
    marginal_brier = base_rate * (1.0 - base_rate)

    # empty-vs-1-event separation (Pi Q3): AUC of occ_prob over count in {0,1} only.
    cnt = count.reshape(-1).float()
    sel = (cnt <= 1.0)
    ev1_auc = binary_auc(occ_prob[sel], (cnt[sel] >= 1.0).float()) if bool(sel.any()) else float("nan")

    return {
        "occupancy_auc": auc,
        "brier": brier,
        "marginal_brier": marginal_brier,
        "beats_marginal": bool(brier < marginal_brier),
        "empty_cos_z_empty": empty_cos,
        "populated_cos_z_empty": pop_cos,
        "empty_vs_populated_margin": margin,
        "empty_vs_one_event_auc": ev1_auc,
        "empty_recall": empty_recall,
        "empty_false_positive_rate": empty_fpr,
        "base_rate_occupied": base_rate,
        "n_empty": n_empty,
        "n_populated": n_pop,
    }


def calibration_report(occ_prob: Any, y_occ: Any, *, n_bins: int = 10) -> dict[str, Any]:
    """Natural-prevalence calibration (Pi Q5): Brier + expected calibration error.

    MUST be computed on the UNCAPPED (natural prevalence) distribution — capping the
    empty fraction during training biases the prior.
    """
    import torch

    p = occ_prob.detach().reshape(-1).float()
    y = y_occ.detach().reshape(-1).float()
    brier = float(((p - y) ** 2).mean())
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.numel()
    bins = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        m = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if not bool(m.any()):
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        w = float(m.sum()) / n
        ece += w * abs(conf - acc)
        bins.append({"lo": lo, "hi": hi, "confidence": conf, "accuracy": acc, "weight": w})
    return {"brier": brier, "ece": ece, "prevalence_occupied": float(y.mean()), "bins": bins,
            "note": "computed on natural (uncapped) prevalence; capped sampler biases the prior (Pi Q5)"}
