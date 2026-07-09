"""Rung-1 read-out decoder — empty/count-0 head + live collapse falsifier (Pi R4).

The rung-1 frozen-decode ceiling trains a read-out decoder ``D`` on FROZEN target
latents ``z+`` -> future ``(token, dt)``. This module supplies the encode-empty
part of ``D``:

  - a supervised empty logit + count head over ``z+``;
  - a decode rule: if empty is predicted, emit "empty / count 0" and SKIP token /
    inter-event-timing reconstruction;
  - the LIVE FALSIFIER metric: empty recall >= 0.95 (and FPR / precision reported,
    since recall alone is gameable — Pi Q3). Below the recall floor => ``z_empty``
    has drifted into the populated manifold, i.e. the collapse-avoidance failed.

Token-order and Delta-t-timing decoding (the rest of ``D``, plus the KS-D<=0.05
timing gate and exact-order>=0.70) are the broader rung-1 deliverable and MUST be
computed on NON-EMPTY cells only (``nonempty_cell_mask``); count-0 is a real class
in exact-count (>=0.80).
"""
from __future__ import annotations

from typing import Any

EMPTY_RECALL_FLOOR = 0.95  # Pi Q3: live collapse falsifier


def build_empty_count_decoder(dim: int) -> Any:
    """A small read-out head: z+ -> (empty_logit, log1p_count_pred)."""
    import torch.nn as nn

    class _EmptyCountDecoder(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.empty_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.count_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, z: Any) -> Any:
            return self.empty_head(z).squeeze(-1), self.count_head(z).squeeze(-1)

    return _EmptyCountDecoder(int(dim))


def train_empty_count_decoder(decoder: Any, z_plus: Any, is_empty: Any, count: Any, *,
                              steps: int = 400, lr: float = 5e-3) -> list[float]:
    """Train the read-out head on FROZEN z+ (target encoder is not updated at rung 1).

    Loss = BCE(empty_logit, is_empty) + Huber(count_pred, log1p(count)) on NON-EMPTY
    rows (empties carry count 0 -> the empty head owns them)."""
    import torch
    import torch.nn.functional as F

    z = z_plus.detach()                      # frozen target latent
    y_empty = is_empty.to(torch.float32)
    y_logcount = torch.log1p(count.clamp_min(0).to(torch.float32))
    nonempty = ~is_empty.to(torch.bool)
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    decoder.train()
    for _ in range(int(steps)):
        empty_logit, count_pred = decoder(z)
        loss = F.binary_cross_entropy_with_logits(empty_logit, y_empty)
        if bool(nonempty.any()):
            loss = loss + F.smooth_l1_loss(count_pred[nonempty], y_logcount[nonempty])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def decode_rule(decoder: Any, z_plus: Any, *, threshold: float = 0.5) -> dict[str, Any]:
    """Decode rule: empties emit count 0 + skip token/timing; only non-empty rows
    are handed to the token/Delta-t decoder. Returns tensors + a decode mask."""
    import torch

    decoder.eval()
    with torch.no_grad():
        empty_logit, count_pred = decoder(z_plus.detach())
        empty_prob = torch.sigmoid(empty_logit)
        decoded_empty = empty_prob > threshold
        decoded_count = torch.where(
            decoded_empty, torch.zeros_like(count_pred), torch.expm1(count_pred).clamp_min(0.0).round()
        )
    return {
        "empty_prob": empty_prob,
        "decoded_empty": decoded_empty,
        "decoded_count": decoded_count,
        "decode_tokens_mask": ~decoded_empty,   # token/timing decode runs on non-empty only
    }


def nonempty_cell_mask(is_empty: Any) -> Any:
    """Mask selecting non-empty rows — order / KS-D timing metrics use this (Pi)."""
    return ~is_empty.to(bool)


def empty_decode_metrics(decoder: Any, z_plus: Any, is_empty: Any, count: Any, *,
                         threshold: float = 0.5) -> dict[str, Any]:
    """Empty-class decode metrics + the live falsifier verdict (Pi Q3)."""
    import torch

    out = decode_rule(decoder, z_plus, threshold=threshold)
    pred_empty = out["decoded_empty"]
    true_empty = is_empty.to(torch.bool)
    tp = int((pred_empty & true_empty).sum())
    fp = int((pred_empty & ~true_empty).sum())
    fn = int((~pred_empty & true_empty).sum())
    tn = int((~pred_empty & ~true_empty).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    # count-0 as a real class + exact-count including 0.
    decoded_count = out["decoded_count"]
    true_count = count.to(torch.float32)
    exact_count = float((decoded_count.round() == true_count.round()).float().mean())
    count0_acc = (
        float((decoded_count[true_empty] == 0).float().mean()) if bool(true_empty.any()) else float("nan")
    )

    passes = (not (recall != recall)) and recall >= EMPTY_RECALL_FLOOR  # recall==recall filters NaN
    return {
        "empty_recall": recall,
        "empty_precision": precision,
        "empty_false_positive_rate": fpr,
        "empty_recall_floor": EMPTY_RECALL_FLOOR,
        "passes_empty_falsifier": bool(passes),
        "exact_count_incl_zero": exact_count,
        "count0_accuracy_on_true_empty": count0_acc,
        "n_empty": int(true_empty.sum()),
        "n_populated": int((~true_empty).sum()),
        "order_timing_scope": "non_empty_cells_only",
        "note": "empty recall < floor => z_empty drifted into the populated manifold (collapse alarm)",
    }
