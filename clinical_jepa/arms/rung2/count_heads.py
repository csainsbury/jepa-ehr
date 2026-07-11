"""Rung-2 sub-gate 2 matched count-head family (Pi v2: authorized to build; no governed data).

ONE matched predictive family (Pi #3): interface A (factorized CONTEXT head) and interface B
(the count channel of the concatenated TARGET representation) are BOTH categorical count-
distribution predictors from context, SAME architecture + matched param budget, varying ONLY
where count enters. Emits a proper count PMF over 0..K_max consumed by
`rung2_count_interface.ranked_probability_score` (a categorical subsumes the zero-hurdle +
count-distribution family and gives a clean PMF for the RPS proper score). numpy-friendly outputs;
torch training loop reused from the Rung-1 head pattern.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _matched_head(input_dim: int, k_max: int, embedding_dim: int) -> Any:
    import torch.nn as nn
    from clinical_jepa.eval.rung2_contract import matched_head_hidden
    h = matched_head_hidden(int(input_dim), int(k_max) + 1, int(embedding_dim))
    return nn.Sequential(nn.Linear(int(input_dim), h), nn.GELU(), nn.Linear(h, int(k_max) + 1))


def train_count_head(z_ctx: Any, counts: Any, embedding_dim: int, *, k_max: int = 20,
                     steps: int = 400, lr: float = 5e-3, target_latents: Any = None) -> Any:
    """Train a categorical count PMF from CONTEXT (proper log-score / cross-entropy → RPS-aligned).

    Interface A: target_latents is None → predict count directly from the context latent.
    Interface B: target_latents given → the head first predicts the concatenated TARGET
    representation (cosine-regressed) and reads count from it — i.e. count enters via the target
    interface. Both share this same categorical family + matched budget (Pi #3)."""
    import torch
    import torch.nn.functional as F
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    y = torch.as_tensor(np.clip(np.asarray(counts), 0, k_max), dtype=torch.long)
    if target_latents is None:                            # interface A: context -> count
        head = _matched_head(Z.shape[1], k_max, embedding_dim)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(Z), y).backward()
            opt.step()
        head.eval()
        return {"interface": "A", "head": head, "predictor": None, "k_max": k_max}
    # interface B: context -> predicted target rep -> count read-out (count enters via the target)
    T = torch.as_tensor(np.asarray(target_latents), dtype=torch.float32)
    from clinical_jepa.eval.rung1_decode import build_matched_head
    predictor = build_matched_head(Z.shape[1], T.shape[1], embedding_dim)
    head = _matched_head(T.shape[1], k_max, embedding_dim)
    opt = torch.optim.AdamW(list(predictor.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        zt = predictor(Z)
        loss = (1 - F.cosine_similarity(zt, T, dim=1)).mean() + F.cross_entropy(head(zt.detach()), y)
        loss.backward()
        opt.step()
    predictor.eval(); head.eval()
    return {"interface": "B", "head": head, "predictor": predictor, "k_max": k_max}


def predict_count_pmf(model: dict[str, Any], z_ctx: Any) -> np.ndarray:
    """Return the predicted count PMF [n, k_max+1] (rows sum to 1) for the RPS proper score."""
    import torch
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    with torch.no_grad():
        feat = model["predictor"](Z) if model["predictor"] is not None else Z
        return torch.softmax(model["head"](feat), dim=1).cpu().numpy()
