"""Rung-2 sub-gate 3 order predictor (Pi v2: authorized to build; frozen T1-T3 only, no governed).

Prediction-achieved order: a predictor maps observable CONTEXT -> the frozen order target rep
`ẑ` (cosine-regressed), and a pairwise-order decoder reads `ẑ` -> P(a≺b). Order fidelity is
ALWAYS scored on the context-only predicted `ẑ` (never the true target `z⁺` ceiling — Pi oracle
#2). T4 (learned VQ) is NOT built here (barred until the oracle). Reuses the Rung-1 matched-budget
head + pairwise reconstruction machinery.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def train_order_predictor(z_ctx: Any, order_targets: Any, ordered_id_lists: list[Any], E: Any,
                          embedding_dim: int, *, steps: int = 300, lr: float = 5e-3) -> dict[str, Any]:
    """Train (i) a CONTEXT->target predictor (cosine) and (ii) a pairwise-order decoder on the
    PREDICTED target `ẑ` (prediction-achieved). Returns the two heads + E."""
    import torch
    import torch.nn.functional as F
    from clinical_jepa.eval.rung1_decode import build_matched_head
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    T = torch.as_tensor(np.asarray(order_targets), dtype=torch.float32)
    Et = torch.as_tensor(np.asarray(E), dtype=torch.float32)
    predictor = build_matched_head(Z.shape[1], T.shape[1], embedding_dim)
    D = int(Et.shape[1])
    pair_head = build_matched_head(T.shape[1] + 2 * D, 1, embedding_dim)   # [ẑ ⊕ E(a) ⊕ E(b)] -> P(a≺b)
    opt = torch.optim.AdamW(list(predictor.parameters()) + list(pair_head.parameters()), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        zt = predictor(Z)
        loss = (1 - F.cosine_similarity(zt, T, dim=1)).mean()
        # pairwise-order loss on the PREDICTED ẑ (detached from the cosine term for stability)
        feats, labels = [], []
        for i, ids in enumerate(ordered_id_lists):
            ids = np.asarray(ids)
            n = len(ids)
            if n < 2:
                continue
            for a in range(min(n, 6)):
                for b in range(min(n, 6)):
                    if a == b:
                        continue
                    feats.append(torch.cat([zt[i].detach(), Et[int(ids[a])], Et[int(ids[b])]]))
                    labels.append(1.0 if a < b else 0.0)
        if feats:
            fb = torch.stack(feats); lb = torch.as_tensor(labels, dtype=torch.float32).view(-1, 1)
            loss = loss + F.binary_cross_entropy_with_logits(pair_head(fb), lb)
        loss.backward()
        opt.step()
    predictor.eval(); pair_head.eval()
    return {"predictor": predictor, "pair_head": pair_head, "E": np.asarray(E), "embedding_dim": embedding_dim}


def predict_precedence(model: dict[str, Any], z_ctx: Any, ordered_id_lists: list[Any]) -> list[np.ndarray]:
    """Prediction-achieved P(a≺b) matrices per window, decoded from the CONTEXT-only predicted ẑ."""
    import torch
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    Et = torch.as_tensor(np.asarray(model["E"]), dtype=torch.float32)
    with torch.no_grad():
        zt = model["predictor"](Z)
        out = []
        for i, ids in enumerate(ordered_id_lists):
            ids = np.asarray(ids); n = len(ids)
            P = np.full((n, n), 0.5)
            for a in range(n):
                for b in range(n):
                    if a == b:
                        continue
                    feat = torch.cat([zt[i], Et[int(ids[a])], Et[int(ids[b])]]).view(1, -1)
                    P[a, b] = float(torch.sigmoid(model["pair_head"](feat)))
            out.append(P)
    return out
