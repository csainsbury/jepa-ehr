"""Rung-2 sub-gate 4 continuous-time / multiplicity head (Pi v2: authorized to build; no governed).

Timestamp-cluster factorization (Pi #4/#5): the Δt=0 point mass is modelled as MULTIPLICITY (how
many events share a timestamp), NOT a hurdle on the gap — the inter-cluster gap is the strictly
positive time to the next DISTINCT timestamp. Two heads trained on CONTEXT (prediction-achieved,
not the target latent):

  * multiplicity head  — context -> categorical cluster-size distribution (feeds gate 4A).
  * inter-cluster-time head — context -> positive Δt quantiles (pinball; feeds gate 4B positive tail).

Marks are NOT modelled here (hence "continuous-time/multiplicity head", not a marked TPP). Reuses
the Rung-1 matched-budget head + pinball machinery.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def train_multiplicity_head(z_ctx: Any, cluster_size: Any, embedding_dim: int, *, m_max: int = 12,
                            steps: int = 400, lr: float = 5e-3) -> Any:
    """Context -> categorical over cluster size 1..m_max (the simultaneity/Δt=0 multiplicity)."""
    import torch
    import torch.nn.functional as F
    from clinical_jepa.eval.rung1_decode import build_matched_head
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    y = torch.as_tensor(np.clip(np.asarray(cluster_size) - 1, 0, m_max - 1), dtype=torch.long)
    head = build_matched_head(Z.shape[1], m_max, embedding_dim)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(head(Z), y).backward()
        opt.step()
    head.eval()
    return {"head": head, "m_max": m_max}


def predict_multiplicity_pmf(model: dict[str, Any], z_ctx: Any) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.softmax(model["head"](torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)), dim=1).cpu().numpy()


def train_intercluster_time_head(z_ctx: Any, gap: Any, embedding_dim: int, *, n_q: int = 9,
                                 steps: int = 400, lr: float = 5e-3) -> tuple[Any, np.ndarray]:
    """Context -> n_q monotone POSITIVE inter-cluster-time quantiles (pinball). Gaps are strictly
    positive (time to the next DISTINCT timestamp), so NO zero hurdle here."""
    import torch
    from clinical_jepa.eval.rung1_decode import build_matched_head
    Z = torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32)
    g = torch.as_tensor(np.asarray(gap, dtype=np.float32)).view(-1, 1)
    qs = torch.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
    head = build_matched_head(Z.shape[1], n_q, embedding_dim)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        q = torch.cumsum(torch.nn.functional.softplus(head(Z)), dim=1)
        e = g - q
        torch.mean(torch.maximum(qs * e, (qs - 1) * e)).backward()
        opt.step()
    head.eval()
    return {"head": head, "n_q": n_q}, qs.cpu().numpy()


def predict_intercluster_quantiles(model: dict[str, Any], z_ctx: Any) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.cumsum(torch.nn.functional.softplus(
            model["head"](torch.as_tensor(np.asarray(z_ctx), dtype=torch.float32))), dim=1).cpu().numpy()
