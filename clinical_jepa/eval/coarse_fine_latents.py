"""Rung-0 coarse/fine latent construction — LEAK-FREE by construction (Pi R5 C1/C2/C3).

CRITICAL (C1): a predicted QUERY is a function of CONTEXT ONLY. Observed future
sub-window cardinalities/events may build/audit TARGETS but must never weight a
query, choose a head, or construct the coarse query. The query functions here take
``ctx_ids`` and nothing target-derived — enforced structurally + by an invariance
test.

  - COARSE query  ẑ_coarse = single-step predicted latent from context.
  - FINE queries  ẑ_k       = the K rolled-out predicted latents from context.
  - COARSE target z_coarse  = target_latent over ALL events in W (silence -> z_empty).
  - FINE targets  z_k        = target_latent over sub-window k (silence -> z_empty).

C2: the pooling identity (event-weighted mean of the K sub-window means == full-W
mean) is a TARGET-side arithmetic identity for POPULATED windows under plain
mean_embed only; empties (routed to the frozen z_empty) break it — the all-empty
convention is z_coarse = z_empty. It is a *candidate* abstraction-edge signal, not
proof. C3: budget-matched targets subsample a frozen event budget B (fixed seed),
applied bilaterally to coarse_B AND fine_B; targets with < B events are a separate
stratum (returned as None), never padded.
"""
from __future__ import annotations

from typing import Any

# ---- queries: CONTEXT ONLY (no target/count argument exists — C1 invariant) --------

def coarse_query(model: Any, ctx_ids: Any) -> Any:
    """ẑ_coarse [B,D] — the single-step predicted future latent from context only."""
    return model.predict_rollout_from_context_ids(ctx_ids, 1)[:, 0, :]


def fine_queries(model: Any, ctx_ids: Any, K: int) -> Any:
    """ẑ_k [B,K,D] — the K rolled-out predicted latents from context only."""
    return model.predict_rollout_from_context_ids(ctx_ids, int(K))


def predicted_counts(model: Any, ctx_ids: Any, K: int) -> dict[str, Any]:
    """Context-only predicted occupancy/count (for the raw-count corroboration route,
    Pi R5 Q3): coarse total vs the K fine sub-window counts."""
    ctx_z = model.mean_embed(ctx_ids)
    fine_occ, fine_cnt = model.predict_occupancy_from_latent(ctx_z, int(K))
    coarse_occ, coarse_cnt = model.predict_occupancy_from_latent(ctx_z, 1)
    return {
        "fine_occ_logit": fine_occ[:, :, 0],
        "fine_logcount": fine_cnt[:, :, 0],
        "coarse_occ_logit": coarse_occ[:, 0, 0],
        "coarse_logcount": coarse_cnt[:, 0, 0],
    }


# ---- targets: TARGET-side (observed events allowed here) ---------------------------

def coarse_target(model: Any, ids: Any, is_empty: Any) -> Any:
    """z_coarse [B,D] — target_latent over all W events (silence -> frozen z_empty)."""
    return model.target_latent(ids, is_empty)


def fine_target(model: Any, ids: Any, is_empty: Any) -> Any:
    """z_k [B,D] for one sub-window (silence -> frozen z_empty)."""
    return model.target_latent(ids, is_empty)


def budget_subsample_ids(event_ids: Any, budget_B: int, seed: int) -> Any | None:
    """Fixed-seed subsample of exactly ``budget_B`` non-pad event ids from a single
    target, or None if the target has < B events (a separate stratum, Pi R5 C3).
    Applied identically to coarse_B and fine_B (bilateral matching)."""
    import numpy as np

    arr = np.asarray(event_ids)
    arr = arr[arr != 0]                       # drop [PAD]
    if len(arr) < int(budget_B):
        return None
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(arr), size=int(budget_B), replace=False))
    return arr[idx].astype(np.int64)


def pooling_identity_residual(model: Any, subwindow_id_arrays: list[Any]) -> float:
    """C2 numerical check on POPULATED windows: max abs difference between the full-W
    mean_embed and the event-weighted mean of the per-sub-window mean_embeds. ~0 for
    plain mean_embed on populated sub-windows; empties (z_empty) are excluded here."""
    import numpy as np
    import torch

    pops = [np.asarray(a)[np.asarray(a) != 0] for a in subwindow_id_arrays]
    pops = [a for a in pops if len(a) > 0]
    if not pops:
        return 0.0
    full = np.concatenate(pops)
    with torch.no_grad():
        z_full = model.mean_embed(torch.as_tensor(full[None, :], dtype=torch.long))[0]
        n = np.array([len(a) for a in pops], dtype=np.float64)
        z_subs = torch.stack([model.mean_embed(torch.as_tensor(a[None, :], dtype=torch.long))[0] for a in pops])
        w = torch.as_tensor(n / n.sum(), dtype=z_subs.dtype)
        z_weighted = (z_subs * w[:, None]).sum(dim=0)
    return float((z_full - z_weighted).abs().max())
