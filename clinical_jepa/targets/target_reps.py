"""Rung-1 parameter-free target representations (Pi R7 #5 / R8).

Each arm is a deterministic, UNTRAINED function of the target span (token ids + aligned
cumulative_days) using the frozen embedding table `E` and the frozen `empty_prototype`
(`z_empty`) — zero retrofit surface, no new encoder. Empties are handled DIMENSIONALLY per
arm so every window in an arm has the same width:

  * mean_embed   : z_empty            (dim D)                       — the incumbent (Rung 1a)
  * tap_concat   : [z_empty ⊕ φ_empty] (dim D+d_time)                — a TIMING target (1b)
  * count_concat : [z_empty ⊕ 0]       (dim D+1)                     — an exact-COUNT target (1b)
  * temporal_slot: [z_empty]×M         (dim M·D)                     — a COARSE temporal target (1b)

The φ time-featurizer is FROZEN (d_time=8, no dev-side search). temporal_slot is
permutation-invariant WITHIN a wall-clock slot — it is a coarse-slot, not exact-order, target.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_contract import D_TIME, M_PRIMARY
from clinical_jepa.targets.block_spans import read_target_span
from clinical_jepa.targets.subwindow_blocks import carve_subwindows

ARM_NAMES = ("mean_embed", "tap_concat", "count_concat", "temporal_slot")


def time_features(tau: Any, log_dt: Any, d_time: int = D_TIME) -> np.ndarray:
    """FROZEN sinusoidal time featurizer over events. `tau` = normalized within-window
    time in [0,1); `log_dt` = log1p inter-event gap. Returns [n_events, d_time]: a
    positional sinusoid bank on tau + a rate bank on log_dt, deterministic."""
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    log_dt = np.asarray(log_dt, dtype=np.float64).reshape(-1)
    n = tau.shape[0]
    n_pairs = max(1, d_time // 2)
    n_tau = n_pairs - 1 if n_pairs >= 2 else n_pairs
    n_dt = n_pairs - n_tau
    cols: list[np.ndarray] = []
    for k in range(n_tau):
        f = 2.0 ** k
        cols.append(np.sin(2 * np.pi * f * tau))
        cols.append(np.cos(2 * np.pi * f * tau))
    for k in range(n_dt):
        s = 2.0 ** k
        cols.append(np.sin(log_dt / s))
        cols.append(np.cos(log_dt / s))
    F = np.stack(cols, axis=-1) if cols else np.zeros((n, 0))
    if F.shape[-1] < d_time:
        F = np.concatenate([F, np.zeros((n, d_time - F.shape[-1]))], axis=-1)
    return F[:, :d_time]


def _mean_embed(model: Any, ids: np.ndarray) -> np.ndarray:
    # Sum over SORTED ids so the mean is a BIT-EXACT function of the multiset: any
    # permutation of the same tokens yields the identical vector (Pi R8 #5 permutation-pair
    # invariance), sidestepping float non-associativity. mean_embed is order-blind by design.
    import torch
    with torch.no_grad():
        canon = np.sort(np.asarray(ids, dtype=np.int64))
        t = torch.as_tensor(canon, dtype=torch.long)
        e = model.embedding(t)                      # [n, D]
        return e.mean(dim=0).detach().cpu().numpy().astype(np.float32)


def _z_empty(model: Any) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model.empty_prototype.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _event_times(block: dict[str, Any], cumulative_days: Any) -> np.ndarray:
    times, _ = read_target_span(block, cumulative_days)
    return np.asarray(times, dtype=np.float64).reshape(-1)


def build_target_rep(
    name: str,
    block: dict[str, Any],
    token_ids: Any,
    cumulative_days: Any,
    *,
    model: Any,
    d_time: int = D_TIME,
    slots: int = M_PRIMARY,
) -> np.ndarray:
    """Build the arm's z+ for one block. token_ids/cumulative_days are the FULL sequence
    arrays; the target span is read via the single-reader `read_target_span`."""
    ids, is_empty = read_target_span(block, token_ids)
    D = int(model.embedding.embedding_dim)
    z_empty = _z_empty(model)

    if name == "mean_embed":
        return z_empty.copy() if is_empty else _mean_embed(model, ids)

    if name == "tap_concat":
        base = z_empty.copy() if is_empty else _mean_embed(model, ids)
        if is_empty or len(ids) == 0:
            phi = np.zeros(d_time, dtype=np.float32)             # φ_empty (frozen)
        else:
            t_query = float(block["t_query"]); W = float(block["window_days"])
            times = _event_times(block, cumulative_days)
            tau = np.clip((times - t_query) / max(W, 1e-9), 0.0, 1.0)
            prev = np.concatenate([[t_query], times[:-1]])
            log_dt = np.log1p(np.clip(times - prev, 0.0, None))
            phi = time_features(tau, log_dt, d_time).mean(axis=0).astype(np.float32)
        return np.concatenate([base, phi]).astype(np.float32)

    if name == "count_concat":
        base = z_empty.copy() if is_empty else _mean_embed(model, ids)
        n = 0 if is_empty else int(block.get("n_target_events", len(ids)))
        return np.concatenate([base, [np.float32(np.log1p(n))]]).astype(np.float32)

    if name == "temporal_slot":
        M = int(slots)
        if is_empty:
            return np.tile(z_empty, M).astype(np.float32)
        t_query = float(block["t_query"]); W = float(block["window_days"])
        seq_len = int(block.get("seq_len", len(token_ids)))
        context_end = int(block.get("context_end_ref", block.get("context_end")))
        subs = carve_subwindows(cumulative_days, seq_len, context_end, t_query, M, W / M)
        parts = []
        for sub in subs:
            sids, s_empty = read_target_span(sub, token_ids)
            parts.append(z_empty.copy() if (s_empty or len(sids) == 0) else _mean_embed(model, sids))
        return np.concatenate(parts).astype(np.float32)

    raise ValueError(f"unknown target rep {name!r}")


def target_dim(name: str, embedding_dim: int, *, d_time: int = D_TIME, slots: int = M_PRIMARY) -> int:
    D = int(embedding_dim)
    return {"mean_embed": D, "tap_concat": D + d_time, "count_concat": D + 1,
            "temporal_slot": slots * D}[name]
