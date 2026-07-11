"""Rung-2 sub-gate 3 FROZEN order-target constructions T1-T3 (Pi v2: authorized to build).

Parameter-free, deterministic functions of the ordered target token sequence using the frozen
embedding table E (zero retrofit surface) — the analogues of the Rung-1 parameter-free arms. T4
(learned VQ codes) is NOT built here: it is barred on governed data until the semi-synthetic
oracle (a separate gate). Bit accounting is explicit (Pi #4). Silence/empty handled by the
frozen z_empty. Sequence length capped at L_max with NO SILENT TRUNCATION (the truncation flag is
returned so coverage is auditable).

  * T1 pooled-ordinal : [mean E ⊕ mean ψ(rank) ⊕ proj(mean(E⊗ψ(rank)))]   (order via a frozen rank tag on a pooled latent)
  * T2 seq-of-latents : [E(id_1) … E(id_L)]  (ordered stack — order is the stack index; CEILING ANCHOR)
  * T3 ordinal-tagged : [E(id_i) ⊕ ψ(rank_i)]_{i=1..L}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung2_contract import ORDER_L_MAX

ORDER_TARGET_NAMES = ("T1_pooled_ordinal", "T2_seq_of_latents", "T3_ordinal_tagged_seq")
D_RANK = 8                                    # frozen ordinal-code width
_MOMENT_PROJ_DIM = 16                         # fixed-random projection of the outer-product moment


def rank_code(ranks: Any, d_rank: int = D_RANK) -> np.ndarray:
    """FROZEN sinusoidal ordinal code ψ(rank) — position within the ordered target (normalized).
    Deterministic; the order-analogue of the Rung-1 time featurizer."""
    r = np.asarray(ranks, dtype=np.float64).reshape(-1)
    n = max(len(r), 1)
    tau = r / n                                # normalized rank in [0,1)
    cols: list[np.ndarray] = []
    for k in range(max(1, d_rank // 2)):
        f = 2.0 ** k
        cols.append(np.sin(2 * np.pi * f * tau)); cols.append(np.cos(2 * np.pi * f * tau))
    F = np.stack(cols, axis=-1)
    if F.shape[-1] < d_rank:
        F = np.concatenate([F, np.zeros((len(r), d_rank - F.shape[-1]))], axis=-1)
    return F[:, :d_rank]


def _moment_projection(D: int, d_rank: int, out_dim: int, seed: int = 20260711) -> np.ndarray:
    """Fixed (seeded) random projection of the flattened E⊗ψ outer product — bounds T1's dim."""
    return np.random.default_rng(seed).standard_normal((D * d_rank, out_dim)) / np.sqrt(D * d_rank)


def _pad_stack(rows: list[np.ndarray], L: int, width: int) -> np.ndarray:
    out = np.zeros((L, width), dtype=np.float32)
    for i in range(min(len(rows), L)):
        out[i] = rows[i]
    return out.reshape(-1)


def build_order_target(name: str, ordered_ids: Any, *, E: np.ndarray, z_empty: np.ndarray,
                       l_max: int = ORDER_L_MAX, d_rank: int = D_RANK) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (z_plus, meta). meta carries the no-silent-truncation flag + exact bit accounting."""
    ids = np.asarray(ordered_ids, dtype=np.int64)
    D = int(np.asarray(E).shape[1])
    n = len(ids)
    truncated = bool(n > l_max)
    L = min(n, l_max) if n > 0 else 0
    meta = {"n_events": int(n), "L_used": int(L), "truncated": truncated,
            "l_max": int(l_max), "empty": bool(n == 0)}

    if n == 0:                                 # silence
        if name == "T1_pooled_ordinal":
            z = np.concatenate([z_empty, np.zeros(d_rank, dtype=np.float32), np.zeros(_MOMENT_PROJ_DIM, dtype=np.float32)])
        elif name == "T2_seq_of_latents":
            z = np.tile(z_empty, l_max).astype(np.float32)
        elif name == "T3_ordinal_tagged_seq":
            z = _pad_stack([], l_max, D + d_rank)
        else:
            raise ValueError(name)
        meta["bits"] = int(z.shape[0] * 32)
        return z.astype(np.float32), meta

    ev = np.asarray(E)[ids[:L]]                 # [L, D]
    psi = rank_code(np.arange(L), d_rank)       # [L, d_rank]
    if name == "T1_pooled_ordinal":
        mean_e = ev.mean(axis=0)
        mean_psi = psi.mean(axis=0)
        outer = (ev[:, :, None] * psi[:, None, :]).mean(axis=0).reshape(-1)   # E⊗ψ moment
        proj = outer @ _moment_projection(D, d_rank, _MOMENT_PROJ_DIM)
        z = np.concatenate([mean_e, mean_psi, proj]).astype(np.float32)
    elif name == "T2_seq_of_latents":
        z = _pad_stack([ev[i] for i in range(L)], l_max, D)   # order = stack index (ceiling anchor)
    elif name == "T3_ordinal_tagged_seq":
        z = _pad_stack([np.concatenate([ev[i], psi[i]]) for i in range(L)], l_max, D + d_rank)
    else:
        raise ValueError(name)
    meta["bits"] = int(z.shape[0] * 32)         # frozen-E targets: dim·32 bit accounting
    return z.astype(np.float32), meta


def order_target_dim(name: str, D: int, *, l_max: int = ORDER_L_MAX, d_rank: int = D_RANK) -> int:
    return {"T1_pooled_ordinal": D + d_rank + _MOMENT_PROJ_DIM,
            "T2_seq_of_latents": l_max * D,
            "T3_ordinal_tagged_seq": l_max * (D + d_rank)}[name]
