"""Rung-0 sufficiency ablation (Pi R5 Q5) — a TRAINED coarse-conditioned fine-render.

Pi requires a real learned two-level model, not a training-free anchoring trick. A
minimal fine-render head predicts each sub-window latent ẑ_k from the context latent,
a positional code, AND the PREDICTED coarse state (never oracle z_coarse — C1). The
FLAT baseline is the SAME head with the coarse slot ZEROED — identical parameters,
FLOPs, prediction calls, optimizer steps, and seeds, so only the coarse INFORMATION
differs. Sufficiency = the two-level model must reduce the per-sub-window drift SLOPE
(drift_k = 1 − cos(ẑ_k, z_k) vs lag k) by a predeclared practical margin, with a
paired bootstrap CI over examples. Diagnostic prototype; the gate decides promotion.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PRACTICAL_DRIFT_IMPROVEMENT = 0.02   # predeclared min drift-slope reduction (per-lag)


def build_fine_render_head(dim: int, K: int, seed: int) -> Any:
    """Head: [ctx_z (dim), coarse (dim), pos_onehot(K)] -> ẑ_k (dim). Flat = coarse zeroed."""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))

    class _Head(nn.Module):
        def __init__(self, d: int, k: int) -> None:
            super().__init__()
            self.dim, self.K = int(d), int(k)
            self.net = nn.Sequential(nn.Linear(2 * d + k, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def rollout(self, ctx_z: Any, coarse: Any, K: int) -> Any:
            N = ctx_z.shape[0]
            eye = torch.eye(self.K, device=ctx_z.device, dtype=ctx_z.dtype)
            outs = [self.net(torch.cat([ctx_z, coarse, eye[k].expand(N, self.K)], dim=-1)) for k in range(K)]
            return torch.stack(outs, dim=1)     # [N, K, D]

    return _Head(int(dim), int(K))


def _coarse_or_zero(coarse: Any, use_coarse: bool) -> Any:
    return coarse if use_coarse else coarse * 0.0


def train_head(head: Any, ctx_z: Any, coarse: Any, targets_z: Any, *, use_coarse: bool,
               steps: int = 300, lr: float = 5e-3) -> None:
    """Cosine-fit ẑ_k -> z_k over all K sub-windows (matched compute across arms)."""
    import torch
    import torch.nn.functional as F

    K = targets_z.shape[1]
    c = _coarse_or_zero(coarse, use_coarse)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    for _ in range(int(steps)):
        pred = head.rollout(ctx_z, c, K)
        loss = (1.0 - F.cosine_similarity(pred.reshape(-1, pred.shape[-1]),
                                          targets_z.reshape(-1, targets_z.shape[-1]), dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def _drift_per_k(head: Any, ctx_z: Any, coarse: Any, targets_z: Any, use_coarse: bool) -> np.ndarray:
    """[N, K] drift = 1 − cos(ẑ_k, z_k), on PREDICTED (or zeroed) coarse — never oracle."""
    import torch
    import torch.nn.functional as F

    K = targets_z.shape[1]
    head.eval()
    with torch.no_grad():
        pred = head.rollout(ctx_z, _coarse_or_zero(coarse, use_coarse), K)
        cos = F.cosine_similarity(pred, targets_z, dim=-1)     # [N, K]
    return (1.0 - cos).cpu().numpy()


def _slope_vs_lag(drift_nk: np.ndarray) -> float:
    """OLS slope of mean drift vs sub-window lag k (higher => faster drift)."""
    K = drift_nk.shape[1]
    if K < 2:
        return 0.0
    return float(np.polyfit(np.arange(K), drift_nk.mean(axis=0), 1)[0])


def compare_flat_vs_2level(dim: int, K: int, ctx_z: Any, coarse_pred: Any, targets_z: Any, *,
                           steps: int = 300, lr: float = 5e-3, seeds: tuple[int, ...] = (0, 1, 2),
                           n_boot: int = 500, boot_seed: int = 0, eval_frac: float = 0.5,
                           practical: float = PRACTICAL_DRIFT_IMPROVEMENT) -> dict[str, Any]:
    """Train matched flat + 2-level heads (multiple seeds) on a TRAIN split, measure the
    per-lag drift SLOPE on a HELD-OUT split (so overfitting a useless coarse cannot pass),
    then paired-bootstrap the improvement (flat − 2level). Sufficiency passes iff the
    improvement CI lower bound clears the predeclared practical margin."""
    N = int(targets_z.shape[0])
    n_train = max(2, int(N * (1.0 - eval_frac)))
    tr = slice(0, n_train)
    ev = slice(n_train, N)
    per_seed = []
    drift2_all, driftf_all = [], []
    for s in seeds:
        h2 = build_fine_render_head(dim, K, s)
        train_head(h2, ctx_z[tr], coarse_pred[tr], targets_z[tr], use_coarse=True, steps=steps, lr=lr)
        hf = build_fine_render_head(dim, K, s)                 # same init seed, coarse zeroed
        train_head(hf, ctx_z[tr], coarse_pred[tr], targets_z[tr], use_coarse=False, steps=steps, lr=lr)
        d2 = _drift_per_k(h2, ctx_z[ev], coarse_pred[ev], targets_z[ev], True)   # HELD-OUT, predicted coarse
        df = _drift_per_k(hf, ctx_z[ev], coarse_pred[ev], targets_z[ev], False)
        drift2_all.append(d2); driftf_all.append(df)
        per_seed.append({"seed": int(s), "slope_2level": _slope_vs_lag(d2), "slope_flat": _slope_vs_lag(df)})

    d2 = np.mean(drift2_all, axis=0)      # [N,K] averaged over seeds
    df = np.mean(driftf_all, axis=0)
    point = _slope_vs_lag(df) - _slope_vs_lag(d2)     # positive => 2-level drifts slower
    N = d2.shape[0]
    rng = np.random.default_rng(boot_seed)
    imps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        imps[b] = _slope_vs_lag(df[idx]) - _slope_vs_lag(d2[idx])
    lo, hi = np.percentile(imps, [2.5, 97.5])
    return {
        "drift_slope_improvement": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "practical_improvement": practical, "sufficiency_ok": bool(lo > practical),
        "mean_slope_2level": _slope_vs_lag(d2), "mean_slope_flat": _slope_vs_lag(df),
        "per_seed": per_seed, "n_examples": int(N), "K": int(K),
        "note": "2-level conditions on PREDICTED coarse (not oracle); matched-compute vs coarse-zeroed flat.",
    }
