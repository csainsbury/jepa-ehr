"""Semi-synthetic oracle GENERATOR (safe-public, fully synthetic — Pi #5/#7).

Realizes the frozen ``oracle_spec`` families as a compact linear-Gaussian mechanism whose true order
is KNOWN, so the reference bracket and the context-only evaluator can be checked against ground truth.
No real data, no governed reads. All randomness is seeded and reproducible from the caller's seed.

Per sequence the generator emits:
  * ``x_ctx``   — observed context features = W·driver + noise (reflect the driver: h, or exogenous z).
  * ``x_hleak`` — an h-SHORTCUT channel = h+noise for h-families, PURE NOISE for the no-h family. A
                  certifier that leans on this channel gets skill on h-families but MUST fail no-h.
  * ``f``       — observed item content features (the certifier knows WHICH items, not their order).
  * ``s_true``  — true order-scores (the certified property). Null sequences draw s independent of driver.
  * ``u``       — nuisance channel; orthogonal cell => u ⟂ order, correlated_leak cell => bounded proxy.
  * ``is_null`` — camouflaged mechanistic null (identical marginals to positives; zero context-order info).

The generator NEVER exposes s_true (the label) to any evaluator input — evaluators read only x_ctx / f
(and, for the negative controls, x_hleak / u). Label-perturbation invariance follows structurally.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.oracle_spec import (
    NUISANCE_LEAK_RHO, StructuralFamily, get_family,
)

D_H = 6         # driver (hidden or exogenous) dimension
D_CTX = 10      # observed context feature dimension
D_ITEM = 4      # observed per-item content feature dimension
L_ITEMS = 8     # items whose order is certified, per sequence
CTX_NOISE = 0.5
HLEAK_NOISE = 0.5


def _family_seed(family_id: str) -> int:
    return int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:4], "big")


@dataclass(frozen=True)
class _Mechanism:
    W_ctx: np.ndarray     # (D_CTX, D_H) driver -> context map
    M: np.ndarray         # (D_H, D_ITEM) driver x item -> order-score coupling
    A: np.ndarray         # (D_H, D_CTX) linear posterior map x_ctx -> E[driver|x_ctx]


def _mechanism(family_id: str) -> _Mechanism:
    """Fixed per-family matrices (deterministic from family_id — part of the frozen mechanism)."""
    rng = np.random.default_rng(_family_seed(family_id))
    W = rng.standard_normal((D_CTX, D_H))
    M = rng.standard_normal((D_H, D_ITEM))
    # Bayes-optimal linear posterior mean of driver given x_ctx = W driver + N(0, CTX_NOISE^2 I):
    prec = (W.T @ W) / (CTX_NOISE ** 2) + np.eye(D_H)
    A = np.linalg.solve(prec, W.T) / (CTX_NOISE ** 2)
    return _Mechanism(W_ctx=W, M=M, A=A)


@dataclass(frozen=True)
class GeneratedCell:
    family_id: str
    kappa: float
    nuisance_cell: str
    x_ctx: np.ndarray      # (N, D_CTX)
    x_hleak: np.ndarray    # (N, D_H)
    f: np.ndarray          # (N, L, D_ITEM)
    s_true: np.ndarray     # (N, L)
    u: np.ndarray          # (N, L)
    is_null: np.ndarray    # (N,) bool


def generate_cell(family: StructuralFamily | str, kappa: float, nuisance_cell: str,
                  n_sequences: int, *, seed: int) -> GeneratedCell:
    """Generate one (family, kappa, nuisance-cell) cell of `n_sequences` fully-synthetic sequences."""
    fam = family if isinstance(family, StructuralFamily) else get_family(family)
    if nuisance_cell not in ("orthogonal", "correlated_leak"):
        raise ValueError(f"unknown nuisance cell: {nuisance_cell!r}")
    mech = _mechanism(fam.family_id)
    rng = np.random.default_rng(seed)
    N, L = n_sequences, L_ITEMS

    # driver: hidden common cause h (has_h) OR an exogenous observable process z (no-h family).
    driver = rng.standard_normal((N, D_H))
    x_ctx = driver @ mech.W_ctx.T + CTX_NOISE * rng.standard_normal((N, D_CTX))
    if fam.has_h:
        x_hleak = driver + HLEAK_NOISE * rng.standard_normal((N, D_H))   # h leaks here
    else:
        x_hleak = rng.standard_normal((N, D_H))                          # NO h -> pure noise

    f = rng.standard_normal((N, L, D_ITEM))
    # true order-score s_k = kappa * driver^T M f_k + sqrt(1-kappa^2) noise (positives);
    # null sequences: s independent of driver (camouflaged content-prior).
    coupling = np.einsum("nd,de,nle->nl", driver, mech.M, f)             # (N, L)
    noise = rng.standard_normal((N, L))
    resid_scale = float(np.sqrt(max(1e-6, 1.0 - kappa ** 2)))
    s_pos = kappa * coupling + resid_scale * noise
    s_null = rng.standard_normal((N, L))                                 # marginals match by construction
    is_null = rng.random(N) < fam.null_mixture_weight
    s_true = np.where(is_null[:, None], s_null, s_pos)

    # nuisance channel u (per item)
    if nuisance_cell == "orthogonal":
        u = rng.standard_normal((N, L))                                  # independent of order
    else:  # correlated_leak: bounded monotone proxy of the TRUE order-score + noise
        sz = (s_true - s_true.mean(axis=1, keepdims=True)) / (s_true.std(axis=1, keepdims=True) + 1e-9)
        rho = NUISANCE_LEAK_RHO
        u = rho * sz + float(np.sqrt(1.0 - rho ** 2)) * rng.standard_normal((N, L))
    return GeneratedCell(fam.family_id, float(kappa), nuisance_cell, x_ctx, x_hleak, f, s_true, u, is_null)


def mechanism_matrices(family_id: str) -> _Mechanism:
    """Exposed for the reference/evaluator predictors (they read the FROZEN maps, never s_true)."""
    return _mechanism(family_id)
