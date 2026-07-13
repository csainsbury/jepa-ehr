"""Fit-once-on-TRAIN recipes + transfer scoring (Pi 2nd-pass REVISE #1 — CORRECTED DESIGN).

A candidate recipe is fitted ONCE on the registered TRAIN families only, then applied UNCHANGED to the
held-out families and frozen off-grid cells. It never fits, selects, or dispatches on a held-out family,
its identity, or its certification κ.

  * ``InvariantLearner`` — a bilinear context⊗item ridge that learns the SHARED cross-family map, so it
    transfers from the train families to the held-out families.
  * ``MemorizerRecipe`` — a nearest-neighbour recipe that stores train (context, item)→order-score and
    copies the nearest neighbour. It succeeds on train-like data but FAILS under the held-out
    distribution shift (heavier-tailed / exogenous drivers) — the deliberate memorizer control.

E-O1 here is the beyond-content-prior Brier skill over the EXACT non-uniform π0 (from class multisets),
NOT over 0.5 (Pi #2). A content-only predictor (π0 itself) scores skill 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.oracle_meta_gen import (
    MetaCell, TRAIN_FAMILIES, exact_pi0, generate_meta_cell,
)

EO1_TIE_ATOL = 1e-9


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    diff = scores[:, None, :] - scores[:, :, None]
    return _sigmoid(diff / max(1e-6, temperature))


def per_sequence_eo1(probs: np.ndarray, true_order: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Per-sequence Brier SKILL of predicted P(a≺b) OVER THE EXACT π0 (from class multisets), on
    eligible non-tied pairs. skill = 1 - Brier(recipe)/Brier(π0). NaN if no eligible pairs."""
    n, L, _ = probs.shape
    pi0 = exact_pi0(class_ids)
    iu, ju = np.triu_indices(L, k=1)
    out = np.full(n, np.nan)
    for s in range(n):
        to = true_order[s]
        elig = np.abs(to[iu] - to[ju]) > EO1_TIE_ATOL
        if not elig.any():
            continue
        y = (to[iu] < to[ju]).astype(float)
        pr = probs[s, iu, ju][elig]
        p0 = pi0[s, iu, ju][elig]
        b_rec = np.mean((pr - y[elig]) ** 2)
        b_pi0 = np.mean((p0 - y[elig]) ** 2)
        out[s] = 1.0 - b_rec / max(1e-9, b_pi0)
    return out


def _design(ctx: np.ndarray, item: np.ndarray, *, use_interaction: bool = True) -> np.ndarray:
    n, L, di = item.shape
    ctx_b = np.repeat(ctx[:, None, :], L, axis=1)
    blocks = [ctx_b, item]
    if use_interaction:
        blocks.append((ctx_b[:, :, :, None] * item[:, :, None, :]).reshape(n, L, ctx.shape[1] * di))
    return np.concatenate(blocks, axis=2).reshape(n * L, -1)


def _calibrate_temperature(recipe, *, seed: int, kappa: float, n: int = 800) -> float:
    """Pick the decode temperature on a DEV TRAIN-family cell (grid, maximise mean E-O1). Dev-only
    selection is permitted; it never touches a held-out family."""
    dev = generate_meta_cell(TRAIN_FAMILIES[0], kappa, "orthogonal", n, seed=seed + 313131)
    best_t, best = 1.0, -1e9
    scores = recipe.predict_scores(dev)
    for t in (0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0):
        e = per_sequence_eo1(pairwise_probs(scores, t), dev.true_order, dev.item_classes)
        m = float(np.nanmean(e))
        if m > best:
            best, best_t = m, t
    return best_t


class InvariantLearner:
    """Bilinear context⊗item ridge fitted ONCE on pooled TRAIN cells; learns the shared map -> transfers."""

    def __init__(self, lam: float = 1.0) -> None:
        self._w = None
        self._lam = lam
        self._L = None
        self._T = 1.0

    def fit_on_train(self, *, seed: int = 0, kappa: float = 0.5, n: int = 2000) -> "InvariantLearner":
        Xs, ys = [], []
        for i, fam in enumerate(TRAIN_FAMILIES):                 # TRAIN families ONLY
            c = generate_meta_cell(fam, kappa, "orthogonal", n, seed=seed + 10 * i)
            X = _design(c.context_features, c.item_features)
            Xs.append(np.concatenate([X, np.ones((X.shape[0], 1))], 1))
            ys.append(c.future_events.reshape(-1).astype(float))
            self._L = c.true_order.shape[1]
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        self._w = np.linalg.solve(X.T @ X + self._lam * np.eye(X.shape[1]), X.T @ y)
        self._T = _calibrate_temperature(self, seed=seed, kappa=kappa)   # DEV calibration (train family)
        return self

    def predict_scores(self, cell: MetaCell) -> np.ndarray:
        X = _design(cell.context_features, cell.item_features)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        return (X @ self._w).reshape(cell.context_features.shape[0], self._L)

    def eo1(self, cell: MetaCell) -> np.ndarray:
        return per_sequence_eo1(pairwise_probs(self.predict_scores(cell), self._T), cell.true_order,
                                cell.item_classes)


def _overfit_design(ctx: np.ndarray, item: np.ndarray) -> np.ndarray:
    """Bilinear design PLUS high-degree context features (squares, cubes). With little regularization
    these fit the TRAIN driver distribution's shape; under a heavier-tailed held-out driver the
    high-degree terms extrapolate wildly."""
    base = _design(ctx, item, use_interaction=True)
    n, L, _ = item.shape
    hi = np.concatenate([ctx ** 2, ctx ** 3], axis=1)               # degree-2/3 context features
    hi_b = np.repeat(hi[:, None, :], L, axis=1).reshape(n * L, -1)
    return np.concatenate([base, hi_b], axis=1)


class MemorizerRecipe:
    """A HIGH-CAPACITY, lightly-regularized recipe (bilinear + high-degree context features). It fits the
    TRAIN driver distribution well but its high-degree terms EXTRAPOLATE badly under the held-out
    heavier-tailed / exogenous driver shift — the deliberate memorizer control that must NOT transfer."""

    def __init__(self, lam: float = 1e-3) -> None:
        self._w = None
        self._L = None
        self._lam = lam
        self._T = 1.0

    def fit_on_train(self, *, seed: int = 0, kappa: float = 0.5, n: int = 2000) -> "MemorizerRecipe":
        Xs, ys = [], []
        for i, fam in enumerate(TRAIN_FAMILIES):
            c = generate_meta_cell(fam, kappa, "orthogonal", n, seed=seed + 10 * i)
            Xs.append(_overfit_design(c.context_features, c.item_features))
            ys.append(c.future_events.reshape(-1).astype(float))
            self._L = c.true_order.shape[1]
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        self._w = np.linalg.solve(X.T @ X + self._lam * np.eye(X.shape[1]), X.T @ y)
        self._T = _calibrate_temperature(self, seed=seed, kappa=kappa)
        return self

    def predict_scores(self, cell: MetaCell) -> np.ndarray:
        X = _overfit_design(cell.context_features, cell.item_features)
        return (X @ self._w).reshape(cell.context_features.shape[0], self._L)

    def eo1(self, cell: MetaCell) -> np.ndarray:
        return per_sequence_eo1(pairwise_probs(self.predict_scores(cell), self._T), cell.true_order,
                                cell.item_classes)


@dataclass(frozen=True)
class TransferResult:
    family_id: str
    mean_eo1_positive: float
    n_positive: int


def transfer_score(recipe, family_id: str, *, kappa: float, seed: int, n: int = 1500) -> TransferResult:
    """Score a fit-once recipe on a held-out family cell WITHOUT any refit. Positive (non-null) sequences."""
    cell = generate_meta_cell(family_id, kappa, "orthogonal", n, seed=seed + 5000)
    pos = ~cell.is_null
    e = recipe.eo1(cell)[pos]
    return TransferResult(family_id, float(np.nanmean(e)), int(pos.sum()))
