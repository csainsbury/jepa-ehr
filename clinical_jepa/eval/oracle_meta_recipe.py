"""Fit-once-on-TRAIN recipes + transfer scoring (Pi keystone GO-WITH-CHANGES).

A recipe is fitted ONCE over the pooled TRAIN families at the TRAIN κ grid only (dev selection frozen
before any held-out generation), then applied UNCHANGED to held-out families and off-grid cells. It
never fits/selects on a held-out family, its identity, or its certification κ (0.15 / 0.35 / 0.60).

  * ``InvariantLearner`` — bilinear (legitimate-context)⊗item ridge that learns the SHARED map; ignores
    the planted shortcut channel; transfers to held-out.
  * ``MemorizerRecipe`` — a linear recipe over the FULL context INCLUDING the shortcut channel; on the
    train families the shortcut leaks order so it clears the gate in-distribution, but on the held-out
    families the shortcut is pure noise so it FAILS to transfer (the structural-transfer falsifier).

Scoring uses the corrected pooled E-O1 (``oracle_meta_refs``) over the MC exact R0.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "big")

from clinical_jepa.eval.oracle_meta_gen import (
    D_ITEM, KAPPA_TRAIN_GRID, MetaCell, TRAIN_FAMILIES, generate_meta_cell,
)
from clinical_jepa.eval import oracle_meta_refs as R

# TRAIN κ grid actually usable for supervised fitting (positive coupling only; κ=0 carries no signal).
FIT_KAPPAS = tuple(k for k in KAPPA_TRAIN_GRID if k > 0)


def _legit_item(item: np.ndarray) -> np.ndarray:
    return item[:, :, :D_ITEM]                            # drop the per-item shortcut channel


def _bilinear(ctx: np.ndarray, item: np.ndarray) -> np.ndarray:
    n, L, di = item.shape
    ctx_b = np.repeat(ctx[:, None, :], L, axis=1)
    cross = (ctx_b[:, :, :, None] * item[:, :, None, :]).reshape(n, L, ctx.shape[1] * di)
    X = np.concatenate([ctx_b, item, cross], axis=2).reshape(n * L, -1)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def _linear(ctx: np.ndarray, item: np.ndarray) -> np.ndarray:
    n, L, _ = item.shape
    ctx_b = np.repeat(ctx[:, None, :], L, axis=1)
    X = np.concatenate([ctx_b, item], axis=2).reshape(n * L, -1)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def _ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _pooled_skill(recipe, cell: MetaCell, positives_only=True):
    probs = R.pairwise_probs(recipe.predict_scores(cell), recipe._T)
    r0 = R.r0_pairwise(cell.family_id, cell.kappa, cell.item_classes)
    b_rec, b_r0, npair = R.per_sequence_briers(probs, cell.true_order, r0)
    mask = (~cell.is_null) if positives_only else np.ones(cell.is_null.shape, bool)
    npair = np.where(mask, npair, 0)
    return R.pooled_eo1_skill(b_rec, b_r0, npair, base_seed=_stable_seed("pooled", cell.family_id))


def _calibrate_temperature(recipe, *, seed: int) -> float:
    """DEV temperature selection on a TRAIN-family DEV cell at a TRAIN-grid κ (never a held-out cell)."""
    dev = generate_meta_cell(TRAIN_FAMILIES[0], FIT_KAPPAS[-1], "orthogonal", 800, seed=seed + 313131)
    r0 = R.r0_pairwise(dev.family_id, dev.kappa, dev.item_classes)
    best_t, best = 1.0, -1e9
    scores = recipe.predict_scores(dev)
    for t in (0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0):
        b_rec, b_r0, npair = R.per_sequence_briers(R.pairwise_probs(scores, t), dev.true_order, r0)
        pt, _, _, _ = R.pooled_eo1_skill(b_rec, b_r0, np.where(~dev.is_null, npair, 0), base_seed=1)
        if pt > best:
            best, best_t = pt, t
    return best_t


class InvariantLearner:
    def __init__(self, lam: float = 1.0) -> None:
        self._w = None; self._lam = lam; self._L = None; self._T = 1.0
        self.fit_provenance: dict = {}

    def fit_on_train(self, *, seed: int = 0, n: int = 1200) -> "InvariantLearner":
        Xs, ys, fams, kaps = [], [], set(), set()
        for i, fam in enumerate(TRAIN_FAMILIES):                 # TRAIN families ...
            for j, kap in enumerate(FIT_KAPPAS):                 # ... over the TRAIN κ grid ONLY
                c = generate_meta_cell(fam, kap, "orthogonal", n, seed=seed + 100 * i + 7 * j)
                Xs.append(_bilinear(c.context_features, _legit_item(c.item_features)))
                ys.append(c.future_events.reshape(-1).astype(float))
                self._L = c.true_order.shape[1]
                fams.add(fam); kaps.add(kap)
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        self._w = _ridge(X, y, self._lam)
        self._T = _calibrate_temperature(self, seed=seed)
        self.fit_provenance = {"families": sorted(fams), "kappas": sorted(kaps)}
        return self

    def predict_scores(self, cell: MetaCell) -> np.ndarray:
        X = _bilinear(cell.context_features, _legit_item(cell.item_features))
        return (X @ self._w).reshape(cell.context_features.shape[0], self._L)


class MemorizerRecipe:
    """Linear recipe over the FULL context (incl the shortcut). Clears the gate on a train-family DEV
    cell (shortcut leaks order there) but FAILS on held-out (shortcut is noise). Structural falsifier."""

    def __init__(self, lam: float = 1.0) -> None:
        self._w = None; self._lam = lam; self._L = None; self._T = 1.0

    def fit_on_train(self, *, seed: int = 0, n: int = 1200) -> "MemorizerRecipe":
        Xs, ys = [], []
        for i, fam in enumerate(TRAIN_FAMILIES):
            for j, kap in enumerate(FIT_KAPPAS):
                c = generate_meta_cell(fam, kap, "orthogonal", n, seed=seed + 100 * i + 7 * j)
                Xs.append(_linear(c.context_features, c.item_features))     # FULL ctx incl shortcut
                ys.append(c.future_events.reshape(-1).astype(float))
                self._L = c.true_order.shape[1]
        X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
        self._w = _ridge(X, y, self._lam)
        self._T = _calibrate_temperature(self, seed=seed)
        return self

    def predict_scores(self, cell: MetaCell) -> np.ndarray:
        X = _linear(cell.context_features, cell.item_features)
        return (X @ self._w).reshape(cell.context_features.shape[0], self._L)


@dataclass(frozen=True)
class TransferResult:
    family_id: str
    point_skill: float
    lower_ci: float
    n_sequences: int


def transfer_score(recipe, family_id: str, *, kappa: float, seed: int, n: int = 1500,
                   support_floor: int = 0) -> TransferResult:
    """Score a fit-once recipe on a held-out cell WITHOUT refit, pooled E-O1 over positives."""
    cell = generate_meta_cell(family_id, kappa, "orthogonal", n, seed=seed + 5000,
                              support_floor=support_floor)
    pt, lo, _, ns = _pooled_skill(recipe, cell)
    return TransferResult(family_id, pt, lo, ns)


def dev_score(recipe, *, kappa: float, seed: int, n: int = 1500) -> TransferResult:
    """Independent TRAIN-family DEV score (in-distribution) — the memorizer must clear the gate here."""
    cell = generate_meta_cell(TRAIN_FAMILIES[0], kappa, "orthogonal", n, seed=seed + 424242)
    pt, lo, _, ns = _pooled_skill(recipe, cell)
    return TransferResult(TRAIN_FAMILIES[0], pt, lo, ns)
