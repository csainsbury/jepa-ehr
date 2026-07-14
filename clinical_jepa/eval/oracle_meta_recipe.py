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


def _item_features(ctx: np.ndarray, item: np.ndarray) -> np.ndarray:
    """Per-item features φ(context, item_k) = [item_k, context ⊗ item_k]. A per-item score f_k = w·φ_k
    with sigmoid(f_a − f_b) is a Bradley-Terry pairwise model; the context-only part cancels in a pair
    so it is omitted (the item⊗context interaction carries the order signal)."""
    n, L, di = item.shape
    ctx_b = np.repeat(ctx[:, None, :], L, axis=1)
    cross = (ctx_b[:, :, :, None] * item[:, :, None, :]).reshape(n, L, ctx.shape[1] * di)
    return np.concatenate([item, cross], axis=2)                          # (N, L, di + D_CTX·di)


def _logistic_fit(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """L2-regularized logistic regression via IRLS (Newton) — the pairwise ranker's calibrated fit."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        Wd = np.maximum(p * (1.0 - p), 1e-6)
        grad = X.T @ (p - y) + l2 * w
        H = (X * Wd[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w = w - np.linalg.solve(H, grad)
    return w


def _pair_dataset(cell: MetaCell, feats: np.ndarray, rng, max_pairs: int) -> tuple:
    """Within-sequence precedence pairs on POSITIVE sequences: X = φ_i − φ_j, y = 1[item i precedes j]."""
    pos = np.nonzero(~cell.is_null)[0]
    L = cell.true_order.shape[1]
    iu, ju = np.triu_indices(L, k=1)
    Xs, ys = [], []
    for s in pos:
        to = cell.true_order[s]
        y = (to[iu] < to[ju]).astype(float)                              # i precedes j
        d = feats[s, iu] - feats[s, ju]
        Xs.append(d); ys.append(y)
    X = np.concatenate(Xs, 0); Y = np.concatenate(ys, 0)
    if X.shape[0] > max_pairs:
        idx = rng.choice(X.shape[0], size=max_pairs, replace=False)
        X, Y = X[idx], Y[idx]
    return X, Y


class _Ranker:
    """Fit-once pairwise-logistic ranker. ``_legit`` restricts the item columns; ``_ctx_interaction``
    toggles the context⊗item features that carry the GENERALIZABLE order signal."""
    _legit = True                       # invariant learner uses legitimate item content only
    _ctx_interaction = True             # ... and the context⊗item interaction (transfers to held-out)

    def __init__(self, l2: float = 1.0) -> None:
        self._w = None; self._l2 = l2; self._L = None; self._T = 1.0
        self.fit_provenance: dict = {}

    def _feats(self, cell: MetaCell) -> np.ndarray:
        item = _legit_item(cell.item_features) if self._legit else cell.item_features
        if self._ctx_interaction:
            return _item_features(cell.context_features, item)            # [item, context⊗item]
        return item                                                      # item content ONLY (no ctx signal)

    def fit_on_train(self, *, seed: int = 0, n: int = 1200, max_pairs: int = 40000):
        rng = np.random.default_rng(seed + 555)
        Xs, Ys, fams, kaps = [], [], set(), set()
        per = max(1, max_pairs // (len(TRAIN_FAMILIES) * len(FIT_KAPPAS)))
        for i, fam in enumerate(TRAIN_FAMILIES):                          # TRAIN families ...
            for j, kap in enumerate(FIT_KAPPAS):                          # ... TRAIN κ grid ONLY
                c = generate_meta_cell(fam, kap, "orthogonal", n, seed=seed + 100 * i + 7 * j)
                X, Y = _pair_dataset(c, self._feats(c), rng, per)
                Xs.append(X); Ys.append(Y); fams.add(fam); kaps.add(kap)
                self._L = c.true_order.shape[1]
        self._w = _logistic_fit(np.concatenate(Xs, 0), np.concatenate(Ys, 0), l2=self._l2)
        self._T = self._calibrate_temperature(seed=seed)                  # DEV scale to match π* sharpness
        self.fit_provenance = {"families": sorted(fams), "kappas": sorted(kaps)}
        return self

    def _calibrate_temperature(self, *, seed: int) -> float:
        """DEV decode-temperature on a TRAIN-family DEV cell at a TRAIN-grid κ (never a held-out cell):
        the ranker's ordering transfers but its logit SCALE is calibrated to the train-κ sharpness, so a
        single decode temperature rescales its confidence to the context-Bayes π* (E-O2 slope ≈ 1)."""
        dev = generate_meta_cell(TRAIN_FAMILIES[0], FIT_KAPPAS[-1], "orthogonal", 1500, seed=seed + 313131)
        scores = self.predict_scores(dev); bayes = R.r_bayes_probs(dev)
        best_t, best_gap = 1.0, 1e9
        for t in np.linspace(0.25, 2.0, 36):
            slope, _ = R.e_o2_calibration(R.pairwise_probs(scores, float(t)), bayes, dev.true_order)
            if abs(slope - 1.0) < best_gap:
                best_gap, best_t = abs(slope - 1.0), float(t)
        return best_t

    def predict_scores(self, cell: MetaCell) -> np.ndarray:
        # ranker score f_k with sigmoid(f_i−f_j)=P(i≺j); NEGATE so the decode convention (higher score
        # = LATER) holds: pairwise_probs(−f) = sigmoid((−f_j)−(−f_i)) = sigmoid(f_i−f_j) = P(i≺j).
        return -(self._feats(cell) @ self._w)                            # (N, L)

    def recipe_hash(self) -> str:
        return self.spec().recipe_hash()

    def artifact(self):
        return _fitted_artifact(self.recipe_hash(), self._w, self._T)


class InvariantLearner(_Ranker):
    _legit = True
    def spec(self):
        return _recipe_spec("invariant_pairwise_ranker", self._l2)


class MemorizerRecipe(_Ranker):
    """Pairwise ranker over the FULL item features (incl the pre-future shortcut) but WITHOUT the
    context⊗item interaction — so it has no generalizable order signal to fall back on. On train the
    shortcut leaks the coupling so it ranks well (succeeds in-distribution); on held-out the shortcut is
    noise, leaving only class content ≈ the content prior, so it FAILS to transfer (structural falsifier)."""
    _legit = False
    _ctx_interaction = False
    def spec(self):
        return _recipe_spec("memorizer_shortcut_ranker", self._l2)


def _pooled_skill(recipe, cell: MetaCell, positives_only=True):
    probs = R.pairwise_probs(recipe.predict_scores(cell), recipe._T)
    r0 = R.r0_pairwise(cell.family_id, cell.kappa, cell.item_classes)
    b_rec, b_r0, npair = R.per_sequence_briers(probs, cell.true_order, r0)
    mask = (~cell.is_null) if positives_only else np.ones(cell.is_null.shape, bool)
    npair = np.where(mask, npair, 0)
    return R.pooled_eo1_skill(b_rec, b_r0, npair, base_seed=_stable_seed("pooled", cell.family_id))


RECIPE_TARGET_BITS = 8
RECIPE_CONTROL_BITS = 8


def _recipe_spec(architecture: str, lam: float):
    from clinical_jepa.eval.oracle_contracts import DecoderSamplerSpec, RecipeSpec, SamplerSpec
    from clinical_jepa.eval.oracle_meta_gen import invariant_hash
    return RecipeSpec(
        architecture=architecture, target_encoder="gaussian_rank_quantile",
        codebook_cfg={"kind": "none"}, losses={"ridge_mse": 1.0}, optimizer="closed_form_ridge",
        schedule="none", bit_accounting={"target_bits": RECIPE_TARGET_BITS, "control_bits": RECIPE_CONTROL_BITS},
        decode_policy="pairwise_sigmoid_temperature",
        sampler_spec=SamplerSpec(n_latent_samples=4, temperature=0.3, aggregation="mean_pairwise_prob",
                                 common_random_numbers=True),
        decoder_sampler_spec=DecoderSamplerSpec(n_decode_samples=1),
        split_ids={"train_grid": list(FIT_KAPPAS)}, seed_policy="sha256",
        evaluator_identity="oracle_meta_eval_v4",
        code_identity=f"{architecture}|lam={lam}|mech={invariant_hash()[:16]}")


def _fitted_artifact(recipe_hash: str, w, T: float):
    from clinical_jepa.eval.oracle_contracts import FittedRecipeArtifact, canonical_hash
    art = canonical_hash({"w_bytes": np.asarray(w, float).round(8).tobytes().hex(), "T": round(float(T), 6)})
    return FittedRecipeArtifact(recipe_hash, art, {"T": round(float(T), 6)})


def sampler_fingerprint(recipe) -> str:
    """Stable fingerprint of the recipe's REGISTERED sampler (used to refuse a sampler mismatch)."""
    from clinical_jepa.eval.oracle_contracts import canonical_hash
    s = recipe.spec().sampler_spec
    return canonical_hash({"n": s.n_latent_samples, "T": s.temperature, "agg": s.aggregation,
                           "crn": s.common_random_numbers, "seed_derivation": s.seed_derivation})


def sampled_pairwise_probs(recipe, cell, *, seed: int) -> np.ndarray:
    """Decode via the recipe's REGISTERED stochastic sampler: draw ``n_latent_samples`` predicted-latent
    perturbations (common-random-number seed derivation), decode each, and aggregate the pairwise
    probabilities. Deterministic given ``seed`` (reproducibility is asserted end-to-end)."""
    sampler = recipe.spec().sampler_spec
    scores = recipe.predict_scores(cell)
    if sampler.n_latent_samples <= 1:
        return R.pairwise_probs(scores, recipe._T)
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(int(sampler.n_latent_samples)):
        noisy = scores + sampler.temperature * rng.standard_normal(scores.shape)
        p = R.pairwise_probs(noisy, recipe._T)
        acc = p if acc is None else acc + p
    return acc / sampler.n_latent_samples


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
