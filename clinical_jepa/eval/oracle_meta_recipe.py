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
from clinical_jepa.eval.rung2_contract import ORACLE_EVALUATOR_IDENTITY
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


def _pairs_from(feats: np.ndarray, ranks: np.ndarray, rng, max_pairs: int) -> tuple:
    """Within-sequence precedence pairs: X = φ_i − φ_j, y = 1[item i precedes j] (rank_i < rank_j)."""
    n, L = ranks.shape
    iu, ju = np.triu_indices(L, k=1)
    y = (ranks[:, iu] < ranks[:, ju]).astype(float).reshape(-1)
    d = (feats[:, iu] - feats[:, ju]).reshape(n * len(iu), -1)
    if d.shape[0] > max_pairs:
        idx = rng.choice(d.shape[0], size=max_pairs, replace=False)
        d, y = d[idx], y[idx]
    return d, y


class RegistryDataLoader:
    """Registry-OWNED training data source (Pi #1). Given a SplitAssignment it yields ONLY capability
    views (context + future) for the TRAIN families over the TRAIN κ grid, and records the EXTERNAL
    access trace. Contamination refusal reads this trace, NOT recipe-reported metadata; a recipe that
    fits only from this loader cannot read a held-out family / κ."""

    def __init__(self, assignment, *, n: int = 1200) -> None:
        if not assignment.seed_ids:
            raise ValueError("registry assignment carries no seed_ids — cannot derive training RNG")
        self.assignment = assignment
        self._n = n
        self._accessed: list[tuple[str, float]] = []

    def _seed_for(self, *parts) -> int:
        """Every train/dev RNG seed is DERIVED from the registry-owned seed IDs (Pi #4): changing a seed
        ID changes the generated data, so seed IDs are consumed, not decorative labels."""
        return registry_seed(self.assignment.seed_ids, *parts)

    def train_iter(self):
        for fam in self.assignment.train:
            for kap in FIT_KAPPAS:
                self._accessed.append((fam, float(kap)))
                c = generate_meta_cell(fam, kap, "orthogonal", self._n,
                                       seed=self._seed_for("train", fam, kap), null_weight=0.0)
                yield c.context_view(), c.future_view()

    def dev_cell(self) -> MetaCell:
        fam, kap = self.assignment.train[0], FIT_KAPPAS[-1]
        dev_id = self.assignment.dev[0] if self.assignment.dev else "dev0"   # dev IDs are CONSUMED (Pi #4)
        self._accessed.append((fam, float(kap)))
        return generate_meta_cell(fam, kap, "orthogonal", 1500,
                                  seed=self._seed_for("dev", dev_id, fam, kap), null_weight=0.0)

    def fit_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed_for("fit_rng"))

    def access_trace(self) -> dict:
        return {"families": sorted({f for f, _ in self._accessed}),
                "kappas": sorted({k for _, k in self._accessed}),
                "split_assignment_hash": self.assignment.assignment_hash(),
                "seed_ids": list(self.assignment.seed_ids)}


def registry_seed(seed_ids, *parts) -> int:
    """Derive a concrete uint32 RNG seed from the registry-owned seed IDs + a role/cell identity (Pi #4)."""
    key = "|".join([str(s) for s in seed_ids] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2 ** 32)


def default_loader(*, seed: int, n: int = 1200) -> RegistryDataLoader:
    from clinical_jepa.eval.oracle_contracts import SplitAssignment
    a = SplitAssignment(train=TRAIN_FAMILIES, dev=(f"dev::{seed}",), sealed_cert=(),
                        family_ids=TRAIN_FAMILIES, seed_ids=(f"s{seed}",))
    return RegistryDataLoader(a, n=n)


class _Ranker:
    """Fit-once pairwise-logistic ranker. Predicts from a ContextView ONLY (no label/family access).
    ``_legit`` restricts the item columns; ``_ctx_interaction`` toggles the context⊗item features that
    carry the GENERALIZABLE order signal."""
    _legit = True
    _ctx_interaction = True

    def __init__(self, l2: float = 1.0) -> None:
        self._w = None; self._l2 = l2; self._L = None; self._T = 1.0
        self._quant = None                                               # frozen quantizer (fit from TRAIN/DEV)
        self.fit_provenance: dict = {}

    # ---- capability-enforced feature extraction: reads ONLY allowlisted context channels ----
    def _feats_from_view(self, context_view) -> np.ndarray:
        ctx = np.asarray(context_view.get("context_features"), float)     # allowlisted
        item = np.asarray(context_view.get("item_features"), float)       # allowlisted
        item = item[:, :, :D_ITEM] if self._legit else item
        return _item_features(ctx, item) if self._ctx_interaction else item

    def fit(self, loader: RegistryDataLoader, *, max_pairs: int = 40000):
        rng = loader.fit_rng()                                            # RNG derived from registry seed IDs
        views = list(loader.train_iter())
        per = max(1, max_pairs // max(1, len(views)))
        Xs, Ys = [], []
        for cv, fv in views:                                             # (context_view, future_view)
            feats = self._feats_from_view(cv)
            ranks = np.asarray(fv.get("future_events"))                  # target-side ordering
            X, Y = _pairs_from(feats, ranks, rng, per)
            Xs.append(X); Ys.append(Y); self._L = ranks.shape[1]
        self._w = _logistic_fit(np.concatenate(Xs, 0), np.concatenate(Ys, 0), l2=self._l2)
        self._T = self._calibrate_temperature(loader.dev_cell())
        # freeze the quantizer endpoints from the TRAIN latents ONLY (Pi #2): applied unchanged to held-out.
        ref = np.concatenate([self.predict_from_view(cv).ravel() for cv, _ in views])
        self._quant = R.fit_frozen_quantizer(ref, RECIPE_TARGET_BITS)
        self.fit_provenance = loader.access_trace()                      # EXTERNAL trace (not self-reported)
        return self

    def fit_on_train(self, *, seed: int = 0, n: int = 1200, max_pairs: int = 40000):
        """Convenience: fit through a default registry-owned loader (for tests / standalone use)."""
        return self.fit(default_loader(seed=seed, n=n), max_pairs=max_pairs)

    def _calibrate_temperature(self, dev: MetaCell) -> float:
        """DEV decode-temperature on a TRAIN dev cell: rescale the ranker's confidence to the context-
        Bayes π* (E-O2 slope ≈ 1). Uses dev labels/π* only — a permitted dev-calibration step."""
        scores = self.predict_from_view(dev.context_view()); bayes = R.r_bayes_probs(dev)
        best_t, best_gap = 1.0, 1e9
        for t in np.linspace(0.25, 2.0, 36):
            slope, _ = R.e_o2_calibration(R.pairwise_probs(scores, float(t)), bayes, dev.true_order)
            if abs(slope - 1.0) < best_gap:
                best_gap, best_t = abs(slope - 1.0), float(t)
        return best_t

    def predict_from_view(self, context_view) -> np.ndarray:
        """The ONLY prediction entry point — a ContextView, never a MetaCell. Negated so higher = later."""
        return -(self._feats_from_view(context_view) @ self._w)          # (N, L)

    def predict_scores(self, cell: MetaCell) -> np.ndarray:              # internal helper: builds the view
        return self.predict_from_view(cell.context_view())

    def recipe_hash(self) -> str:
        return self.spec().recipe_hash()

    def artifact(self):
        return _fitted_artifact(self.recipe_hash(), self._w, self._T, self._quant)


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


class LabelPeekingRecipe(InvariantLearner):
    """A recipe that CHEATS by reaching for the eval label through the ContextView. The capability view
    DENIES it (CapabilityError) — the boundary is physical, not a convention (Pi #1 deny-test)."""
    def spec(self):
        return _recipe_spec("label_peeking_recipe", self._l2)

    def predict_from_view(self, context_view) -> np.ndarray:
        context_view.get("true_order")                                   # DENIED -> CapabilityError
        return super().predict_from_view(context_view)


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
        codebook_cfg={"kind": "frozen_uniform", "bits": RECIPE_TARGET_BITS, "endpoints": "fit_from_train_dev"},
        losses={"ridge_mse": 1.0}, optimizer="closed_form_ridge",
        schedule="none", bit_accounting={"target_bits": RECIPE_TARGET_BITS, "control_bits": RECIPE_CONTROL_BITS},
        decode_policy="pairwise_sigmoid_temperature",
        sampler_spec=SamplerSpec(n_latent_samples=4, temperature=0.3, aggregation="mean_pairwise_prob",
                                 common_random_numbers=True),
        decoder_sampler_spec=DecoderSamplerSpec(n_decode_samples=1),
        split_ids={"train_grid": list(FIT_KAPPAS)}, seed_policy="sha256",
        evaluator_identity=ORACLE_EVALUATOR_IDENTITY,
        code_identity=f"{architecture}|lam={lam}|mech={invariant_hash()[:16]}")


def expected_sampler_fingerprint() -> str:
    """The canonical registered-sampler fingerprint (identical for every recipe) — the value the pure
    verdict pins ``identities['sampler_fingerprint']`` against (Pi #3)."""
    return sampler_fingerprint(_SamplerFP())


class _SamplerFP:
    def spec(self):
        return _recipe_spec("invariant_pairwise_ranker", 1.0)


def _fitted_artifact(recipe_hash: str, w, T: float, quant=None):
    """Full-byte fitted-parameter identity (Pi #8): exact dtype/shape/contiguous bytes, not rounded.
    Binds the FROZEN quantizer endpoints too (Pi #2 — the code is part of the fitted identity)."""
    from clinical_jepa.eval.oracle_contracts import FittedRecipeArtifact, canonical_hash
    wa = np.ascontiguousarray(w, dtype=np.float64)
    art = canonical_hash({"w_bytes": wa.tobytes().hex(), "dtype": str(wa.dtype), "shape": list(wa.shape),
                          "T_bytes": np.float64(T).tobytes().hex(),
                          "quantizer": quant.identity() if quant is not None else None})
    return FittedRecipeArtifact(recipe_hash, art, {"T": float(T)})


def sequence_bits(bit_accounting: dict, n_items: int) -> int:
    """The ONE total-sequence-bit formula: n_items × target_bits (the quantized per-item latent). Both
    the candidate and the U6 controls are accounted under this same formula (Pi #7)."""
    return int(n_items) * int(bit_accounting.get("target_bits", 8))


def validate_bit_budget(recipe, n_items: int) -> None:
    """Fail-closed bit-accounting (Pi #2): matched controls must use EXACTLY the candidate's target bits
    — not merely ``<=`` — so the U6 comparison is a genuinely matched-bandwidth test."""
    ba = recipe.spec().bit_accounting
    if int(ba.get("control_bits", 8)) != int(ba.get("target_bits", 8)):
        raise RuntimeError("bit-budget mismatch: matched controls must use EXACTLY the candidate's bits")
    if sequence_bits(ba, n_items) <= 0:
        raise RuntimeError("non-positive sequence bit budget")


def sampler_fingerprint(recipe) -> str:
    """Stable fingerprint of the recipe's REGISTERED sampler (used to refuse a sampler mismatch)."""
    from clinical_jepa.eval.oracle_contracts import canonical_hash
    s = recipe.spec().sampler_spec
    return canonical_hash({"n": s.n_latent_samples, "T": s.temperature, "agg": s.aggregation,
                           "crn": s.common_random_numbers, "seed_derivation": s.seed_derivation})


def _context_id(cell) -> str:
    """A stable per-context identifier for the registered sampler seed policy (content of the context)."""
    cf = np.ascontiguousarray(cell.context_features, dtype=np.float64)
    return hashlib.sha256(cf.tobytes()).hexdigest()[:16]


def _sampler_rng(recipe_hash: str, context_id: str, sample_idx: int) -> np.random.Generator:
    """The REGISTERED seed-derivation policy: sha256(recipe_hash|context_id|sample_idx) → RNG (Pi #2)."""
    h = hashlib.sha256(f"{recipe_hash}|{context_id}|{sample_idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def sampled_pairwise_probs(recipe, cell, *, seed: int | None = None) -> np.ndarray:
    """Decode via the recipe's REGISTERED stochastic sampler: draw ``n_latent_samples`` predicted-latent
    perturbations under the DECLARED sha256(recipe_hash|context_id|sample_idx) seed policy, quantize each
    with the recipe's FROZEN quantizer, decode, and aggregate the pairwise probabilities. Deterministic
    from the recipe + context identity (``seed`` is ignored — the policy is intrinsic, Pi #2)."""
    sampler = recipe.spec().sampler_spec
    quant = recipe._quant
    if quant is None:
        raise RuntimeError("recipe has no frozen quantizer — fit() must run before scoring")
    base = recipe.predict_from_view(cell.context_view())              # ContextView latent
    scores, _ = quant.apply(base)                                     # FROZEN, non-adaptive quantization
    if sampler.n_latent_samples <= 1:
        return R.pairwise_probs(scores, recipe._T)
    rhash, cid = recipe.recipe_hash(), _context_id(cell)
    acc = None
    for k in range(int(sampler.n_latent_samples)):
        rng = _sampler_rng(rhash, cid, k)                            # registered per-sample seed derivation
        noisy, _ = quant.apply(base + sampler.temperature * rng.standard_normal(base.shape))
        p = R.pairwise_probs(noisy, recipe._T)
        acc = p if acc is None else acc + p
    return acc / sampler.n_latent_samples


def sampled_clip_diagnostic(recipe, cell) -> float:
    """Max frozen-quantizer clip fraction over the base + sampled latents (overflow diagnostic, Pi #2)."""
    base = recipe.predict_from_view(cell.context_view())
    fracs = [recipe._quant.apply(base)[1]]
    rhash, cid = recipe.recipe_hash(), _context_id(cell)
    s = recipe.spec().sampler_spec
    for k in range(int(s.n_latent_samples)):
        rng = _sampler_rng(rhash, cid, k)
        fracs.append(recipe._quant.apply(base + s.temperature * rng.standard_normal(base.shape))[1])
    return float(max(fracs))


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
