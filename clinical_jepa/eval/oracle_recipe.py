"""Candidate-recipe boundary (Pi 2nd-pass Phase 3) — the load-bearing certified object.

Certification scores a REGISTERED candidate recipe's sampled context → predicted latent → decoded
order, ``D(ẑ(context))`` — NOT a hand-coded Bayes predictor and NOT the target-side ceiling ``D(z⁺)``.
The recipe reads ONLY capability-restricted views: ``fit`` sees a ``SplitViews`` bundle of the context
and future (target-side) views — NO eval-label / family view is ever put in the bundle;
``predict_latent`` sees the ContextView ONLY (never labels / family ID / null flag). ``recipe_hash`` is
recomputed from the immutable spec.

Boundary guards:
  * ``assert_predictor_context_only`` — predict must run from a ContextView without reaching a
    non-context channel (a label read raises CapabilityError), and its output must be invariant to
    perturbing the eval cell's future/labels.
  * ``assert_labels_eval_only`` — runs ``fit`` on instrumented bundles and inspects the recorded access
    trace: no EvalLabelView / FamilyMetadata channel may have been served (they are not even in the
    bundle, so a compliant recipe cannot reach them; the trace makes that auditable).

Toy recipes here (Good / ContextBlind / LabelPeeking) demonstrate the boundary; the property-specific
bad recipes that must fail ONE scored gate live with the references/integration phases. All synthetic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from clinical_jepa.eval.oracle_contracts import (
    CONTEXT_CHANNELS, FUTURE_CHANNELS, CapabilityError, DecodedOrder, DecoderSamplerSpec,
    FittedRecipeArtifact, RecipeSpec, RestrictedView, SamplerSpec, canonical_hash,
)
from clinical_jepa.eval.oracle_literal_gen import LiteralCell


@dataclass(frozen=True)
class SplitViews:
    """The ONLY data a recipe's fit receives per split: context (input) + future (target-side). No
    eval-label or family view is ever bundled here — that is the anti-tailoring boundary."""
    context: RestrictedView
    future: RestrictedView

    def accessed(self) -> frozenset[str]:
        return self.context.accessed_channels() | self.future.accessed_channels()


def split_views(cell: LiteralCell) -> SplitViews:
    return SplitViews(context=cell.context_view(), future=cell.future_view())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_probs_from_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(item i precedes item j) = sigmoid((s_j - s_i)/T): a HIGHER predicted order-score means LATER,
    so i precedes j when s_i < s_j. Returns (N, L, L) calibrated pairwise probabilities."""
    diff = scores[:, None, :] - scores[:, :, None]        # diff[n,i,j] = s_j - s_i
    return _sigmoid(diff / max(1e-6, temperature))


class CandidateRecipe(ABC):
    """The certification boundary. predict_latent is CONTEXT-ONLY by contract."""
    _temperature: float = 1.0

    @abstractmethod
    def spec(self) -> RecipeSpec: ...

    def recipe_hash(self) -> str:
        return self.spec().recipe_hash()

    @abstractmethod
    def fit(self, train: SplitViews, dev: SplitViews) -> FittedRecipeArtifact: ...

    @staticmethod
    def encode_target(future: RestrictedView) -> np.ndarray:
        """z⁺ from the realized future ordering — ceiling/target-side only, NEVER a certification input."""
        return np.asarray(future.get("future_events"), dtype=float)

    @abstractmethod
    def predict_latent(self, context: RestrictedView, sampler: SamplerSpec, seed: int) -> np.ndarray:
        """ẑ (per-item predicted order-scores) from the ContextView ONLY."""

    def decode_order(self, z_hat: np.ndarray, decoder_sampler: DecoderSamplerSpec,
                     seed: int) -> DecodedOrder:
        probs = pairwise_probs_from_scores(z_hat, self._temperature)
        perms = None
        if decoder_sampler.return_permutations and decoder_sampler.n_decode_samples > 0:
            rng = np.random.default_rng(seed)
            noisy = z_hat[None] + decoder_sampler.temperature * rng.standard_normal(
                (decoder_sampler.n_decode_samples, *z_hat.shape))
            perms = np.argsort(noisy, axis=-1)
        return DecodedOrder(pairwise_probs=probs, sampled_permutations=perms)


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _design(context: RestrictedView) -> np.ndarray:
    """Per-item design matrix from context. The true order-score is BILINEAR (a hidden factor recovered
    from context, coupled to the item feature), so the design includes the context⊗item interaction
    terms — a purely additive [context, item] model cannot represent context×item coupling."""
    ctx = np.asarray(context.get("context_features"), dtype=float)         # (N, D_CTX)
    item = np.asarray(context.get("item_features"), dtype=float)           # (N, L, D_ITEM)
    n, L, di = item.shape
    ctx_b = np.repeat(ctx[:, None, :], L, axis=1)                          # (N, L, D_CTX)
    cross = (ctx_b[:, :, :, None] * item[:, :, None, :]).reshape(n, L, ctx.shape[1] * di)  # ctx⊗item
    X = np.concatenate([ctx_b, item, cross], axis=2).reshape(n * L, -1)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


class GoodContextRecipe(CandidateRecipe):
    """Fits a context+item -> order-score ridge on the target-side future ordering, then predicts from
    context ONLY. Recovers order for positive cells; never reads labels at predict time."""

    def __init__(self) -> None:
        self._w: np.ndarray | None = None
        self._L: int | None = None

    def spec(self) -> RecipeSpec:
        return RecipeSpec(architecture="ridge_context_order", target_encoder="rank_identity",
                          codebook_cfg={"K": 0}, losses={"mse": 1.0}, optimizer="closed_form",
                          schedule="none", bit_accounting={"target_bits": 8},
                          decode_policy="pairwise_sigmoid", sampler_spec=SamplerSpec(),
                          decoder_sampler_spec=DecoderSamplerSpec(), split_ids={"v": 1},
                          seed_policy="sha256", evaluator_identity="oracle_eval_v3",
                          code_identity="good_context_recipe")

    def fit(self, train: SplitViews, dev: SplitViews) -> FittedRecipeArtifact:
        X = _design(train.context)                                         # context-only design
        y = self.encode_target(train.future).reshape(-1)                   # target-side ordering
        self._w = _ridge_fit(X, y)
        self._L = int(np.asarray(train.future.get("future_events")).shape[1])
        art = canonical_hash({"w_sig": round(float(np.abs(self._w).sum()), 6), "L": self._L})
        return FittedRecipeArtifact(self.recipe_hash(), art, {"n_train": int(X.shape[0])})

    def predict_latent(self, context: RestrictedView, sampler: SamplerSpec, seed: int) -> np.ndarray:
        assert self._w is not None, "fit before predict"
        X = _design(context)
        n = np.asarray(context.get("context_features")).shape[0]
        return (X @ self._w).reshape(n, self._L)


class ContextBlindRecipe(CandidateRecipe):
    """A CONTEXT-BLIND predictor: constant ẑ => pairwise 0.5. Must PASS null and FAIL U1 (Pi
    clarification: a zero predictor must not falsely fire on null)."""

    def spec(self) -> RecipeSpec:
        return RecipeSpec(architecture="context_blind", target_encoder="rank_identity",
                          codebook_cfg={"K": 0}, losses={}, optimizer="none", schedule="none",
                          bit_accounting={"target_bits": 8}, decode_policy="pairwise_sigmoid",
                          sampler_spec=SamplerSpec(), decoder_sampler_spec=DecoderSamplerSpec(),
                          split_ids={"v": 1}, seed_policy="sha256", evaluator_identity="oracle_eval_v3",
                          code_identity="context_blind_recipe")

    def fit(self, train: SplitViews, dev: SplitViews) -> FittedRecipeArtifact:
        return FittedRecipeArtifact(self.recipe_hash(), canonical_hash({"blind": True}))

    def predict_latent(self, context: RestrictedView, sampler: SamplerSpec, seed: int) -> np.ndarray:
        ctx = np.asarray(context.get("context_features"))
        L = np.asarray(context.get("item_features")).shape[1]
        return np.zeros((ctx.shape[0], L))                                 # ignores context entirely


class LabelPeekingRecipe(CandidateRecipe):
    """A recipe that CHEATS by reaching for the eval label through the ContextView. The capability view
    denies it (CapabilityError), so ``assert_predictor_context_only`` catches it."""

    def spec(self) -> RecipeSpec:
        return RecipeSpec(architecture="label_peek", target_encoder="rank_identity", codebook_cfg={},
                          losses={}, optimizer="none", schedule="none", bit_accounting={},
                          decode_policy="pairwise_sigmoid", sampler_spec=SamplerSpec(),
                          decoder_sampler_spec=DecoderSamplerSpec(), split_ids={}, seed_policy="sha256",
                          evaluator_identity="oracle_eval_v3", code_identity="label_peeking_recipe")

    def fit(self, train: SplitViews, dev: SplitViews) -> FittedRecipeArtifact:
        return FittedRecipeArtifact(self.recipe_hash(), canonical_hash({"peek": True}))

    def predict_latent(self, context: RestrictedView, sampler: SamplerSpec, seed: int) -> np.ndarray:
        return np.asarray(context.get("true_order"), dtype=float)          # DENIED -> CapabilityError


# ----------------------------------------------------------------------------------------------
# boundary guards
# ----------------------------------------------------------------------------------------------
def assert_predictor_context_only(recipe: CandidateRecipe, cell: LiteralCell, *,
                                  sampler: SamplerSpec | None = None, seed: int = 0) -> bool:
    """True iff predict runs from the ContextView without reaching a non-context channel AND its output
    is invariant to perturbing the eval cell's future/labels. A label read (CapabilityError) => False."""
    smp = sampler or SamplerSpec()
    try:
        out1 = recipe.predict_latent(cell.context_view(), smp, seed)
        rng = np.random.default_rng(seed + 1)
        rand_order = rng.standard_normal(cell.true_order.shape)
        perturbed = replace(cell, true_order=rand_order,
                            future_events=np.argsort(np.argsort(rand_order, 1), 1),
                            nuisance_u=rng.standard_normal(cell.nuisance_u.shape))
        out2 = recipe.predict_latent(perturbed.context_view(), smp, seed)
    except CapabilityError:
        return False                                                       # tried to read a label channel
    return bool(np.allclose(out1, out2))                                   # context unchanged => invariant


def assert_labels_eval_only(recipe: CandidateRecipe, train_cell: LiteralCell,
                            dev_cell: LiteralCell) -> bool:
    """True iff ``fit`` reads only context/future channels. fit is handed instrumented SplitViews
    bundles (no eval-label/family view present); we then inspect the recorded access trace."""
    train, dev = split_views(train_cell), split_views(dev_cell)
    recipe.fit(train, dev)
    allowed = set(CONTEXT_CHANNELS) | set(FUTURE_CHANNELS)
    return train.accessed().issubset(allowed) and dev.accessed().issubset(allowed)
