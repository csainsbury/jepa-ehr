"""FROZEN second-pass shared contracts (Pi 2nd-pass Phase 0).

Data/view schemas, recipe/artifact/sampler identity, split assignment, and canonical hashing that the
registry, generator, recipe API, references, and calibration all read. Pure structure + deterministic
hashing; NO data, NO RNG, NO governed reads.

Load-bearing anti-tailoring devices (Pi 2nd-pass #1/#2):
  * CAPABILITY-RESTRICTED VIEWS — a recipe's context-only predictor is handed a ``ContextView`` that
    physically cannot read family ID / null flag / kappa / labels. Reading a non-allowed channel raises.
  * RecipeSpec (immutable spec) is SEPARATE from FittedRecipeArtifact (fitted identity): the same spec
    can produce different fitted artifacts, and both hashes are pinned.
  * ``recipe_hash`` is recomputed from the spec here; a recipe-supplied string is never trusted.
  * Three DISTINCT randomness layers are named: the paired-bootstrap estimator, the independent-seed OC
    studies, and the stochastic decoder sampler (SamplerSpec / DecoderSamplerSpec).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

CONTRACTS_VERSION = "clinical-jepa-oracle-contracts-v3"


# ----------------------------------------------------------------------------------------------
# Canonical deterministic hashing (shared by every identity in the oracle).
# ----------------------------------------------------------------------------------------------
def _canonical(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return _canonical(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (tuple, list)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float) and obj == int(obj):
        return float(obj)          # 1 and 1.0 hash identically; keep floats floats
    return obj


def canonical_hash(payload: Any) -> str:
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------------------------
# Capability-restricted views. A view exposes ONLY its allowlisted channels; anything else raises.
# The recipe NEVER receives a FamilyMetadata / EvalLabelView; that is the anti-tailoring boundary.
# ----------------------------------------------------------------------------------------------
class CapabilityError(KeyError):
    """Raised when code reaches for a channel outside a view's declared capability."""


class RestrictedView:
    """Read-only, allowlisted key/value view. Records the channels actually accessed (provenance) so
    ``assert_labels_eval_only`` can inspect a real access trace, not just a perturbation result."""
    __slots__ = ("_name", "_allowed", "_data", "_accessed")

    def __init__(self, name: str, allowed: Iterable[str], data: Mapping[str, Any]):
        self._name = name
        self._allowed = frozenset(allowed)
        extra = set(data) - self._allowed
        if extra:
            raise CapabilityError(f"{name}: data carries non-allowed channels {sorted(extra)}")
        self._data = dict(data)
        self._accessed: set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    def allowed_channels(self) -> frozenset[str]:
        return self._allowed

    def accessed_channels(self) -> frozenset[str]:
        return frozenset(self._accessed)

    def get(self, key: str) -> Any:
        if key not in self._allowed:
            raise CapabilityError(f"{self._name}: channel {key!r} is not in this view's capability")
        self._accessed.add(key)
        return self._data.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._allowed

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"RestrictedView({self._name!r}, allowed={sorted(self._allowed)})"


# the fixed capability allowlists (channels are the discriminated GeneratedCell tags — see generator)
CONTEXT_CHANNELS = ("context_features", "item_features", "context_timestamps", "observed_covariates")
FUTURE_CHANNELS = ("future_multiset", "future_events", "future_timestamps")
EVAL_LABEL_CHANNELS = ("pi0", "true_order", "null_flag", "kappa", "nuisance_labels", "family_labels")
FAMILY_META_CHANNELS = ("family_id", "has_h", "hidden_state", "true_mechanism_params")


def context_view(data: Mapping[str, Any]) -> RestrictedView:
    return RestrictedView("ContextView", CONTEXT_CHANNELS, data)


def future_view(data: Mapping[str, Any]) -> RestrictedView:
    return RestrictedView("FutureView", FUTURE_CHANNELS, data)


def eval_label_view(data: Mapping[str, Any]) -> RestrictedView:
    return RestrictedView("EvalLabelView", EVAL_LABEL_CHANNELS, data)


def family_metadata_view(data: Mapping[str, Any]) -> RestrictedView:
    return RestrictedView("FamilyMetadata", FAMILY_META_CHANNELS, data)


# ----------------------------------------------------------------------------------------------
# Sampler specs — the stochastic-decoder randomness layer (distinct from bootstrap / OC seeds).
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplerSpec:
    n_latent_samples: int = 1          # 1 == deterministic point-mass special case
    temperature: float = 1.0
    seed_derivation: str = "sha256(recipe_hash|context_id|sample_idx)"
    aggregation: str = "mean_pairwise_prob"
    common_random_numbers: bool = True


@dataclass(frozen=True)
class DecoderSamplerSpec:
    n_decode_samples: int = 1
    temperature: float = 1.0
    seed_derivation: str = "sha256(artifact_hash|zhat_id|sample_idx)"
    return_permutations: bool = False


# ----------------------------------------------------------------------------------------------
# Recipe specification (immutable) vs fitted artifact identity (separate hash) — Pi 2nd-pass #1.
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RecipeSpec:
    architecture: str
    target_encoder: str
    codebook_cfg: dict[str, Any]
    losses: dict[str, float]
    optimizer: str
    schedule: str
    bit_accounting: dict[str, Any]
    decode_policy: str
    sampler_spec: SamplerSpec
    decoder_sampler_spec: DecoderSamplerSpec
    split_ids: dict[str, Any]
    seed_policy: str
    evaluator_identity: str
    code_identity: str
    contracts_version: str = CONTRACTS_VERSION

    def recipe_hash(self) -> str:
        """Externally recomputed identity — a recipe-supplied hash string is never trusted."""
        return canonical_hash(self)


@dataclass(frozen=True)
class FittedRecipeArtifact:
    """A fitted result. Its identity (checkpoint/params) is SEPARATE from the spec; both are pinned."""
    originating_recipe_hash: str
    artifact_hash: str            # content hash of the fitted parameters / checkpoint
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitAssignment:
    """Owned by the REGISTRY, not the recipe (Pi 2nd-pass #1)."""
    train: tuple[str, ...]
    dev: tuple[str, ...]
    sealed_cert: tuple[str, ...]
    family_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]

    def assignment_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class EvaluatorAssignment:
    """Registry-owned EVALUATION plan (Pi #4): the held-out family × endpoint-κ inventory the evaluator
    MUST score, plus the seed IDs every eval/OC RNG is derived from. ``compute_unlock`` does not choose
    its own held-out inventory — it consumes this, and the verdict pins the inventory to the canonical
    held-out families."""
    held_out_families: tuple[str, ...]
    endpoints: tuple[float, ...]
    seed_ids: tuple[str, ...]

    def assignment_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class DecodedOrder:
    """Output of decode_order: CALIBRATED pairwise probabilities (P(item i precedes item j)), and
    optionally sampled permutations — not arbitrary scores (Pi 2nd-pass #1)."""
    pairwise_probs: Any                 # np.ndarray (n_items, n_items), row i col j = P(i before j)
    sampled_permutations: Any = None    # optional np.ndarray (n_samples, n_items)
