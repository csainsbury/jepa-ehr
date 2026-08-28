"""Seed-agnostic frozen-R0 residual prototype for BP011 J04c-v3.

The module is deliberately in-memory and CPU-only.  It contains no production
seed values, file access, device selection, or result persistence.  The guarded
CLI is the only input/output boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clinical_jepa.arms.v0f.own_latent import J04Encoder, SharedLatentPredictor
from clinical_jepa.eval.j04c_falsifier import (
    NUIS_COMP_0, NUIS_COMP_1, NUIS_ORDER_0, NUIS_ORDER_1, TIME_NUIS,
    SyntheticFactorSplit, _state_dict_bytes, _with_preserved_cpu_rng,
    balanced_accuracy, component_seed,
)
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor
from clinical_jepa.eval.j04c_stage1 import (
    _adamw, _fresh_encoder, _fresh_predictor, _prefix_inputs, _tensor_rows,
    optimizer_membership, pretraining_indices,
)
from clinical_jepa.targets.next_event_contract import construct_latent_targets, latent_objective

NAMESPACE = "BP011-J04C-V3-R0RESID-BETA"
CONTRACT_SHA256 = "c765bfe6e68db25a74f7a3aa7a999a4ac6cba7ac88661b840e99d3db192eabba"
TARGET_COMMIT = "c27f20d4e7a67b29c8f3e23dc5f2cab45e170f81"
IMPLEMENTATION_PATHS = (
    "clinical_jepa/eval/j04c_v3_r0resid.py",
    "scripts/bp_clinjepa_011_j04c_v3_r0resid_beta.py",
    "tests/test_bp_clinjepa_011_j04c_v3_r0resid.py",
)
SOURCE_DIGESTS = {
    "clinical_jepa/arms/v0f/own_latent.py": "f3cd838225f8099c79d604036961cc64170c98a87b925fd960550846b7c50dfb",
    "clinical_jepa/targets/next_event_contract.py": "7903f587996a2fd82fde5b13316f853cd06b2c1c8c13eca642dedcad7f8c755d",
    "clinical_jepa/eval/j04c_falsifier.py": "206bf1d59d36a180168b6bb0954d68db50f46b3a0828c6c4c9bbc2e19c843a0a",
    "clinical_jepa/eval/j04c_stage1.py": "167300f6a075b07a2cfcc53fe15c8b507120a9eeb0a553da37841a69b2511bb2",
    "clinical_jepa/eval/j04c_initialization_bridge.py": "099237fc24382f1015df64d53a1017918d3a10d0bb741d6eed98522f4f2b23d9",
    "clinical_jepa/eval/j04c_nuisance_bridge.py": "521d64d60cd91f041cf61c1bd94a2a45fb401772963d5ac4667781aa861eea2e",
    "scripts/bp_clinjepa_011_j04c_l1_beta_3x3.py": "5f375810e2214be46b5e3590995f441709e00108e224659d98ba187cb0b5bf72",
    "tests/test_bp_clinjepa_011_j04c_stage1.py": "976de4041031b82a7dedc6a2b8b192f209b98028ce37054a73c718a55b079f31",
}
C0_HEAD_COMPONENT_CODE = 40
EMA_MOMENTUM = 0.996
READOUT_RIDGE = 1e-3
PRODUCTION_PATH_COUNT = 123


class PrototypeInvariantError(RuntimeError):
    """Fail-closed implementation invariant error (never a scientific result)."""


@dataclass(frozen=True)
class SeedManifest:
    schema: Literal["BP011-J04C-V3-R0RESID-SEEDS-V1"]
    generator_seed: int
    model_seeds: tuple[int, int, int]
    train_target_shuffle_roots: tuple[int, int, int]
    capacity_probe_roots: tuple[int, int, int]
    capacity_cal_original_roots: tuple[int, int, int]
    capacity_cal_intervention_roots: tuple[int, int, int]
    nuisance_intervention_root: int
    bootstrap_root: int


@dataclass(frozen=True)
class ApprovedSeedEnvelope:
    schema: Literal["BP011-J04C-V3-R0RESID-SEED-APPROVAL-V1"]
    manifest_sha256: str
    historical_inventory_sha256: str
    expected_generated_audit_sha256: str
    production_path_count: Literal[123]


@dataclass(frozen=True)
class BuildProvenance:
    schema: Literal["BP011-J04C-V3-R0RESID-BUILD-PROVENANCE-V1"]
    target_commit: str
    implementation_commit: str
    clean_tree: bool
    source_digests: dict[str, str]
    implementation_digests: dict[str, str]
    python_version: str
    numpy_version: str
    torch_version: str
    platform_machine: str
    platform_system: str
    blas_fingerprint: str


@dataclass
class FrozenResidualCondition:
    name: str
    base_encoder: J04Encoder
    adapter: "TokenwiseResidualAdapter | None"
    teacher_adapter: "EMAResidualAdapter | None"
    predictor: SharedLatentPredictor | None
    training: dict[str, object]


@dataclass(frozen=True)
class FeatureBlocks:
    z0: np.ndarray
    delta_z: np.ndarray


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray


@dataclass(frozen=True)
class ReadoutFit:
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool
    zero_short_circuit: bool = False


class TokenwiseResidualAdapter(nn.Module):
    """Exact bias-free 16→16→16 nonlinear adapter applied on the final axis."""

    def __init__(self) -> None:
        super().__init__()
        self.W1 = nn.Linear(16, 16, bias=False)
        self.W2 = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.W1.weight.copy_(torch.eye(16, dtype=self.W1.weight.dtype))
            self.W2.weight.zero_()
        self.assert_initial_state()

    def assert_initial_state(self) -> None:
        identity = torch.eye(16, dtype=self.W1.weight.dtype, device=self.W1.weight.device)
        if self.W1.bias is not None or self.W2.bias is not None:
            raise PrototypeInvariantError("residual adapter biases are forbidden")
        if not torch.equal(self.W1.weight, identity) or not bool(torch.count_nonzero(self.W2.weight) == 0):
            raise PrototypeInvariantError("residual adapter identity/zero initialization failed")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim < 2 or h.shape[-1] != 16:
            raise ValueError("adapter input final dimension must be 16")
        normalized = F.layer_norm(h, (16,), weight=None, bias=None, eps=1e-5)
        return self.W2(F.gelu(self.W1(normalized)))


class EMAResidualAdapter(nn.Module):
    """Non-trainable EMA copy; updates are explicitly controlled by the trainer."""

    def __init__(self, online: TokenwiseResidualAdapter, momentum: float = EMA_MOMENTUM) -> None:
        super().__init__()
        if momentum != EMA_MOMENTUM:
            raise ValueError("EMA momentum must be exactly 0.996")
        online.assert_initial_state()
        self.adapter = copy.deepcopy(online)
        self.momentum = momentum
        self.successful_updates = 0
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.adapter(h)

    @torch.no_grad()
    def update_after_success(self, online: TokenwiseResidualAdapter) -> None:
        online_values = dict(online.named_parameters())
        for name, target in self.adapter.named_parameters():
            source = online_values[name]
            target.mul_(self.momentum).add_(source.detach(), alpha=1.0 - self.momentum)
        self.successful_updates += 1


def freeze_encoder(model_seed: int) -> J04Encoder:
    encoder = _fresh_encoder(model_seed).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return encoder


def state_sha256(module: nn.Module) -> str:
    return hashlib.sha256(_state_dict_bytes(module)).hexdigest()


def apply_tokenwise_before_pooling(
    sequence: torch.Tensor, pooled: torch.Tensor, context_valid: torch.Tensor,
    adapter: TokenwiseResidualAdapter | EMAResidualAdapter | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return accepted z0 and masked mean adapter output, asserting exact pool seam."""
    if sequence.ndim != 3 or sequence.shape[-1] != 16 or pooled.shape != (sequence.shape[0], 16):
        raise ValueError("sequence [B,N,16] and pooled [B,16] required")
    if context_valid.shape != sequence.shape[:2] or context_valid.dtype != torch.bool:
        raise ValueError("context_valid must be bool [B,N]")
    counts = context_valid.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise PrototypeInvariantError("zero-valid-position context row")
    mask = context_valid.unsqueeze(-1).to(sequence.dtype)
    recomputed = (sequence * mask).sum(dim=1) / counts.to(sequence.dtype).clamp_min(1)
    if not torch.equal(pooled, recomputed):
        raise PrototypeInvariantError("pinned encoder pooled output is not exactly reproducible")
    if adapter is None:
        delta = torch.zeros_like(pooled)
    else:
        adapted = adapter(sequence)
        delta = (adapted * mask).sum(dim=1) / counts.to(sequence.dtype)
    return pooled, delta


def encode_context(
    encoder: J04Encoder, adapter: TokenwiseResidualAdapter | None,
    type_ids: torch.Tensor, transformed_intervals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = type_ids.ne(0)
    if bool((valid.sum(dim=1) == 0).any()):
        raise PrototypeInvariantError("zero-valid-position context row")
    with torch.set_grad_enabled(adapter is not None and any(p.requires_grad for p in adapter.parameters())):
        with torch.no_grad():
            _, sequence, pooled = encoder(type_ids, transformed_intervals, causal=False)
        return apply_tokenwise_before_pooling(sequence, pooled, valid, adapter)


def composite_l1_target(
    encoder: J04Encoder, teacher_adapter: EMAResidualAdapter,
    target_ids: torch.Tensor, target_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    valid = target_ids.ne(0)
    if bool((valid.sum(dim=1) == 0).any()):
        raise PrototypeInvariantError("zero-valid-position target row")
    with torch.no_grad():
        blocks, _, _ = encoder(target_ids, target_times, causal=True)
        if blocks.ndim != 4 or tuple(blocks.shape[1:]) != (4, target_ids.shape[1], 16):
            raise PrototypeInvariantError("target block axes must be [B,4,K,16]")
        composite = blocks + teacher_adapter(blocks)
        target, mask, identities = construct_latent_targets(composite, valid, "L1_AVG")
    expected = (target_ids.shape[0], target_ids.shape[1], 16)
    if target.shape != expected or mask.shape != target_ids.shape or identities != [1, 2, 3, 4]:
        raise PrototypeInvariantError("composite L1 target shape or identity order mismatch")
    return target, mask, identities, composite


def l1_directional_objective(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base, parts = latent_objective(prediction, target, valid)
    unit = F.normalize(prediction, dim=-1, eps=1e-8)
    per_identity = unit.var(dim=0, correction=0).mean(dim=-1)
    if per_identity.shape != (prediction.shape[1],):
        raise PrototypeInvariantError("directional variance identity shape mismatch")
    penalty = 5.0 * torch.clamp(0.010 - per_identity, min=0.0).mean()
    return base + penalty, {
        "cosine": parts["cosine"],
        "directional": penalty,
        "v_direction_mean": per_identity.mean(),
        "v_direction_min": per_identity.min(),
    }


def first_w2_gradient_is_finite_nonzero(adapter: TokenwiseResidualAdapter) -> bool:
    gradient = adapter.W2.weight.grad
    return gradient is not None and bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient) > 0)


def make_r0_base(model_seed: int) -> FrozenResidualCondition:
    return FrozenResidualCondition("R0_BASE", freeze_encoder(model_seed), None, None, None, {})


def _all_optimizer_tensors_finite(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not bool(torch.isfinite(parameter).all()):
                return False
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    return False
    return True


def successful_optimizer_update(
    loss: torch.Tensor, optimizer: torch.optim.Optimizer,
    ema: EMAResidualAdapter | None = None,
    online_adapter: TokenwiseResidualAdapter | None = None,
) -> None:
    """Perform exactly one fail-closed update and only then advance EMA."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("TRAINING_NONFINITE_PRE")
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("TRAINING_NONFINITE_PRE")
    try:
        optimizer.step()
    except Exception as error:
        raise PrototypeInvariantError("TRAINING_STEP_EXCEPTION") from error
    if not _all_optimizer_tensors_finite(optimizer):
        raise FloatingPointError("TRAINING_NONFINITE_POST")
    if ema is not None:
        if online_adapter is None:
            raise PrototypeInvariantError("EMA update requires online adapter")
        ema.update_after_success(online_adapter)


def target_tuple_permutation(root: int, n: int) -> np.ndarray:
    if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root < 2**32:
        raise ValueError("shuffle root must be uint32")
    rng = _rng((root, 7301))
    permutation = rng.permutation(n)
    if np.array_equal(permutation, np.arange(n)):
        raise PrototypeInvariantError("SEED_IDENTITY_PERMUTATION")
    return permutation


def shuffled_target_split(split: SyntheticFactorSplit, permutation: np.ndarray) -> SyntheticFactorSplit:
    permutation = np.asarray(permutation, dtype=np.int64)
    n = split.S.shape[0]
    if permutation.shape != (n,) or not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError("one complete row permutation required")
    return SyntheticFactorSplit(
        split.prefix_type_ids, split.prefix_intervals,
        split.target_type_ids[permutation].copy(), split.target_intervals[permutation].copy(),
        split.S, split.X, split.N, split.L_after,
    )


def _loss_window(values: Sequence[float], first: bool) -> float:
    if not values or not all(math.isfinite(x) for x in values):
        raise FloatingPointError("nonfinite or empty training history")
    sample = values[:100] if first else values[-100:]
    return float(np.mean(np.asarray(sample, dtype=np.float64)))


def _training_summary(
    totals: Sequence[float], cosine: Sequence[float], directional: Sequence[float],
    v_min: Sequence[float], successful: int, attempted: int, ema_updates: int,
    *, c0: bool = False,
) -> dict[str, object]:
    components: dict[str, float | None]
    if c0:
        components = {name: None for name in (
            "cosine_first", "cosine_last", "directional_first", "directional_last",
            "v_direction_min_first", "v_direction_min_last",
        )}
    else:
        components = {
            "cosine_first": _loss_window(cosine, True),
            "cosine_last": _loss_window(cosine, False),
            "directional_first": _loss_window(directional, True),
            "directional_last": _loss_window(directional, False),
            "v_direction_min_first": _loss_window(v_min, True),
            "v_direction_min_last": _loss_window(v_min, False),
        }
    return {
        "attempted_steps": attempted, "successful_steps": successful,
        "optimizer_steps": successful, "ema_updates": ema_updates,
        "first_100_mean_total": _loss_window(totals, True),
        "last_100_mean_total": _loss_window(totals, False),
        "component_scalars": components,
    }


def _validate_frozen_encoder(encoder: J04Encoder) -> bytes:
    if encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise PrototypeInvariantError("E0 must be eval, frozen, and gradient-free")
    return _state_dict_bytes(encoder)


def _train_l1_arm(
    name: str, train: SyntheticFactorSplit, transform: Any, model_seed: int,
    encoder: J04Encoder, schedule: Iterable[np.ndarray], *, train_adapter: bool,
    shuffled_split: SyntheticFactorSplit | None = None,
    expected_steps: int,
) -> FrozenResidualCondition:
    e0_before = _validate_frozen_encoder(encoder)
    adapter = TokenwiseResidualAdapter()
    teacher = EMAResidualAdapter(adapter)
    if not train_adapter:
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    predictor = _fresh_predictor(model_seed).train()
    set_identity_gelu_predictor(predictor)
    parts: tuple[nn.Module, ...] = (adapter, predictor) if train_adapter else (predictor,)
    parameters = [p for module in parts for p in module.parameters() if p.requires_grad]
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership(parts, optimizer):
        raise PrototypeInvariantError("optimizer membership mismatch")
    source_targets = train if shuffled_split is None else shuffled_split
    target_times_all = transform.transform(source_targets.target_intervals).astype(np.float32, copy=False)
    totals: list[float] = []
    cosine: list[float] = []
    directional: list[float] = []
    v_min: list[float] = []
    attempted = successful = 0
    for indices in schedule:
        attempted += 1
        prefix_ids, prefix_times = _prefix_inputs(train, transform, "L1_AVG", indices)
        target_ids = _tensor_rows(source_targets.target_type_ids, indices, dtype=torch.long)
        target_times = _tensor_rows(target_times_all, indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        z0, delta = encode_context(encoder, adapter if train_adapter else None, prefix_ids, prefix_times)
        target, valid, _, _ = composite_l1_target(encoder, teacher, target_ids, target_times)
        prediction = predictor(z0 + delta, "L1_AVG")
        loss, metrics = l1_directional_objective(prediction, target, valid)
        loss.backward()
        if attempted == 1 and train_adapter and not first_w2_gradient_is_finite_nonzero(adapter):
            raise PrototypeInvariantError("candidate first W2 gradient is not finite nonzero")
        successful_optimizer_update(loss, optimizer, teacher if train_adapter else None,
                                    adapter if train_adapter else None)
        successful += 1
        totals.append(float(loss.detach()))
        cosine.append(float(metrics["cosine"].detach()))
        directional.append(float(metrics["directional"].detach()))
        v_min.append(float(metrics["v_direction_min"].detach()))
    if attempted != expected_steps or successful != expected_steps:
        raise PrototypeInvariantError("successful-step schedule mismatch")
    expected_ema = expected_steps if train_adapter else 0
    if teacher.successful_updates != expected_ema or _state_dict_bytes(encoder) != e0_before:
        raise PrototypeInvariantError("EMA count or E0 immutability mismatch")
    predictor.eval()
    adapter.eval()
    return FrozenResidualCondition(name, encoder, adapter, teacher, predictor,
        _training_summary(totals, cosine, directional, v_min, successful, attempted, expected_ema))


def train_pred_only(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, base_encoder: J04Encoder,
) -> FrozenResidualCondition:
    return _train_l1_arm("PRED_ONLY", train, transform, model_seed, base_encoder,
                         pretraining_indices(model_seed), train_adapter=False, expected_steps=2000)


def train_pred_only_shuffled(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, base_encoder: J04Encoder,
    target_shuffle_root: int,
) -> FrozenResidualCondition:
    permutation = target_tuple_permutation(target_shuffle_root, 8192)
    return _train_l1_arm("PRED_ONLY_SHUFFLED", train, transform, model_seed, base_encoder,
                         pretraining_indices(model_seed), train_adapter=False,
                         shuffled_split=shuffled_target_split(train, permutation), expected_steps=2000)


def train_resid_candidate(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, base_encoder: J04Encoder,
) -> FrozenResidualCondition:
    return _train_l1_arm("RESID_CANDIDATE", train, transform, model_seed, base_encoder,
                         pretraining_indices(model_seed), train_adapter=True, expected_steps=2000)


def train_resid_shuffled(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, base_encoder: J04Encoder,
    target_shuffle_root: int,
) -> FrozenResidualCondition:
    permutation = target_tuple_permutation(target_shuffle_root, 8192)
    return _train_l1_arm("RESID_SHUFFLED", train, transform, model_seed, base_encoder,
                         pretraining_indices(model_seed), train_adapter=True,
                         shuffled_split=shuffled_target_split(train, permutation), expected_steps=2000)


def make_c0_head(model_seed: int) -> nn.Linear:
    def factory() -> nn.Module:
        head = nn.Linear(16, 1, bias=True)
        nn.init.xavier_uniform_(head.weight)
        with torch.no_grad():
            head.bias.zero_()
        return head
    head = _with_preserved_cpu_rng(component_seed(model_seed, C0_HEAD_COMPONENT_CODE), factory)
    if not isinstance(head, nn.Linear) or head.bias is None or bool(torch.count_nonzero(head.bias)):
        raise PrototypeInvariantError("C0 head initialization mismatch")
    return head


def _train_c0(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, encoder: J04Encoder,
    schedule: Iterable[np.ndarray], *, expected_steps: int,
) -> FrozenResidualCondition:
    e0_before = _validate_frozen_encoder(encoder)
    adapter = TokenwiseResidualAdapter().train()
    head = make_c0_head(model_seed).train()
    optimizer = _adamw(list(adapter.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership((adapter, head), optimizer):
        raise PrototypeInvariantError("C0 optimizer membership mismatch")
    totals: list[float] = []
    attempted = successful = 0
    for indices in schedule:
        attempted += 1
        prefix_ids, prefix_times = _prefix_inputs(train, transform, "L1_AVG", indices)
        labels = _tensor_rows(train.S[:, 0], indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, delta = encode_context(encoder, adapter, prefix_ids, prefix_times)
        logits = head(delta).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        loss.backward()
        if attempted == 1 and not first_w2_gradient_is_finite_nonzero(adapter):
            raise PrototypeInvariantError("C0 first W2 gradient is not finite nonzero")
        successful_optimizer_update(loss, optimizer)
        successful += 1
        totals.append(float(loss.detach()))
    if attempted != expected_steps or successful != expected_steps or _state_dict_bytes(encoder) != e0_before:
        raise PrototypeInvariantError("C0 step count or E0 immutability mismatch")
    # The direct head is deliberately not retained in the returned condition.
    adapter.eval()
    return FrozenResidualCondition("C0_DIRECT", encoder, adapter, None, None,
        _training_summary(totals, (), (), (), successful, attempted, 0, c0=True))


def train_c0_direct(
    train: SyntheticFactorSplit, transform: Any, model_seed: int, base_encoder: J04Encoder,
) -> FrozenResidualCondition:
    return _train_c0(train, transform, model_seed, base_encoder,
                     pretraining_indices(model_seed), expected_steps=2000)


def extract_feature_blocks(
    condition: FrozenResidualCondition, split: SyntheticFactorSplit, transform: Any,
) -> FeatureBlocks:
    before = _state_dict_bytes(condition.base_encoder)
    adapter_before = None if condition.adapter is None else _state_dict_bytes(condition.adapter)
    condition.base_encoder.eval()
    if condition.adapter is not None:
        condition.adapter.eval()
    with torch.no_grad():
        ids, times = _prefix_inputs(split, transform, "L1_AVG")
        _, sequence, pooled = condition.base_encoder(ids, times, causal=False)
        z0, delta = apply_tokenwise_before_pooling(sequence, pooled, ids.ne(0), condition.adapter)
    if _state_dict_bytes(condition.base_encoder) != before or (
        condition.adapter is not None and _state_dict_bytes(condition.adapter) != adapter_before
    ):
        raise PrototypeInvariantError("feature extraction mutated representation")
    return FeatureBlocks(
        np.ascontiguousarray(z0.numpy().astype("<f8", copy=False)),
        np.ascontiguousarray(delta.numpy().astype("<f8", copy=False)),
    )


def _integer_seed(value: object, *, production: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed roots must be Python int and not bool")
    lower = 1 << 31 if production else 0
    upper = (1 << 32) - 1 if production else (1 << 31) - 1
    if not lower <= value <= upper:
        raise ValueError("seed root outside permitted half-domain")
    return value


def _rng(path: Sequence[int]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(path))))


def canonical_array_bytes(name: str, value: np.ndarray | torch.Tensor, *, dtype: str | None = None) -> bytes:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().newbyteorder("<")
    elif array.dtype.byteorder == "=":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    logical = name.encode("utf-8")
    dtype_bytes = array.dtype.str.encode("ascii")
    if array.ndim > 255 or len(logical) >= 2**32 or len(dtype_bytes) >= 2**16:
        raise ValueError("array hash metadata outside encoding domain")
    encoded = bytearray(b"BP011-ARRAY-V1\0")
    encoded += len(logical).to_bytes(4, "little") + logical
    encoded += len(dtype_bytes).to_bytes(2, "little") + dtype_bytes
    encoded += array.ndim.to_bytes(1, "little")
    for dimension in array.shape:
        encoded += int(dimension).to_bytes(8, "little", signed=False)
    encoded += array.tobytes(order="C")
    return bytes(encoded)


def array_sha256(name: str, value: np.ndarray | torch.Tensor, *, dtype: str | None = None) -> str:
    return hashlib.sha256(canonical_array_bytes(name, value, dtype=dtype)).hexdigest()


def canonical_state_sha256(module: nn.Module) -> str:
    payload = bytearray(b"BP011-STATE-V1\0")
    for name, tensor in sorted(module.state_dict().items()):
        payload += canonical_array_bytes(name, tensor)
    return hashlib.sha256(bytes(payload)).hexdigest()


def _as_float64_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    if result.ndim != 2 or result.shape[1] != 16 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [n,16] float64")
    return result


def fit_standardization(probe: np.ndarray) -> tuple[np.ndarray, Standardization]:
    values = _as_float64_matrix(probe, name="PROBE feature block")
    constant = np.array([np.equal(values[:, j], values[0, j]).all() for j in range(16)], dtype=np.bool_)
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale < 1e-8, 1.0, scale).astype("<f8", copy=False)
    transformed = np.empty_like(values)
    transformed[:, ~constant] = (values[:, ~constant] - mean[~constant]) / scale[~constant]
    transformed[:, constant] = np.float64(0.0)
    if np.signbit(transformed[:, constant]).any():
        raise PrototypeInvariantError("constant canonicalization produced negative zero")
    return np.ascontiguousarray(transformed), Standardization(
        np.ascontiguousarray(mean.astype("<f8")), np.ascontiguousarray(scale), np.ascontiguousarray(constant),
    )


def apply_standardization(values: np.ndarray, state: Standardization) -> np.ndarray:
    matrix = _as_float64_matrix(values, name="evaluation feature block")
    result = np.empty_like(matrix)
    result[:, ~state.constant] = (matrix[:, ~state.constant] - state.mean[~state.constant]) / state.scale[~state.constant]
    result[:, state.constant] = np.float64(0.0)
    if np.signbit(result[:, state.constant]).any() or not np.isfinite(result).all():
        raise PrototypeInvariantError("invalid frozen standardization")
    return np.ascontiguousarray(result)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _readout_terms(
    parameters: np.ndarray, X: np.ndarray, y: np.ndarray, *,
    base_logits: np.ndarray | None, intercept: bool, ridge: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if intercept:
        logits = parameters[0] + X @ parameters[1:]
        regularized = parameters[1:]
        design = np.column_stack((np.ones(X.shape[0], dtype=np.float64), X))
        ridge_diag = np.concatenate(([0.0], np.full(X.shape[1], ridge)))
    else:
        logits = X @ parameters
        if base_logits is not None:
            logits = logits + base_logits
        regularized = parameters
        design = X
        ridge_diag = np.full(X.shape[1], ridge)
    objective = float(np.mean(np.logaddexp(0.0, logits) - y * logits)
                      + 0.5 * ridge * np.dot(regularized, regularized))
    probabilities = _sigmoid(logits)
    gradient = design.T @ (probabilities - y) / y.size + ridge_diag * parameters
    weights = probabilities * (1.0 - probabilities)
    hessian = (design.T * weights) @ design / y.size
    hessian.flat[::hessian.shape[0] + 1] += ridge_diag
    hessian = (hessian + hessian.T) / 2.0
    return objective, gradient, hessian, np.ascontiguousarray(logits)


def fit_deterministic_logistic(
    X: np.ndarray, y: np.ndarray, *, base_logits: np.ndarray | None = None,
    intercept: bool = True, ridge: float = READOUT_RIDGE, max_iterations: int = 1000,
) -> ReadoutFit:
    features = _as_float64_matrix(X, name="readout design")
    labels = np.ascontiguousarray(np.asarray(y, dtype="<f8"))
    if labels.shape != (features.shape[0],) or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("binary labels must match readout rows")
    if not np.any(labels == 0.0) or not np.any(labels == 1.0):
        raise ValueError("both classes are required")
    if base_logits is not None:
        frozen = np.ascontiguousarray(np.asarray(base_logits, dtype="<f8"))
        if frozen.shape != labels.shape or not np.isfinite(frozen).all():
            raise ValueError("base logits must be finite and row-matched")
        if intercept:
            raise ValueError("additive frozen-offset readout has no residual intercept")
    else:
        frozen = None
    if ridge != READOUT_RIDGE or max_iterations != 1000:
        raise ValueError("readout ridge/iteration cap are frozen")
    size = 17 if intercept else 16
    parameters = np.zeros(size, dtype=np.float64)
    for iteration in range(max_iterations + 1):
        objective, gradient, hessian, _ = _readout_terms(
            parameters, features, labels, base_logits=frozen, intercept=intercept, ridge=ridge,
        )
        if not (math.isfinite(objective) and np.isfinite(gradient).all() and np.isfinite(hessian).all()):
            raise PrototypeInvariantError("READOUT_INVALID")
        if float(np.linalg.norm(gradient, ord=np.inf)) <= 1e-10:
            coefficients = parameters[1:] if intercept else parameters
            readout_intercept = float(parameters[0]) if intercept else 0.0
            return ReadoutFit(np.ascontiguousarray(coefficients), readout_intercept, iteration, True)
        if iteration == max_iterations:
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise PrototypeInvariantError("READOUT_INVALID") from error
        accepted = False
        for k in range(41):
            candidate = parameters - math.ldexp(1.0, -k) * step
            candidate_objective, _, _, _ = _readout_terms(
                candidate, features, labels, base_logits=frozen, intercept=intercept, ridge=ridge,
            )
            if np.isfinite(candidate).all() and math.isfinite(candidate_objective) and candidate_objective < objective:
                parameters = candidate
                accepted = True
                break
        if not accepted:
            raise PrototypeInvariantError("READOUT_INVALID")
    raise PrototypeInvariantError("READOUT_INVALID")


def readout_logits(fit: ReadoutFit, X: np.ndarray, *, base_logits: np.ndarray | None = None) -> np.ndarray:
    features = _as_float64_matrix(X, name="readout evaluation")
    logits = fit.intercept + features @ fit.coefficients
    if base_logits is not None:
        frozen = np.ascontiguousarray(np.asarray(base_logits, dtype="<f8"))
        if fit.intercept != 0.0 or frozen.shape != (features.shape[0],):
            raise PrototypeInvariantError("frozen-offset readout invariant")
        logits = logits + frozen
    if not np.isfinite(logits).all():
        raise PrototypeInvariantError("READOUT_INVALID")
    return np.ascontiguousarray(logits)


def fit_additive_readout(
    residual_probe: np.ndarray, labels: np.ndarray, base_probe_logits: np.ndarray,
    standardization: Standardization | None = None,
) -> tuple[ReadoutFit, Standardization, np.ndarray]:
    if standardization is None:
        standardized, standardization = fit_standardization(residual_probe)
    else:
        standardized = apply_standardization(residual_probe, standardization)
    if np.equal(standardized, np.float64(0.0)).all() and not np.signbit(standardized).any():
        fit = ReadoutFit(np.zeros(16, dtype=np.float64), 0.0, 0, True, True)
        return fit, standardization, np.ascontiguousarray(np.asarray(base_probe_logits, dtype="<f8")).copy()
    fit = fit_deterministic_logistic(standardized, labels, base_logits=base_probe_logits, intercept=False)
    return fit, standardization, readout_logits(fit, standardized, base_logits=base_probe_logits)


def binary_row_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(logits, dtype="<f8"))
    truth = np.ascontiguousarray(np.asarray(labels, dtype="<f8"))
    if values.shape != truth.shape or values.ndim != 1:
        raise ValueError("logits and labels must be matching vectors")
    result = np.logaddexp(0.0, values) - truth * values
    if not np.isfinite(result).all():
        raise PrototypeInvariantError("READOUT_INVALID")
    return np.ascontiguousarray(result)


def assay_summary(logits: np.ndarray, labels: np.ndarray, logical_name: str) -> dict[str, object]:
    nll = binary_row_nll(logits, labels)
    predictions = (np.asarray(logits) >= 0.0).astype(np.uint8)
    return {
        "nll_mean": float(nll.mean()),
        "balanced_accuracy": balanced_accuracy(predictions, np.asarray(labels, dtype=np.uint8)),
        "row_nll_hash": array_sha256(logical_name + ".nll", nll, dtype="<f8"),
        "logit_hash": array_sha256(logical_name + ".logits", np.asarray(logits), dtype="<f8"),
    }


def nuisance_intervention(split: SyntheticFactorSplit, root: int) -> SyntheticFactorSplit:
    permutation = _rng((root, 7501)).permutation(split.S.shape[0])
    if np.array_equal(permutation, np.arange(split.S.shape[0])):
        raise PrototypeInvariantError("SEED_IDENTITY_PERMUTATION")
    nuisance = split.N[permutation].copy()
    types = split.prefix_type_ids.copy()
    intervals = split.prefix_intervals.copy()
    types[:, 4] = np.where(nuisance[:, 0] == 0, NUIS_COMP_0, NUIS_COMP_1)
    types[:, 5] = np.where(nuisance[:, 1] == 0, NUIS_ORDER_0, NUIS_ORDER_1)
    types[:, 6] = TIME_NUIS
    intervals[:, 6] = np.where(nuisance[:, 2] == 0, 1.0, 4.0)
    result = SyntheticFactorSplit(
        types, intervals, split.target_type_ids.copy(), split.target_intervals.copy(),
        split.S.copy(), split.X.copy(), nuisance, split.L_after.copy(),
    )
    preserved = (
        np.array_equal(result.S, split.S), np.array_equal(result.X, split.X),
        np.array_equal(result.target_type_ids, split.target_type_ids),
        np.array_equal(result.target_intervals, split.target_intervals),
        np.array_equal(result.L_after, split.L_after),
        np.array_equal(result.prefix_type_ids[:, :4], split.prefix_type_ids[:, :4]),
        np.array_equal(result.prefix_intervals[:, :6], split.prefix_intervals[:, :6]),
    )
    if not all(preserved):
        raise PrototypeInvariantError("nuisance intervention changed protected values")
    return result


def correspondence_permutation(root: int, n: int, path_code: int) -> np.ndarray:
    if path_code not in (7401, 7402, 7403):
        raise ValueError("correspondence path code must identify PROBE/original/intervention")
    permutation = _rng((root, path_code)).permutation(n)
    if np.array_equal(permutation, np.arange(n)):
        raise PrototypeInvariantError("SEED_IDENTITY_PERMUTATION")
    return permutation


def correspondence_null(delta: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    values = _as_float64_matrix(delta, name="correspondence residual")
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (values.shape[0],) or not np.array_equal(np.sort(order), np.arange(values.shape[0])):
        raise ValueError("whole-row permutation required")
    result = np.ascontiguousarray(values[order])
    if not np.array_equal(np.sort(result, axis=0), np.sort(values, axis=0)):
        raise PrototypeInvariantError("correspondence null changed marginals")
    # Whole-row indexing preserves cross-coordinate pairs by construction.  The
    # reduction order changes, so covariance equality is numerical rather than
    # bytewise even though the row multiset is exact.
    if not np.allclose(np.cov(result, rowvar=False), np.cov(values, rowvar=False), rtol=1e-12, atol=1e-15):
        raise PrototypeInvariantError("correspondence null changed covariance")
    return result


CONTRAST_NAMES = ("d_C0", "d_R0", "d_SHUF", "d_CAP", "r_SHUF", "r_CAP")
VIEW_NAMES = ("original", "intervention")


def expected_contrast_keys() -> list[tuple[int, str, str]]:
    return [(model, view, name) for model in range(3) for view in VIEW_NAMES for name in CONTRAST_NAMES]


def bootstrap_simultaneous_lcbs(
    row_contrasts: Mapping[tuple[int, str, str], np.ndarray], bootstrap_root: int,
    *, replicates: int = 10000, supplied_indices: np.ndarray | None = None,
) -> tuple[float, list[dict[str, object]], np.ndarray]:
    keys = expected_contrast_keys()
    if list(row_contrasts.keys()) != keys:
        raise PrototypeInvariantError("bootstrap family must contain exactly ordered 36 contrasts")
    arrays = [np.ascontiguousarray(np.asarray(row_contrasts[key], dtype="<f8")) for key in keys]
    n = arrays[0].size
    if n == 0 or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays):
        raise PrototypeInvariantError("invalid bootstrap contrast rows")
    observed = np.asarray([a.mean() for a in arrays], dtype=np.float64)
    if supplied_indices is None:
        if replicates != 10000 or n != 2048:
            raise ValueError("production bootstrap is exactly 10000x2048")
        indices = _rng((bootstrap_root, 7601)).integers(
            0, n, size=(replicates, n), dtype=np.int64, endpoint=False,
        )
    else:
        indices = np.ascontiguousarray(np.asarray(supplied_indices, dtype=np.int64))
        if indices.shape != (replicates, n) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("tiny bootstrap indices shape/range mismatch")
    centered_max = np.empty(replicates, dtype=np.float64)
    stacked = np.stack(arrays)
    for start in range(0, replicates, 64):
        rows = indices[start:start + 64]
        bootstrap_means = stacked[:, rows].mean(axis=2).T
        centered_max[start:start + rows.shape[0]] = np.max(bootstrap_means - observed, axis=1)
    critical = float(np.quantile(centered_max, 0.95, method="linear"))
    contrasts = [
        {"name": name, "view": view, "model_index": model,
         "observed": float(observed[index]), "lcb95": float(observed[index] - critical)}
        for index, (model, view, name) in enumerate(keys)
    ]
    return critical, contrasts, indices


def evaluate_terminal_outcome(
    contrasts: Sequence[Mapping[str, object]], *, valid: bool,
    residual_nonzero_nonconstant: bool,
) -> tuple[str, bool, dict[str, bool]]:
    if not valid:
        return "INVALID", False, {}
    if [(c["model_index"], c["view"], c["name"]) for c in contrasts] != expected_contrast_keys():
        raise PrototypeInvariantError("gate family is not exact and ordered")
    passing = {(int(c["model_index"]), str(c["view"]), str(c["name"])): float(c["lcb95"]) > 0
               for c in contrasts}
    eligible = all(passing[(m, v, "d_C0")] for m in range(3) for v in VIEW_NAMES)
    gates = {
        "d_r0": all(passing[(m, v, "d_R0")] for m in range(3) for v in VIEW_NAMES),
        "d_shuf": all(passing[(m, v, "d_SHUF")] for m in range(3) for v in VIEW_NAMES),
        "d_cap": all(passing[(m, v, "d_CAP")] for m in range(3) for v in VIEW_NAMES),
        "r_shuf": all(passing[(m, v, "r_SHUF")] for m in range(3) for v in VIEW_NAMES),
        "r_cap": all(passing[(m, v, "r_CAP")] for m in range(3) for v in VIEW_NAMES),
        "residual_nonzero_nonconstant": bool(residual_nonzero_nonconstant),
    }
    if not eligible:
        return "INELIGIBLE", False, gates
    if not gates["d_r0"]:
        return "SCIENTIFIC_RED", True, gates
    if not all(gates.values()):
        return "SCIENTIFIC_RED", True, gates
    return "GREEN", True, gates


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_canonical_json(raw: bytes) -> object:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrototypeInvariantError("INPUT_NONCANONICAL") from error
    if canonical_json_bytes(value) != raw:
        raise PrototypeInvariantError("INPUT_NONCANONICAL")
    return value


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, object], keys: Sequence[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys) or list(value) != list(keys):
        # Input key order is canonical sorted order after parsing; compare sets only here.
        if not isinstance(value, dict) or set(value) != set(keys):
            raise PrototypeInvariantError(f"{label} exact-key schema mismatch")


def seed_manifest_from_dict(value: object) -> SeedManifest:
    if not isinstance(value, dict) or set(value) != {
        "schema", "generator_seed", "model_seeds", "train_target_shuffle_roots",
        "capacity_probe_roots", "capacity_cal_original_roots", "capacity_cal_intervention_roots",
        "nuisance_intervention_root", "bootstrap_root",
    }:
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if value["schema"] != "BP011-J04C-V3-R0RESID-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    tuple_fields = (
        "model_seeds", "train_target_shuffle_roots", "capacity_probe_roots",
        "capacity_cal_original_roots", "capacity_cal_intervention_roots",
    )
    if any(not isinstance(value[name], list) or len(value[name]) != 3 for name in tuple_fields):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    manifest = SeedManifest(
        value["schema"], value["generator_seed"], tuple(value["model_seeds"]),
        tuple(value["train_target_shuffle_roots"]), tuple(value["capacity_probe_roots"]),
        tuple(value["capacity_cal_original_roots"]), tuple(value["capacity_cal_intervention_roots"]),
        value["nuisance_intervention_root"], value["bootstrap_root"],
    )
    validate_seed_manifest(manifest)
    return manifest


def validate_seed_manifest(manifest: SeedManifest) -> None:
    if manifest.schema != "BP011-J04C-V3-R0RESID-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    values = [manifest.generator_seed, *manifest.model_seeds, *manifest.train_target_shuffle_roots,
              *manifest.capacity_probe_roots, *manifest.capacity_cal_original_roots,
              *manifest.capacity_cal_intervention_roots, manifest.nuisance_intervention_root,
              manifest.bootstrap_root]
    for value in values:
        _integer_seed(value, production=True)
    if len(values) != len(set(values)):
        raise PrototypeInvariantError("SEED_COLLISION")


def approved_envelope_from_dict(value: object) -> ApprovedSeedEnvelope:
    keys = {"schema", "manifest_sha256", "historical_inventory_sha256",
            "expected_generated_audit_sha256", "production_path_count"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != "BP011-J04C-V3-R0RESID-SEED-APPROVAL-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    for name in ("manifest_sha256", "historical_inventory_sha256", "expected_generated_audit_sha256"):
        if not _is_digest(value[name]):
            raise PrototypeInvariantError("INPUT_SCHEMA")
    if value["production_path_count"] != 123 or isinstance(value["production_path_count"], bool):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return ApprovedSeedEnvelope(**value)


def build_provenance_from_dict(value: object) -> BuildProvenance:
    keys = {"schema", "target_commit", "implementation_commit", "clean_tree", "source_digests",
            "implementation_digests", "python_version", "numpy_version", "torch_version",
            "platform_machine", "platform_system", "blas_fingerprint"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != "BP011-J04C-V3-R0RESID-BUILD-PROVENANCE-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if value["target_commit"] != TARGET_COMMIT or value["clean_tree"] is not True:
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    if value["source_digests"] != SOURCE_DIGESTS:
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    implementation_commit = value["implementation_commit"]
    implementation_digests = value["implementation_digests"]
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40 \
       or any(character not in "0123456789abcdef" for character in implementation_commit):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    if not isinstance(implementation_digests, dict) or set(implementation_digests) != set(IMPLEMENTATION_PATHS) \
       or any(not _is_digest(digest) for digest in implementation_digests.values()):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    if any(not isinstance(value[name], str) or not value[name] for name in (
        "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system", "blas_fingerprint"
    )):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    return BuildProvenance(**value)


def _path_record(purpose: str, path: Sequence[int]) -> dict[str, object]:
    values = list(path)
    if not purpose.isascii() or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 2**32 for x in values):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return {"purpose": purpose, "path": values}


def generated_seed_audit(manifest: SeedManifest) -> list[dict[str, object]]:
    validate_seed_manifest(manifest)
    records: list[dict[str, object]] = []
    records.extend(_path_record("GENERATOR_SPLIT", (manifest.generator_seed, split)) for split in (1, 3, 6))
    records.append(_path_record("TRAIN_NUISANCE", (manifest.generator_seed, 1, 7101)))
    for model_index, model_seed in enumerate(manifest.model_seeds):
        records.extend((
            _path_record("E0_INIT", (model_seed, 1)),
            _path_record("PREDICTOR_INIT", (model_seed, 2)),
            _path_record("C0_HEAD_INIT", (model_seed, 40)),
        ))
        records.extend(_path_record("TRAIN_SCHEDULE", (model_seed, epoch, 6101)) for epoch in range(32))
        records.append(_path_record("TARGET_SHUFFLE", (manifest.train_target_shuffle_roots[model_index], 7301)))
        records.extend((
            _path_record("CORRESPONDENCE_PROBE", (manifest.capacity_probe_roots[model_index], 7401)),
            _path_record("CORRESPONDENCE_CAL_ORIGINAL", (manifest.capacity_cal_original_roots[model_index], 7402)),
            _path_record("CORRESPONDENCE_CAL_INTERVENTION", (manifest.capacity_cal_intervention_roots[model_index], 7403)),
        ))
    records.extend((
        _path_record("NUISANCE_INTERVENTION", (manifest.nuisance_intervention_root, 7501)),
        _path_record("BOOTSTRAP", (manifest.bootstrap_root, 7601)),
    ))
    records.sort(key=lambda record: (record["purpose"], record["path"]))
    unique = {(record["purpose"], tuple(record["path"])) for record in records}
    if len(records) != PRODUCTION_PATH_COUNT or len(unique) != PRODUCTION_PATH_COUNT:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return records


def validate_historical_inventory(value: object) -> tuple[set[int], set[tuple[int, ...]]]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "through_rows", "roots", "records", "source_artifact_digests"
    } or value.get("schema") != "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    roots = value["roots"]
    records = value["records"]
    if (not isinstance(roots, list) or roots != sorted(set(roots)) or
        any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in roots) or
        not isinstance(records, list)):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    paths: list[tuple[int, ...]] = []
    previous: tuple[str, tuple[int, ...]] | None = None
    for record in records:
        if not isinstance(record, dict) or set(record) != {"purpose", "path"}:
            raise PrototypeInvariantError("INPUT_SCHEMA")
        purpose, raw_path = record["purpose"], record["path"]
        if not isinstance(purpose, str) or not purpose.isascii() or not isinstance(raw_path, list) or not raw_path:
            raise PrototypeInvariantError("INPUT_SCHEMA")
        path = tuple(raw_path)
        if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in path) or path[0] not in roots:
            raise PrototypeInvariantError("INPUT_SCHEMA")
        current = (purpose, path)
        if previous is not None and current <= previous:
            raise PrototypeInvariantError("INPUT_SCHEMA")
        previous = current
        paths.append(path)
    return set(roots), set(paths)


def validate_seed_audit(
    manifest: SeedManifest, manifest_raw: bytes, envelope: ApprovedSeedEnvelope,
    inventory: object, inventory_raw: bytes,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if sha256_hex(manifest_raw) != envelope.manifest_sha256 or sha256_hex(inventory_raw) != envelope.historical_inventory_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    records = generated_seed_audit(manifest)
    audit_digest = sha256_hex(canonical_json_bytes(records))
    if audit_digest != envelope.expected_generated_audit_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    historical_roots, historical_paths = validate_historical_inventory(inventory)
    production_paths = {tuple(record["path"]) for record in records}
    production_roots = {path[0] for path in production_paths}
    if production_paths & historical_paths or production_roots & historical_roots:
        raise PrototypeInvariantError("SEED_COLLISION")
    return records, {
        "manifest_sha256": envelope.manifest_sha256,
        "historical_inventory_sha256": envelope.historical_inventory_sha256,
        "generated_audit_sha256": audit_digest,
        "production_path_count": 123,
        "historical_path_count": len(historical_paths),
        "path_intersection_count": 0, "root_intersection_count": 0,
    }


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


_FORBIDDEN_KEY_FRAGMENTS = ("seed", "permutation", "raw", "tensor", "checkpoint", "absolute_path")


def validate_recursive_output(value: object, *, failure: bool = False, key: str = "root") -> None:
    if isinstance(value, (np.ndarray, torch.Tensor, nn.Module)):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if isinstance(value, float) and not math.isfinite(value):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if isinstance(value, str) and value.startswith("/"):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
            lower = nested_key.lower()
            # These are the only contract-required aggregate metadata keys containing "seed".
            allowed_seed_keys = {"seed_audit", "all_seed_beats"}
            if any(fragment in lower for fragment in _FORBIDDEN_KEY_FRAGMENTS) and nested_key not in allowed_seed_keys:
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
            validate_recursive_output(nested_value, failure=failure, key=nested_key)
    elif isinstance(value, (list, tuple)):
        if len(value) > 36:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
        for item in value:
            validate_recursive_output(item, failure=failure, key=key)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


FAILURE_CODES = frozenset({
    "INPUT_NONCANONICAL", "INPUT_SCHEMA", "PROVENANCE_DIGEST", "PROVENANCE_CONTENT",
    "SEED_AUDIT_DIGEST", "SEED_COLLISION", "SEED_IDENTITY_PERMUTATION", "GENERATION_INVARIANT",
    "TRAINING_NONFINITE_PRE", "TRAINING_STEP_EXCEPTION", "TRAINING_NONFINITE_POST", "TRAINING_INVARIANT",
    "READOUT_INVALID", "BOOTSTRAP_INVALID", "SERIALIZATION_INVALID",
})
FAILURE_PHASES = frozenset({"INPUT", "PROVENANCE", "SEED_AUDIT", "GENERATION", "TRAINING", "READOUT", "BOOTSTRAP", "SERIALIZATION"})


def failure_artifact(phase: str, error_code: str) -> dict[str, object]:
    if phase not in FAILURE_PHASES or error_code not in FAILURE_CODES:
        phase, error_code = "SERIALIZATION", "SERIALIZATION_INVALID"
    result = {
        "schema": "BP011-J04C-V3-R0RESID-INVALID-V1", "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256, "terminal_outcome": "INVALID",
        "phase": phase, "error_code": error_code,
    }
    validate_failure_schema(result)
    return result


def validate_failure_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"
    } or value.get("schema") != "BP011-J04C-V3-R0RESID-INVALID-V1" or value.get("namespace") != NAMESPACE \
       or value.get("contract_sha256") != CONTRACT_SHA256 or value.get("terminal_outcome") != "INVALID" \
       or value.get("phase") not in FAILURE_PHASES or value.get("error_code") not in FAILURE_CODES:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    validate_recursive_output(value, failure=True)


SUCCESS_ROOT_KEYS = {
    "schema", "namespace", "contract_sha256", "claim_ceiling", "provenance", "seed_audit", "fixed",
    "checks", "models", "bootstrap", "diagnostics", "recovery", "descriptive_targets", "valid", "eligible",
    "scientific_gates", "terminal_outcome",
}


def validate_success_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != SUCCESS_ROOT_KEYS:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if value.get("schema") != "BP011-J04C-V3-R0RESID-RESULT-V1" or value.get("namespace") != NAMESPACE \
       or value.get("contract_sha256") != CONTRACT_SHA256 or value.get("valid") is not True \
       or value.get("terminal_outcome") not in {"GREEN", "SCIENTIFIC_RED", "INELIGIBLE"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(value.get("models"), list) or len(value["models"]) != 3:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("replicates") != 10000 or bootstrap.get("family_size") != 36 \
       or not isinstance(bootstrap.get("contrasts"), list) or len(bootstrap["contrasts"]) != 36:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    validate_recursive_output(value)


def _hash_named_arrays(values: Sequence[tuple[str, np.ndarray]]) -> str:
    payload = bytearray(b"BP011-STATE-V1\0")
    for name, value in values:
        payload += canonical_array_bytes(name, value)
    return hashlib.sha256(bytes(payload)).hexdigest()


def _readout_summary(fit: ReadoutFit, state: Standardization, logical: str) -> dict[str, object]:
    return {
        "preprocess_hash": _hash_named_arrays(((logical + ".mean", state.mean), (logical + ".scale", state.scale))),
        "coefficient_hash": _hash_named_arrays(((logical + ".coefficients", fit.coefficients),
                                                  (logical + ".intercept", np.asarray([fit.intercept], dtype="<f8")))),
        "constant_coordinate_mask_hash": array_sha256(logical + ".constant", state.constant),
        "iterations": fit.iterations, "converged": fit.converged,
        "zero_short_circuit": fit.zero_short_circuit,
    }


def _fit_base_bundle(
    probe: np.ndarray, probe_labels: np.ndarray, views: Mapping[str, np.ndarray], logical: str,
) -> tuple[dict[str, object], dict[str, np.ndarray], Standardization]:
    standardized, state = fit_standardization(probe)
    fit = fit_deterministic_logistic(standardized, probe_labels)
    logits = {name: readout_logits(fit, apply_standardization(values, state)) for name, values in views.items()}
    logits["probe"] = readout_logits(fit, standardized)
    return _readout_summary(fit, state, logical), logits, state


def _fit_additive_bundle(
    probe: np.ndarray, probe_labels: np.ndarray, base_probe_logits: np.ndarray,
    views: Mapping[str, np.ndarray], base_view_logits: Mapping[str, np.ndarray], logical: str,
) -> tuple[dict[str, object], dict[str, np.ndarray], Standardization]:
    fit, state, probe_logits = fit_additive_readout(probe, probe_labels, base_probe_logits)
    logits = {"probe": probe_logits}
    for name, values in views.items():
        standardized = apply_standardization(values, state)
        if fit.zero_short_circuit:
            logits[name] = np.ascontiguousarray(base_view_logits[name]).copy()
        else:
            logits[name] = readout_logits(fit, standardized, base_logits=base_view_logits[name])
    return _readout_summary(fit, state, logical), logits, state


def _fit_residual_only_bundle(
    probe: np.ndarray, probe_labels: np.ndarray, views: Mapping[str, np.ndarray], logical: str,
) -> tuple[dict[str, object], dict[str, np.ndarray], Standardization]:
    standardized, state = fit_standardization(probe)
    if np.equal(standardized, 0.0).all():
        # Residual-only constant blocks still fit an intercept-only model through the base solver.
        fit = fit_deterministic_logistic(standardized, probe_labels)
    else:
        fit = fit_deterministic_logistic(standardized, probe_labels)
    logits = {name: readout_logits(fit, apply_standardization(values, state)) for name, values in views.items()}
    logits["probe"] = readout_logits(fit, standardized)
    return _readout_summary(fit, state, logical), logits, state


def target_learning_row_loss(
    arm: FrozenResidualCondition, target_teacher: FrozenResidualCondition,
    split: SyntheticFactorSplit, transform: Any,
) -> np.ndarray:
    if arm.predictor is None or target_teacher.teacher_adapter is None:
        raise ValueError("diagnostic requires predictor arm and candidate EMA target")
    arm.predictor.eval()
    with torch.no_grad():
        ids, times = _prefix_inputs(split, transform, "L1_AVG")
        target_ids = torch.as_tensor(split.target_type_ids, dtype=torch.long)
        target_times = torch.as_tensor(transform.transform(split.target_intervals), dtype=torch.float32)
        z0, delta = encode_context(arm.base_encoder, arm.adapter if arm.name.startswith("RESID_") else None, ids, times)
        target, valid, _, _ = composite_l1_target(
            target_teacher.base_encoder, target_teacher.teacher_adapter, target_ids, target_times,
        )
        prediction = arm.predictor(z0 + delta, "L1_AVG")
        unit_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
        counts = valid.sum(dim=1)
        if bool((counts == 0).any()):
            raise PrototypeInvariantError("diagnostic zero target count")
        row = (unit_loss * valid.to(unit_loss.dtype)).sum(dim=1) / counts.to(unit_loss.dtype)
    result = np.ascontiguousarray(row.numpy().astype("<f8", copy=False))
    if not np.isfinite(result).all():
        raise PrototypeInvariantError("nonfinite target diagnostic")
    return result


def paired_percentile_interval(rows: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    values = np.ascontiguousarray(np.asarray(rows, dtype="<f8"))
    bootstrap_indices = np.asarray(indices, dtype=np.int64)
    means = np.empty(bootstrap_indices.shape[0], dtype=np.float64)
    for start in range(0, bootstrap_indices.shape[0], 128):
        selected = bootstrap_indices[start:start + 128]
        means[start:start + selected.shape[0]] = values[selected].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025, method="linear")), \
        float(np.quantile(means, 0.975, method="linear"))


def classify_diagnostics(
    terminal: str, eligible: bool, gates: Mapping[str, bool],
    train_summaries: Sequence[Mapping[str, object]], all_seed_beats: Mapping[str, bool],
) -> str:
    if not eligible:
        return "NOT_INTERPRETED"
    if terminal == "GREEN":
        return "SUPPORTED_RESIDUAL_MECHANISM"
    original_pass = all(bool(gates[name + "_original"]) for name in ("d_r0", "d_shuf", "d_cap", "r_shuf", "r_cap")) \
        if "d_r0_original" in gates else False
    intervention_pass = all(bool(gates[name + "_intervention"]) for name in ("d_r0", "d_shuf", "d_cap", "r_shuf", "r_cap")) \
        if "d_r0_intervention" in gates else True
    if original_pass and not intervention_pass:
        return "NUISANCE_FAILURE"
    pred_only = [item for item in train_summaries if item["arm"] == "PRED_ONLY"]
    if any(float(item["last_100_mean_total"]) >= float(item["first_100_mean_total"]) for item in pred_only):
        return "PREDICTOR_FIT_LIMITATION"
    if not all_seed_beats["g_PS"]:
        return "R0_TARGET_ACCESS_GENERALIZATION_LIMITATION"
    composition_failed = not all(bool(gates[name]) for name in ("d_r0", "d_shuf", "d_cap", "r_shuf", "r_cap"))
    if all_seed_beats["g_CP"] and all_seed_beats["g_CS"] and composition_failed:
        return "TARGET_TO_REPRESENTATION_FAILURE"
    if (not all_seed_beats["g_CP"] or not all_seed_beats["g_CS"]) and not bool(gates["d_r0"]):
        return "RESIDUAL_OPTIMIZATION_FAILURE"
    return "UNRESOLVED"


def _training_diagnostic_records(model_index: int, arms: Mapping[str, FrozenResidualCondition]) -> list[dict[str, object]]:
    order = ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT")
    return [{
        "model_index": model_index, "arm": name,
        "first_100_mean_total": arms[name].training["first_100_mean_total"],
        "last_100_mean_total": arms[name].training["last_100_mean_total"],
    } for name in order]


def _residual_nonzero_nonconstant(*blocks: np.ndarray) -> bool:
    return all(not np.equal(block, 0.0).all() and any(not np.equal(block[:, j], block[0, j]).all() for j in range(16))
               for block in blocks)


def _state_hashes(arms: Mapping[str, FrozenResidualCondition]) -> dict[str, str]:
    return {
        "e0": canonical_state_sha256(arms["RESID_CANDIDATE"].base_encoder),
        "candidate_adapter": canonical_state_sha256(arms["RESID_CANDIDATE"].adapter),
        "candidate_predictor": canonical_state_sha256(arms["RESID_CANDIDATE"].predictor),
        "candidate_ema_adapter": canonical_state_sha256(arms["RESID_CANDIDATE"].teacher_adapter),
        "shuffled_adapter": canonical_state_sha256(arms["RESID_SHUFFLED"].adapter),
        "shuffled_predictor": canonical_state_sha256(arms["RESID_SHUFFLED"].predictor),
        "shuffled_ema_adapter": canonical_state_sha256(arms["RESID_SHUFFLED"].teacher_adapter),
        "c0_adapter": canonical_state_sha256(arms["C0_DIRECT"].adapter),
        "pred_only_predictor": canonical_state_sha256(arms["PRED_ONLY"].predictor),
        "pred_only_shuffled_predictor": canonical_state_sha256(arms["PRED_ONLY_SHUFFLED"].predictor),
    }


def _recovery_records(
    row_contrasts: Mapping[tuple[int, str, str], np.ndarray], indices: np.ndarray,
) -> list[dict[str, object]]:
    records = []
    for model in range(3):
        for view in VIEW_NAMES:
            numerator = row_contrasts[(model, view, "d_R0")]
            denominator = row_contrasts[(model, view, "d_C0")]
            d_r0 = numerator.mean()
            d_c0 = denominator.mean()
            observed = float(d_r0 / d_c0)
            ratios = np.empty(indices.shape[0], dtype=np.float64)
            bounded = True
            for start in range(0, indices.shape[0], 128):
                selected = indices[start:start + 128]
                num = numerator[selected].mean(axis=1)
                den = denominator[selected].mean(axis=1)
                if not np.isfinite(den).all() or np.any(den <= 0):
                    bounded = False
                    break
                ratios[start:start + selected.shape[0]] = num / den
            if bounded:
                records.append({"model_index": model, "view": view, "estimate": observed,
                                "interval_status": "FINITE",
                                "lower95": float(np.quantile(ratios, 0.025, method="linear")),
                                "upper95": float(np.quantile(ratios, 0.975, method="linear"))})
            else:
                records.append({"model_index": model, "view": view, "estimate": observed,
                                "interval_status": "UNBOUNDED_DENOMINATOR", "lower95": None, "upper95": None})
    return records


def _fixed_contract() -> dict[str, object]:
    return {
        "generator_count": 1, "model_count": 3, "train_n": 8192, "probe_n": 2048, "cal_n": 2048,
        "batch_size": 128, "updates": 2000, "ema_momentum": 0.996,
        "adapter": "W2_GELU_W1_PARAMETER_FREE_LN_TOKENWISE", "recipe": "L1_AVG",
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001,
                      "betas": [0.9, 0.999], "eps": 1e-8},
        "directional_weight": 5.0, "directional_floor": 0.010, "readout_ridge": 0.001,
        "bootstrap_replicates": 10000, "bootstrap_family_size": 36,
    }


def run_production_beta(
    manifest: SeedManifest, provenance: BuildProvenance, seed_audit: Mapping[str, object],
    *, build_provenance_sha256: str, approved_envelope_sha256: str,
) -> dict[str, object]:
    """Execute the frozen production contract after the CLI has validated all inputs.

    This function contains no seed selection or I/O.  Callers must not use it
    without the separately authorized sealed inputs; tests exercise only its
    constituent tiny-fixture operations.
    """
    from clinical_jepa.eval.j04c_falsifier import (
        TRAIN, CAL_OOD, PROBE_FIT, fit_stage0_time_transform, generate_factor_split,
    )
    from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance

    validate_seed_manifest(manifest)
    if not _is_digest(build_provenance_sha256) or not _is_digest(approved_envelope_sha256):
        raise PrototypeInvariantError("PROVENANCE_DIGEST")
    train = independent_train_nuisance(
        generate_factor_split(manifest.generator_seed, TRAIN, 8192), manifest.generator_seed,
    )
    probe = generate_factor_split(manifest.generator_seed, PROBE_FIT, 2048)
    cal = generate_factor_split(manifest.generator_seed, CAL_OOD, 2048)
    cal_intervention = nuisance_intervention(cal, manifest.nuisance_intervention_root)
    transform = fit_stage0_time_transform(train)

    models: list[dict[str, object]] = []
    row_contrasts: dict[tuple[int, str, str], np.ndarray] = {}
    diagnostic_train: list[dict[str, object]] = []
    target_loss_summaries: list[dict[str, object]] = []
    target_loss_rows: dict[tuple[int, str, str], np.ndarray] = {}
    descriptive_pending: list[tuple[str, int, str, dict[str, object], np.ndarray]] = []
    residual_ok = True

    for model_index, model_seed in enumerate(manifest.model_seeds):
        encoder = freeze_encoder(model_seed)
        e0_bytes = _state_dict_bytes(encoder)
        shuffle = target_tuple_permutation(manifest.train_target_shuffle_roots[model_index], 8192)
        shuffled_train = shuffled_target_split(train, shuffle)
        arms = {
            "PRED_ONLY": _train_l1_arm(
                "PRED_ONLY", train, transform, model_seed, encoder, pretraining_indices(model_seed),
                train_adapter=False, expected_steps=2000,
            ),
            "PRED_ONLY_SHUFFLED": _train_l1_arm(
                "PRED_ONLY_SHUFFLED", train, transform, model_seed, encoder, pretraining_indices(model_seed),
                train_adapter=False, shuffled_split=shuffled_train, expected_steps=2000,
            ),
            "RESID_CANDIDATE": _train_l1_arm(
                "RESID_CANDIDATE", train, transform, model_seed, encoder, pretraining_indices(model_seed),
                train_adapter=True, expected_steps=2000,
            ),
            "RESID_SHUFFLED": _train_l1_arm(
                "RESID_SHUFFLED", train, transform, model_seed, encoder, pretraining_indices(model_seed),
                train_adapter=True, shuffled_split=shuffled_train, expected_steps=2000,
            ),
            "C0_DIRECT": _train_c0(
                train, transform, model_seed, encoder, pretraining_indices(model_seed), expected_steps=2000,
            ),
        }
        if _state_dict_bytes(encoder) != e0_bytes:
            raise PrototypeInvariantError("TRAINING_INVARIANT")
        diagnostic_train.extend(_training_diagnostic_records(model_index, arms))

        feature: dict[str, dict[str, FeatureBlocks]] = {}
        for arm_name in ("RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT"):
            feature[arm_name] = {
                "probe": extract_feature_blocks(arms[arm_name], probe, transform),
                "original": extract_feature_blocks(arms[arm_name], cal, transform),
                "intervention": extract_feature_blocks(arms[arm_name], cal_intervention, transform),
            }
        candidate = feature["RESID_CANDIDATE"]
        for arm_name in ("RESID_SHUFFLED", "C0_DIRECT"):
            for view in ("probe", *VIEW_NAMES):
                if candidate[view].z0.tobytes() != feature[arm_name][view].z0.tobytes():
                    raise PrototypeInvariantError("baseline representation mismatch")
        residual_ok &= _residual_nonzero_nonconstant(
            candidate["probe"].delta_z, candidate["original"].delta_z,
            candidate["intervention"].delta_z,
        )
        null_blocks = {
            "probe": correspondence_null(candidate["probe"].delta_z,
                correspondence_permutation(manifest.capacity_probe_roots[model_index], 2048, 7401)),
            "original": correspondence_null(candidate["original"].delta_z,
                correspondence_permutation(manifest.capacity_cal_original_roots[model_index], 2048, 7402)),
            "intervention": correspondence_null(candidate["intervention"].delta_z,
                correspondence_permutation(manifest.capacity_cal_intervention_roots[model_index], 2048, 7403)),
        }
        probe_labels = probe.S[:, 0].astype(np.float64)
        cal_labels = cal.S[:, 0].astype(np.float64)
        z0_views = {view: candidate[view].z0 for view in VIEW_NAMES}
        readouts: dict[str, dict[str, object]] = {}
        all_logits: dict[str, dict[str, np.ndarray]] = {}
        summary, logits, _ = _fit_base_bundle(candidate["probe"].z0, probe_labels, z0_views,
                                               f"m{model_index}.R0_BASE")
        readouts["R0_BASE"], all_logits["R0_BASE"] = summary, logits
        for output_name, arm_name, blocks in (
            ("RESID_CANDIDATE_ADDITIVE", "RESID_CANDIDATE", candidate),
            ("RESID_SHUFFLED_ADDITIVE", "RESID_SHUFFLED", feature["RESID_SHUFFLED"]),
            ("RESID_CORRESPONDENCE_NULL_ADDITIVE", "RESID_CORRESPONDENCE_NULL", null_blocks),
            ("C0_DIRECT_ADDITIVE", "C0_DIRECT", feature["C0_DIRECT"]),
        ):
            probe_delta = blocks["probe"].delta_z if isinstance(blocks["probe"], FeatureBlocks) else blocks["probe"]
            view_delta = {view: (blocks[view].delta_z if isinstance(blocks[view], FeatureBlocks) else blocks[view])
                          for view in VIEW_NAMES}
            summary, logits, _ = _fit_additive_bundle(
                probe_delta, probe_labels, all_logits["R0_BASE"]["probe"], view_delta,
                all_logits["R0_BASE"], f"m{model_index}.{output_name}",
            )
            readouts[output_name], all_logits[output_name] = summary, logits
        for output_name, blocks in (
            ("RESID_CANDIDATE_RESIDUAL_ONLY", candidate),
            ("RESID_SHUFFLED_RESIDUAL_ONLY", feature["RESID_SHUFFLED"]),
            ("RESID_CORRESPONDENCE_NULL_RESIDUAL_ONLY", null_blocks),
        ):
            probe_delta = blocks["probe"].delta_z if isinstance(blocks["probe"], FeatureBlocks) else blocks["probe"]
            view_delta = {view: (blocks[view].delta_z if isinstance(blocks[view], FeatureBlocks) else blocks[view])
                          for view in VIEW_NAMES}
            summary, logits, _ = _fit_residual_only_bundle(
                probe_delta, probe_labels, view_delta, f"m{model_index}.{output_name}",
            )
            readouts[output_name], all_logits[output_name] = summary, logits

        readout_order = (
            "R0_BASE", "RESID_CANDIDATE_ADDITIVE", "RESID_CANDIDATE_RESIDUAL_ONLY",
            "RESID_SHUFFLED_ADDITIVE", "RESID_SHUFFLED_RESIDUAL_ONLY",
            "RESID_CORRESPONDENCE_NULL_ADDITIVE", "RESID_CORRESPONDENCE_NULL_RESIDUAL_ONLY",
            "C0_DIRECT_ADDITIVE",
        )
        views_output: dict[str, object] = {}
        nll: dict[str, dict[str, np.ndarray]] = {}
        for view in VIEW_NAMES:
            nll[view] = {name: binary_row_nll(all_logits[name][view], cal_labels) for name in readout_order}
            views_output[view] = {name: assay_summary(all_logits[name][view], cal_labels,
                                                      f"m{model_index}.{view}.{name}")
                                  for name in readout_order}
            rows_for_view = {
                "d_C0": nll[view]["R0_BASE"] - nll[view]["C0_DIRECT_ADDITIVE"],
                "d_R0": nll[view]["R0_BASE"] - nll[view]["RESID_CANDIDATE_ADDITIVE"],
                "d_SHUF": nll[view]["RESID_SHUFFLED_ADDITIVE"] - nll[view]["RESID_CANDIDATE_ADDITIVE"],
                "d_CAP": nll[view]["RESID_CORRESPONDENCE_NULL_ADDITIVE"] - nll[view]["RESID_CANDIDATE_ADDITIVE"],
                "r_SHUF": nll[view]["RESID_SHUFFLED_RESIDUAL_ONLY"] - nll[view]["RESID_CANDIDATE_RESIDUAL_ONLY"],
                "r_CAP": nll[view]["RESID_CORRESPONDENCE_NULL_RESIDUAL_ONLY"] - nll[view]["RESID_CANDIDATE_RESIDUAL_ONLY"],
            }
            for contrast_name in CONTRAST_NAMES:
                row_contrasts[(model_index, view, contrast_name)] = np.ascontiguousarray(rows_for_view[contrast_name])

        for view_name, split in (("original", cal), ("intervention", cal_intervention)):
            for arm_name in ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED"):
                rows = target_learning_row_loss(arms[arm_name], arms["RESID_CANDIDATE"], split, transform)
                target_loss_rows[(model_index, view_name, arm_name)] = rows
                target_loss_summaries.append({
                    "model_index": model_index, "view": view_name, "arm": arm_name,
                    "mean_l1_cosine_loss": float(rows.mean()),
                    "row_loss_hash": array_sha256(f"m{model_index}.{view_name}.{arm_name}.target_loss", rows),
                })

        for target_name, target_index in (("ORDER", 1), ("TIME", 2)):
            probe_target = probe.S[:, target_index].astype(np.float64)
            cal_target = cal.S[:, target_index].astype(np.float64)
            base_summary, base_target_logits, _ = _fit_base_bundle(
                candidate["probe"].z0, probe_target, z0_views, f"m{model_index}.{target_name}.R0",
            )
            offset_summary, candidate_target_logits, _ = _fit_additive_bundle(
                candidate["probe"].delta_z, probe_target, base_target_logits["probe"],
                {view: candidate[view].delta_z for view in VIEW_NAMES}, base_target_logits,
                f"m{model_index}.{target_name}.candidate",
            )
            for view in VIEW_NAMES:
                base_nll = binary_row_nll(base_target_logits[view], cal_target)
                candidate_nll = binary_row_nll(candidate_target_logits[view], cal_target)
                record = {
                    "target": target_name, "model_index": model_index, "view": view,
                    "r0_readout_hash": base_summary["coefficient_hash"],
                    "candidate_offset_readout_hash": offset_summary["coefficient_hash"],
                    "r0_nll_mean": float(base_nll.mean()),
                    "r0_balanced_accuracy": balanced_accuracy((base_target_logits[view] >= 0).astype(np.uint8), cal_target.astype(np.uint8)),
                    "r0_row_nll_hash": array_sha256(f"m{model_index}.{target_name}.{view}.r0", base_nll),
                    "candidate_nll_mean": float(candidate_nll.mean()),
                    "candidate_balanced_accuracy": balanced_accuracy((candidate_target_logits[view] >= 0).astype(np.uint8), cal_target.astype(np.uint8)),
                    "candidate_row_nll_hash": array_sha256(f"m{model_index}.{target_name}.{view}.candidate", candidate_nll),
                }
                descriptive_pending.append((target_name, model_index, view, record, base_nll - candidate_nll))

        models.append({
            "model_index": model_index,
            "state_hashes": _state_hashes(arms),
            "training": {name: arms[name].training for name in (
                "RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT", "PRED_ONLY", "PRED_ONLY_SHUFFLED")},
            "readouts": {name: readouts[name] for name in readout_order},
            "views": views_output,
        })

    critical, contrasts, bootstrap_indices = bootstrap_simultaneous_lcbs(
        row_contrasts, manifest.bootstrap_root,
    )
    terminal, eligible, gates = evaluate_terminal_outcome(
        contrasts, valid=True, residual_nonzero_nonconstant=residual_ok,
    )
    diagnostic_contrasts: list[dict[str, object]] = []
    all_seed_local: dict[str, list[bool]] = {name: [] for name in ("g_PS", "g_CP", "g_CS")}
    for model_index in range(3):
        local_rows = {
            "g_PS": target_loss_rows[(model_index, "original", "PRED_ONLY_SHUFFLED")] - target_loss_rows[(model_index, "original", "PRED_ONLY")],
            "g_CP": target_loss_rows[(model_index, "original", "PRED_ONLY")] - target_loss_rows[(model_index, "original", "RESID_CANDIDATE")],
            "g_CS": target_loss_rows[(model_index, "original", "RESID_SHUFFLED")] - target_loss_rows[(model_index, "original", "RESID_CANDIDATE")],
        }
        for name in ("g_PS", "g_CP", "g_CS"):
            observed, lower, upper = paired_percentile_interval(local_rows[name], bootstrap_indices)
            beats = lower > 0
            all_seed_local[name].append(beats)
            diagnostic_contrasts.append({"name": name, "model_index": model_index, "observed": observed,
                                         "lower95": lower, "upper95": upper, "beats": beats})
    all_seed_beats = {name: all(values) for name, values in all_seed_local.items()}
    view_gates = dict(gates)
    contrast_lookup = {(c["model_index"], c["view"], c["name"]): c["lcb95"] > 0 for c in contrasts}
    for name, contract_name in (("d_r0", "d_R0"), ("d_shuf", "d_SHUF"), ("d_cap", "d_CAP"),
                                ("r_shuf", "r_SHUF"), ("r_cap", "r_CAP")):
        for view in VIEW_NAMES:
            view_gates[name + "_" + view] = all(contrast_lookup[(m, view, contract_name)] for m in range(3))
    classification = classify_diagnostics(terminal, eligible, view_gates, diagnostic_train, all_seed_beats)

    descriptive_targets: list[dict[str, object]] = []
    for _, _, _, record, rows in descriptive_pending:
        observed, lower, upper = paired_percentile_interval(rows, bootstrap_indices)
        record.update({"d_r0_observed": observed, "lower95": lower, "upper95": upper})
        descriptive_targets.append(record)
    # Pending records were generated model-major; enforce target-major contract order.
    descriptive_targets.sort(key=lambda item: (("ORDER", "TIME").index(item["target"]), item["model_index"],
                                                VIEW_NAMES.index(item["view"])))

    provenance_output = {
        "build_provenance_sha256": build_provenance_sha256, "target_commit": provenance.target_commit,
        "implementation_commit": provenance.implementation_commit, "clean_tree": provenance.clean_tree,
        "expected_source_digests": dict(SOURCE_DIGESTS),
        "verified_actual_source_digests": dict(provenance.source_digests),
        "implementation_digests": dict(provenance.implementation_digests),
        "python_version": provenance.python_version, "numpy_version": provenance.numpy_version,
        "torch_version": provenance.torch_version, "platform_machine": provenance.platform_machine,
        "platform_system": provenance.platform_system, "blas_fingerprint": provenance.blas_fingerprint,
    }
    audit_output = {"approved_envelope_sha256": approved_envelope_sha256, **dict(seed_audit)}
    checks = {name: True for name in (
        "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "target_axes", "initial_zero",
        "first_w2_gradient", "optimizer_membership", "successful_steps", "ema_steps", "readout_deterministic",
        "baseline_frozen", "nuisance_preserved", "family_exact", "diagnostics_nonpromoting", "output_allowlist",
    )}
    result = {
        "schema": "BP011-J04C-V3-R0RESID-RESULT-V1", "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256,
        "claim_ceiling": "one-generator directional public-synthetic frozen-R0 residual beta only",
        "provenance": provenance_output, "seed_audit": audit_output, "fixed": _fixed_contract(),
        "checks": checks, "models": models,
        "bootstrap": {"replicates": 10000, "family_size": 36, "quantile_method": "linear",
                      "critical_value": critical, "contrasts": contrasts},
        "diagnostics": {"classification": classification, "all_seed_beats": all_seed_beats,
                        "train_summaries": diagnostic_train, "target_loss_summaries": target_loss_summaries,
                        "target_loss_contrasts": diagnostic_contrasts},
        "recovery": _recovery_records(row_contrasts, bootstrap_indices) if eligible else [],
        "descriptive_targets": descriptive_targets, "valid": True, "eligible": eligible,
        "scientific_gates": gates, "terminal_outcome": terminal,
    }
    validate_success_schema(result)
    return result


def _require_mapping_keys(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    return value


def _validate_success_details(value: dict[str, object]) -> None:
    provenance = _require_mapping_keys(value["provenance"], {
        "build_provenance_sha256", "target_commit", "implementation_commit", "clean_tree",
        "expected_source_digests", "verified_actual_source_digests", "implementation_digests",
        "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system",
        "blas_fingerprint",
    })
    if provenance["target_commit"] != TARGET_COMMIT or provenance["clean_tree"] is not True \
       or provenance["expected_source_digests"] != SOURCE_DIGESTS \
       or provenance["verified_actual_source_digests"] != SOURCE_DIGESTS:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(provenance["implementation_commit"], str) or len(provenance["implementation_commit"]) != 40 \
       or any(character not in "0123456789abcdef" for character in provenance["implementation_commit"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    implementation_digests = provenance["implementation_digests"]
    if not isinstance(implementation_digests, dict) or set(implementation_digests) != set(IMPLEMENTATION_PATHS) \
       or any(not _is_digest(digest) for digest in implementation_digests.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    seed_audit = _require_mapping_keys(value["seed_audit"], {
        "approved_envelope_sha256", "manifest_sha256", "historical_inventory_sha256",
        "generated_audit_sha256", "production_path_count", "historical_path_count",
        "path_intersection_count", "root_intersection_count",
    })
    if seed_audit["production_path_count"] != 123 or seed_audit["path_intersection_count"] != 0 \
       or seed_audit["root_intersection_count"] != 0:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    fixed = _require_mapping_keys(value["fixed"], set(_fixed_contract()))
    if fixed != _fixed_contract():
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    check_names = {
        "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "target_axes", "initial_zero",
        "first_w2_gradient", "optimizer_membership", "successful_steps", "ema_steps", "readout_deterministic",
        "baseline_frozen", "nuisance_preserved", "family_exact", "diagnostics_nonpromoting", "output_allowlist",
    }
    checks = _require_mapping_keys(value["checks"], check_names)
    if any(item is not True for item in checks.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    state_names = {
        "e0", "candidate_adapter", "candidate_predictor", "candidate_ema_adapter", "shuffled_adapter",
        "shuffled_predictor", "shuffled_ema_adapter", "c0_adapter", "pred_only_predictor",
        "pred_only_shuffled_predictor",
    }
    training_names = {"RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT", "PRED_ONLY", "PRED_ONLY_SHUFFLED"}
    readout_order = (
        "R0_BASE", "RESID_CANDIDATE_ADDITIVE", "RESID_CANDIDATE_RESIDUAL_ONLY",
        "RESID_SHUFFLED_ADDITIVE", "RESID_SHUFFLED_RESIDUAL_ONLY",
        "RESID_CORRESPONDENCE_NULL_ADDITIVE", "RESID_CORRESPONDENCE_NULL_RESIDUAL_ONLY",
        "C0_DIRECT_ADDITIVE",
    )
    training_keys = {"attempted_steps", "successful_steps", "optimizer_steps", "ema_updates",
                     "first_100_mean_total", "last_100_mean_total", "component_scalars"}
    component_keys = {"cosine_first", "cosine_last", "directional_first", "directional_last",
                      "v_direction_min_first", "v_direction_min_last"}
    readout_keys = {"preprocess_hash", "coefficient_hash", "constant_coordinate_mask_hash",
                    "iterations", "converged", "zero_short_circuit"}
    assay_keys = {"nll_mean", "balanced_accuracy", "row_nll_hash", "logit_hash"}
    for model_index, model in enumerate(value["models"]):
        model = _require_mapping_keys(model, {"model_index", "state_hashes", "training", "readouts", "views"})
        if model["model_index"] != model_index:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
        _require_mapping_keys(model["state_hashes"], state_names)
        training = _require_mapping_keys(model["training"], training_names)
        for name, summary in training.items():
            summary = _require_mapping_keys(summary, training_keys)
            components = _require_mapping_keys(summary["component_scalars"], component_keys)
            if name == "C0_DIRECT":
                if any(item is not None for item in components.values()):
                    raise PrototypeInvariantError("SERIALIZATION_INVALID")
            elif any(item is None for item in components.values()):
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
        readouts = _require_mapping_keys(model["readouts"], set(readout_order))
        for summary in readouts.values():
            _require_mapping_keys(summary, readout_keys)
        views = _require_mapping_keys(model["views"], set(VIEW_NAMES))
        for view_summary in views.values():
            view_summary = _require_mapping_keys(view_summary, set(readout_order))
            for assay in view_summary.values():
                _require_mapping_keys(assay, assay_keys)
    bootstrap = _require_mapping_keys(value["bootstrap"], {
        "replicates", "family_size", "quantile_method", "critical_value", "contrasts"
    })
    if bootstrap["replicates"] != 10000 or bootstrap["family_size"] != 36 or bootstrap["quantile_method"] != "linear":
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    contrasts = bootstrap["contrasts"]
    if not isinstance(contrasts, list) or len(contrasts) != 36:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for expected, contrast in zip(expected_contrast_keys(), contrasts):
        contrast = _require_mapping_keys(contrast, {"name", "view", "model_index", "observed", "lcb95"})
        if (contrast["model_index"], contrast["view"], contrast["name"]) != expected:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    diagnostics = _require_mapping_keys(value["diagnostics"], {
        "classification", "all_seed_beats", "train_summaries", "target_loss_summaries", "target_loss_contrasts"
    })
    if diagnostics["classification"] not in {
        "NOT_INTERPRETED", "SUPPORTED_RESIDUAL_MECHANISM", "NUISANCE_FAILURE", "PREDICTOR_FIT_LIMITATION",
        "R0_TARGET_ACCESS_GENERALIZATION_LIMITATION", "TARGET_TO_REPRESENTATION_FAILURE",
        "RESIDUAL_OPTIMIZATION_FAILURE", "UNRESOLVED",
    }:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    _require_mapping_keys(diagnostics["all_seed_beats"], {"g_PS", "g_CP", "g_CS"})
    train_records = diagnostics["train_summaries"]
    if not isinstance(train_records, list) or len(train_records) != 15:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    arm_order = ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT")
    for index, item in enumerate(train_records):
        item = _require_mapping_keys(item, {"model_index", "arm", "first_100_mean_total", "last_100_mean_total"})
        if (item["model_index"], item["arm"]) != (index // 5, arm_order[index % 5]):
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    target_records = diagnostics["target_loss_summaries"]
    target_arm_order = ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED")
    expected_targets = [(m, v, a) for m in range(3) for v in VIEW_NAMES for a in target_arm_order]
    if not isinstance(target_records, list) or len(target_records) != 24:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for expected, item in zip(expected_targets, target_records):
        item = _require_mapping_keys(item, {"model_index", "view", "arm", "mean_l1_cosine_loss", "row_loss_hash"})
        if (item["model_index"], item["view"], item["arm"]) != expected:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    diagnostic_contrasts = diagnostics["target_loss_contrasts"]
    expected_diagnostic = [(m, name) for m in range(3) for name in ("g_PS", "g_CP", "g_CS")]
    if not isinstance(diagnostic_contrasts, list) or len(diagnostic_contrasts) != 9:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for expected, item in zip(expected_diagnostic, diagnostic_contrasts):
        item = _require_mapping_keys(item, {"name", "model_index", "observed", "lower95", "upper95", "beats"})
        if (item["model_index"], item["name"]) != expected:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    recovery = value["recovery"]
    if not isinstance(recovery, list) or len(recovery) != (6 if value["eligible"] else 0):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for expected, item in zip([(m, v) for m in range(3) for v in VIEW_NAMES], recovery):
        item = _require_mapping_keys(item, {"model_index", "view", "estimate", "interval_status", "lower95", "upper95"})
        if (item["model_index"], item["view"]) != expected or item["interval_status"] not in {"FINITE", "UNBOUNDED_DENOMINATOR"}:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
        if item["interval_status"] == "FINITE":
            if item["lower95"] is None or item["upper95"] is None:
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
        elif item["lower95"] is not None or item["upper95"] is not None:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    descriptions = value["descriptive_targets"]
    expected_descriptions = [(target, m, v) for target in ("ORDER", "TIME") for m in range(3) for v in VIEW_NAMES]
    description_keys = {"target", "model_index", "view", "r0_readout_hash", "candidate_offset_readout_hash",
                        "r0_nll_mean", "r0_balanced_accuracy", "r0_row_nll_hash", "candidate_nll_mean",
                        "candidate_balanced_accuracy", "candidate_row_nll_hash", "d_r0_observed", "lower95", "upper95"}
    if not isinstance(descriptions, list) or len(descriptions) != 12:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for expected, item in zip(expected_descriptions, descriptions):
        item = _require_mapping_keys(item, description_keys)
        if (item["target"], item["model_index"], item["view"]) != expected:
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    _require_mapping_keys(value["scientific_gates"], {
        "d_r0", "d_shuf", "d_cap", "r_shuf", "r_cap", "residual_nonzero_nonconstant"
    })
    for current_key, current_value in _walk_items(value):
        if current_key.endswith("_hash") or current_key.endswith("_sha256"):
            if not _is_digest(current_value):
                raise PrototypeInvariantError("SERIALIZATION_INVALID")


def _walk_items(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_items(item)


# Wrap the root validator with the exact recursive nested schema above.
_original_validate_success_schema = validate_success_schema

def validate_success_schema(value: object) -> None:
    _original_validate_success_schema(value)
    assert isinstance(value, dict)
    _validate_success_details(value)
