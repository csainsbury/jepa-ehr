"""Safe-public next-event target contract for BP-CLINJEPA-011 J04b.

This module operates only on caller-supplied in-memory arrays/tensors.  It has
no dataset, filesystem, checkpoint, vocabulary, or training dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

PAD_ID = 0
END_OBS_ID = 1


@dataclass(frozen=True)
class NextEventTargetBatch:
    type_ids: torch.Tensor
    transformed_intervals: torch.Tensor
    latent_mask: torch.Tensor
    type_mask: torch.Tensor
    interval_mask: torch.Tensor
    attention_mask: torch.Tensor


def build_next_event_targets(
    observed_types: Sequence[Sequence[int]],
    observed_intervals: Sequence[Sequence[float]],
    *,
    k: int = 4,
    known_endpoints: Sequence[bool] | None = None,
    time_transform: "TimeTransform | None" = None,
) -> NextEventTargetBatch:
    """Build observed targets followed by at most one END_OBS, then PAD.

    ``observed_types`` contains only observed ordinary event IDs.  A known
    observation endpoint is represented only when fewer than ``k`` observed
    events are supplied.  Censored examples set ``known_endpoints=False``.
    """
    if k <= 0 or not observed_types:
        raise ValueError("k and batch size must be positive")
    if len(observed_types) != len(observed_intervals):
        raise ValueError("type/interval batch sizes differ")
    if known_endpoints is None:
        known_endpoints = [False] * len(observed_types)
    if len(known_endpoints) != len(observed_types):
        raise ValueError("known_endpoints batch size differs")

    b = len(observed_types)
    ids = torch.zeros((b, k), dtype=torch.long)
    intervals = torch.zeros((b, k), dtype=torch.float32)
    latent = torch.zeros((b, k), dtype=torch.bool)
    type_mask = torch.zeros((b, k), dtype=torch.bool)
    interval_mask = torch.zeros((b, k), dtype=torch.bool)
    for row, (types, times, known) in enumerate(zip(observed_types, observed_intervals, known_endpoints)):
        if len(types) != len(times):
            raise ValueError("each type/interval sequence must have equal length")
        if len(types) < 1:
            raise ValueError("each example requires at least one observed target")
        r = min(k, len(types))
        vals = torch.as_tensor(types[:r], dtype=torch.long)
        if bool(((vals < 2) | (vals > 17)).any()):
            raise ValueError("observed synthetic event IDs must be in 2..17")
        raw = torch.as_tensor(times[:r], dtype=torch.float32)
        if bool((raw < 0).any()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("intervals must be finite and nonnegative")
        ids[row, :r] = vals
        intervals[row, :r] = raw
        latent[row, :r] = True
        type_mask[row, :r] = True
        interval_mask[row, :r] = True
        if r < k and bool(known):
            ids[row, r] = END_OBS_ID
            type_mask[row, r] = True
    transformed = intervals if time_transform is None else time_transform.transform(intervals)
    transformed = transformed * interval_mask.to(transformed.dtype)
    attention = ids.ne(PAD_ID)
    return NextEventTargetBatch(ids, transformed, latent, type_mask, interval_mask, attention)


class TimeTransform:
    """TRAIN-fitted ``log1p(dt / unit)`` standardization."""

    def __init__(self, *, unit: float = 1.0) -> None:
        if unit <= 0:
            raise ValueError("unit must be positive")
        self.unit = float(unit)
        self.mu: float | None = None
        self.sigma: float | None = None

    def fit_train(self, observed_intervals: Sequence[float] | np.ndarray | torch.Tensor) -> "TimeTransform":
        values = np.asarray(observed_intervals, dtype=np.float64)
        if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("TRAIN intervals must be nonempty, finite, and nonnegative")
        a = np.log1p(values / self.unit)
        sigma = float(a.std(ddof=0))
        if not sigma > 0:
            raise ValueError("TRAIN transformed interval sigma must be positive")
        self.mu, self.sigma = float(a.mean()), sigma
        return self

    def state_bytes(self) -> bytes:
        self._require_fit()
        return np.asarray([self.unit, self.mu, self.sigma], dtype="<f8").tobytes()

    def transform(self, delta: np.ndarray | torch.Tensor):
        self._require_fit()
        if torch.is_tensor(delta):
            if bool((delta < 0).any()):
                raise ValueError("intervals must be nonnegative")
            return (torch.log1p(delta / self.unit) - self.mu) / self.sigma
        values = np.asarray(delta)
        if np.any(values < 0):
            raise ValueError("intervals must be nonnegative")
        return (np.log1p(values / self.unit) - self.mu) / self.sigma

    def inverse_median(self, transformed_mean: np.ndarray | torch.Tensor):
        self._require_fit()
        if torch.is_tensor(transformed_mean):
            return torch.clamp(self.unit * torch.expm1(self.mu + self.sigma * transformed_mean), min=0.0)
        return np.maximum(0.0, self.unit * np.expm1(self.mu + self.sigma * np.asarray(transformed_mean)))

    def _require_fit(self) -> None:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("TimeTransform must be fit on TRAIN intervals first")


def position_causal_attention(valid_positions: torch.Tensor) -> torch.Tensor:
    """Return allowed ``[batch, query, key]`` target attention relation."""
    if valid_positions.ndim != 2 or valid_positions.dtype != torch.bool:
        raise ValueError("valid_positions must be a rank-2 bool tensor")
    k = valid_positions.shape[1]
    causal = torch.ones((k, k), dtype=torch.bool, device=valid_positions.device).tril()
    return causal.unsqueeze(0) & valid_positions[:, :, None] & valid_positions[:, None, :]


def resolve_layer_sets(layer_count: int) -> tuple[list[int], list[int]]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    top = list(range(max(1, layer_count - 3), layer_count + 1))
    m = min(4, layer_count)
    if m == 1:
        spaced = [1]
    else:
        spaced = []
        for i in range(m):
            # floor(x + .5), not Python's ties-to-even round.
            value = 1 + int(np.floor(i * (layer_count - 1) / (m - 1) + 0.5))
            if value not in spaced:
                spaced.append(value)
    return top, spaced


def construct_latent_targets(
    block_states: torch.Tensor,
    latent_mask: torch.Tensor,
    recipe: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Construct normalized L0/L1/L2 teacher targets.

    ``block_states`` is ``[B,L,K,D]`` post-residual teacher state.  Returned
    masks identify target-unit identities, excluding END_OBS and PAD.
    """
    if block_states.ndim != 4 or latent_mask.shape != (block_states.shape[0], block_states.shape[2]):
        raise ValueError("expected states [B,L,K,D] and mask [B,K]")
    _, layers, _, d = block_states.shape
    top, spaced = resolve_layer_sets(layers)
    normalized = F.layer_norm(block_states, (d,), weight=None, bias=None, eps=1e-5)
    if recipe == "L0_EMA_POOL":
        per_position = normalized[:, [x - 1 for x in top]].mean(dim=1)
        weights = latent_mask.to(per_position.dtype).unsqueeze(-1)
        target = (per_position * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return target, latent_mask.any(dim=1), top
    if recipe == "L1_AVG":
        return normalized[:, [x - 1 for x in top]].mean(dim=1), latent_mask, top
    if recipe == "L2_SEP":
        if layers == 1:
            raise ValueError("L2 separation is undefined for one layer")
        # [B,K,S,D]
        target = normalized[:, [x - 1 for x in spaced]].permute(0, 2, 1, 3)
        return target, latent_mask[:, :, None].expand(-1, -1, len(spaced)), spaced
    raise ValueError(f"unknown recipe: {recipe}")


def latent_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Complete masked ``L_cosine + L_variance`` objective."""
    if prediction.shape != target.shape or prediction.shape[:-1] != valid_mask.shape:
        raise ValueError("prediction, target, and valid mask shapes disagree")
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=eps)
    per_unit = 1.0 - cosine
    flat_loss = per_unit.reshape(per_unit.shape[0], -1)
    flat_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every example requires a valid latent target unit")
    per_example = (flat_loss * flat_mask).sum(dim=1) / counts
    l_cosine = per_example.mean()

    flat_pred = prediction.reshape(prediction.shape[0], -1, prediction.shape[-1])
    eligible_variances = []
    for identity in range(flat_pred.shape[1]):
        rows = flat_mask[:, identity]
        if int(rows.sum()) >= 2:
            eligible_variances.append(flat_pred[rows, identity].var(dim=0, correction=0).mean())
    if not eligible_variances:
        raise ValueError("no target-unit identity has two valid examples")
    v_pred = torch.stack(eligible_variances).mean()
    l_variance = 0.01 * torch.clamp(0.05 - v_pred, min=0.0)
    total = l_cosine + l_variance
    return total, {"cosine": l_cosine, "variance": l_variance, "v_pred": v_pred}


def latent_output_accounting(
    recipe: str,
    *,
    r: int,
    d: int = 16,
    k: int = 4,
    layer_count: int = 4,
) -> dict[str, int | str]:
    """Analytic predictor-only output and compute accounting, not training FLOPs."""
    if not 0 <= r <= k:
        raise ValueError("r must be in 0..k")
    _, spaced = resolve_layer_sets(layer_count)
    if recipe == "L0_EMA_POOL":
        if r < 1:
            raise ValueError("L0 requires an observed target")
        allocated_elements = valid_elements = d
        allocated_units = valid_units = predictor_calls = 1
    elif recipe == "L1_AVG":
        allocated_elements, valid_elements = k * d, r * d
        allocated_units, valid_units, predictor_calls = k, r, k
    elif recipe == "L2_SEP":
        if layer_count == 1:
            raise ValueError("L2 separation is undefined for one layer")
        selected_layers = len(spaced)
        allocated_elements, valid_elements = k * selected_layers * d, r * selected_layers * d
        allocated_units, valid_units = k * selected_layers, r * selected_layers
        predictor_calls = k * selected_layers
    else:
        raise ValueError(f"unknown recipe: {recipe}")
    predictor_activation_elements = predictor_calls * (16 + 32 + 16)
    predictor_macs = predictor_calls * (16 * 32 + 32 * 16)
    return {
        "allocated_output_elements": allocated_elements,
        "valid_output_elements": valid_elements,
        "allocated_target_units": allocated_units,
        "valid_target_units": valid_units,
        "predictor_calls": predictor_calls,
        "predictor_activation_elements": predictor_activation_elements,
        "predictor_macs": predictor_macs,
        "predictor_flops": 2 * predictor_macs,
        "predictor_macs_label": "analytic predictor-only MACs",
        "predictor_flops_label": "analytic predictor-only FLOPs; not measured whole-training FLOPs",
    }
