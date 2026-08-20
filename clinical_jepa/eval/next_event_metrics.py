"""Pure safe-public metrics for the BP-CLINJEPA-011 next-event contract."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def type_cross_entropy(logits: torch.Tensor, event_ids: torch.Tensor, type_mask: torch.Tensor) -> torch.Tensor:
    """Return per-slot CE with declared unscored slots, including PAD, zeroed."""
    if logits.shape[:-1] != event_ids.shape or logits.shape[-1] != 17 or type_mask.shape != event_ids.shape:
        raise ValueError("logits [B,K,17], event IDs [B,K], and type mask [B,K] required")
    if type_mask.dtype != torch.bool:
        raise ValueError("type_mask must be boolean")
    if bool(((event_ids < 0) | (event_ids > 17)).any()):
        raise ValueError("event IDs must be PAD=0 or a class ID in 1..17")
    if bool(((event_ids < 1) & type_mask).any()):
        raise ValueError("PAD cannot be scored under the declared type mask")
    safe_ids = event_ids.masked_fill(~type_mask, 1)
    losses = F.cross_entropy(logits.reshape(-1, 17), (safe_ids - 1).reshape(-1), reduction="none").reshape(event_ids.shape)
    return losses.masked_fill(~type_mask, 0.0)


def gaussian_interval_nll(y: torch.Tensor, mean: torch.Tensor, raw_scale: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    sigma = F.softplus(raw_scale) + eps
    return torch.log(sigma) + 0.5 * ((y - mean) / sigma).square() + 0.5 * math.log(2.0 * math.pi)


def gaussian_crps(y: torch.Tensor, mean: torch.Tensor, raw_scale: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    sigma = F.softplus(raw_scale) + eps
    z = (y - mean) / sigma
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def inverse_median_raw_time(transformed_mean, *, train_mu: float, train_sigma: float, unit: float = 1.0):
    if torch.is_tensor(transformed_mean):
        return torch.clamp(unit * torch.expm1(train_mu + train_sigma * transformed_mean), min=0.0)
    return np.maximum(0.0, unit * np.expm1(train_mu + train_sigma * np.asarray(transformed_mean)))


def masked_mean_within_example(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask shapes differ")
    flat_values, flat_mask = values.reshape(values.shape[0], -1), mask.reshape(mask.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every example must contain a scored unit")
    return ((flat_values * flat_mask).sum(dim=1) / counts).mean()


def balanced_accuracy(true: Sequence[int], predicted: Sequence[int]) -> float:
    y, p = np.asarray(true), np.asarray(predicted)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("balanced accuracy requires both true classes")
    return float(0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()))


def position_marginal_probabilities(type_ids: np.ndarray, mask: np.ndarray, *, class_count: int = 17, pseudocount: float = 0.5) -> np.ndarray:
    ids, valid = np.asarray(type_ids), np.asarray(mask, dtype=bool)
    out = np.empty((ids.shape[1], class_count), dtype=np.float64)
    for position in range(ids.shape[1]):
        counts = np.full(class_count, pseudocount, dtype=np.float64)
        for event_id in ids[valid[:, position], position]:
            if not 1 <= int(event_id) <= class_count:
                raise ValueError("valid event ID outside non-PAD classes")
            counts[int(event_id) - 1] += 1
        out[position] = counts / counts.sum()
    return out


def recurrence_probabilities(position_marginal: np.ndarray, context_ids: Sequence[int], *, pseudocount: float = 0.5) -> np.ndarray:
    marginal = np.asarray(position_marginal, dtype=np.float64)
    class_count = marginal.shape[-1]
    counts = np.full(class_count, pseudocount, dtype=np.float64)
    for event_id in context_ids:
        if event_id != 0:
            counts[int(event_id) - 1] += 1
    context = counts / counts.sum()
    return 0.5 * marginal + 0.5 * context


def population_feature_variance(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("variance requires at least two examples")
    return float(np.var(x, axis=0, ddof=0).mean())


def effective_rank(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("effective rank requires a rank-2 array")
    minimum_examples = max(32, 2 * x.shape[1])
    if x.shape[0] < minimum_examples:
        raise ValueError(f"effective rank requires at least {minimum_examples} examples for width {x.shape[1]}")
    centered = x - x.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / x.shape[0]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = eigenvalues.sum()
    if trace <= 1e-12:
        return 0.0
    probabilities = eigenvalues / trace
    nz = probabilities > 0
    return float(np.exp(-(probabilities[nz] * np.log(probabilities[nz])).sum()))


def off_diagonal_cosine(values: np.ndarray, *, eps: float = 1e-8) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("off-diagonal cosine requires at least two examples")
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    normalized = x / norms
    rows, cols = np.triu_indices(x.shape[0], k=1)
    return float((normalized[rows] * normalized[cols]).sum(axis=1).mean())


def tie_permutation(local_sequence_key: str, timestamp: str, occurrence_index: int, group_size: int, *, namespace: str = "clinjepa-j04-tie-v1") -> np.ndarray:
    if group_size < 0:
        raise ValueError("group size must be nonnegative")
    message = f"{namespace}|{local_sequence_key}|{timestamp}|{occurrence_index}|{group_size}".encode("utf-8")
    digest = hashlib.blake2b(message, digest_size=32).digest()
    seed64 = int.from_bytes(digest[:8], "little")
    return np.random.Generator(np.random.PCG64(seed64)).permutation(group_size)


def order_pair_mask(timestamps: Sequence[float], valid: Sequence[bool] | None = None) -> np.ndarray:
    times = np.asarray(timestamps)
    n = len(times)
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    mask = np.zeros((n, n), dtype=bool)
    for earlier in range(n):
        for later in range(earlier + 1, n):
            mask[earlier, later] = bool(keep[earlier] and keep[later] and times[earlier] < times[later])
    return mask
