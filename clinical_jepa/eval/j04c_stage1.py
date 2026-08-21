"""Pure in-memory BP-CLINJEPA-011 J04c v2 Stage 1 training/evaluation.

This module has no filesystem, checkpoint, data-loader, subprocess, network, or
runtime device-selection behavior.  Callers supply only safe-public synthetic
splits and all results are aggregate values.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clinical_jepa.arms.v0f.own_latent import (
    EMATeacher,
    J04Encoder,
    NonAutoregressiveNextEventsHead,
    SharedLatentPredictor,
)
from clinical_jepa.eval.j04c_falsifier import (
    CAL_ID,
    CAL_OOD,
    PROBE_FIT,
    TRAIN,
    SyntheticFactorSplit,
    _normalized_diagnostics,
    _state_dict_bytes,
    _with_preserved_cpu_rng,
    balanced_accuracy,
    component_seed,
    fit_stage0_time_transform,
    initialized_teacher_reference_calibration,
    make_r0_encoder,
    make_readout_initializations,
    nuisance_only,
    nuisance_stratified_permutations,
    position_specificity,
    sever_student_leak,
    time_free_model_intervals,
)
from clinical_jepa.eval.next_event_metrics import gaussian_interval_nll, type_cross_entropy
from clinical_jepa.targets.next_event_contract import construct_latent_targets, latent_objective

GENERATOR_SEED = 1100
MODEL_SEED = 2100
CONDITION_NAMES = (
    "L0_EMA_POOL", "L1_AVG", "L2_SEP", "C0_DIRECT",
    "C0_NUISANCE_ONLY", "C0_TIME_FREE", "C0_STUDENT_LEAK",
)
FROZEN_CONDITION_NAMES = CONDITION_NAMES + ("R0_INIT",)
LATENT_NAMES = CONDITION_NAMES[:3]
THRESHOLD_DIGEST = "70fc5e63fc357a2341ecad421f52978a1fb8cc60ae33ad52e1080ce4beb71181"


@dataclass
class TrainedCondition:
    name: str
    encoder: J04Encoder
    teacher: EMATeacher | None
    predictor: SharedLatentPredictor | None
    training: dict[str, object]


@dataclass
class FrozenRepresentations:
    H: torch.Tensor
    z: torch.Tensor


def _rng_permutation(seed_items: list[int], n: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_items)))
    return rng.permutation(n)


def pretraining_indices(model_seed: int = MODEL_SEED, *, n: int = 8192, batch_size: int = 128, total: int = 2000) -> Iterable[np.ndarray]:
    if n != 8192 or batch_size != 128 or total != 2000:
        raise ValueError("production pretraining schedule is exactly 8192/128/2000")
    emitted = 0
    for epoch in range(32):
        permutation = _rng_permutation([model_seed, epoch, 6101], n)
        count = 64 if epoch <= 30 else 16
        for batch in range(count):
            emitted += 1
            yield permutation[batch * batch_size:(batch + 1) * batch_size]
    if emitted != total:
        raise RuntimeError("pretraining exposure mismatch")


def tiny_pretraining_indices(model_seed: int, *, n: int, batch_size: int, epochs: int) -> Iterable[np.ndarray]:
    """Tiny test-only analogue: one permutation and all contiguous batches per epoch."""
    if n <= 0 or batch_size <= 0 or n % batch_size or epochs <= 0:
        raise ValueError("positive divisible tiny schedule required")
    for epoch in range(epochs):
        permutation = _rng_permutation([model_seed, epoch, 6101], n)
        for start in range(0, n, batch_size):
            yield permutation[start:start + batch_size]


def probe_indices(model_seed: int, factor_code: int, *, n: int = 2048, batch_size: int = 256, total: int = 250) -> Iterable[np.ndarray]:
    if factor_code not in (0, 1, 2) or (n, batch_size, total) != (2048, 256, 250):
        raise ValueError("production probe schedule is factor 0..2 and 2048/256/250")
    emitted = 0
    for epoch in range(32):
        permutation = _rng_permutation([model_seed, factor_code, epoch, 6201], n)
        count = 8 if epoch <= 30 else 2
        for batch in range(count):
            emitted += 1
            yield permutation[batch * batch_size:(batch + 1) * batch_size]
    if emitted != total:
        raise RuntimeError("probe exposure mismatch")


def head_indices(model_seed: int, *, n: int = 2048, batch_size: int = 128, total: int = 250) -> Iterable[np.ndarray]:
    if (n, batch_size, total) != (2048, 128, 250):
        raise ValueError("production head schedule is exactly 2048/128/250")
    emitted = 0
    for epoch in range(16):
        permutation = _rng_permutation([model_seed, epoch, 6301], n)
        count = 16 if epoch <= 14 else 10
        for batch in range(count):
            emitted += 1
            yield permutation[batch * batch_size:(batch + 1) * batch_size]
    if emitted != total:
        raise RuntimeError("head exposure mismatch")


def _fresh_encoder(model_seed: int) -> J04Encoder:
    module = _with_preserved_cpu_rng(component_seed(model_seed, 1), J04Encoder)
    if not isinstance(module, J04Encoder):
        raise RuntimeError("encoder factory returned wrong type")
    return module


def _fresh_predictor(model_seed: int) -> SharedLatentPredictor:
    module = _with_preserved_cpu_rng(component_seed(model_seed, 2), SharedLatentPredictor)
    if not isinstance(module, SharedLatentPredictor):
        raise RuntimeError("predictor factory returned wrong type")
    return module


def _fresh_head(model_seed: int) -> NonAutoregressiveNextEventsHead:
    _, head = make_readout_initializations(model_seed)
    return head


def _adamw(parameters: list[nn.Parameter], *, lr: float, weight_decay: float) -> torch.optim.AdamW:
    if len(parameters) != len({id(p) for p in parameters}) or not all(p.requires_grad for p in parameters):
        raise RuntimeError("optimizer parameters must be unique trainables")
    optimizer = torch.optim.AdamW(
        [{"params": parameters}], lr=lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=weight_decay,
    )
    if len(optimizer.param_groups) != 1 or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in parameters]:
        raise RuntimeError("optimizer one-group identity invariant failed")
    return optimizer


def optimizer_membership(module_parts: tuple[nn.Module, ...], optimizer: torch.optim.Optimizer) -> bool:
    intended = [p for module in module_parts for p in module.parameters() if p.requires_grad]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    return len(optimizer.param_groups) == 1 and [id(p) for p in actual] == [id(p) for p in intended]


def _tensor_rows(values: np.ndarray, indices: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values[indices], dtype=dtype)


def _prefix_inputs(split: SyntheticFactorSplit, transform, condition: str, indices: np.ndarray | slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    selected = split if condition != "C0_NUISANCE_ONLY" else nuisance_only(split)
    types = selected.prefix_type_ids
    if condition == "C0_TIME_FREE":
        times = time_free_model_intervals(selected).astype(np.float32, copy=False)
    else:
        times = transform.transform(selected.prefix_intervals).astype(np.float32, copy=False)
    if indices is not None:
        types, times = types[indices], times[indices]
    return torch.as_tensor(types, dtype=torch.long), torch.as_tensor(times, dtype=torch.float32)


def append_student_leak(H: torch.Tensor, l_after: np.ndarray | torch.Tensor) -> FrozenRepresentations:
    bits = torch.as_tensor(l_after, dtype=H.dtype, device=H.device)
    if bits.shape != (H.shape[0], 3) or H.ndim != 3 or H.shape[-1] != 16:
        raise ValueError("H [B,N,16] and L_after [B,3] required")
    leak = torch.zeros((H.shape[0], 1, 16), dtype=H.dtype, device=H.device)
    leak[:, 0, :3] = bits.mul(2.0).sub(1.0)
    extended = torch.cat((H, leak), dim=1)
    return FrozenRepresentations(extended, extended.mean(dim=1))


def direct_objective(
    logits: torch.Tensor, mean: torch.Tensor, raw_scale: torch.Tensor,
    target_ids: torch.Tensor, target_intervals: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = torch.ones_like(target_ids, dtype=torch.bool)
    ce = type_cross_entropy(logits, target_ids, mask)
    interval = gaussian_interval_nll(target_intervals, mean, raw_scale)
    type_loss = ce.mean(dim=1).mean()
    interval_loss = interval.mean(dim=1).mean()
    return (ce + interval).mean(dim=1).mean(), {"type": type_loss, "interval": interval_loss}


def _finite_loss(loss: torch.Tensor) -> None:
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite scalar loss")


def _finite_gradients(parameters: Iterable[nn.Parameter]) -> bool:
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)


def _loss_summary(total: list[float], components: dict[str, list[float]]) -> dict[str, object]:
    if not total or not all(np.isfinite(total)) or not all(np.isfinite(x) for values in components.values() for x in values):
        raise FloatingPointError("non-finite or absent loss history")
    return {
        "first_100_mean_total": float(np.mean(total[:100])),
        "last_100_mean_total": float(np.mean(total[-100:])),
        "components": {
            name: {"first_100_mean": float(np.mean(values[:100])), "last_100_mean": float(np.mean(values[-100:]))}
            for name, values in components.items()
        },
    }


def train_latent_condition(recipe: str, train: SyntheticFactorSplit, transform, model_seed: int = MODEL_SEED) -> TrainedCondition:
    if recipe not in LATENT_NAMES:
        raise ValueError("unknown latent recipe")
    encoder = _fresh_encoder(model_seed).train()
    teacher = EMATeacher(encoder, momentum=0.996).eval()
    predictor = _fresh_predictor(model_seed).train()
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership((encoder, predictor), optimizer):
        raise RuntimeError("latent optimizer membership mismatch")
    target_times_all = transform.transform(train.target_intervals).astype(np.float32, copy=False)
    totals: list[float] = []
    components = {"cosine": [], "variance": [], "v_pred": []}
    successful = 0
    for indices in pretraining_indices(model_seed):
        prefix_ids, prefix_times = _prefix_inputs(train, transform, recipe, indices)
        target_ids = _tensor_rows(train.target_type_ids, indices, dtype=torch.long)
        target_times = _tensor_rows(target_times_all, indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, _, pooled = encoder(prefix_ids, prefix_times, causal=False)
        teacher_blocks, _, _ = teacher(target_ids, target_times, causal=True)
        latent_mask = torch.ones_like(target_ids, dtype=torch.bool)
        target, valid, selected = construct_latent_targets(teacher_blocks, latent_mask, recipe)
        if recipe == "L2_SEP" and selected != [1, 2, 3, 4]:
            raise RuntimeError("L2 identity order mismatch")
        prediction = predictor(pooled, recipe)
        loss, parts = latent_objective(prediction, target, valid)
        _finite_loss(loss)
        loss.backward()
        metadata = teacher.step_and_update(encoder, optimizer)
        if metadata != {
            "gradients_finite": True, "step_called": True, "ema_updated": True,
            "successful_steps": successful + 1,
        }:
            raise FloatingPointError("latent gradient/step/EMA invariant failed")
        successful += 1
        totals.append(float(loss.detach()))
        for name in components:
            components[name].append(float(parts[name].detach()))
    if successful != 2000 or teacher.successful_steps != 2000:
        raise RuntimeError("latent successful-step mismatch")
    return TrainedCondition(recipe, encoder, teacher, predictor, {
        "attempted_steps": 2000, "successful_steps": successful,
        "optimizer_steps": successful, "ema_updates": teacher.successful_steps,
        "losses": _loss_summary(totals, components),
    })


def train_direct_condition(name: str, train: SyntheticFactorSplit, transform, model_seed: int = MODEL_SEED) -> TrainedCondition:
    if name not in CONDITION_NAMES[3:]:
        raise ValueError("unknown direct condition")
    encoder = _fresh_encoder(model_seed).train()
    head = _fresh_head(model_seed).train()
    parameters = list(encoder.parameters()) + list(head.parameters())
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership((encoder, head), optimizer):
        raise RuntimeError("direct optimizer membership mismatch")
    target_times_all = transform.transform(train.target_intervals).astype(np.float32, copy=False)
    totals: list[float] = []
    components = {"type": [], "interval": []}
    successful = 0
    for indices in pretraining_indices(model_seed):
        prefix_ids, prefix_times = _prefix_inputs(train, transform, name, indices)
        target_ids = _tensor_rows(train.target_type_ids, indices, dtype=torch.long)
        target_times = _tensor_rows(target_times_all, indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, H, _ = encoder(prefix_ids, prefix_times, causal=False)
        if name == "C0_STUDENT_LEAK":
            representation = append_student_leak(H, train.L_after[indices])
            H = representation.H
        logits, time_output, _ = head(H, torch.ones(H.shape[:2], dtype=torch.bool))
        loss, parts = direct_objective(logits, time_output[..., 0], time_output[..., 1], target_ids, target_times)
        _finite_loss(loss)
        loss.backward()
        if not _finite_gradients(parameters):
            raise FloatingPointError("non-finite direct gradient")
        optimizer.step()
        successful += 1
        totals.append(float(loss.detach()))
        for component in components:
            components[component].append(float(parts[component].detach()))
    if successful != 2000:
        raise RuntimeError("direct successful-step mismatch")
    return TrainedCondition(name, encoder, None, None, {
        "attempted_steps": 2000, "successful_steps": successful,
        "optimizer_steps": successful, "ema_updates": 0,
        "losses": _loss_summary(totals, components),
    })


def freeze_encoder(condition: TrainedCondition) -> None:
    condition.encoder.eval()
    for parameter in condition.encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if condition.teacher is not None:
        condition.teacher.eval()
    if condition.predictor is not None:
        condition.predictor.eval()


def frozen_representations(condition: TrainedCondition, split: SyntheticFactorSplit, transform, *, leak_override: np.ndarray | None = None) -> FrozenRepresentations:
    condition.encoder.eval()
    with torch.no_grad():
        ids, times = _prefix_inputs(split, transform, condition.name)
        _, H, z = condition.encoder(ids, times, causal=False)
        if condition.name == "C0_STUDENT_LEAK":
            return append_student_leak(H, split.L_after if leak_override is None else leak_override)
        return FrozenRepresentations(H.detach(), z.detach())


def fit_probe(probe: nn.Linear, z: torch.Tensor, labels: np.ndarray, model_seed: int, factor_code: int) -> dict[str, object]:
    if z.shape != (2048, 16) or labels.shape != (2048,):
        raise ValueError("production probe fit requires 2048 pooled rows")
    probe.train()
    optimizer = _adamw(list(probe.parameters()), lr=1e-2, weight_decay=1e-3)
    losses: list[float] = []
    successful = 0
    target = torch.as_tensor(labels, dtype=torch.float32)
    for indices in probe_indices(model_seed, factor_code):
        idx = torch.as_tensor(indices, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(probe(z[idx]).squeeze(-1), target[idx])
        _finite_loss(loss)
        loss.backward()
        if not _finite_gradients(probe.parameters()):
            raise FloatingPointError("non-finite probe gradient")
        optimizer.step()
        successful += 1
        losses.append(float(loss.detach()))
    if successful != 250:
        raise RuntimeError("probe successful-step mismatch")
    probe.eval()
    return {"attempted_steps": 250, "successful_steps": successful, "optimizer_steps": successful,
            "first_100_mean_loss": float(np.mean(losses[:100])), "last_100_mean_loss": float(np.mean(losses[-100:]))}


def fit_frozen_head(head: NonAutoregressiveNextEventsHead, H: torch.Tensor, split: SyntheticFactorSplit, transform, model_seed: int) -> dict[str, object]:
    if H.shape[0] != 2048 or H.shape[-1] != 16:
        raise ValueError("production head fit requires 2048 representation rows")
    head.train()
    optimizer = _adamw(list(head.parameters()), lr=3e-4, weight_decay=1e-4)
    target_ids = torch.as_tensor(split.target_type_ids, dtype=torch.long)
    target_times = torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32, copy=False))
    losses: list[float] = []
    types: list[float] = []
    intervals: list[float] = []
    successful = 0
    for indices in head_indices(model_seed):
        idx = torch.as_tensor(indices, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        h = H[idx]
        logits, time_output, _ = head(h, torch.ones(h.shape[:2], dtype=torch.bool))
        loss, parts = direct_objective(logits, time_output[..., 0], time_output[..., 1], target_ids[idx], target_times[idx])
        _finite_loss(loss)
        loss.backward()
        if not _finite_gradients(head.parameters()):
            raise FloatingPointError("non-finite readout-head gradient")
        optimizer.step()
        successful += 1
        losses.append(float(loss.detach())); types.append(float(parts["type"].detach())); intervals.append(float(parts["interval"].detach()))
    if successful != 250:
        raise RuntimeError("head successful-step mismatch")
    head.eval()
    return {"attempted_steps": 250, "successful_steps": successful, "optimizer_steps": successful,
            "first_100_mean_total": float(np.mean(losses[:100])), "last_100_mean_total": float(np.mean(losses[-100:])),
            "first_100_mean_type": float(np.mean(types[:100])), "last_100_mean_type": float(np.mean(types[-100:])),
            "first_100_mean_interval": float(np.mean(intervals[:100])), "last_100_mean_interval": float(np.mean(intervals[-100:]))}


def fit_condition_readouts(condition: TrainedCondition, probe_fit: SyntheticFactorSplit, transform, model_seed: int = MODEL_SEED) -> tuple[tuple[nn.Linear, ...], NonAutoregressiveNextEventsHead, dict[str, object]]:
    before = _state_dict_bytes(condition.encoder)
    reps = frozen_representations(condition, probe_fit, transform)
    probes, head = make_readout_initializations(model_seed)
    probe_reports = [fit_probe(probes[f], reps.z, probe_fit.S[:, f], model_seed, f) for f in range(3)]
    head_report = fit_frozen_head(head, reps.H, probe_fit, transform, model_seed)
    after = _state_dict_bytes(condition.encoder)
    if before != after or any(p.grad is not None for p in condition.encoder.parameters()):
        raise RuntimeError("frozen encoder mutated during readout fitting")
    return probes, head, {"probes": probe_reports, "head": head_report, "encoder_bytes_unchanged": True, "encoder_grads_absent": True}


def evaluate_readouts(condition: TrainedCondition, probes: tuple[nn.Linear, ...], head: NonAutoregressiveNextEventsHead, split: SyntheticFactorSplit, transform, *, leak_override: np.ndarray | None = None) -> tuple[dict[str, object], list[np.ndarray]]:
    reps = frozen_representations(condition, split, transform, leak_override=leak_override)
    predictions: list[np.ndarray] = []
    factor_metrics = []
    with torch.no_grad():
        for factor, probe in enumerate(probes):
            logits = probe(reps.z).squeeze(-1)
            prediction = logits.ge(0).to(torch.uint8).cpu().numpy()
            predictions.append(prediction)
            factor_metrics.append({"factor_code": factor, "balanced_accuracy": balanced_accuracy(prediction, split.S[:, factor])})
        logits, time_output, _ = head(reps.H, torch.ones(reps.H.shape[:2], dtype=torch.bool))
        target_ids = torch.as_tensor(split.target_type_ids, dtype=torch.long)
        target_times = torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32, copy=False))
        mask = torch.ones_like(target_ids, dtype=torch.bool)
        type_nll = float(type_cross_entropy(logits, target_ids, mask).mean().item())
        interval_nll = float(gaussian_interval_nll(target_times, time_output[..., 0], time_output[..., 1]).mean().item())
    if not np.isfinite(type_nll) or not np.isfinite(interval_nll):
        raise FloatingPointError("non-finite evaluation metric")
    return {"factors": factor_metrics, "type_nll": type_nll, "interval_nll": interval_nll}, predictions


def one_run_negative_control(predictions: np.ndarray, labels: np.ndarray, nuisance: np.ndarray, factor_code: int, *, generator_seed: int = GENERATOR_SEED) -> dict[str, object]:
    permutations = nuisance_stratified_permutations(labels, nuisance, generator_seed, CAL_OOD, factor_code)
    observed = balanced_accuracy(predictions, labels)
    null = np.asarray([balanced_accuracy(predictions, row) for row in permutations], dtype=np.float64)
    critical = float(np.quantile(null, 1.0 - 0.05 / 7.0, method="linear"))
    return {"factor_code": factor_code, "T_obs_smoke": observed, "c_a_smoke": critical,
            "passes": bool(observed <= critical), "permutation_count": 1000}


def threshold_reference_report() -> tuple[dict[str, object], str]:
    report = initialized_teacher_reference_calibration((1101, 1102, 1103), (2101, 2102, 2103))
    rows = []
    for arm_name in LATENT_NAMES:
        for identity in report["arms"][arm_name]["identities"]:
            rows.append([arm_name, identity["identity_index"],
                         identity["normalized_variance"]["threshold_midpoint"],
                         identity["effective_rank"]["threshold_midpoint"]])
    digest = hashlib.sha256(json.dumps(rows, allow_nan=False, separators=(",", ":")).encode("ascii")).hexdigest()
    if len(rows) != 21 or digest != THRESHOLD_DIGEST:
        raise RuntimeError("accepted labeled threshold-row digest mismatch")
    return report, digest


def trained_collapse_diagnostics(condition: TrainedCondition, split: SyntheticFactorSplit, transform, reference: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if condition.name not in LATENT_NAMES or condition.teacher is None or condition.predictor is None:
        raise ValueError("trained latent condition required")
    ids, times = _prefix_inputs(split, transform, condition.name)
    target_ids = torch.as_tensor(split.target_type_ids, dtype=torch.long)
    target_times = torch.as_tensor(transform.transform(split.target_intervals).astype(np.float32, copy=False))
    with torch.no_grad():
        _, _, pooled = condition.encoder(ids, times, causal=False)
        prediction = condition.predictor(pooled, condition.name)
        blocks, _, _ = condition.teacher(target_ids, target_times, causal=True)
        target, mask, _ = construct_latent_targets(blocks, torch.ones_like(target_ids, dtype=torch.bool), condition.name)
    pred_flat = prediction[:, None, :] if prediction.ndim == 2 else prediction.reshape(2048, -1, 16)
    target_flat = target[:, None, :] if target.ndim == 2 else target.reshape(2048, -1, 16)
    thresholds = reference["arms"][condition.name]["identities"]
    if pred_flat.shape[1] != len(thresholds):
        raise RuntimeError("trained collapse identity count mismatch")
    rows = []
    for identity, threshold in enumerate(thresholds):
        metrics = _normalized_diagnostics(pred_flat[:, identity].cpu().numpy())
        target_metrics = _normalized_diagnostics(target_flat[:, identity].cpu().numpy())
        variance_threshold = threshold["normalized_variance"]["threshold_midpoint"]
        rank_threshold = threshold["effective_rank"]["threshold_midpoint"]
        variance_pass = bool(metrics["normalized_variance"] > variance_threshold)
        rank_pass = bool(metrics["effective_rank"] > rank_threshold)
        rows.append({"arm_name": condition.name, "identity_index": identity,
                     "normalized_variance": metrics["normalized_variance"], "variance_threshold": variance_threshold,
                     "variance_pass": variance_pass, "effective_rank": metrics["effective_rank"],
                     "rank_threshold": rank_threshold, "rank_pass": rank_pass,
                     "both_metrics_pass": variance_pass and rank_pass,
                     "teacher_target_effective_rank": target_metrics["effective_rank"]})
    position = None
    if condition.name != "L0_EMA_POOL":
        position = position_specificity(prediction, target, mask[:, :, 0] if mask.ndim == 3 else mask)
        position = {key: value for key, value in position.items() if key != "per_example"}
        position["teacher_target_effective_rank_by_identity"] = [row["teacher_target_effective_rank"] for row in rows]
    return rows, position


def hierarchical_complete_rule(predictions: np.ndarray, labels: np.ndarray, comparator_labels: np.ndarray, *, factor_code: int, simulation_index: int, n_boot: int = 10000) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    comparator_labels = np.asarray(comparator_labels, dtype=np.uint8)
    if predictions.ndim != 3 or predictions.shape[:2] != (3, 3) or labels.shape != (3, predictions.shape[2]) or comparator_labels.shape != labels.shape:
        raise ValueError("lexicographic [3,3,n] predictions and [3,n] labels required")
    n = predictions.shape[2]
    run_d = np.asarray([[balanced_accuracy(predictions[g, m], labels[g]) - balanced_accuracy(predictions[g, m], comparator_labels[g]) for m in range(3)] for g in range(3)])
    point = float(run_d.mean())
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([5102, factor_code, simulation_index, b])))
        selected_generators = rng.integers(0, 3, size=3)
        values = []
        for g in selected_generators:
            selected_models = rng.integers(0, 3, size=3)
            for m in selected_models:
                sequence_indices = rng.integers(0, n, size=n)
                values.append(balanced_accuracy(predictions[g, m, sequence_indices], labels[g, sequence_indices]) - balanced_accuracy(predictions[g, m, sequence_indices], comparator_labels[g, sequence_indices]))
        boot[b] = float(np.mean(values))
    lower = float(np.quantile(boot, 0.005, method="linear"))
    return {"point": point, "lower_0_005": lower, "bootstrap_replicates": n_boot}


def simulated_complete_rule(labels: np.ndarray, nuisance: np.ndarray, generator_seeds: tuple[int, int, int], *, factor_code: int, simulation_index: int, gap: float, delta: float, delta_power: float, powered: bool, n_boot: int = 10000) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.uint8)
    nuisance = np.asarray(nuisance, dtype=np.uint8)
    if labels.ndim != 2 or labels.shape[0] != 3 or nuisance.shape != (3, labels.shape[1], 3):
        raise ValueError("three generator panels required")
    n = labels.shape[1]
    comparator = np.empty_like(labels)
    for g, seed in enumerate(generator_seeds):
        comparator[g] = nuisance_stratified_permutations(labels[g], nuisance[g], seed, CAL_OOD, factor_code)[0]
    w = delta_power / gap if powered else 0.0
    predictions = np.empty((3, 3, n), dtype=np.uint8)
    for run_index in range(9):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([5101, factor_code, simulation_index, run_index])))
        prediction = (rng.random(n) < 0.5).astype(np.uint8)
        replacements = rng.random(n) < w
        g, m = divmod(run_index, 3)
        prediction[replacements] = labels[g, replacements]
        predictions[g, m] = prediction
    result = hierarchical_complete_rule(predictions, labels, comparator, factor_code=factor_code, simulation_index=simulation_index, n_boot=n_boot)
    result.update({"powered_fixture": powered, "replacement_probability": w,
                   "passes": bool(result["point"] >= delta and result["lower_0_005"] > 0.0)})
    return result


def complete_rule_calibration(labels: np.ndarray, nuisance: np.ndarray, generator_seeds: tuple[int, int, int], calibration: dict[str, float | int], *, simulations: int = 200, n_boot: int = 10000, minimum_passes: int = 160) -> dict[str, object]:
    if (simulations, n_boot, minimum_passes) != (200, 10000, 160):
        raise ValueError("production complete-rule calibration is exactly 200/10000/160")
    passed = sum(bool(simulated_complete_rule(labels, nuisance, generator_seeds,
        factor_code=int(calibration["factor_code"]), simulation_index=s,
        gap=float(calibration["G"]), delta=float(calibration["delta"]),
        delta_power=float(calibration["delta_power"]), powered=True, n_boot=n_boot)["passes"])
        for s in range(simulations))
    return {"simulations": simulations, "bootstrap_replicates": n_boot, "minimum_passes": minimum_passes,
            "pass_count": passed, "passes": passed >= minimum_passes}


def reduced_complete_rule_dry(
    cal_ood_splits: tuple[SyntheticFactorSplit, SyntheticFactorSplit, SyntheticFactorSplit],
    calibrations: list[dict[str, float | int]],
) -> list[dict[str, object]]:
    generator_seeds = (1101, 1102, 1103)
    if len(cal_ood_splits) != 3 or any(split.S.shape != (2048, 3) or split.N.shape != (2048, 3) for split in cal_ood_splits):
        raise ValueError("exactly three 2,048-row CAL-OOD panels required")
    nuisance = np.stack([split.N for split in cal_ood_splits])
    output = []
    for factor, calibration in enumerate(calibrations):
        labels = np.stack([split.S[:, factor] for split in cal_ood_splits])
        powered = simulated_complete_rule(labels, nuisance, generator_seeds, factor_code=factor, simulation_index=0,
            gap=float(calibration["G"]), delta=float(calibration["delta"]), delta_power=float(calibration["delta_power"]), powered=True, n_boot=100)
        null = simulated_complete_rule(labels, nuisance, generator_seeds, factor_code=factor, simulation_index=1,
            gap=float(calibration["G"]), delta=float(calibration["delta"]), delta_power=float(calibration["delta_power"]), powered=False, n_boot=100)
        output.append({"factor_code": factor, "powered": powered, "null": null,
                       "powered_passes": powered["passes"], "null_fails": not null["passes"]})
    return output
