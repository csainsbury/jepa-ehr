"""Bounded L0 initialization intervention for the BP011 J04c bridge."""
from __future__ import annotations

import torch

from clinical_jepa.arms.v0f.own_latent import EMATeacher
from clinical_jepa.eval.j04c_stage1 import (
    TrainedCondition, _adamw, _finite_loss, _fresh_encoder, _fresh_predictor,
    _loss_summary, _prefix_inputs, _tensor_rows, optimizer_membership,
    pretraining_indices,
)
from clinical_jepa.targets.next_event_contract import construct_latent_targets, latent_objective


def set_identity_type_embedding(encoder) -> None:
    """Set a label-free deterministic identity code with unit-norm non-padding rows."""
    weight = encoder.type_embedding.weight
    if tuple(weight.shape) != (18, 16) or encoder.type_embedding.padding_idx != 0:
        raise ValueError("expected J04 18x16 type embedding with padding row zero")
    with torch.no_grad():
        weight.zero_()
        for token_id in range(1, 17):
            weight[token_id, token_id - 1] = 1.0
        weight[17].fill_(-0.25)
    if not bool(torch.all(weight[0] == 0)) or not bool(torch.allclose(weight[1:].norm(dim=1), torch.ones(17))):
        raise RuntimeError("deterministic identity embedding invariant failed")


def train_l0_identity_embedding(train, transform, model_seed: int) -> TrainedCondition:
    encoder = _fresh_encoder(model_seed).train()
    set_identity_type_embedding(encoder)
    teacher = EMATeacher(encoder, momentum=0.996).eval()
    predictor = _fresh_predictor(model_seed).train()
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership((encoder, predictor), optimizer):
        raise RuntimeError("latent optimizer membership mismatch")
    target_times_all = transform.transform(train.target_intervals).astype("float32", copy=False)
    totals = []
    components = {"cosine": [], "variance": [], "v_pred": []}
    successful = 0
    for indices in pretraining_indices(model_seed):
        prefix_ids, prefix_times = _prefix_inputs(train, transform, "L0_EMA_POOL", indices)
        target_ids = _tensor_rows(train.target_type_ids, indices, dtype=torch.long)
        target_times = _tensor_rows(target_times_all, indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, _, pooled = encoder(prefix_ids, prefix_times, causal=False)
        teacher_blocks, _, _ = teacher(target_ids, target_times, causal=True)
        latent_mask = torch.ones_like(target_ids, dtype=torch.bool)
        target, valid, _ = construct_latent_targets(teacher_blocks, latent_mask, "L0_EMA_POOL")
        prediction = predictor(pooled, "L0_EMA_POOL")
        loss, parts = latent_objective(prediction, target, valid)
        _finite_loss(loss)
        loss.backward()
        metadata = teacher.step_and_update(encoder, optimizer)
        if metadata != {"gradients_finite": True, "step_called": True, "ema_updated": True,
                        "successful_steps": successful + 1}:
            raise FloatingPointError("latent gradient/step/EMA invariant failed")
        successful += 1
        totals.append(float(loss.detach()))
        for name in components:
            components[name].append(float(parts[name].detach()))
    if successful != 2000 or teacher.successful_steps != 2000:
        raise RuntimeError("latent successful-step mismatch")
    return TrainedCondition("L0_EMA_POOL", encoder, teacher, predictor, {
        "attempted_steps": 2000, "successful_steps": successful,
        "optimizer_steps": successful, "ema_updates": teacher.successful_steps,
        "losses": _loss_summary(totals, components),
        "initialization_intervention": "label-free deterministic unit-norm identity type embedding",
    })


def set_identity_gelu_predictor(predictor) -> None:
    """Initialize the existing 16-32-16 GELU MLP to return LayerNorm(z) exactly."""
    if tuple(predictor.linear1.weight.shape) != (32, 16) or tuple(predictor.linear2.weight.shape) != (16, 32):
        raise ValueError("expected SharedLatentPredictor 16-32-16 MLP")
    identity = torch.eye(16, dtype=predictor.linear1.weight.dtype, device=predictor.linear1.weight.device)
    with torch.no_grad():
        predictor.linear1.weight.copy_(torch.cat((identity, -identity), dim=0))
        predictor.linear1.bias.zero_()
        predictor.linear2.weight.copy_(torch.cat((identity, -identity), dim=1))
        predictor.linear2.bias.zero_()


def train_l0_decoupled_seeds(
    train, transform, encoder_seed: int, training_seed: int, *, schedule_seed: int | None = None,
    identity_predictor: bool = False, total_steps: int = 2000,
    student_variance_weight: float = 0.0, student_variance_floor: float = 0.05,
    directional_variance_weight: float = 0.0, directional_variance_floor: float = 0.005,
) -> TrainedCondition:
    """Train standard L0 with separately selectable encoder, predictor, and order seeds."""
    if schedule_seed is None:
        schedule_seed = training_seed
    if total_steps <= 0 or total_steps > 2000:
        raise ValueError("total_steps must be in 1..2000")
    if student_variance_weight < 0 or student_variance_floor <= 0:
        raise ValueError("student variance weight/floor must be non-negative/positive")
    if directional_variance_weight < 0 or directional_variance_floor <= 0:
        raise ValueError("directional variance weight/floor must be non-negative/positive")
    encoder = _fresh_encoder(encoder_seed).train()
    teacher = EMATeacher(encoder, momentum=0.996).eval()
    predictor = _fresh_predictor(training_seed).train()
    if identity_predictor:
        set_identity_gelu_predictor(predictor)
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    if not optimizer_membership((encoder, predictor), optimizer):
        raise RuntimeError("latent optimizer membership mismatch")
    target_times_all = transform.transform(train.target_intervals).astype("float32", copy=False)
    totals = []
    components = {"cosine": [], "variance": [], "v_pred": []}
    if student_variance_weight:
        components.update({"student_variance_penalty": [], "v_student": []})
    if directional_variance_weight:
        components.update({"directional_variance_penalty": [], "v_direction": []})
    successful = 0
    for indices in pretraining_indices(schedule_seed):
        if successful == total_steps:
            break
        prefix_ids, prefix_times = _prefix_inputs(train, transform, "L0_EMA_POOL", indices)
        target_ids = _tensor_rows(train.target_type_ids, indices, dtype=torch.long)
        target_times = _tensor_rows(target_times_all, indices, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, _, pooled = encoder(prefix_ids, prefix_times, causal=False)
        teacher_blocks, _, _ = teacher(target_ids, target_times, causal=True)
        target, valid, _ = construct_latent_targets(teacher_blocks, torch.ones_like(target_ids, dtype=torch.bool), "L0_EMA_POOL")
        prediction = predictor(pooled, "L0_EMA_POOL")
        loss, parts = latent_objective(prediction, target, valid)
        if student_variance_weight:
            v_student = pooled.var(dim=0, correction=0).mean()
            student_penalty = student_variance_weight * torch.clamp(student_variance_floor - v_student, min=0.0)
            loss = loss + student_penalty
            parts = dict(parts, student_variance_penalty=student_penalty, v_student=v_student)
        if directional_variance_weight:
            unit_prediction = torch.nn.functional.normalize(prediction, dim=-1, eps=1e-8)
            v_direction = unit_prediction.var(dim=0, correction=0).mean()
            directional_penalty = directional_variance_weight * torch.clamp(directional_variance_floor - v_direction, min=0.0)
            loss = loss + directional_penalty
            parts = dict(parts, directional_variance_penalty=directional_penalty, v_direction=v_direction)
        _finite_loss(loss)
        loss.backward()
        metadata = teacher.step_and_update(encoder, optimizer)
        if metadata != {"gradients_finite": True, "step_called": True, "ema_updated": True,
                        "successful_steps": successful + 1}:
            raise FloatingPointError("latent gradient/step/EMA invariant failed")
        successful += 1
        totals.append(float(loss.detach()))
        for name in components:
            components[name].append(float(parts[name].detach()))
    if successful != total_steps or teacher.successful_steps != total_steps:
        raise RuntimeError("latent successful-step mismatch")
    return TrainedCondition("L0_EMA_POOL", encoder, teacher, predictor, {
        "attempted_steps": total_steps, "successful_steps": successful,
        "optimizer_steps": successful, "ema_updates": teacher.successful_steps,
        "losses": _loss_summary(totals, components),
        "encoder_seed": encoder_seed, "predictor_seed": training_seed, "schedule_seed": schedule_seed,
        "identity_predictor_initialization": identity_predictor,
        "student_variance_weight": student_variance_weight,
        "student_variance_floor": student_variance_floor,
        "directional_variance_weight": directional_variance_weight,
        "directional_variance_floor": directional_variance_floor,
    })
