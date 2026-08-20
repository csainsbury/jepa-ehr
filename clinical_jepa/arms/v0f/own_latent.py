"""Exact CPU-safe public architecture for BP-CLINJEPA-011 J04b.

There is deliberately no training loop, checkpoint I/O, device selection, or
candidate runner in this module.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
from torch import nn

D_MODEL = 16
K_TARGET = 4


def sinusoidal_code(position: int | torch.Tensor, *, d: int = D_MODEL, dtype=torch.float32, device=None) -> torch.Tensor:
    if d % 2:
        raise ValueError("sinusoidal width must be even")
    pos = torch.as_tensor(position, dtype=dtype, device=device).reshape(-1, 1)
    pair = torch.arange(0, d, 2, dtype=dtype, device=pos.device)
    angle = pos / torch.pow(torch.tensor(10000.0, dtype=dtype, device=pos.device), pair / d)
    out = torch.empty((pos.shape[0], d), dtype=dtype, device=pos.device)
    out[:, 0::2], out[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return out[0] if torch.as_tensor(position).ndim == 0 else out


def _initialize(module: nn.Module, *, embedding_or_query: torch.Tensor | None = None) -> None:
    handled: set[int] = set()
    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            nn.init.zeros_(child.in_proj_bias)
            nn.init.xavier_uniform_(child.out_proj.weight)
            nn.init.zeros_(child.out_proj.bias)
            handled.add(id(child.out_proj))
    for child in module.modules():
        if isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=D_MODEL ** -0.5)
            if child.padding_idx is not None:
                with torch.no_grad():
                    child.weight[child.padding_idx].zero_()
        elif isinstance(child, nn.Linear) and id(child) not in handled:
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm) and child.elementwise_affine:
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
    if embedding_or_query is not None:
        nn.init.normal_(embedding_or_query, mean=0.0, std=D_MODEL ** -0.5)


class _EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(16, eps=1e-5)
        self.attention = nn.MultiheadAttention(16, 4, dropout=0.0, bias=True, batch_first=True)
        self.ln2 = nn.LayerNorm(16, eps=1e-5)
        self.linear1 = nn.Linear(16, 64, bias=True)
        self.linear2 = nn.Linear(64, 16, bias=True)
        self.gelu = nn.GELU(approximate="none")

    def forward(self, x: torch.Tensor, valid: torch.Tensor, causal: bool) -> torch.Tensor:
        norm = self.ln1(x)
        length = x.shape[1]
        forbidden = None
        if causal:
            forbidden = torch.ones((length, length), dtype=torch.bool, device=x.device).triu(1)
        attended, _ = self.attention(norm, norm, norm, attn_mask=forbidden, key_padding_mask=~valid, need_weights=False)
        a = x + attended
        x = a + self.linear2(self.gelu(self.linear1(self.ln2(a))))
        return x * valid.unsqueeze(-1).to(x.dtype)


class J04Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(18, 16, padding_idx=0)
        self.time_projection = nn.Linear(1, 16, bias=True)
        self.input_norm = nn.LayerNorm(16, eps=1e-5)
        self.blocks = nn.ModuleList([_EncoderBlock() for _ in range(4)])
        self.final_norm = nn.LayerNorm(16, eps=1e-5)
        _initialize(self)

    def forward(
        self,
        type_ids: torch.Tensor,
        transformed_intervals: torch.Tensor,
        *,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type_ids.ndim != 2 or transformed_intervals.shape != type_ids.shape:
            raise ValueError("type_ids and transformed_intervals must be [B,N]")
        if type_ids.shape[1] > 8 and not causal:
            raise ValueError("student context limit is eight events")
        valid = type_ids.ne(0)
        positions = sinusoidal_code(torch.arange(type_ids.shape[1], device=type_ids.device), dtype=self.type_embedding.weight.dtype)
        time_part = self.time_projection(transformed_intervals.unsqueeze(-1)) * valid.unsqueeze(-1)
        x = self.input_norm(self.type_embedding(type_ids) + time_part + positions.unsqueeze(0))
        x = x * valid.unsqueeze(-1).to(x.dtype)
        block_states = []
        for block in self.blocks:
            x = block(x, valid, causal)
            block_states.append(x)
        sequence = self.final_norm(x) * valid.unsqueeze(-1).to(x.dtype)
        pooled = sequence.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.stack(block_states, dim=1), sequence, pooled


class EMATeacher(nn.Module):
    """Non-trainable EMA copy with explicit eligible/rejected-step semantics."""

    def __init__(self, online: nn.Module, *, momentum: float = 0.996) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0,1)")
        self.model = copy.deepcopy(online).float().eval()
        self.momentum = float(momentum)
        self.successful_steps = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def step_and_update(
        self,
        online: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, bool | int]:
        """Atomically reject non-finite gradients or step once then update EMA.

        Parameters with ``grad is None`` are eligible.  No optimizer or EMA
        state is touched when any existing accumulated gradient is non-finite.
        """
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if not gradients_finite:
            return {
                "gradients_finite": False,
                "step_called": False,
                "ema_updated": False,
                "successful_steps": self.successful_steps,
            }

        optimizer.step()
        online_parameters = dict(online.named_parameters())
        for name, teacher_parameter in self.model.named_parameters():
            teacher_parameter.mul_(self.momentum).add_(online_parameters[name].detach().float(), alpha=1.0 - self.momentum)
        online_buffers = dict(online.named_buffers())
        for name, teacher_buffer in self.model.named_buffers():
            source = online_buffers[name].detach()
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(self.momentum).add_(source.float(), alpha=1.0 - self.momentum)
            else:
                teacher_buffer.copy_(source)
        self.successful_steps += 1
        self.model.eval()
        return {
            "gradients_finite": True,
            "step_called": True,
            "ema_updated": True,
            "successful_steps": self.successful_steps,
        }

    def forward(self, *args, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwargs)


class SharedLatentPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(16, eps=1e-5)
        self.linear1 = nn.Linear(16, 32, bias=True)
        self.linear2 = nn.Linear(32, 16, bias=True)
        self.gelu = nn.GELU(approximate="none")
        _initialize(self)

    def predict_with_codes(self, pooled: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        z = pooled
        while z.ndim < codes.ndim:
            z = z.unsqueeze(1)
        return self.linear2(self.gelu(self.linear1(self.norm(z + codes))))

    def forward(self, pooled: torch.Tensor, recipe: str, *, k: int = K_TARGET, layers: Iterable[int] = (1, 2, 3, 4)) -> torch.Tensor:
        if recipe == "L0_EMA_POOL":
            return self.predict_with_codes(pooled, torch.zeros_like(pooled))
        positions = sinusoidal_code(torch.arange(k, device=pooled.device), dtype=pooled.dtype)
        if recipe == "L1_AVG":
            return self.predict_with_codes(pooled, positions.unsqueeze(0))
        if recipe == "L2_SEP":
            selected = list(layers)
            layer_codes = sinusoidal_code(torch.tensor([k + layer for layer in selected], device=pooled.device), dtype=pooled.dtype)
            codes = (positions[:, None, :] + layer_codes[None, :, :]) / math.sqrt(2.0)
            return self.predict_with_codes(pooled, codes.unsqueeze(0))
        raise ValueError(f"unknown recipe: {recipe}")


class NonAutoregressiveNextEventsHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(4, 16))
        self.query_norm = nn.LayerNorm(16, eps=1e-5)
        self.context_norm = nn.LayerNorm(16, eps=1e-5)
        self.cross_attention = nn.MultiheadAttention(16, 4, dropout=0.0, bias=True, batch_first=True)
        self.hidden_norm = nn.LayerNorm(16, eps=1e-5)
        self.linear1 = nn.Linear(16, 64, bias=True)
        self.linear2 = nn.Linear(64, 16, bias=True)
        self.output_norm = nn.LayerNorm(16, eps=1e-5)
        self.type_output = nn.Linear(16, 17, bias=True)
        self.time_output = nn.Linear(16, 2, bias=True)
        self.gelu = nn.GELU(approximate="none")
        _initialize(self, embedding_or_query=self.queries)

    def forward(self, context_states: torch.Tensor, prefix_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_states.ndim != 3 or prefix_mask.shape != context_states.shape[:2]:
            raise ValueError("context_states [B,N,D] and prefix_mask [B,N] required")
        q = self.queries.unsqueeze(0).expand(context_states.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(q), self.context_norm(context_states), self.context_norm(context_states),
            key_padding_mask=~prefix_mask, need_weights=False,
        )
        a = q + attended
        b = a + self.linear2(self.gelu(self.linear1(self.hidden_norm(a))))
        output = self.output_norm(b)
        return self.type_output(output), self.time_output(output), output


def parameter_count_report(
    online: J04Encoder,
    teacher: EMATeacher,
    predictor: SharedLatentPredictor,
    head: NonAutoregressiveNextEventsHead,
) -> dict[str, int]:
    total = lambda module: sum(parameter.numel() for parameter in module.parameters())
    trainable = lambda module: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    embedding = online.type_embedding.weight.numel()
    normalization = sum(p.numel() for m in online.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters(recurse=False))
    return {
        "embedding_parameters": embedding,
        "online_normalization_parameters": normalization,
        "online_encoder": total(online),
        "online_encoder_trainable": trainable(online),
        "ema_teacher": total(teacher.model),
        "ema_teacher_trainable": trainable(teacher.model),
        "predictor": total(predictor),
        "head": total(head),
        "latent_stored_total": total(online) + total(teacher.model) + total(predictor),
        "latent_trainable": trainable(online) + trainable(predictor),
        "encoder_head_trainable": trainable(online) + trainable(head),
    }
