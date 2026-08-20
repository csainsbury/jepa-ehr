from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

from clinical_jepa.arms.v0f.own_latent import (
    EMATeacher,
    J04Encoder,
    NonAutoregressiveNextEventsHead,
    SharedLatentPredictor,
    parameter_count_report,
    sinusoidal_code,
)
from clinical_jepa.eval.next_event_metrics import gaussian_interval_nll, type_cross_entropy


def test_exact_architecture_initialization_and_attention_modes():
    torch.manual_seed(20)
    encoder = J04Encoder().eval()
    assert encoder.type_embedding.weight.shape == (18, 16)
    assert torch.equal(encoder.type_embedding.weight[0], torch.zeros(16))
    assert len(encoder.blocks) == 4
    assert all(block.attention.num_heads == 4 and block.linear1.in_features == 16 and block.linear1.out_features == 64 for block in encoder.blocks)
    assert all(block.attention.dropout == 0.0 and block.gelu.approximate == "none" for block in encoder.blocks)
    for module in encoder.modules():
        if isinstance(module, nn.LayerNorm):
            assert module.eps == 1e-5
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            biases = [p for name, p in module.named_parameters(recurse=False) if "bias" in name]
            assert all(torch.equal(bias, torch.zeros_like(bias)) for bias in biases)
    code = sinusoidal_code(torch.tensor([0, 1]))
    torch.testing.assert_close(code[0, 0::2], torch.zeros(8))
    torch.testing.assert_close(code[0, 1::2], torch.ones(8))

    ids = torch.tensor([[2, 3, 4, 5]])
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    causal, _, _ = encoder(ids, times, causal=True)
    bidirectional, _, _ = encoder(ids, times, causal=False)
    changed = ids.clone(); changed[0, 3] = 17
    causal_changed, _, _ = encoder(changed, times, causal=True)
    bidirectional_changed, _, _ = encoder(changed, times, causal=False)
    torch.testing.assert_close(causal[:, :, :3], causal_changed[:, :, :3], atol=1e-6, rtol=0)
    assert not torch.allclose(bidirectional[:, :, :3], bidirectional_changed[:, :, :3])


def test_parameter_totals_and_component_report():
    online = J04Encoder()
    teacher = EMATeacher(online)
    predictor = SharedLatentPredictor()
    head = NonAutoregressiveNextEventsHead()
    report = parameter_count_report(online, teacher, predictor, head)
    assert report == {
        "embedding_parameters": 288,
        "online_normalization_parameters": 320,
        "online_encoder": 13_504,
        "online_encoder_trainable": 13_504,
        "ema_teacher": 13_504,
        "ema_teacher_trainable": 0,
        "predictor": 1_104,
        "head": 3_731,
        "latent_stored_total": 28_112,
        "latent_trainable": 14_608,
        "encoder_head_trainable": 17_235,
    }


def test_predictor_l0_l1_l2_shapes_same_parameters_fixed_codes():
    torch.manual_seed(21)
    predictor = SharedLatentPredictor()
    pooled = torch.randn(3, 16)
    count = sum(p.numel() for p in predictor.parameters())
    assert predictor(pooled, "L0_EMA_POOL").shape == (3, 16)
    assert predictor(pooled, "L1_AVG").shape == (3, 4, 16)
    assert predictor(pooled, "L2_SEP", layers=[1, 2, 3, 4]).shape == (3, 4, 4, 16)
    assert sum(p.numel() for p in predictor.parameters()) == count == 1_104
    assert not any("code" in name for name, _ in predictor.named_parameters())
    assert not list(predictor.buffers())


def test_head_shapes_class_order_and_zero_interval_gradients():
    torch.manual_seed(22)
    head = NonAutoregressiveNextEventsHead()
    context = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    logits, time_params, output = head(context, mask)
    assert logits.shape == (2, 4, 17) and time_params.shape == (2, 4, 2) and output.shape == (2, 4, 16)
    manual_logits = torch.arange(17.0).reshape(1, 1, 17)
    losses = type_cross_entropy(manual_logits, torch.tensor([[1]]), torch.tensor([[True]]))
    torch.testing.assert_close(losses, -torch.log_softmax(manual_logits, -1)[0, 0, 0].reshape(1, 1))
    losses_last = type_cross_entropy(manual_logits, torch.tensor([[17]]), torch.tensor([[True]]))
    torch.testing.assert_close(losses_last, -torch.log_softmax(manual_logits, -1)[0, 0, 16].reshape(1, 1))
    mean = time_params[..., 0]
    raw_scale = time_params[..., 1]
    zero_y = torch.zeros_like(mean)
    nll = gaussian_interval_nll(zero_y, mean, raw_scale).mean()
    assert torch.isfinite(nll)
    nll.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


class _BufferedToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.unused = nn.Parameter(torch.tensor([5.0]))
        self.register_buffer("floating", torch.tensor([2.0]))
        self.register_buffer("integer", torch.tensor([3], dtype=torch.int64))


def _assert_state_dict_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def _assert_optimizer_state_exact(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for key, value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][key]
            if torch.is_tensor(value):
                assert torch.equal(value, expected_value), (parameter_id, key)
            else:
                assert value == expected_value


def test_f9_atomic_eligible_step_updates_exact_optimizer_ema_buffers_and_counter():
    online = _BufferedToy()
    teacher = EMATeacher(online, momentum=0.5)
    optimizer = torch.optim.AdamW(online.parameters(), lr=0.1, weight_decay=0.0)
    online.weight.square().sum().backward()
    assert online.unused.grad is None  # None-grad parameters remain eligible.
    before_teacher = copy.deepcopy(teacher.model.state_dict())
    online.floating.fill_(6.0); online.integer.fill_(9)

    metadata = teacher.step_and_update(online, optimizer)

    assert metadata == {
        "gradients_finite": True, "step_called": True, "ema_updated": True, "successful_steps": 1,
    }
    torch.testing.assert_close(online.weight, torch.tensor([0.9]))
    optimizer_state = optimizer.state[online.weight]
    assert optimizer_state["step"].item() == 1
    torch.testing.assert_close(optimizer_state["exp_avg"], torch.tensor([0.2]))
    torch.testing.assert_close(optimizer_state["exp_avg_sq"], torch.tensor([0.004]))
    assert online.unused not in optimizer.state
    expected_weight = 0.5 * before_teacher["weight"] + 0.5 * online.weight.detach()
    expected_float = 0.5 * before_teacher["floating"] + 0.5 * online.floating
    torch.testing.assert_close(teacher.model.weight, expected_weight)
    torch.testing.assert_close(teacher.model.floating, expected_float)
    assert teacher.model.integer.item() == 9
    assert teacher.successful_steps == 1
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert not teacher.model.training
    assert not hasattr(teacher, "update")


def test_f9_atomic_lr_zero_step_still_advances_optimizer_and_ema_once():
    online = _BufferedToy()
    teacher = EMATeacher(online, momentum=0.5)
    optimizer = torch.optim.AdamW(online.parameters(), lr=0.0, weight_decay=0.0)
    online.weight.square().sum().backward()
    online_before = copy.deepcopy(online.state_dict())
    teacher_before = copy.deepcopy(teacher.model.state_dict())

    metadata = teacher.step_and_update(online, optimizer)

    assert metadata == {
        "gradients_finite": True, "step_called": True, "ema_updated": True, "successful_steps": 1,
    }
    _assert_state_dict_exact(online.state_dict(), online_before)
    _assert_state_dict_exact(teacher.model.state_dict(), teacher_before)
    assert optimizer.state[online.weight]["step"].item() == 1


def test_f9b_real_nonfinite_gradient_rejects_full_step_and_state_transition():
    online = _BufferedToy()
    teacher = EMATeacher(online, momentum=0.5)
    optimizer = torch.optim.AdamW(online.parameters(), lr=0.1, weight_decay=0.0)
    online.weight.square().sum().backward()
    teacher.step_and_update(online, optimizer)
    optimizer.zero_grad(set_to_none=True)
    (online.weight * torch.tensor(float("inf"))).sum().backward()
    assert bool(torch.isinf(online.weight.grad).all())
    online_snapshot = copy.deepcopy(online.state_dict())
    teacher_snapshot = copy.deepcopy(teacher.model.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    successful_steps = teacher.successful_steps

    metadata = teacher.step_and_update(online, optimizer)

    assert metadata == {
        "gradients_finite": False, "step_called": False, "ema_updated": False,
        "successful_steps": successful_steps,
    }
    _assert_state_dict_exact(online.state_dict(), online_snapshot)
    _assert_state_dict_exact(teacher.model.state_dict(), teacher_snapshot)
    _assert_optimizer_state_exact(optimizer.state_dict(), optimizer_snapshot)
    assert teacher.successful_steps == successful_steps


def test_same_seed_paired_encoder_and_predictor_initialization_is_byte_identical():
    torch.manual_seed(23); encoder_a = J04Encoder()
    torch.manual_seed(23); encoder_b = J04Encoder()
    torch.manual_seed(24); predictor_a = SharedLatentPredictor()
    torch.manual_seed(24); predictor_b = SharedLatentPredictor()
    _assert_state_dict_exact(encoder_a.state_dict(), encoder_b.state_dict())
    _assert_state_dict_exact(predictor_a.state_dict(), predictor_b.state_dict())
    encoder_a_bytes = b"".join(value.detach().contiguous().numpy().tobytes() for value in encoder_a.state_dict().values())
    encoder_b_bytes = b"".join(value.detach().contiguous().numpy().tobytes() for value in encoder_b.state_dict().values())
    predictor_a_bytes = b"".join(value.detach().contiguous().numpy().tobytes() for value in predictor_a.state_dict().values())
    predictor_b_bytes = b"".join(value.detach().contiguous().numpy().tobytes() for value in predictor_b.state_dict().values())
    assert encoder_a_bytes == encoder_b_bytes and predictor_a_bytes == predictor_b_bytes


def test_teacher_forward_is_eval_float32_and_no_device_auto_selection():
    online = J04Encoder()
    teacher = EMATeacher(online)
    online.train()
    ids, times = torch.tensor([[2, 3]]), torch.tensor([[0.0, 1.0]])
    blocks, sequence, pooled = teacher(ids, times, causal=True)
    assert blocks.dtype == sequence.dtype == pooled.dtype == torch.float32
    assert not teacher.model.training and all(not value.requires_grad for value in (blocks, sequence, pooled))
    production_modules = [
        __import__("clinical_jepa.arms.v0f.own_latent", fromlist=["*"]),
        __import__("clinical_jepa.targets.next_event_contract", fromlist=["*"]),
        __import__("clinical_jepa.eval.next_event_metrics", fromlist=["*"]),
    ]
    forbidden = ["cuda" + ".", "cuda" + "(", "device(" + "\"cuda", "torch.save", "torch.load"]
    for module in production_modules:
        source = inspect.getsource(module).lower()
        assert not any(token in source for token in forbidden), module.__name__
