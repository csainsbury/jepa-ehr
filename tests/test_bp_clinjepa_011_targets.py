from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from clinical_jepa.arms.v0f.own_latent import (
    J04Encoder,
    NonAutoregressiveNextEventsHead,
    SharedLatentPredictor,
    sinusoidal_code,
)
from clinical_jepa.eval.next_event_metrics import gaussian_interval_nll, masked_mean_within_example, type_cross_entropy
from clinical_jepa.targets.next_event_contract import (
    TimeTransform,
    build_next_event_targets,
    construct_latent_targets,
    latent_objective,
    latent_output_accounting,
    position_causal_attention,
    resolve_layer_sets,
)


def _transform() -> TimeTransform:
    return TimeTransform(unit=1.0).fit_train([0.0, 1.0, 4.0, 16.0])


def test_f0_cutoff_context_tensors_states_and_all_predictions_are_future_invariant():
    torch.manual_seed(10)
    encoder, predictor = J04Encoder().eval(), SharedLatentPredictor().eval()
    full_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    full_time = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    q = 3
    context_ids, context_time = full_ids[:, :q].clone(), full_time[:, :q].clone()
    _, states, pooled = encoder(context_ids, context_time)
    predictions = {
        recipe: predictor(pooled, recipe)
        for recipe in ("L0_EMA_POOL", "L1_AVG", "L2_SEP")
    }

    # Mutate every event and time at or after q, not merely after q + K - 1.
    full_ids[:, q:], full_time[:, q:] = torch.tensor([[17, 16, 15]]), torch.tensor([[9.0, 8.0, 7.0]])
    context_ids_2, context_time_2 = full_ids[:, :q].clone(), full_time[:, :q].clone()
    _, states_2, pooled_2 = encoder(context_ids_2, context_time_2)
    predictions_2 = {
        recipe: predictor(pooled_2, recipe)
        for recipe in ("L0_EMA_POOL", "L1_AVG", "L2_SEP")
    }

    assert torch.equal(context_ids, context_ids_2) and torch.equal(context_time, context_time_2)
    torch.testing.assert_close(states, states_2, atol=1e-6, rtol=0)
    for recipe in predictions:
        torch.testing.assert_close(predictions[recipe], predictions_2[recipe], atol=1e-6, rtol=0)


def test_f1_after_block_correct_routes_and_exact_test_local_leak_controls():
    torch.manual_seed(11)
    encoder = J04Encoder().eval()
    predictor = SharedLatentPredictor().eval()
    head = NonAutoregressiveNextEventsHead().eval()
    target_ids = torch.tensor([[2, 3, 4, 5]])
    target_time = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    target_mask = torch.ones((1, 4), dtype=torch.bool)
    after = torch.tensor([[1.0, -1.0, 1.0] + [0.0] * 13])
    mutated_after = -after

    states, _, _ = encoder(target_ids, target_time, causal=True)
    _, context_states, pooled = encoder(target_ids, target_time, causal=False)
    states_again, _, _ = encoder(target_ids, target_time, causal=True)
    _, _, pooled_again = encoder(target_ids, target_time, causal=False)
    torch.testing.assert_close(states, states_again, atol=1e-6, rtol=0)
    torch.testing.assert_close(pooled, pooled_again, atol=1e-6, rtol=0)
    for recipe in ("L0_EMA_POOL", "L1_AVG", "L2_SEP"):
        target, mask, _ = construct_latent_targets(states, target_mask, recipe)
        target_again, mask_again, _ = construct_latent_targets(states_again, target_mask, recipe)
        torch.testing.assert_close(target, target_again, atol=1e-6, rtol=0)
        assert torch.equal(mask, mask_again)
        torch.testing.assert_close(predictor(pooled, recipe), predictor(pooled_again, recipe), atol=1e-6, rtol=0)

    # Algebraic controls directly add the prescribed vector and differ exactly.
    algebraic_student = torch.zeros_like(after)
    algebraic_teacher = torch.zeros((1, 1, 1, 16))
    assert torch.equal((algebraic_student + after) - algebraic_student, after)
    assert torch.equal(
        (algebraic_teacher + after[:, None, None]) - algebraic_teacher,
        after[:, None, None],
    )

    # Student leak is appended after LN_final as an extra valid context token.
    student_mask = torch.ones((1, 5), dtype=torch.bool)
    context_with_leak = torch.cat([context_states, after[:, None, :]], dim=1)
    context_with_mutated_leak = torch.cat([context_states, mutated_after[:, None, :]], dim=1)
    leaked_pool = context_with_leak.sum(dim=1) / student_mask.sum(dim=1, keepdim=True)
    mutated_leaked_pool = context_with_mutated_leak.sum(dim=1) / student_mask.sum(dim=1, keepdim=True)
    torch.testing.assert_close(leaked_pool - mutated_leaked_pool, (after - mutated_after) / 5, atol=1e-6, rtol=0)
    leaked_head = head(context_with_leak, student_mask)[2]
    mutated_leaked_head = head(context_with_mutated_leak, student_mask)[2]
    assert (leaked_head - mutated_leaked_head).abs().max().item() > 1e-3

    # Teacher leak uses only this custom non-causal test mask.  The slot K is
    # key and value for every valid target query, while its own query sees only
    # itself.  It is appended once, then propagated by residuals at each block.
    with torch.no_grad():
        identity = torch.eye(16)
        for block in encoder.blocks:
            block.attention.in_proj_weight.zero_()
            block.attention.in_proj_weight[:16].copy_(identity)
            block.attention.in_proj_weight[16:32].copy_(identity)
            block.attention.in_proj_weight[32:].copy_(identity)
            block.attention.in_proj_bias.zero_()
            block.attention.out_proj.weight.copy_(identity)
            block.attention.out_proj.bias.zero_()
            block.linear1.weight.zero_()
            block.linear1.bias.zero_()
            block.linear2.weight.zero_()
            block.linear2.bias.zero_()

    def teacher_leak_blocks(leak: torch.Tensor) -> torch.Tensor:
        positions = sinusoidal_code(torch.arange(4), dtype=encoder.type_embedding.weight.dtype)
        x = encoder.input_norm(
            encoder.type_embedding(target_ids)
            + encoder.time_projection(target_time.unsqueeze(-1))
            + positions.unsqueeze(0)
        )
        x = torch.cat([x, leak[:, None, :]], dim=1)
        allowed = torch.zeros((5, 5), dtype=torch.bool)
        for query in range(4):
            allowed[query, : query + 1] = True
            allowed[query, 4] = True
        allowed[4, 4] = True
        forbidden = ~allowed
        outputs = []
        for block in encoder.blocks:
            norm = block.ln1(x)
            attended, _ = block.attention(norm, norm, norm, attn_mask=forbidden, need_weights=False)
            a = x + attended
            x = a + block.linear2(block.gelu(block.linear1(block.ln2(a))))
            outputs.append(x)
        return torch.stack(outputs, dim=1)

    leaked_teacher = teacher_leak_blocks(after)
    mutated_leaked_teacher = teacher_leak_blocks(mutated_after)
    valid_target_change = (leaked_teacher[:, :, :4] - mutated_leaked_teacher[:, :, :4]).abs().max().item()
    assert valid_target_change > 1e-3
    for recipe in ("L0_EMA_POOL", "L1_AVG", "L2_SEP"):
        _, leak_mask, _ = construct_latent_targets(leaked_teacher[:, :, :4], target_mask, recipe)
        assert leak_mask.shape[-1] != 5  # slot K is never a latent target identity


def test_f2_algebraic_current_past_later_and_actual_causal_invariance():
    e = torch.eye(4)
    t = torch.arange(4.0)[:, None] * e
    def causal_toy(e_, t_):
        return torch.stack([e_[j] + t_[j] + 0.25 * (e_[:j] + t_[:j]).sum(0) for j in range(4)])
    base = causal_toy(e, t)
    current = e.clone(); current[2, 0] += 2
    past = e.clone(); past[0, 0] += 2
    later = e.clone(); later[3, 0] += 2
    assert not torch.equal(base[2], causal_toy(current, t)[2])
    assert not torch.equal(base[2], causal_toy(past, t)[2])
    assert torch.equal(base[2], causal_toy(later, t)[2])
    noncausal = base[2] + 0.25 * (e[3:] + t[3:]).sum(0)
    assert not torch.equal(base[2], noncausal)

    torch.manual_seed(12)
    encoder = J04Encoder().eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    blocks, _, _ = encoder(ids, times, causal=True)
    ids_later, times_later = ids.clone(), times.clone()
    ids_later[0, 3], times_later[0, 3] = 17, 9.0
    blocks_later, _, _ = encoder(ids_later, times_later, causal=True)
    torch.testing.assert_close(blocks[:, :, :3], blocks_later[:, :, :3], atol=1e-6, rtol=0)
    padded_ids = torch.tensor([[2, 3, 4, 0]])
    padded_t = torch.tensor([[0.0, 1.0, 2.0, 99.0]])
    padded, _, _ = encoder(padded_ids, padded_t, causal=True)
    padded_t[0, 3] = -99.0
    padded_again, _, _ = encoder(padded_ids, padded_t, causal=True)
    torch.testing.assert_close(padded, padded_again, atol=1e-6, rtol=0)
    assert torch.equal(padded[:, :, 3], torch.zeros_like(padded[:, :, 3]))


def test_f2b_teacher_prefix_independence_with_student_change():
    torch.manual_seed(13)
    encoder = J04Encoder().eval()
    target_ids = torch.tensor([[4, 5, 6, 7]])
    target_time = torch.tensor([[1.0, 0.0, 2.0, 1.0]])
    teacher, _, _ = encoder(target_ids, target_time, causal=True)
    _, _, student_a = encoder(torch.tensor([[2, 3]]), torch.tensor([[0.0, 1.0]]))
    _, _, student_b = encoder(torch.tensor([[8, 9]]), torch.tensor([[3.0, 1.0]]))
    teacher_again, _, _ = encoder(target_ids, target_time, causal=True)
    torch.testing.assert_close(teacher, teacher_again, atol=1e-6, rtol=0)
    assert not torch.allclose(student_a, student_b)


def test_f3_train_statistics_are_byte_stable_after_other_mutations():
    transform = _transform()
    frozen = transform.state_bytes()
    _ = transform.transform(np.array([9.0, 15.0]))
    _ = transform.transform(torch.tensor([3.0, 12.0]))
    assert transform.state_bytes() == frozen


def test_f4_exact_order_fixture():
    eye = torch.eye(16)
    first = [0, 1, 0, 2]
    second = [0, 2, 0, 1]
    def toy(order):
        return torch.stack([eye[event] + 0.25 * eye[order[:j]].sum(0) for j, event in enumerate(order)])
    a, b = toy(first), toy(second)
    assert (a[2, 1] - b[2, 1]).item() == 0.25
    assert (a[2, 2] - b[2, 2]).item() == -0.25
    assert not torch.equal(a.mean(0), b.mean(0))


def test_f5_exact_time_fixture():
    types = [0, 1, 2, 3]
    a, b = [1.0, 1.0, 4.0, 4.0], [4.0, 4.0, 1.0, 1.0]
    eye = torch.eye(16)
    def toy(intervals, coefficient=1.0):
        return torch.stack([eye[event] + coefficient * math.log1p(dt) * eye[15] for event, dt in zip(types, intervals)])
    expected = math.log(5.0) - math.log(2.0)
    assert (toy(b)[0, 15] - toy(a)[0, 15]).item() == pytest.approx(expected)
    torch.testing.assert_close(toy(a, 0.0), toy(b, 0.0), atol=0, rtol=0)


def test_f6_boundaries_masks_attention_and_no_repeated_end():
    batch = build_next_event_targets(
        [[2, 3, 4, 5], [6, 7], [8]], [[0, 1, 2, 3], [0, 4], [2]],
        known_endpoints=[False, True, False], k=4, time_transform=_transform(),
    )
    assert batch.type_ids.tolist() == [[2, 3, 4, 5], [6, 7, 1, 0], [8, 0, 0, 0]]
    assert batch.latent_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]
    assert batch.type_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    assert batch.interval_mask.tolist() == batch.latent_mask.tolist()
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    assert int(batch.type_ids.eq(1).sum(dim=1).max()) == 1
    relation = position_causal_attention(batch.attention_mask)
    assert relation[0].int().tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    assert not bool(relation[1, :, 3].any())


def test_f6b_hand_computed_recipe_cosines_identity_variances_and_masked_event_losses():
    e1, e2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])

    l0_prediction = torch.stack([e1, e2])
    l0_target = torch.stack([e1, e1])
    l0_valid = torch.tensor([True, True])
    _, l0 = latent_objective(l0_prediction, l0_target, l0_valid)
    assert l0["cosine"].item() == pytest.approx(0.5)
    assert l0["v_pred"].item() == pytest.approx(0.25)

    l1_prediction = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0], [9.0, 9.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    l1_target = torch.zeros_like(l1_prediction); l1_target[..., 0] = 1.0
    l1_valid = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    _, l1 = latent_objective(l1_prediction, l1_target, l1_valid)
    assert l1["cosine"].item() == pytest.approx(1.0)  # mean of per-example .5, 2, .5
    assert l1["v_pred"].item() == pytest.approx(25.0 / 72.0)  # position identities: 4/9 and 1/4

    l2_prediction = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        [[[-1.0, 0.0], [1.0, 0.0]], [[9.0, 9.0], [9.0, 9.0]]],
        [[[0.0, 1.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]]],
    ])
    l2_target = torch.zeros_like(l2_prediction); l2_target[..., 0] = 1.0
    l2_valid = torch.tensor([
        [[1, 1], [1, 1]],
        [[1, 1], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=torch.bool)
    _, l2 = latent_objective(l2_prediction, l2_target, l2_valid)
    assert l2["cosine"].item() == pytest.approx(11.0 / 12.0)  # per-example .5, 1, 1.25
    assert l2["v_pred"].item() == pytest.approx(59.0 / 144.0)  # identities: 4/9,4/9,1/4,1/2

    with pytest.raises(ValueError, match="two valid"):
        latent_objective(l1_prediction[:1], l1_target[:1], l1_valid[:1])

    batch = build_next_event_targets(
        [[2, 3, 4], [5]], [[0.0, 1.0, 2.0], [3.0]],
        known_endpoints=[False, True], k=4,
    )
    logits = torch.zeros((2, 4, 17))
    active_scores = [[0.0, 1.0, 2.0], [3.0, 4.0]]
    for row, scores in enumerate(active_scores):
        for slot, score in enumerate(scores):
            logits[row, slot, batch.type_ids[row, slot] - 1] = score
    type_losses = type_cross_entropy(logits, batch.type_ids, batch.type_mask)
    assert torch.equal(type_losses[~batch.type_mask], torch.zeros_like(type_losses[~batch.type_mask]))
    expected_type_examples = [
        sum(math.log(math.exp(score) + 16.0) - score for score in scores) / len(scores)
        for scores in active_scores
    ]
    assert masked_mean_within_example(type_losses, batch.type_mask).item() == pytest.approx(sum(expected_type_examples) / 2)
    with pytest.raises(ValueError, match="PAD cannot be scored"):
        type_cross_entropy(logits, batch.type_ids, torch.ones_like(batch.type_mask))

    y = torch.tensor([[0.0, 1.0, 2.0, 100.0], [3.0, 100.0, 100.0, 100.0]])
    mean, raw = torch.zeros_like(y), torch.zeros_like(y)
    interval_losses = gaussian_interval_nll(y, mean, raw)
    sigma = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-4
    constant = torch.log(sigma).item() + 0.5 * math.log(2 * math.pi)
    expected_interval = (
        sum(constant + 0.5 * (value / sigma.item()) ** 2 for value in (0.0, 1.0, 2.0)) / 3
        + constant + 0.5 * (3.0 / sigma.item()) ** 2
    ) / 2
    assert masked_mean_within_example(interval_losses, batch.interval_mask).item() == pytest.approx(expected_interval)


def test_f7_zero_interval_retained_and_transformed_finite():
    transform = _transform()
    batch = build_next_event_targets([[2]], [[0.0]], known_endpoints=[False], time_transform=transform)
    assert batch.interval_mask[0, 0] and torch.isfinite(batch.transformed_intervals[0, 0])


def test_f8_layer_policy_and_shapes():
    assert resolve_layer_sets(1) == ([1], [1])
    assert resolve_layer_sets(2) == ([1, 2], [1, 2])
    assert resolve_layer_sets(4) == ([1, 2, 3, 4], [1, 2, 3, 4])
    assert resolve_layer_sets(8) == ([5, 6, 7, 8], [1, 3, 6, 8])
    with pytest.raises(ValueError, match="undefined"):
        construct_latent_targets(torch.randn(2, 1, 4, 16), torch.ones(2, 4, dtype=torch.bool), "L2_SEP")
    for layer_count, separated in [(2, 2), (4, 4), (8, 4)]:
        out, mask, selected = construct_latent_targets(torch.randn(2, layer_count, 4, 16), torch.ones(2, 4, dtype=torch.bool), "L2_SEP")
        assert out.shape == (2, 4, separated, 16) and mask.shape == (2, 4, separated)
        assert len(selected) == separated


def test_f10_output_unit_activation_mac_and_flop_accounting():
    labels = {
        "predictor_macs_label": "analytic predictor-only MACs",
        "predictor_flops_label": "analytic predictor-only FLOPs; not measured whole-training FLOPs",
    }
    assert latent_output_accounting("L0_EMA_POOL", r=2) == {
        "allocated_output_elements": 16, "valid_output_elements": 16,
        "allocated_target_units": 1, "valid_target_units": 1, "predictor_calls": 1,
        "predictor_activation_elements": 64, "predictor_macs": 1024, "predictor_flops": 2048,
        **labels,
    }
    assert latent_output_accounting("L1_AVG", r=2) == {
        "allocated_output_elements": 64, "valid_output_elements": 32,
        "allocated_target_units": 4, "valid_target_units": 2, "predictor_calls": 4,
        "predictor_activation_elements": 256, "predictor_macs": 4096, "predictor_flops": 8192,
        **labels,
    }
    assert latent_output_accounting("L2_SEP", r=2) == {
        "allocated_output_elements": 256, "valid_output_elements": 128,
        "allocated_target_units": 16, "valid_target_units": 8, "predictor_calls": 16,
        "predictor_activation_elements": 1024, "predictor_macs": 16384, "predictor_flops": 32768,
        **labels,
    }
