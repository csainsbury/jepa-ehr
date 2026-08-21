from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import clinical_jepa.eval.j04c_stage1 as stage1
from clinical_jepa.arms.v0f.own_latent import EMATeacher, NonAutoregressiveNextEventsHead
from clinical_jepa.eval.j04c_falsifier import (
    CAL_OOD, PROBE_FIT, TRAIN, _state_dict_bytes, fit_stage0_time_transform,
    full_panel_negative_control, generate_factor_split, make_initialized_teacher,
    nuisance_only, nuisance_stratified_permutations,
)
from clinical_jepa.eval.j04c_stage1 import (
    CONDITION_NAMES, THRESHOLD_DIGEST, TrainedCondition, _adamw, _fresh_encoder,
    _fresh_head, _fresh_predictor, append_student_leak, direct_objective,
    freeze_encoder, frozen_representations, head_indices, hierarchical_complete_rule,
    one_run_negative_control, optimizer_membership, pretraining_indices, probe_indices,
    tiny_pretraining_indices,
)
from clinical_jepa.eval.next_event_metrics import gaussian_interval_nll, type_cross_entropy


def test_s1_00_exact_scope_safe_module_and_no_forbidden_split_import_or_call():
    module_source = inspect.getsource(stage1)
    runner_source = (Path(__file__).parents[1] / "scripts" / "bp_clinjepa_011_j04c_stage1.py").read_text(encoding="utf-8")
    forbidden_ids = {"open", "write_text", "write_bytes", "save", "load", "checkpoint", "cuda", "DataLoader"}
    module_tree = ast.parse(module_source)
    runner_tree = ast.parse(runner_source)
    for tree in (module_tree, runner_tree):
        imported_roots = {alias.name.split(".")[0] for node in ast.walk(tree)
                          if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not imported_roots.intersection({"subprocess", "socket", "requests", "urllib", "pathlib"})
        executable_ids = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        executable_ids.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        assert not executable_ids.intersection(forbidden_ids)
    module_names = {node.id for node in ast.walk(module_tree) if isinstance(node, ast.Name)}
    module_names.update(node.attr for node in ast.walk(module_tree) if isinstance(node, ast.Attribute))
    assert not module_names.intersection({"ID_DEV", "OOD_DEV"})

    calls = [node for node in ast.walk(runner_tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "generate_factor_split"]
    assert len(calls) == 5
    direct_signatures = {(node.args[0].id, node.args[1].id, node.args[2].value)
                         for node in calls if isinstance(node.args[0], ast.Name) and node.args[0].id == "GENERATOR_SEED"}
    assert direct_signatures == {
        ("GENERATOR_SEED", "TRAIN", 8192), ("GENERATOR_SEED", "PROBE_FIT", 2048),
        ("GENERATOR_SEED", "CAL_ID", 2048), ("GENERATOR_SEED", "CAL_OOD", 2048),
    }
    comprehensions = [node for node in ast.walk(runner_tree) if isinstance(node, ast.GeneratorExp)
                      and isinstance(node.elt, ast.Call) and isinstance(node.elt.func, ast.Name)
                      and node.elt.func.id == "generate_factor_split"]
    assert len(comprehensions) == 1
    comprehension = comprehensions[0]
    assert (comprehension.elt.args[0].id, comprehension.elt.args[1].id, comprehension.elt.args[2].value) == ("seed", "CAL_OOD", 2048)
    assert len(comprehension.generators) == 1 and comprehension.generators[0].target.id == "seed"
    assert tuple(item.value for item in comprehension.generators[0].iter.elts) == (1101, 1102, 1103)


def test_s1_01_exact_zero_based_schedules_contiguous_and_counts():
    pre = list(pretraining_indices(2100))
    assert len(pre) == 2000
    for epoch in (0, 1, 30, 31):
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2100, epoch, 6101]))).permutation(8192)
        start = epoch * 64
        count = 64 if epoch <= 30 else 16
        np.testing.assert_array_equal(np.concatenate(pre[start:start + count]), expected[:count * 128])
    for factor in range(3):
        batches = list(probe_indices(2100, factor))
        assert len(batches) == 250
        expected0 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2100, factor, 0, 6201]))).permutation(2048)
        np.testing.assert_array_equal(np.concatenate(batches[:8]), expected0)
        expected31 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2100, factor, 31, 6201]))).permutation(2048)
        np.testing.assert_array_equal(np.concatenate(batches[-2:]), expected31[:512])
    heads = list(head_indices(2100)); assert len(heads) == 250
    expected15 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2100, 15, 6301]))).permutation(2048)
    np.testing.assert_array_equal(np.concatenate(heads[-10:]), expected15[:1280])


def test_s1_01_tiny_schedule_has_no_drop_repeat_per_epoch():
    batches = list(tiny_pretraining_indices(9, n=12, batch_size=3, epochs=2))
    assert len(batches) == 8
    np.testing.assert_array_equal(np.sort(np.concatenate(batches[:4])), np.arange(12))
    np.testing.assert_array_equal(np.sort(np.concatenate(batches[4:])), np.arange(12))


def test_s1_02_component_order_bytes_and_exact_optimizer_group():
    encoder_a, predictor_a, head_a = _fresh_encoder(2100), _fresh_predictor(2100), _fresh_head(2100)
    head_b, predictor_b, encoder_b = _fresh_head(2100), _fresh_predictor(2100), _fresh_encoder(2100)
    assert _state_dict_bytes(encoder_a) == _state_dict_bytes(encoder_b)
    assert _state_dict_bytes(predictor_a) == _state_dict_bytes(predictor_b)
    assert _state_dict_bytes(head_a) == _state_dict_bytes(head_b)
    parameters = list(encoder_a.parameters()) + list(predictor_a.parameters())
    optimizer = _adamw(parameters, lr=3e-4, weight_decay=1e-4)
    assert optimizer_membership((encoder_a, predictor_a), optimizer)
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-4 and group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 1e-4


def test_s1_03_latent_source_has_prefix_teacher_target_and_only_ema_step():
    source = inspect.getsource(stage1.train_latent_condition)
    assert "_prefix_inputs(train" in source
    assert "train.target_type_ids" in source and "teacher(target_ids, target_times, causal=True)" in source
    assert "construct_latent_targets" in source and "latent_objective" in source
    assert "teacher.step_and_update(encoder, optimizer)" in source
    assert "optimizer.step()" not in source
    encoder = torch.nn.Linear(2, 2); teacher = EMATeacher(encoder)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    optimizer.zero_grad(); encoder(torch.ones(2, 2)).sum().backward()
    metadata = teacher.step_and_update(encoder, optimizer)
    assert metadata["step_called"] and metadata["ema_updated"] and metadata["successful_steps"] == 1


def test_s1_04_direct_objective_is_hand_mean_example_ce_plus_all_four_nll():
    torch.manual_seed(3)
    logits = torch.randn(3, 4, 17)
    means = torch.randn(3, 4); scales = torch.randn(3, 4)
    ids = torch.tensor([[13, 13, 14, 15], [15, 14, 13, 13], [13, 14, 14, 15]])
    targets = torch.tensor([[1., 1., 4., 4.], [4., 4., 1., 1.], [1., 1., 4., 4.]])
    loss, parts = direct_objective(logits, means, scales, ids, targets)
    ce = type_cross_entropy(logits, ids, torch.ones_like(ids, dtype=torch.bool))
    nll = gaussian_interval_nll(targets, means, scales)
    assert torch.equal(loss, (ce + nll).mean(dim=1).mean())
    assert torch.equal(parts["type"], ce.mean(dim=1).mean())
    assert torch.equal(parts["interval"], nll.mean(dim=1).mean())
    assert nll.numel() == 12


def test_s1_05_controls_exact_nonmutating_time_free_zeros_are_direct_model_inputs():
    split = generate_factor_split(1100, TRAIN, 32); transform = fit_stage0_time_transform(split)
    before = split.prefix_type_ids.copy(), split.prefix_intervals.copy(), split.target_intervals.copy()
    ids, times = stage1._prefix_inputs(split, transform, "C0_TIME_FREE")
    assert torch.count_nonzero(times) == 0
    assert torch.equal(ids, torch.as_tensor(split.prefix_type_ids))
    controlled = nuisance_only(split)
    cids, ctimes = stage1._prefix_inputs(split, transform, "C0_NUISANCE_ONLY")
    assert torch.equal(cids, torch.as_tensor(controlled.prefix_type_ids))
    np.testing.assert_allclose(ctimes.numpy(), transform.transform(controlled.prefix_intervals).astype(np.float32))
    np.testing.assert_array_equal(split.prefix_type_ids, before[0]); np.testing.assert_array_equal(split.prefix_intervals, before[1]); np.testing.assert_array_equal(split.target_intervals, before[2])


def test_s1_05_student_leak_single_append_after_encoder_and_pooled_over_eight():
    H = torch.zeros((2, 7, 16)); bits = np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    result = append_student_leak(H, bits)
    assert result.H.shape == (2, 8, 16)
    assert torch.equal(result.H[:, :7], H)
    expected = torch.tensor([[-1., 1., -1.], [1., -1., 1.]])
    assert torch.equal(result.H[:, 7, :3], expected)
    assert torch.count_nonzero(result.H[:, 7, 3:]) == 0
    assert torch.allclose(result.z, result.H.mean(dim=1))


def test_s1_07_frozen_representation_does_not_change_encoder_bytes_or_grads():
    split = generate_factor_split(1100, PROBE_FIT, 16); transform = fit_stage0_time_transform(split)
    encoder, teacher = make_initialized_teacher(2100)
    condition = TrainedCondition("L1_AVG", encoder, teacher, _fresh_predictor(2100), {})
    freeze_encoder(condition); before = _state_dict_bytes(encoder)
    reps = frozen_representations(condition, split, transform)
    assert reps.H.shape == (16, 7, 16) and reps.z.shape == (16, 16)
    assert _state_dict_bytes(encoder) == before and all(p.grad is None for p in encoder.parameters())
    assert not reps.H.requires_grad and not reps.z.requires_grad


def test_s1_08_readout_initializations_fresh_and_hard_prediction_includes_zero():
    probes_a, head_a = stage1.make_readout_initializations(2100)
    probes_b, head_b = stage1.make_readout_initializations(2100)
    assert all(a.weight.data_ptr() != b.weight.data_ptr() for a, b in zip(probes_a, probes_b))
    assert head_a.queries.data_ptr() != head_b.queries.data_ptr()
    logits = torch.tensor([-0.1, 0.0, 0.2])
    assert logits.ge(0).to(torch.uint8).tolist() == [0, 1, 1]


def test_s1_09_one_run_fixed_prediction_null_and_existing_full_panel_nonidentical_vectors():
    split = generate_factor_split(1100, CAL_OOD, 2048)
    predictions = split.S[:, 0].copy(); before = predictions.copy()
    report = one_run_negative_control(predictions, split.S[:, 0], split.N, 0)
    np.testing.assert_array_equal(predictions, before)
    perms = nuisance_stratified_permutations(split.S[:, 0], split.N, 1100, CAL_OOD, 0)
    assert report["T_obs_smoke"] == 1.0
    assert report["c_a_smoke"] == float(np.quantile([stage1.balanced_accuracy(predictions, row) for row in perms], 1 - .05 / 7, method="linear"))
    labels = np.stack([np.roll(split.S[:, 0], g) for g in range(3)])
    panel_perms = np.stack([np.roll(perms, g, axis=1) for g in range(3)])
    panel = np.empty((3, 3, 2048), dtype=np.uint8)
    for g in range(3):
        for m in range(3):
            panel[g, m] = np.roll(labels[g], 1 + g * 3 + m)
    assert len({row.tobytes() for row in panel.reshape(9, 2048)}) == 9
    observed, null, critical = full_panel_negative_control(panel, labels, panel_perms)
    assert np.isfinite(observed) and null.shape == (1000,) and critical == np.quantile(null, 1 - .05 / 7, method="linear")


def test_s1_10_labeled_threshold_rows_have_accepted_digest():
    values = {
        "L0_EMA_POOL": [(0, .001612472761926322, .9797216947964541)],
        "L1_AVG": [(0,.0073294466242287545,.6817864540520566),(1,.009084693182175193,.7181395968102731),(2,.006799179258077162,.8263945101552667),(3,.005144090574691897,.7668899730229419)],
        "L2_SEP": [(0,.00604369628505158,.6789513306844063),(1,.008786627625603205,.7117601449971063),(2,.007069421992188329,.6804449634388821),(3,.007938774708153601,.6885657638937837),(4,.006982095556465977,.7074056228474246),(5,.009864751019384176,.7315442922152853),(6,.009674733437987576,.722059182691564),(7,.009016713472598066,.7230056912665725),(8,.005964791231239042,.8242411756696744),(9,.0072216078677750685,.8374906080556898),(10,.0067315352164640145,.8585243038231307),(11,.0060535864673108,.846613791400767),(12,.0041179271953726854,.8519787604573966),(13,.005558459923963963,.7949046025837255),(14,.0048515051763988815,.745955398402419),(15,.005010465330992578,.7450713247018552)],
    }
    rows = [[arm, identity, variance, rank] for arm in stage1.LATENT_NAMES for identity, variance, rank in values[arm]]
    digest = hashlib.sha256(json.dumps(rows, allow_nan=False, separators=(",", ":")).encode("ascii")).hexdigest()
    assert len(rows) == 21 and digest == THRESHOLD_DIGEST


def test_s1_11_position_specificity_is_descriptive_with_eligibility():
    target = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    report = stage1.position_specificity(target, target, torch.ones((2, 4), dtype=torch.bool))
    assert report["aggregate"] == pytest.approx(1.0)
    assert report["eligible_example_count"] == 2 and report["eligible_position_count"] == 8
    assert report["gated"] is False


def test_s1_13_hierarchical_bootstrap_tiny_and_production_defaults_frozen():
    n = 64
    labels = np.stack([np.roll(np.arange(n) % 2, g) for g in range(3)]).astype(np.uint8)
    comparator = 1 - labels
    predictions = np.broadcast_to(labels[:, None, :], (3, 3, n)).copy()
    report = hierarchical_complete_rule(predictions, labels, comparator, factor_code=0, simulation_index=7, n_boot=3)
    assert report["point"] == 1.0 and report["bootstrap_replicates"] == 3 and report["lower_0_005"] == 1.0
    signature = inspect.signature(stage1.complete_rule_calibration)
    assert signature.parameters["simulations"].default == 200
    assert signature.parameters["n_boot"].default == 10000
    assert signature.parameters["minimum_passes"].default == 160
    sim_source = inspect.getsource(stage1.simulated_complete_rule)
    assert "[5101, factor_code, simulation_index, run_index]" in sim_source
    assert sim_source.index("rng.random(n) < 0.5") < sim_source.index("rng.random(n) < w")
    reduced_source = inspect.getsource(stage1.reduced_complete_rule_dry)
    assert "np.roll" not in reduced_source
    assert "np.stack([split.S[:, factor] for split in cal_ood_splits])" in reduced_source
    assert "np.stack([split.N for split in cal_ood_splits])" in reduced_source
    assert "generator_seeds = (1101, 1102, 1103)" in reduced_source


def test_s1_14_runner_schema_authority_booleans_and_exact_gate_membership():
    source = (Path(__file__).parents[1] / "scripts" / "bp_clinjepa_011_j04c_stage1.py").read_text(encoding="utf-8")
    for token in ('"schema_version": "bp-clinjepa-011-j04c-stage1-v1"', '"training_performed": True',
                  '"candidate_c_a_computed": False', '"cal_ood_smoke_c_a_computed": True',
                  '"id_dev_generated": False', '"ood_dev_generated": False',
                  '"thresholds_modified": False', '"seed_selected": False'):
        assert token in source
    assert '"stage1_smoke_pass": all(smoke_gate.values())' in source
    assert set(CONDITION_NAMES) == {"L0_EMA_POOL", "L1_AVG", "L2_SEP", "C0_DIRECT", "C0_NUISANCE_ONLY", "C0_TIME_FREE", "C0_STUDENT_LEAK"}
