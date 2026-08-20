from __future__ import annotations

import ast
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import clinical_jepa.eval.j04c_falsifier as falsifier
from clinical_jepa.eval.j04c_falsifier import (
    CAL_ID,
    CAL_OOD,
    ID_DEV,
    OOD_DEV,
    PROBE_FIT,
    TRAIN,
    SyntheticFactorSplit,
    _expanded_conditional_means,
    _normalized_diagnostics,
    analytic_llr_report,
    balanced_accuracy,
    component_seed,
    fit_stage0_time_transform,
    full_panel_negative_control,
    generate_factor_split,
    initialization_audit,
    make_initialized_teacher,
    make_r0_encoder,
    model_free_factor_calibration,
    nuisance_only,
    nuisance_stratified_permutations,
    position_specificity,
    readout_exposure_spec,
    sever_student_leak,
    split_fixture_hash,
    time_free_model_intervals,
)


def _hand_row_loop(seed: int, split_code: int, n: int):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, split_code])))
    s, x, nuisance = [], [], []
    correlated = split_code in (TRAIN, CAL_ID, ID_DEV)
    for _ in range(n):
        sr = (rng.random(3) < 0.5).astype(np.uint8)
        xf = (rng.random(3) < 0.15).astype(np.uint8)
        nd = (rng.random(3) < (0.05 if correlated else 0.5)).astype(np.uint8)
        s.append(sr); x.append(sr ^ xf); nuisance.append(sr ^ nd if correlated else nd)
    return np.asarray(s), np.asarray(x), np.asarray(nuisance)


def _all_row_vectorized(seed: int, split_code: int, n: int):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, split_code])))
    s = (rng.random((n, 3)) < 0.5).astype(np.uint8)
    x = s ^ (rng.random((n, 3)) < 0.15).astype(np.uint8)
    draws = (rng.random((n, 3)) < (0.05 if split_code in (TRAIN, CAL_ID, ID_DEV) else 0.5)).astype(np.uint8)
    nuisance = s ^ draws if split_code in (TRAIN, CAL_ID, ID_DEV) else draws
    return s, x, nuisance


def test_s0_00_ast_authority_guard_for_both_production_additions():
    module_source = inspect.getsource(falsifier)
    script_path = Path(__file__).parents[1] / "scripts" / "bp_clinjepa_011_j04c_stage0.py"
    script_source = script_path.read_text(encoding="utf-8")
    for source in (module_source, script_source):
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not imported_roots.intersection({"subprocess", "socket", "requests", "urllib", "pathlib"})
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        call_names = {
            func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            for func in calls
        }
        assert not call_names.intersection({"open", "write_text", "write_bytes", "save", "load", "backward", "step", "train"})
        assert "cuda" not in source.lower()
        assert "DataLoader" not in source and "checkpoint" not in source.lower()
    script_tree = ast.parse(script_source)
    split_calls = [
        node for node in ast.walk(script_tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "generate_factor_split"
    ]
    assert split_calls and all(not (len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in (4, 5)) for node in split_calls)
    assert "OOD_DEV" not in script_source and "ID_DEV" not in script_source
    assert "[7101, generator_seed, 5]" in module_source


def test_s0_01_exact_per_row_rng_shapes_dtypes_replay_and_split_independence():
    split = generate_factor_split(1101, TRAIN, 64)
    expected_s, expected_x, expected_n = _hand_row_loop(1101, TRAIN, 64)
    np.testing.assert_array_equal(split.S, expected_s)
    np.testing.assert_array_equal(split.X, expected_x)
    np.testing.assert_array_equal(split.N, expected_n)
    vector_s, vector_x, vector_n = _all_row_vectorized(1101, TRAIN, 64)
    assert not (np.array_equal(split.S, vector_s) and np.array_equal(split.X, vector_x) and np.array_equal(split.N, vector_n))
    replay = generate_factor_split(1101, TRAIN, 64)
    assert split_fixture_hash(split) == split_fixture_hash(replay)
    assert split.prefix_type_ids.shape == (64, 7) and split.prefix_type_ids.dtype == np.int64
    assert split.prefix_intervals.shape == (64, 7) and split.prefix_intervals.dtype == np.float64
    assert split.target_type_ids.shape == (64, 4) and split.target_type_ids.dtype == np.int64
    assert split.target_intervals.shape == (64, 4) and split.target_intervals.dtype == np.float64
    assert split.S.dtype == split.X.dtype == split.N.dtype == split.L_after.dtype == np.uint8
    assert split_fixture_hash(split) != split_fixture_hash(generate_factor_split(1101, CAL_ID, 64))


def test_s0_02_exact_composition_order_times_and_signal_controls():
    split = generate_factor_split(1102, CAL_OOD, 256)
    assert np.all(split.prefix_type_ids[:, 0] == 2)
    assert np.all(split.prefix_intervals[:, 0] == 0)
    assert np.all(split.prefix_intervals[:, [1, 2, 4, 5]] == 1)
    assert np.all(split.prefix_type_ids[:, 3] == 7) and np.all(split.prefix_type_ids[:, 6] == 12)
    for row in range(256):
        base = [13, 13, 14, 15] if split.S[row, 0] == 0 else [13, 14, 14, 15]
        if split.S[row, 1] == 1:
            base.reverse()
        assert split.target_type_ids[row].tolist() == base
        expected_time = [1, 1, 4, 4] if split.S[row, 2] == 0 else [4, 4, 1, 1]
        assert split.target_intervals[row].tolist() == expected_time
    before_hash = split_fixture_hash(split)
    controlled = nuisance_only(split)
    assert np.all(controlled.prefix_type_ids[:, 1:3] == 16)
    assert np.all(controlled.prefix_intervals[:, 3] == 2.5)
    np.testing.assert_array_equal(controlled.prefix_type_ids[:, 4:], split.prefix_type_ids[:, 4:])
    np.testing.assert_array_equal(controlled.prefix_intervals[:, 4:], split.prefix_intervals[:, 4:])
    zero_inputs = time_free_model_intervals(split)
    assert np.count_nonzero(zero_inputs) == 0
    assert split_fixture_hash(split) == before_hash
    np.testing.assert_array_equal(controlled.target_intervals, split.target_intervals)


def test_s0_03_probe_fit_independent_nuisance_and_separate_seed_sequence():
    train = generate_factor_split(1103, TRAIN, 2048)
    probe = generate_factor_split(1103, PROBE_FIT, 2048)
    assert np.mean(train.N == train.S) > 0.92
    assert 0.47 < np.mean(probe.N == probe.S) < 0.53
    hand = _hand_row_loop(1103, PROBE_FIT, 2048)
    np.testing.assert_array_equal(probe.S, hand[0])
    np.testing.assert_array_equal(probe.N, hand[2])
    assert split_fixture_hash(train) != split_fixture_hash(probe)


def test_s0_04_time_transform_exact_train_prefix_then_target_and_cutoff_stability():
    train = generate_factor_split(1101, TRAIN, 128)
    transform = fit_stage0_time_transform(train)
    population = np.concatenate((train.prefix_intervals, train.target_intervals), axis=1).reshape(-1)
    logs = np.log1p(population)
    assert transform.mu == float(logs.mean())
    assert transform.sigma == float(logs.std(ddof=0))
    state = transform.state_bytes()
    cal = generate_factor_split(1101, CAL_OOD, 128)
    cal.prefix_intervals[:] = 99.0
    cal.target_intervals[:] = 77.0
    _ = transform.transform(cal.prefix_intervals)
    assert transform.state_bytes() == state


def _module_bytes(module: torch.nn.Module) -> bytes:
    return b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in module.state_dict().values())


def test_s0_05_component_seed_init_teacher_readouts_and_ambient_rng_preservation():
    expected = int(np.random.SeedSequence([2101, 1]).generate_state(1, dtype=np.uint64)[0] & np.uint64((1 << 63) - 1))
    assert component_seed(2101, 1) == expected
    torch.manual_seed(8765)
    state = torch.random.get_rng_state().clone()
    r0 = make_r0_encoder(2101)
    online, teacher = make_initialized_teacher(2101)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert _module_bytes(r0) == _module_bytes(online) == _module_bytes(teacher.model)
    assert not r0.training and not teacher.model.training
    assert all(not parameter.requires_grad for parameter in r0.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    audit = initialization_audit(2101)
    assert audit == {
        "ambient_cpu_rng_preserved": True,
        "r0_online_teacher_byte_identical": True,
        "readouts_deterministic": True,
        "readouts_independent_storage": True,
    }


def test_s0_06_frozen_readout_exposure_arithmetic_and_seed_tuples():
    spec = readout_exposure_spec()
    assert spec["probe"] == {
        "batch_size": 256, "batches_per_epoch": 8, "total_batches": 250,
        "full_epochs": 31, "partial_epoch_batches": 2,
        "epoch_shuffle_seed_tuple": ["model_seed", "factor_code", "epoch", 6201],
    }
    assert spec["head"] == {
        "batch_size": 128, "batches_per_epoch": 16, "total_batches": 250,
        "full_epochs": 15, "partial_epoch_batches": 10,
        "epoch_shuffle_seed_tuple": ["model_seed", "epoch", 6301],
    }
    assert 31 * 8 + 2 == 15 * 16 + 10 == 250


def test_s0_07_nuisance_stratified_permutations_and_same_panel_statistic():
    split = generate_factor_split(1101, CAL_OOD, 2048)
    permutations = nuisance_stratified_permutations(split.S[:, 0], split.N, 1101, CAL_OOD, 0)
    assert permutations.shape == (1000, 2048)
    for nuisance_tuple in np.ndindex(2, 2, 2):
        rows = np.all(split.N == nuisance_tuple, axis=1)
        for k in (0, 1, 999):
            np.testing.assert_array_equal(np.sort(permutations[k, rows]), np.sort(split.S[rows, 0]))
    labels = np.stack([split.S[:, 0], split.S[:, 0], split.S[:, 0]])
    perms = np.stack([permutations, permutations, permutations])
    predictions = np.broadcast_to(labels[:, None, :], (3, 3, 2048)).copy()
    observed, null, critical = full_panel_negative_control(predictions, labels, perms)
    assert observed == 1.0 and null.shape == (1000,)
    expected0 = np.mean([balanced_accuracy(predictions[g, m], perms[g, 0]) for g in range(3) for m in range(3)])
    assert null[0] == expected0
    assert critical == float(np.quantile(null, 1 - 0.05 / 7, method="linear"))
    with pytest.raises(ValueError, match="exact"):
        full_panel_negative_control(predictions[:, :, :-1], labels[:, :-1], perms[:, :, :-1])


def test_s0_08_student_leak_exact_key_leaves_truth_and_falls_to_null():
    split = generate_factor_split(1102, CAL_OOD, 2048)
    truth_before = split.S.copy()
    severed = sever_student_leak(split.L_after, 1102)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7101, 1102, 5])))
    expected = split.L_after[rng.permutation(2048)]
    np.testing.assert_array_equal(severed, expected)
    np.testing.assert_array_equal(split.S, truth_before)
    assert balanced_accuracy(split.L_after[:, 0], split.S[:, 0]) == 1.0
    severed_ba = balanced_accuracy(severed[:, 0], split.S[:, 0])
    assert 0.45 < severed_ba < 0.55


def test_s0_09_factor_equations_and_analytic_llr_dominance():
    calibration = model_free_factor_calibration()
    assert [row["factor_code"] for row in calibration] == [0, 1, 2]
    for row in calibration:
        assert row["G"] == pytest.approx(1.0 - row["M_null"])
        assert row["delta"] == pytest.approx(max(0.05, 0.25 * row["G"]))
        assert row["delta_power"] == pytest.approx(min(row["G"], row["delta"] + 0.10))
        assert row["G"] >= 0.10 and row["delta_power"] > row["delta"]
    llr = analytic_llr_report()
    assert llr["signal_llr_magnitude"] == pytest.approx(math.log(0.85 / 0.15))
    assert llr["nuisance_llr_magnitude"] == pytest.approx(math.log(0.95 / 0.05))
    assert llr["nuisance_dominates_on_disagreement"] is True


def test_s0_10_conditional_means_expand_back_to_duplicate_rows_and_identity_order():
    factors = np.asarray(list(np.ndindex(2, 2, 2)) * 4, dtype=np.uint8)
    values = np.arange(32 * 21 * 3, dtype=np.float64).reshape(32, 21, 3)
    expanded = _expanded_conditional_means(values, factors)
    assert expanded.shape == values.shape
    for factor_tuple in np.ndindex(2, 2, 2):
        rows = np.all(factors == factor_tuple, axis=1)
        expected = values[rows].mean(axis=0)
        np.testing.assert_allclose(expanded[rows], np.broadcast_to(expected, expanded[rows].shape))
        assert expanded[rows].shape[0] == 4
    _, spaced = falsifier.resolve_layer_sets(4)
    assert spaced == [1, 2, 3, 4]
    assert [j * len(spaced) + s for j in range(4) for s in range(4)] == list(range(16))


def test_s0_10a_source_freezes_target_only_ascending_sixteen_batch_traversal():
    source = inspect.getsource(falsifier.initialized_teacher_reference_calibration)
    assert "range(0, 2048, 128)" in source
    assert "cal.target_type_ids[start:stop]" in source
    assert "cal.target_intervals" in source
    assert "teacher(ids, times, causal=True)" in source
    assert "prefix_type_ids" not in source and "prefix_intervals" not in source
    assert "len(observed_slices) != 16" in source
    assert "torch.no_grad()" in source


def test_s0_11_normalized_variance_rank_zero_rules_and_strict_quantiles():
    collapse = np.zeros((2048, 16)); collapse[:, 0] = 1
    diagnostics = _normalized_diagnostics(collapse)
    assert diagnostics["normalized_variance"] == 0.0
    assert diagnostics["effective_rank"] == 0.0
    reference = np.zeros((2048, 16)); reference[:1024, 0] = 1; reference[1024:, 1] = 1
    reference_diag = _normalized_diagnostics(reference)
    assert reference_diag["normalized_variance"] > diagnostics["normalized_variance"]
    assert reference_diag["effective_rank"] > diagnostics["effective_rank"]
    q99 = float(np.quantile(np.zeros(9), 0.99, method="linear"))
    q01 = float(np.quantile(np.full(9, reference_diag["normalized_variance"]), 0.01, method="linear"))
    assert q99 < q01 and 0.5 * (q99 + q01) > 0


def test_s0_12_position_specificity_positive_zero_masked_and_empty_omission():
    target = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    positive = position_specificity(target.clone(), target, torch.ones((2, 3), dtype=torch.bool))
    assert positive["aggregate"] == pytest.approx(1.0)
    assert positive["eligible_example_count"] == 2 and positive["eligible_position_count"] == 6
    all_same = torch.ones((1, 3, 2))
    zero = position_specificity(all_same, all_same, torch.ones((1, 3), dtype=torch.bool))
    assert zero["aggregate"] == pytest.approx(0.0)
    masked = position_specificity(target, target, torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
    assert masked["eligible_example_count"] == 1 and masked["eligible_position_count"] == 2
    empty = position_specificity(target[:1], target[:1], torch.tensor([[1, 0, 0]], dtype=torch.bool))
    assert empty["aggregate"] is None and empty["eligible_example_count"] == 0
    assert empty["per_example"].numel() == 0 and empty["gated"] is False


def test_s0_12_l2_position_comparisons_stay_within_layer():
    target = torch.zeros((1, 3, 2, 3))
    for position in range(3):
        target[0, position, :, position] = 1.0
    result = position_specificity(target.clone(), target, torch.ones((1, 3), dtype=torch.bool))
    assert result["aggregate"] == pytest.approx(1.0)
    assert result["eligible_position_count"] == 3
    assert result["eligible_identity_comparison_count"] == 6


def test_s0_13_stage0_json_authority_fields_and_no_candidate_outputs_in_source():
    script_path = Path(__file__).parents[1] / "scripts" / "bp_clinjepa_011_j04c_stage0.py"
    source = script_path.read_text(encoding="utf-8")
    assert '"schema_version": "bp-clinjepa-011-j04c-stage0-v1"' in source
    assert '"authority": "instrumentation/no-training calibration only"' in source
    assert '"candidate_c_a_computed": False' in source
    assert '"training_performed": False' in source
    assert '"ood_dev_generated": False' in source
    assert '"all_reference_separations_pass": True' in source
    assert "predictions" not in source and "examples" not in source


def test_fixture_hash_includes_schema_and_all_arrays():
    split = generate_factor_split(1101, TRAIN, 8)
    original = split_fixture_hash(split)
    changed_types = split.prefix_type_ids.copy(); changed_types[0, 0] += 1
    changed = SyntheticFactorSplit(
        changed_types, split.prefix_intervals, split.target_type_ids, split.target_intervals,
        split.S, split.X, split.N, split.L_after,
    )
    assert split_fixture_hash(changed) != original
    assert len(original) == 64 and all(character in "0123456789abcdef" for character in original)
