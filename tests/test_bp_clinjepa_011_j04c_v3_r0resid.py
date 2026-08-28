from __future__ import annotations

import numpy as np
import pytest
import torch

from clinical_jepa.eval.j04c_falsifier import TRAIN, generate_factor_split, fit_stage0_time_transform
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import _fresh_predictor, _prefix_inputs
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_gelu_predictor
from clinical_jepa.eval.j04c_v3_r0resid import (
    EMAResidualAdapter, TokenwiseResidualAdapter, apply_tokenwise_before_pooling,
    composite_l1_target, first_w2_gradient_is_finite_nonzero, freeze_encoder,
    l1_directional_objective,
)

# All fixture roots are deliberately in the test-only low half.
TEST_GENERATOR_SEED = 101
TEST_MODEL_SEED = 202


def test_r0_real_pinned_encoder_seam_composite_target_and_first_gradient():
    train = independent_train_nuisance(generate_factor_split(TEST_GENERATOR_SEED, TRAIN, 8), TEST_GENERATOR_SEED)
    transform = fit_stage0_time_transform(train)
    encoder = freeze_encoder(TEST_MODEL_SEED)
    adapter = TokenwiseResidualAdapter()
    teacher = EMAResidualAdapter(adapter)
    predictor = _fresh_predictor(TEST_MODEL_SEED)
    set_identity_gelu_predictor(predictor)

    ids, times = _prefix_inputs(train, transform, "L1_AVG")
    target_ids = torch.as_tensor(train.target_type_ids, dtype=torch.long)
    target_times = torch.as_tensor(transform.transform(train.target_intervals), dtype=torch.float32)
    with torch.no_grad():
        _, sequence, pooled = encoder(ids, times, causal=False)
    z0, delta = apply_tokenwise_before_pooling(sequence, pooled, ids.ne(0), adapter)
    assert torch.equal(z0, pooled)
    assert torch.count_nonzero(delta) == 0

    target, valid, identities, blocks = composite_l1_target(encoder, teacher, target_ids, target_times)
    assert blocks.shape == (8, 4, 4, 16)
    assert target.shape == (8, 4, 16)
    assert identities == [1, 2, 3, 4]
    prediction = predictor(z0 + delta, "L1_AVG")
    loss, _ = l1_directional_objective(prediction, target, valid)
    loss.backward()
    assert first_w2_gradient_is_finite_nonzero(adapter)
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_adapter_is_tokenwise_nonlinear_before_pooling():
    adapter = TokenwiseResidualAdapter()
    with torch.no_grad():
        adapter.W2.weight.copy_(torch.eye(16))
    tokens = torch.zeros(1, 2, 16)
    tokens[0, 0, 0] = 3.0
    tokens[0, 1, 1] = 3.0
    pre_pool = adapter(tokens).mean(dim=1)
    post_pool = adapter(tokens.mean(dim=1))
    assert not torch.allclose(pre_pool, post_pool)
    assert adapter.W1.bias is None and adapter.W2.bias is None

from clinical_jepa.eval.j04c_stage1 import tiny_pretraining_indices
from clinical_jepa.eval.j04c_v3_r0resid import (
    _train_c0, _train_l1_arm, successful_optimizer_update, shuffled_target_split,
    target_tuple_permutation,
)


def _tiny_training_fixture():
    train = independent_train_nuisance(generate_factor_split(TEST_GENERATOR_SEED, TRAIN, 8), TEST_GENERATOR_SEED)
    return train, fit_stage0_time_transform(train)


def test_r1_tiny_exact_candidate_pred_only_c0_and_ema_semantics():
    train, transform = _tiny_training_fixture()
    encoder = freeze_encoder(TEST_MODEL_SEED)
    before = [p.detach().clone() for p in encoder.parameters()]
    candidate = _train_l1_arm(
        "RESID_CANDIDATE", train, transform, TEST_MODEL_SEED, encoder,
        tiny_pretraining_indices(TEST_MODEL_SEED, n=8, batch_size=4, epochs=1),
        train_adapter=True, expected_steps=2,
    )
    assert candidate.training["successful_steps"] == 2
    assert candidate.training["ema_updates"] == 2
    assert candidate.teacher_adapter.successful_updates == 2
    assert all(torch.equal(a, b) for a, b in zip(before, encoder.parameters()))
    assert all(p.grad is None for p in encoder.parameters())

    pred_only = _train_l1_arm(
        "PRED_ONLY", train, transform, TEST_MODEL_SEED, encoder,
        tiny_pretraining_indices(TEST_MODEL_SEED, n=8, batch_size=4, epochs=1),
        train_adapter=False, expected_steps=2,
    )
    assert pred_only.training["ema_updates"] == 0
    assert all(not p.requires_grad for p in pred_only.adapter.parameters())

    c0 = _train_c0(
        train, transform, TEST_MODEL_SEED, encoder,
        tiny_pretraining_indices(TEST_MODEL_SEED, n=8, batch_size=4, epochs=1), expected_steps=2,
    )
    assert c0.training["successful_steps"] == 2
    assert c0.predictor is None and c0.teacher_adapter is None


def test_r1_joint_target_tuple_shuffle_and_failed_step_no_ema():
    train, _ = _tiny_training_fixture()
    permutation = target_tuple_permutation(303, 8)
    shuffled = shuffled_target_split(train, permutation)
    assert np.array_equal(shuffled.target_type_ids, train.target_type_ids[permutation])
    assert np.array_equal(shuffled.target_intervals, train.target_intervals[permutation])
    assert sorted(map(tuple, shuffled.target_type_ids)) == sorted(map(tuple, train.target_type_ids))
    assert np.array_equal(shuffled.prefix_type_ids, train.prefix_type_ids)

    adapter = TokenwiseResidualAdapter()
    ema = EMAResidualAdapter(adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    bad = adapter.W2.weight.sum() * torch.tensor(float("inf"))
    bad.backward()
    with pytest.raises(FloatingPointError, match="TRAINING_NONFINITE_PRE"):
        successful_optimizer_update(bad, optimizer, ema, adapter)
    assert ema.successful_updates == 0

from clinical_jepa.eval.j04c_falsifier import CAL_OOD
from clinical_jepa.eval.j04c_v3_r0resid import (
    CONTRAST_NAMES, VIEW_NAMES, apply_standardization, array_sha256,
    bootstrap_simultaneous_lcbs, canonical_array_bytes, correspondence_null,
    correspondence_permutation, evaluate_terminal_outcome, expected_contrast_keys,
    fit_additive_readout, fit_deterministic_logistic, fit_standardization,
    nuisance_intervention, readout_logits,
)


def test_r2_newton_readout_converges_uniquely_and_frozen_zero_offset_is_exact():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 16))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.float64)
    standardized, state = fit_standardization(X)
    first = fit_deterministic_logistic(standardized, y)
    second = fit_deterministic_logistic(standardized.copy(), y.copy())
    assert np.array_equal(first.coefficients, second.coefficients)
    assert first.intercept == second.intercept and first.converged
    base = readout_logits(first, standardized)
    zero = np.zeros_like(X)
    offset, _, copied = fit_additive_readout(zero, y, base)
    assert offset.zero_short_circuit and offset.intercept == 0.0
    assert copied.tobytes() == base.tobytes()


def test_r2_constant_nearly_constant_and_extreme_readout_fixtures():
    probe = np.ones((40, 16), dtype=np.float64)
    probe[:, 1] += np.arange(40) * 1e-12
    transformed, state = fit_standardization(probe)
    assert np.equal(transformed[:, 0], 0.0).all()
    assert not np.signbit(transformed[:, 0]).any()
    cal = apply_standardization(np.full((12, 16), 99.0), state)
    assert np.equal(cal[:, 0], 0.0).all()
    X = np.zeros((20, 16)); X[:, 0] = np.linspace(-1e4, 1e4, 20)
    y = np.array([0] * 10 + [1] * 10, dtype=np.float64)
    fit = fit_deterministic_logistic(X, y)
    assert fit.converged and np.isfinite(fit.coefficients).all()


def test_r2_nuisance_intervention_and_correspondence_null_exactness():
    split = generate_factor_split(TEST_GENERATOR_SEED, CAL_OOD, 32)
    changed = nuisance_intervention(split, 404)
    assert np.array_equal(changed.S, split.S)
    assert np.array_equal(changed.target_type_ids, split.target_type_ids)
    assert np.array_equal(changed.prefix_type_ids[:, :4], split.prefix_type_ids[:, :4])
    values = np.arange(32 * 16, dtype=np.float64).reshape(32, 16)
    permutation = correspondence_permutation(505, 32, 7401)
    null = correspondence_null(values, permutation)
    assert np.array_equal(null, values[permutation])
    assert not np.array_equal(null, values)


def _tiny_contrast_rows(value=0.2):
    return {key: np.full(12, value, dtype=np.float64) for key in expected_contrast_keys()}


def test_r2_exact_36_family_paired_bootstrap_lcb_and_gate_order():
    rows = _tiny_contrast_rows()
    rng = np.random.default_rng(8)
    indices = rng.integers(0, 12, size=(40, 12), dtype=np.int64)
    critical, contrasts, used = bootstrap_simultaneous_lcbs(
        rows, 606, replicates=40, supplied_indices=indices,
    )
    assert critical == pytest.approx(0.0, abs=1e-15)
    assert len(contrasts) == 36 and np.array_equal(used, indices)
    outcome, eligible, gates = evaluate_terminal_outcome(
        contrasts, valid=True, residual_nonzero_nonconstant=True,
    )
    assert (outcome, eligible) == ("GREEN", True) and all(gates.values())
    ineligible = [dict(c, lcb95=(-0.1 if c["name"] == "d_C0" else c["lcb95"])) for c in contrasts]
    assert evaluate_terminal_outcome(ineligible, valid=True, residual_nonzero_nonconstant=True)[0] == "INELIGIBLE"
    red = [dict(c, lcb95=(-0.1 if c["name"] == "d_R0" else c["lcb95"])) for c in contrasts]
    assert evaluate_terminal_outcome(red, valid=True, residual_nonzero_nonconstant=True)[0] == "SCIENTIFIC_RED"
    assert evaluate_terminal_outcome([], valid=False, residual_nonzero_nonconstant=False)[0] == "INVALID"


def test_r2_canonical_array_hash_has_metadata_and_is_stable():
    a = np.array([[1.0, 2.0]], dtype="<f8")
    assert canonical_array_bytes("golden", a).startswith(b"BP011-ARRAY-V1\0")
    assert array_sha256("golden", a) == array_sha256("golden", a.copy())
    assert array_sha256("other", a) != array_sha256("golden", a)

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from clinical_jepa.eval.j04c_v3_r0resid import (
    ApprovedSeedEnvelope, SeedManifest, canonical_json_bytes, classify_diagnostics,
    failure_artifact, generated_seed_audit, parse_canonical_json,
    seed_manifest_from_dict, validate_failure_schema, validate_historical_inventory,
    validate_recursive_output, validate_seed_manifest,
)


def _production_manifest():
    roots = iter(range(2**31 + 100, 2**31 + 118))
    return SeedManifest(
        "BP011-J04C-V3-R0RESID-SEEDS-V1", next(roots),
        tuple(next(roots) for _ in range(3)), tuple(next(roots) for _ in range(3)),
        tuple(next(roots) for _ in range(3)), tuple(next(roots) for _ in range(3)),
        tuple(next(roots) for _ in range(3)), next(roots), next(roots),
    )


def test_r2_manifest_high_half_collision_guards_and_exact_123_path_audit():
    manifest = _production_manifest()
    validate_seed_manifest(manifest)
    records = generated_seed_audit(manifest)
    assert len(records) == 123
    assert records == sorted(records, key=lambda item: (item["purpose"], item["path"]))
    collided = SeedManifest(
        manifest.schema, manifest.generator_seed, manifest.model_seeds,
        manifest.train_target_shuffle_roots, manifest.capacity_probe_roots,
        manifest.capacity_cal_original_roots, manifest.capacity_cal_intervention_roots,
        manifest.generator_seed, manifest.bootstrap_root,
    )
    with pytest.raises(Exception, match="SEED_COLLISION"):
        validate_seed_manifest(collided)
    low = SeedManifest(
        manifest.schema, 1, manifest.model_seeds, manifest.train_target_shuffle_roots,
        manifest.capacity_probe_roots, manifest.capacity_cal_original_roots,
        manifest.capacity_cal_intervention_roots, manifest.nuisance_intervention_root,
        manifest.bootstrap_root,
    )
    with pytest.raises(ValueError, match="outside permitted"):
        validate_seed_manifest(low)
    with pytest.raises(Exception, match="IDENTITY"):
        target_tuple_permutation(1, 1)


def test_r2_canonical_input_rejection_and_historical_inventory_guards():
    value = {"a": 1, "b": [2]}
    canonical = canonical_json_bytes(value)
    assert parse_canonical_json(canonical) == value
    with pytest.raises(Exception, match="INPUT_NONCANONICAL"):
        parse_canonical_json(b'{"b":[2], "a":1}')
    inventory = {
        "schema": "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1", "through_rows": "11202",
        "roots": [10], "records": [{"purpose": "X", "path": [10, 1]}],
        "source_artifact_digests": {},
    }
    roots, paths = validate_historical_inventory(inventory)
    assert roots == {10} and paths == {(10, 1)}
    inventory["records"][0]["path"][0] = 11
    with pytest.raises(Exception, match="INPUT_SCHEMA"):
        validate_historical_inventory(inventory)


def test_r2_recursive_schema_rejects_alias_arrays_paths_nan_and_failure_is_exact():
    for bad in (
        {"alias": np.zeros(2)}, {"alias": "/secret"}, {"alias": float("nan")},
        {"hidden_raw_values": [1]}, {"items": list(range(37))},
    ):
        with pytest.raises(Exception, match="SERIALIZATION_INVALID"):
            validate_recursive_output(bad)
    artifact = failure_artifact("INPUT", "INPUT_SCHEMA")
    validate_failure_schema(artifact)
    assert set(artifact) == {"schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"}
    assert json.loads(canonical_json_bytes(artifact)) == artifact


def _valid_success_schema_fixture():
    import clinical_jepa.eval.j04c_v3_r0resid as r0

    digest = "0" * 64
    training_components = {
        "cosine_first": 1.0, "cosine_last": 0.5,
        "directional_first": 0.1, "directional_last": 0.1,
        "v_direction_min_first": 0.02, "v_direction_min_last": 0.02,
    }
    def training(c0=False):
        return {
            "attempted_steps": 2000, "successful_steps": 2000,
            "optimizer_steps": 2000, "ema_updates": 0 if c0 else 2000,
            "first_100_mean_total": 1.0, "last_100_mean_total": 0.5,
            "component_scalars": {key: None for key in training_components} if c0 else dict(training_components),
        }
    def readout():
        return {
            "preprocess_hash": digest, "coefficient_hash": digest,
            "constant_coordinate_mask_hash": digest, "iterations": 1,
            "converged": True, "zero_short_circuit": False,
        }
    def assay():
        return {"nll_mean": 0.5, "balanced_accuracy": 0.6, "row_nll_hash": digest, "logit_hash": digest}
    state_names = (
        "e0", "candidate_adapter", "candidate_predictor", "candidate_ema_adapter",
        "shuffled_adapter", "shuffled_predictor", "shuffled_ema_adapter",
        "c0_adapter", "pred_only_predictor", "pred_only_shuffled_predictor",
    )
    readout_names = (
        "R0_BASE", "RESID_CANDIDATE_ADDITIVE", "RESID_CANDIDATE_RESIDUAL_ONLY",
        "RESID_SHUFFLED_ADDITIVE", "RESID_SHUFFLED_RESIDUAL_ONLY",
        "RESID_CORRESPONDENCE_NULL_ADDITIVE", "RESID_CORRESPONDENCE_NULL_RESIDUAL_ONLY",
        "C0_DIRECT_ADDITIVE",
    )
    models = []
    for model_index in range(3):
        models.append({
            "model_index": model_index,
            "state_hashes": {name: digest for name in state_names},
            "training": {
                "RESID_CANDIDATE": training(), "RESID_SHUFFLED": training(),
                "C0_DIRECT": training(c0=True), "PRED_ONLY": training(),
                "PRED_ONLY_SHUFFLED": training(),
            },
            "readouts": {name: readout() for name in readout_names},
            "views": {view: {name: assay() for name in readout_names} for view in r0.VIEW_NAMES},
        })
    contrasts = [{
        "model_index": model, "view": view, "name": name, "observed": 0.1, "lcb95": -0.1,
    } for model, view, name in r0.expected_contrast_keys()]
    arm_order = ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED", "C0_DIRECT")
    train_summaries = [{
        "model_index": model, "arm": arm, "first_100_mean_total": 1.0, "last_100_mean_total": 0.5,
    } for model in range(3) for arm in arm_order]
    target_arms = ("PRED_ONLY", "PRED_ONLY_SHUFFLED", "RESID_CANDIDATE", "RESID_SHUFFLED")
    target_summaries = [{
        "model_index": model, "view": view, "arm": arm,
        "mean_l1_cosine_loss": 0.5, "row_loss_hash": digest,
    } for model in range(3) for view in r0.VIEW_NAMES for arm in target_arms]
    target_contrasts = [{
        "model_index": model, "name": name, "observed": 0.1,
        "lower95": -0.1, "upper95": 0.2, "beats": False,
    } for model in range(3) for name in ("g_PS", "g_CP", "g_CS")]
    descriptions = [{
        "target": target, "model_index": model, "view": view,
        "r0_readout_hash": digest, "candidate_offset_readout_hash": digest,
        "r0_nll_mean": 0.5, "r0_balanced_accuracy": 0.6, "r0_row_nll_hash": digest,
        "candidate_nll_mean": 0.5, "candidate_balanced_accuracy": 0.6,
        "candidate_row_nll_hash": digest, "d_r0_observed": 0.0,
        "lower95": -0.1, "upper95": 0.1,
    } for target in ("ORDER", "TIME") for model in range(3) for view in r0.VIEW_NAMES]
    check_names = (
        "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "target_axes",
        "initial_zero", "first_w2_gradient", "optimizer_membership", "successful_steps",
        "ema_steps", "readout_deterministic", "baseline_frozen", "nuisance_preserved",
        "family_exact", "diagnostics_nonpromoting", "output_allowlist",
    )
    return {
        "schema": "BP011-J04C-V3-R0RESID-RESULT-V1", "namespace": r0.NAMESPACE,
        "contract_sha256": r0.CONTRACT_SHA256,
        "claim_ceiling": "one-generator directional public-synthetic frozen-R0 residual beta only",
        "provenance": {
            "build_provenance_sha256": digest, "target_commit": r0.TARGET_COMMIT,
            "implementation_commit": "1" * 40, "clean_tree": True,
            "expected_source_digests": dict(r0.SOURCE_DIGESTS),
            "verified_actual_source_digests": dict(r0.SOURCE_DIGESTS),
            "implementation_digests": {path: digest for path in r0.IMPLEMENTATION_PATHS},
            "python_version": "test", "numpy_version": "test", "torch_version": "test",
            "platform_machine": "test", "platform_system": "test", "blas_fingerprint": "test",
        },
        "seed_audit": {
            "approved_envelope_sha256": digest, "manifest_sha256": digest,
            "historical_inventory_sha256": digest, "generated_audit_sha256": digest,
            "production_path_count": 123, "historical_path_count": 1,
            "path_intersection_count": 0, "root_intersection_count": 0,
        },
        "fixed": r0._fixed_contract(), "checks": {name: True for name in check_names},
        "models": models,
        "bootstrap": {
            "replicates": 10000, "family_size": 36, "quantile_method": "linear",
            "critical_value": 0.1, "contrasts": contrasts,
        },
        "diagnostics": {
            "classification": "NOT_INTERPRETED",
            "all_seed_beats": {"g_PS": False, "g_CP": False, "g_CS": False},
            "train_summaries": train_summaries, "target_loss_summaries": target_summaries,
            "target_loss_contrasts": target_contrasts,
        },
        "recovery": [], "descriptive_targets": descriptions, "valid": True, "eligible": False,
        "scientific_gates": {
            "d_r0": False, "d_shuf": False, "d_cap": False, "r_shuf": False,
            "r_cap": False, "residual_nonzero_nonconstant": False,
        },
        "terminal_outcome": "INELIGIBLE",
    }


def test_r2_build_provenance_binds_external_implementation_commit_and_digests():
    import clinical_jepa.eval.j04c_v3_r0resid as r0
    digest = "0" * 64
    value = {
        "schema": "BP011-J04C-V3-R0RESID-BUILD-PROVENANCE-V1",
        "target_commit": r0.TARGET_COMMIT, "implementation_commit": "1" * 40,
        "clean_tree": True, "source_digests": dict(r0.SOURCE_DIGESTS),
        "implementation_digests": {path: digest for path in r0.IMPLEMENTATION_PATHS},
        "python_version": "test", "numpy_version": "test", "torch_version": "test",
        "platform_machine": "test", "platform_system": "test", "blas_fingerprint": "test",
    }
    parsed = r0.build_provenance_from_dict(value)
    assert parsed.implementation_commit == "1" * 40
    assert parsed.implementation_digests == value["implementation_digests"]
    with pytest.raises(Exception, match="PROVENANCE_CONTENT"):
        r0.build_provenance_from_dict(dict(value, implementation_commit="not-a-commit"))
    with pytest.raises(Exception, match="PROVENANCE_CONTENT"):
        r0.build_provenance_from_dict(dict(value, implementation_digests={r0.IMPLEMENTATION_PATHS[0]: digest}))


def test_r2_complete_positive_success_fixture_passes_recursive_schema():
    import clinical_jepa.eval.j04c_v3_r0resid as r0
    value = _valid_success_schema_fixture()
    r0.validate_success_schema(value)
    assert value["diagnostics"]["all_seed_beats"] == {"g_PS": False, "g_CP": False, "g_CS": False}


def test_r2_failed_post_step_does_not_advance_ema_or_retry():
    adapter = TokenwiseResidualAdapter()
    ema = EMAResidualAdapter(adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    loss = adapter.W2.weight.sum()
    loss.backward()
    original_step = optimizer.step
    calls = {"n": 0}
    def corrupting_step():
        calls["n"] += 1
        original_step()
        with torch.no_grad():
            adapter.W2.weight[0, 0] = float("nan")
    optimizer.step = corrupting_step
    with pytest.raises(FloatingPointError, match="TRAINING_NONFINITE_POST"):
        successful_optimizer_update(loss, optimizer, ema, adapter)
    assert calls["n"] == 1 and ema.successful_updates == 0


def test_r2_diagnostic_precedence_and_eligibility_guard():
    train = [{"arm": "PRED_ONLY", "first_100_mean_total": 1.0, "last_100_mean_total": 1.1}]
    gates = {name: False for name in ("d_r0", "d_shuf", "d_cap", "r_shuf", "r_cap")}
    beats = {"g_PS": True, "g_CP": True, "g_CS": True}
    assert classify_diagnostics("INELIGIBLE", False, gates, train, beats) == "NOT_INTERPRETED"
    assert classify_diagnostics("SCIENTIFIC_RED", True, gates, train, beats) == "PREDICTOR_FIT_LIMITATION"
    assert classify_diagnostics("GREEN", True, gates, train, beats) == "SUPPORTED_RESIDUAL_MECHANISM"


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_r2_static_guards_no_forbidden_runtime_capabilities_or_existing_mutation():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "clinical_jepa/eval/j04c_v3_r0resid.py"
    runner_path = root / "scripts/bp_clinjepa_011_j04c_v3_r0resid_beta.py"
    forbidden_imports = {"socket", "requests", "urllib", "subprocess", "httpx", "sklearn"}
    assert "subprocess" in _import_roots(ast.parse("from subprocess import run"))
    for path in (module_path, runner_path):
        tree = ast.parse(path.read_text())
        assert _import_roots(tree).isdisjoint(forbidden_imports)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert not any(isinstance(call.func, ast.Attribute) and call.func.attr in {"cuda", "save", "load"} for call in calls)
    assert not any((root / name).exists() for name in ("results", "checkpoints"))


def test_r2_runner_invalid_invocation_emits_only_allowlisted_failure(capfd):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/bp_clinjepa_011_j04c_v3_r0resid_beta.py"
    spec = importlib.util.spec_from_file_location("bp011_r0_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    assert runner.main([]) == 2
    out, err = capfd.readouterr()
    value = json.loads(out)
    validate_failure_schema(value)
    assert err == "" and out.endswith("\n") and len(out.splitlines()) == 1

from dataclasses import asdict
from clinical_jepa.eval.j04c_v3_r0resid import (
    _recovery_records, sha256_hex, target_learning_row_loss, validate_seed_audit,
)


def test_r2_array_hash_golden_vector_and_random_correspondence_covariance():
    value = np.array([[1.0, 2.0]], dtype="<f8")
    assert array_sha256("golden", value) == "78948cdb6ab5bf0205896b34403f5375a74fd0964e162d4d45ab521aa0ed0b8a"
    random = np.random.default_rng(17).normal(size=(64, 16))
    permutation = np.random.default_rng(18).permutation(64)
    null = correspondence_null(random, permutation)
    assert np.allclose(np.cov(null, rowvar=False), np.cov(random, rowvar=False), rtol=1e-12, atol=1e-15)


def test_r2_trusted_envelope_inventory_digest_mismatch_fails_closed():
    manifest = _production_manifest()
    manifest_raw = canonical_json_bytes(asdict(manifest))
    inventory = {
        "schema": "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1", "through_rows": "11200-11202",
        "roots": [10], "records": [{"purpose": "HISTORICAL", "path": [10, 1]}],
        "source_artifact_digests": {},
    }
    inventory_raw = canonical_json_bytes(inventory)
    audit_digest = sha256_hex(canonical_json_bytes(generated_seed_audit(manifest)))
    envelope = ApprovedSeedEnvelope(
        "BP011-J04C-V3-R0RESID-SEED-APPROVAL-V1", sha256_hex(manifest_raw),
        sha256_hex(inventory_raw), audit_digest, 123,
    )
    records, summary = validate_seed_audit(manifest, manifest_raw, envelope, inventory, inventory_raw)
    assert len(records) == 123 and summary["path_intersection_count"] == 0
    bad = ApprovedSeedEnvelope(envelope.schema, "0" * 64, envelope.historical_inventory_sha256,
                               envelope.expected_generated_audit_sha256, 123)
    with pytest.raises(Exception, match="SEED_AUDIT_DIGEST"):
        validate_seed_audit(manifest, manifest_raw, bad, inventory, inventory_raw)


def test_r2_target_learning_diagnostic_uses_candidate_ema_common_target():
    train, transform = _tiny_training_fixture()
    encoder = freeze_encoder(TEST_MODEL_SEED)
    candidate = _train_l1_arm(
        "RESID_CANDIDATE", train, transform, TEST_MODEL_SEED, encoder,
        tiny_pretraining_indices(TEST_MODEL_SEED, n=8, batch_size=4, epochs=1),
        train_adapter=True, expected_steps=2,
    )
    pred_only = _train_l1_arm(
        "PRED_ONLY", train, transform, TEST_MODEL_SEED, encoder,
        tiny_pretraining_indices(TEST_MODEL_SEED, n=8, batch_size=4, epochs=1),
        train_adapter=False, expected_steps=2,
    )
    candidate_rows = target_learning_row_loss(candidate, candidate, train, transform)
    pred_rows = target_learning_row_loss(pred_only, candidate, train, transform)
    assert candidate_rows.shape == pred_rows.shape == (8,)
    assert np.isfinite(candidate_rows).all() and np.isfinite(pred_rows).all()
    assert not np.array_equal(candidate_rows, pred_rows)


def test_r2_recovery_finite_and_unbounded_denominator_branches_are_ordered():
    rows = _tiny_contrast_rows()
    indices = np.tile(np.arange(12), (20, 1))
    finite = _recovery_records(rows, indices)
    assert len(finite) == 6
    assert [(item["model_index"], item["view"]) for item in finite] == [(m, v) for m in range(3) for v in VIEW_NAMES]
    assert all(item["interval_status"] == "FINITE" for item in finite)
    mixed = _tiny_contrast_rows()
    denominator = np.array([1.0] * 7 + [-1.0] * 5)
    for model in range(3):
        for view in VIEW_NAMES:
            mixed[(model, view, "d_C0")] = denominator
    bad_indices = np.vstack((np.arange(12), np.arange(7, 12).repeat(3)[:12]))
    unbounded = _recovery_records(mixed, bad_indices)
    assert all(item["interval_status"] == "UNBOUNDED_DENOMINATOR" for item in unbounded)
    assert all(item["lower95"] is None and item["upper95"] is None for item in unbounded)
