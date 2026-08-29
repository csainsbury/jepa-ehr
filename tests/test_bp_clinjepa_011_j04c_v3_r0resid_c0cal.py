from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import tiny_pretraining_indices
from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_v3_r0resid_1m as one
from clinical_jepa.eval import j04c_v3_r0resid_c0cal as cal

LOW = 100
HIGH = 2**31
DIGEST = "0" * 64


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def manifest() -> cal.C0CalSeedManifest:
    return cal.C0CalSeedManifest(
        "BP011-J04C-V3-R0RESID-C0CAL-SEEDS-V1", HIGH + 101, HIGH + 202, HIGH + 303,
    )


def test_manifest_and_exact_39_path_audit():
    value = manifest()
    cal.validate_seed_manifest(value)
    records = cal.generated_seed_audit(value)
    assert len(records) == 39
    assert records == sorted(records, key=lambda item: (item["purpose"], item["path"]))
    purposes = [item["purpose"] for item in records]
    assert purposes.count("GENERATOR_SPLIT") == 3
    assert purposes.count("TRAIN_NUISANCE") == 1
    assert purposes.count("E0_INIT") == purposes.count("C0_HEAD_INIT") == 1
    assert "PREDICTOR_INIT" not in purposes
    assert purposes.count("TRAIN_SCHEDULE") == 32
    assert purposes.count("BOOTSTRAP") == 1
    assert {item["path"][0] for item in records} == {
        value.generator_seed, value.model_seed, value.bootstrap_root,
    }
    with pytest.raises(cal.PrototypeInvariantError, match="SEED_COLLISION"):
        cal.validate_seed_manifest(cal.C0CalSeedManifest(value.schema, HIGH + 1, HIGH + 1, HIGH + 2))
    with pytest.raises(cal.PrototypeInvariantError, match="INPUT_SCHEMA"):
        cal.seed_manifest_from_dict({
            "schema": value.schema, "generator_seed": 3, "model_seed": 4, "bootstrap_root": 5,
        })
    with pytest.raises(cal.PrototypeInvariantError, match="INPUT_SCHEMA"):
        cal.approved_envelope_from_dict({
            "schema": "BP011-J04C-V3-R0RESID-C0CAL-SEED-APPROVAL-V1",
            "manifest_sha256": DIGEST, "historical_inventory_sha256": DIGEST,
            "expected_generated_audit_sha256": DIGEST, "production_path_count": 39.0,
        })


def _full_closed_records():
    roots = list(range(1000, 1018))
    records = []
    def add(purpose, *path): records.append({"purpose": purpose, "path": list(path)})
    for split in (1, 3, 6): add("GENERATOR_SPLIT", roots[0], split)
    add("TRAIN_NUISANCE", roots[0], 1, 7101)
    for model in roots[1:4]:
        add("E0_INIT", model, 1); add("PREDICTOR_INIT", model, 2); add("C0_HEAD_INIT", model, 40)
        for epoch in range(32): add("TRAIN_SCHEDULE", model, epoch, 6101)
    for root in roots[4:7]: add("TARGET_SHUFFLE", root, 7301)
    for root in roots[7:10]: add("CORRESPONDENCE_PROBE", root, 7401)
    for root in roots[10:13]: add("CORRESPONDENCE_CAL_ORIGINAL", root, 7402)
    for root in roots[13:16]: add("CORRESPONDENCE_CAL_INTERVENTION", root, 7403)
    add("NUISANCE_INTERVENTION", roots[16], 7501); add("BOOTSTRAP", roots[17], 7601)
    records.sort(key=lambda item: (item["purpose"], item["path"]))
    return roots, records


def closed_inventory(monkeypatch, *, extra_roots=()):
    full_roots, full_records = _full_closed_records()
    full_digest = full.sha256_hex(canonical(full_records))
    monkeypatch.setattr(one, "CLOSED_LINEAGE_AUDIT_SHA256", full_digest)
    one_manifest = one.OneModelSeedManifest(
        "BP011-J04C-V3-R0RESID-1M-SEEDS-V1", HIGH + 2001, HIGH + 2002, HIGH + 2003,
    )
    one_records = one.generated_seed_audit(one_manifest)
    one_digest = full.sha256_hex(canonical(one_records))
    monkeypatch.setattr(cal, "CLOSED_1M_AUDIT_SHA256", one_digest)
    records = [dict(item, purpose=one.CLOSED_LINEAGE_PURPOSE_PREFIX + item["purpose"])
               for item in full_records]
    records += [dict(item, purpose=cal.CLOSED_1M_PURPOSE_PREFIX + item["purpose"])
                for item in one_records]
    records.sort(key=lambda item: (item["purpose"], item["path"]))
    roots = sorted(set(full_roots) | {HIGH + 2001, HIGH + 2002, HIGH + 2003} | set(extra_roots))
    return {
        "schema": "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1", "through_rows": "closed-1m",
        "roots": roots, "records": records,
        "source_artifact_digests": {
            one.CLOSED_LINEAGE_SOURCE_KEY: full_digest,
            cal.CLOSED_1M_SOURCE_KEY: one_digest,
        },
    }


def test_seed_envelope_requires_both_closed_lineages_and_zero_collision(monkeypatch):
    value = manifest()
    manifest_raw = canonical({
        "schema": value.schema, "generator_seed": value.generator_seed,
        "model_seed": value.model_seed, "bootstrap_root": value.bootstrap_root,
    })
    audit = cal.generated_seed_audit(value)
    inventory = closed_inventory(monkeypatch)
    inventory_raw = canonical(inventory)
    envelope = cal.C0CalApprovedSeedEnvelope(
        "BP011-J04C-V3-R0RESID-C0CAL-SEED-APPROVAL-V1",
        full.sha256_hex(manifest_raw), full.sha256_hex(inventory_raw),
        full.sha256_hex(canonical(audit)), 39,
    )
    records, summary = cal.validate_seed_audit(value, manifest_raw, envelope, inventory, inventory_raw)
    assert records == audit and summary["production_path_count"] == 39
    incomplete = dict(inventory)
    incomplete["source_artifact_digests"] = {
        one.CLOSED_LINEAGE_SOURCE_KEY: inventory["source_artifact_digests"][one.CLOSED_LINEAGE_SOURCE_KEY],
    }
    incomplete_raw = canonical(incomplete)
    incomplete_envelope = cal.C0CalApprovedSeedEnvelope(
        envelope.schema, envelope.manifest_sha256, full.sha256_hex(incomplete_raw),
        envelope.expected_generated_audit_sha256, 39,
    )
    with pytest.raises(cal.PrototypeInvariantError, match="SEED_AUDIT_DIGEST"):
        cal.validate_seed_audit(value, manifest_raw, incomplete_envelope, incomplete, incomplete_raw)
    collided = closed_inventory(monkeypatch, extra_roots=(value.generator_seed,))
    collided_raw = canonical(collided)
    collided_envelope = cal.C0CalApprovedSeedEnvelope(
        envelope.schema, envelope.manifest_sha256, full.sha256_hex(collided_raw),
        envelope.expected_generated_audit_sha256, 39,
    )
    with pytest.raises(cal.PrototypeInvariantError, match="SEED_COLLISION"):
        cal.validate_seed_audit(value, manifest_raw, collided_envelope, collided, collided_raw)


def test_bias_free_head_is_deterministic_nonzero_and_initial_adapter_is_zero():
    first = cal.make_bias_free_c0_head(202)
    second = cal.make_bias_free_c0_head(202)
    assert first.bias is None
    assert torch_equal(first.weight.detach().numpy(), second.weight.detach().numpy())
    assert np.isfinite(first.weight.detach().numpy()).all()
    assert np.count_nonzero(first.weight.detach().numpy()) > 0
    adapter = full.TokenwiseResidualAdapter()
    values = np.random.default_rng(1).normal(size=(5, 16)).astype(np.float32)
    with np.errstate(all="raise"):
        output = adapter(torch_from(values)).detach().numpy()
    assert np.array_equal(output, np.zeros_like(output))


def torch_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left, right)


def torch_from(value: np.ndarray):
    import torch
    return torch.from_numpy(value)


def test_conditioned_training_tiny_real_encoder_and_unchanged_assay():
    generator_seed, model_seed = 101, 202
    train = independent_train_nuisance(generate_factor_split(generator_seed, TRAIN, 64), generator_seed)
    probe = generate_factor_split(generator_seed, PROBE_FIT, 64)
    cal_split = generate_factor_split(generator_seed, CAL_OOD, 64)
    transform = fit_stage0_time_transform(train)
    encoder = full.freeze_encoder(model_seed)
    before = full._state_dict_bytes(encoder)
    base = full.FrozenResidualCondition("R0_BASE", encoder, None, None, None, {})
    train_features = full.extract_feature_blocks(base, train, transform)
    train_summary, train_logits, _ = full._fit_base_bundle(
        train_features.z0, train.S[:, 0].astype(float), {}, "tiny.train_r0",
    )
    offsets_before = train_logits["probe"].copy()
    schedule = tiny_pretraining_indices(model_seed, n=64, batch_size=32, epochs=1)
    product = cal.train_c0_conditioned(
        train, transform, model_seed, encoder, train_logits["probe"], schedule, expected_steps=2,
    )
    assert full._state_dict_bytes(encoder) == before
    assert np.array_equal(train_logits["probe"], offsets_before)
    assert train_summary["converged"] and product.condition.training["successful_steps"] == 2
    assert product.condition.training["ema_updates"] == 0
    probe_base = full.extract_feature_blocks(base, probe, transform)
    cal_base = full.extract_feature_blocks(base, cal_split, transform)
    probe_c0 = full.extract_feature_blocks(product.condition, probe, transform)
    cal_c0 = full.extract_feature_blocks(product.condition, cal_split, transform)
    assert probe_base.z0.tobytes() == probe_c0.z0.tobytes()
    assert cal_base.z0.tobytes() == cal_c0.z0.tobytes()
    y_probe = probe.S[:, 0].astype(float); y_cal = cal_split.S[:, 0].astype(float)
    r0_summary, r0_logits, _ = full._fit_base_bundle(
        probe_base.z0, y_probe, {"cal": cal_base.z0}, "tiny.probe_r0",
    )
    c0_summary, c0_logits, _ = full._fit_additive_bundle(
        probe_c0.delta_z, y_probe, r0_logits["probe"], {"cal": cal_c0.delta_z},
        {"cal": r0_logits["cal"]}, "tiny.c0",
    )
    assert r0_summary["converged"] and c0_summary["converged"]
    rows = full.binary_row_nll(r0_logits["cal"], y_cal) - full.binary_row_nll(c0_logits["cal"], y_cal)
    indices = np.tile(np.arange(64), (20, 1))
    critical, contrasts, used = cal.bootstrap_one_lcb(rows, LOW, replicates=20, supplied_indices=indices)
    assert np.array_equal(used, indices) and critical == pytest.approx(0.0, abs=1e-15)
    assert contrasts[0]["name"] == "d_C0" and np.isfinite(contrasts[0]["lcb95"])


def test_single_contrast_bootstrap_and_terminal_order():
    rows = np.full(12, 0.3)
    indices = np.tile(np.arange(12), (20, 1))
    critical, contrasts, _ = cal.bootstrap_one_lcb(rows, LOW, replicates=20, supplied_indices=indices)
    assert critical == pytest.approx(0.0, abs=1e-15)
    assert contrasts == [{"name": "d_C0", "observed": pytest.approx(0.3), "lcb95": pytest.approx(0.3)}]
    assert cal.evaluate_terminal_outcome(contrasts, valid=True) == ("ELIGIBLE", True, {"d_c0": True})
    red = [dict(contrasts[0], lcb95=0.0)]
    assert cal.evaluate_terminal_outcome(red, valid=True) == ("INELIGIBLE", False, {"d_c0": False})
    assert cal.evaluate_terminal_outcome(red, valid=False) == ("INVALID", False, {})
    with pytest.raises(cal.PrototypeInvariantError, match="BOOTSTRAP_INVALID"):
        cal.bootstrap_one_lcb(rows, LOW, replicates=20, supplied_indices=np.zeros((20, 11), dtype=int))


def training_summary() -> dict[str, object]:
    return {
        "attempted_steps": 2000, "successful_steps": 2000, "optimizer_steps": 2000,
        "ema_updates": 0, "first_100_mean_total": 0.6, "last_100_mean_total": 0.4,
        "component_scalars": {name: None for name in (
            "cosine_first", "cosine_last", "directional_first", "directional_last",
            "v_direction_min_first", "v_direction_min_last",
        )},
    }


def readout_summary() -> dict[str, object]:
    return {
        "preprocess_hash": DIGEST, "coefficient_hash": DIGEST,
        "constant_coordinate_mask_hash": DIGEST, "iterations": 4,
        "converged": True, "zero_short_circuit": False,
    }


def assay_summary() -> dict[str, object]:
    return {"nll_mean": 0.5, "balanced_accuracy": 0.6, "row_nll_hash": DIGEST, "logit_hash": DIGEST}


def success_fixture(*, eligible=True) -> dict[str, object]:
    contrast = {"name": "d_C0", "observed": 0.2, "lcb95": 0.1 if eligible else 0.0}
    return {
        "schema": "BP011-J04C-V3-R0RESID-C0CAL-RESULT-V1", "namespace": cal.NAMESPACE,
        "contract_sha256": cal.CONTRACT_SHA256, "claim_ceiling": "SAFE_PUBLIC_C0_ASSAY_CALIBRATION_ONLY",
        "provenance": {
            "build_provenance_sha256": DIGEST, "target_commit": cal.TARGET_COMMIT,
            "implementation_commit": "a" * 40, "clean_tree": True,
            "source_digests": cal.SOURCE_DIGESTS,
            "implementation_digests": {path: DIGEST for path in cal.IMPLEMENTATION_PATHS},
            "python_version": "3.10", "numpy_version": "2", "torch_version": "2",
            "platform_machine": "x86_64", "platform_system": "Linux", "blas_fingerprint": "sha256:x",
        },
        "seed_audit": {
            "approved_envelope_sha256": DIGEST, "manifest_sha256": DIGEST,
            "historical_inventory_sha256": DIGEST, "generated_audit_sha256": DIGEST,
            "production_path_count": 39, "historical_path_count": 1,
            "path_intersection_count": 0, "root_intersection_count": 0,
        },
        "fixed": cal._fixed_contract(),
        "checks": {name: True for name in (
            "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "successful_steps",
            "train_r0_only", "train_r0_frozen", "c0_head_bias_free", "first_w2_gradient_nonzero",
            "readout_deterministic", "baseline_frozen", "family_exact", "output_allowlist",
        )},
        "model": {
            "state_hashes": {name: DIGEST for name in (
                "e0", "c0_adapter", "c0_training_head", "train_r0_logit_hash",
            )},
            "training": {"C0_CONDITIONED": training_summary()},
            "readouts": {name: readout_summary() for name in (
                "TRAIN_R0_OPTIMIZATION_ONLY", "PROBE_R0_BASE", "C0_CONDITIONED_ADDITIVE",
            )},
            "cal": {name: assay_summary() for name in ("R0_BASE", "C0_CONDITIONED_ADDITIVE")},
        },
        "bootstrap": {"replicates": 10000, "family_size": 1, "quantile_method": "linear",
                      "critical_value": 0.1, "contrasts": [contrast]},
        "valid": True, "eligible": eligible, "scientific_gates": {"d_c0": eligible},
        "terminal_outcome": "ELIGIBLE" if eligible else "INELIGIBLE",
    }


def test_complete_success_failure_schemas_and_nested_rejection():
    cal.validate_success_schema(success_fixture())
    cal.validate_success_schema(success_fixture(eligible=False))
    for mutation in ("extra_root", "bad_training", "extra_contrast", "raw_hidden"):
        value = json.loads(json.dumps(success_fixture()))
        if mutation == "extra_root": value["extra"] = 1
        elif mutation == "bad_training": value["model"]["training"] = ["wrong"]
        elif mutation == "extra_contrast": value["bootstrap"]["contrasts"].append(value["bootstrap"]["contrasts"][0])
        else: value["model"]["raw_hidden"] = [1, 2]
        with pytest.raises(cal.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
            cal.validate_success_schema(value)
    for field, bad_value in (
        ("production_path_count", 39.0), ("historical_path_count", -1),
        ("path_intersection_count", False), ("root_intersection_count", 0.0),
    ):
        value = json.loads(json.dumps(success_fixture()))
        value["seed_audit"][field] = bad_value
        with pytest.raises(cal.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
            cal.validate_success_schema(value)
    for field, bad_value in (("attempted_steps", 2000.0), ("ema_updates", False)):
        value = json.loads(json.dumps(success_fixture()))
        value["model"]["training"]["C0_CONDITIONED"][field] = bad_value
        with pytest.raises(cal.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
            cal.validate_success_schema(value)
    for field, bad_value in (("replicates", 10000.0), ("family_size", True)):
        value = json.loads(json.dumps(success_fixture()))
        value["bootstrap"][field] = bad_value
        with pytest.raises(cal.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
            cal.validate_success_schema(value)
    value = json.loads(json.dumps(success_fixture()))
    value["bootstrap"]["contrasts"] = [["wrong"]]
    with pytest.raises(cal.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
        cal.validate_success_schema(value)
    failure = cal.failure_artifact("READOUT", "READOUT_INVALID")
    cal.validate_failure_schema(failure)
    assert failure["terminal_outcome"] == "INVALID"


def test_build_provenance_pins_2cc_source_and_new_implementation_only():
    value = {
        "schema": "BP011-J04C-V3-R0RESID-C0CAL-BUILD-PROVENANCE-V1",
        "target_commit": cal.TARGET_COMMIT, "implementation_commit": "b" * 40, "clean_tree": True,
        "source_digests": cal.SOURCE_DIGESTS,
        "implementation_digests": {path: DIGEST for path in cal.IMPLEMENTATION_PATHS},
        "python_version": "3.10", "numpy_version": "2", "torch_version": "2",
        "platform_machine": "x86_64", "platform_system": "Linux", "blas_fingerprint": "sha256:x",
    }
    parsed = cal.build_provenance_from_dict(value)
    assert parsed.target_commit == cal.TARGET_COMMIT and parsed.implementation_commit == "b" * 40
    with pytest.raises(cal.PrototypeInvariantError, match="PROVENANCE_CONTENT"):
        cal.build_provenance_from_dict(dict(value, target_commit="c" * 40))
    source = Path("clinical_jepa/eval/j04c_v3_r0resid_1m.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cal.SOURCE_DIGESTS[str(source)]


def test_runner_fail_closed_without_inputs(capfd):
    path = Path("scripts/bp_clinjepa_011_j04c_v3_r0resid_c0cal_beta.py")
    spec = importlib.util.spec_from_file_location("bp011_c0cal_runner_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    assert runner.main([]) == 2
    captured = capfd.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    cal.validate_failure_schema(value)
    assert value["phase"] == "INPUT" and value["error_code"] == "INPUT_SCHEMA"
    defaults = {
        "PROVENANCE": "PROVENANCE_CONTENT", "SEED_AUDIT": "SEED_AUDIT_DIGEST",
        "GENERATION": "GENERATION_INVARIANT", "TRAINING": "TRAINING_INVARIANT",
        "READOUT": "READOUT_INVALID", "BOOTSTRAP": "BOOTSTRAP_INVALID",
        "SERIALIZATION": "SERIALIZATION_INVALID",
    }
    for phase, code in defaults.items():
        assert runner._normalize_failure(RuntimeError("generic"), phase) == (phase, code)
    for code, phase in (
        ("TRAINING_NONFINITE_PRE", "TRAINING"), ("READOUT_INVALID", "READOUT"),
        ("BOOTSTRAP_INVALID", "BOOTSTRAP"), ("SERIALIZATION_INVALID", "SERIALIZATION"),
    ):
        assert runner._normalize_failure(RuntimeError(code), "GENERATION") == (phase, code)


def test_module_runner_static_guards_and_no_candidate_execution():
    module_path = Path("clinical_jepa/eval/j04c_v3_r0resid_c0cal.py")
    runner_path = Path("scripts/bp_clinjepa_011_j04c_v3_r0resid_c0cal_beta.py")
    module_tree = ast.parse(module_path.read_text()); runner_tree = ast.parse(runner_path.read_text())
    forbidden_imports = {"pathlib", "subprocess", "socket", "requests", "urllib"}
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_imports for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_imports
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "exec", "eval"}
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"cuda", "save", "load"}
    reads = [node for node in ast.walk(runner_tree)
             if isinstance(node, ast.Attribute) and node.attr == "read_bytes"]
    assert len(reads) == 4
    text = runner_path.read_text(); module_text = module_path.read_text()
    assert "subprocess" not in text and "fallback" not in text and "glob(" not in text
    assert "train_resid_candidate(" not in module_text and "train_c0_direct(" not in module_text
    assert "frozen_offsets + residual_logits" in module_text and "bias=False" in module_text
    markers = [f'notify("{phase}")' for phase in (
        "GENERATION", "TRAINING", "READOUT", "BOOTSTRAP", "SERIALIZATION",
    )]
    assert all(marker in module_text for marker in markers)
    assert [module_text.index(marker) for marker in markers] == sorted(module_text.index(marker) for marker in markers)
