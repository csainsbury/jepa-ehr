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

LOW = 100
HIGH = 2**31
DIGEST = "0" * 64


def manifest() -> one.OneModelSeedManifest:
    return one.OneModelSeedManifest(
        "BP011-J04C-V3-R0RESID-1M-SEEDS-V1", HIGH + 101, HIGH + 202, HIGH + 303,
    )


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_manifest_and_exact_40_path_audit():
    value = manifest()
    one.validate_seed_manifest(value)
    records = one.generated_seed_audit(value)
    assert len(records) == 40
    assert records == sorted(records, key=lambda item: (item["purpose"], item["path"]))
    purposes = [item["purpose"] for item in records]
    assert purposes.count("GENERATOR_SPLIT") == 3
    assert purposes.count("TRAIN_NUISANCE") == 1
    assert purposes.count("E0_INIT") == purposes.count("PREDICTOR_INIT") == purposes.count("C0_HEAD_INIT") == 1
    assert purposes.count("TRAIN_SCHEDULE") == 32
    assert purposes.count("BOOTSTRAP") == 1
    assert {item["path"][0] for item in records} == {value.generator_seed, value.model_seed, value.bootstrap_root}
    with pytest.raises(one.PrototypeInvariantError, match="SEED_COLLISION"):
        one.validate_seed_manifest(one.OneModelSeedManifest(value.schema, HIGH + 1, HIGH + 1, HIGH + 2))
    with pytest.raises(one.PrototypeInvariantError, match="INPUT_SCHEMA"):
        one.seed_manifest_from_dict({"schema": value.schema, "generator_seed": 3, "model_seed": 4, "bootstrap_root": 5})


def closed_inventory(monkeypatch, *, extra_roots=()):
    roots = list(range(1000, 1018))
    records = []
    def add(purpose, *path):
        records.append({"purpose": purpose, "path": list(path)})
    for split in (1, 3, 6): add("GENERATOR_SPLIT", roots[0], split)
    add("TRAIN_NUISANCE", roots[0], 1, 7101)
    for model in roots[1:4]:
        add("E0_INIT", model, 1); add("PREDICTOR_INIT", model, 2); add("C0_HEAD_INIT", model, 40)
        for epoch in range(32): add("TRAIN_SCHEDULE", model, epoch, 6101)
    for root in roots[4:7]: add("TARGET_SHUFFLE", root, 7301)
    for root in roots[7:10]: add("CORRESPONDENCE_PROBE", root, 7401)
    for root in roots[10:13]: add("CORRESPONDENCE_CAL_ORIGINAL", root, 7402)
    for root in roots[13:16]: add("CORRESPONDENCE_CAL_INTERVENTION", root, 7403)
    add("NUISANCE_INTERVENTION", roots[16], 7501)
    add("BOOTSTRAP", roots[17], 7601)
    records.sort(key=lambda item: (item["purpose"], item["path"]))
    assert len(records) == 123
    audit_sha = full.sha256_hex(canonical(records))
    monkeypatch.setattr(one, "CLOSED_LINEAGE_AUDIT_SHA256", audit_sha)
    prefixed = [dict(item, purpose=one.CLOSED_LINEAGE_PURPOSE_PREFIX + item["purpose"]) for item in records]
    return {
        "schema": "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1", "through_rows": "closed-production",
        "roots": sorted(set(roots) | set(extra_roots)), "records": prefixed,
        "source_artifact_digests": {one.CLOSED_LINEAGE_SOURCE_KEY: audit_sha},
    }


def test_seed_envelope_digest_and_historical_collision_gates(monkeypatch):
    value = manifest()
    manifest_value = {
        "schema": value.schema, "generator_seed": value.generator_seed,
        "model_seed": value.model_seed, "bootstrap_root": value.bootstrap_root,
    }
    manifest_raw = canonical(manifest_value)
    audit = one.generated_seed_audit(value)
    incomplete = {
        "schema": "BP011-HISTORICAL-SEED-PATH-INVENTORY-V1", "through_rows": "incomplete",
        "roots": [0], "records": [{"purpose": "OLD", "path": [0]}], "source_artifact_digests": {},
    }
    incomplete_raw = canonical(incomplete)
    incomplete_envelope = one.OneModelApprovedSeedEnvelope(
        "BP011-J04C-V3-R0RESID-1M-SEED-APPROVAL-V1", full.sha256_hex(manifest_raw),
        full.sha256_hex(incomplete_raw), full.sha256_hex(canonical(audit)), 40,
    )
    with pytest.raises(one.PrototypeInvariantError, match="SEED_AUDIT_DIGEST"):
        one.validate_seed_audit(value, manifest_raw, incomplete_envelope, incomplete, incomplete_raw)

    inventory = closed_inventory(monkeypatch)
    inventory_raw = canonical(inventory)
    envelope = one.OneModelApprovedSeedEnvelope(
        incomplete_envelope.schema, full.sha256_hex(manifest_raw), full.sha256_hex(inventory_raw),
        full.sha256_hex(canonical(audit)), 40,
    )
    records, summary = one.validate_seed_audit(value, manifest_raw, envelope, inventory, inventory_raw)
    assert records == audit and summary["production_path_count"] == 40
    collided = closed_inventory(monkeypatch, extra_roots=(value.generator_seed,))
    collided_raw = canonical(collided)
    collided_envelope = one.OneModelApprovedSeedEnvelope(
        envelope.schema, envelope.manifest_sha256, full.sha256_hex(collided_raw),
        envelope.expected_generated_audit_sha256, 40,
    )
    with pytest.raises(one.PrototypeInvariantError, match="SEED_COLLISION"):
        one.validate_seed_audit(value, manifest_raw, collided_envelope, collided, collided_raw)


def test_two_contrast_paired_bootstrap_and_terminal_precedence():
    rows = {"d_C0": np.full(12, 0.3), "d_R0": np.full(12, 0.2)}
    indices = np.tile(np.arange(12), (20, 1))
    critical, contrasts, used = one.bootstrap_two_lcbs(rows, LOW, replicates=20, supplied_indices=indices)
    assert critical == pytest.approx(0.0, abs=1e-15)
    assert np.array_equal(used, indices)
    assert [item["name"] for item in contrasts] == ["d_C0", "d_R0"]
    assert one.evaluate_terminal_outcome(contrasts, valid=True) == (
        "SUPPORTED", True, {"d_c0": True, "d_r0": True},
    )
    ineligible = [dict(contrasts[0], lcb95=0.0), contrasts[1]]
    assert one.evaluate_terminal_outcome(ineligible, valid=True)[0] == "INELIGIBLE"
    unsupported = [contrasts[0], dict(contrasts[1], lcb95=0.0)]
    assert one.evaluate_terminal_outcome(unsupported, valid=True)[0] == "NOT_SUPPORTED"
    with pytest.raises(one.PrototypeInvariantError, match="BOOTSTRAP_INVALID"):
        one.bootstrap_two_lcbs({"d_R0": rows["d_R0"], "d_C0": rows["d_C0"]}, LOW,
                               replicates=20, supplied_indices=indices)

    asymmetric = {"d_C0": np.array([0.0, 0.0, 0.0, 4.0]),
                  "d_R0": np.array([0.0, 4.0, 4.0, 4.0])}
    asymmetric_indices = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                   [1, 1, 1, 1], [3, 3, 3, 3]])
    critical, contrasts, _ = one.bootstrap_two_lcbs(
        asymmetric, LOW, replicates=5, supplied_indices=asymmetric_indices,
    )
    assert critical == pytest.approx(2.6)
    assert contrasts == [
        {"name": "d_C0", "observed": 1.0, "lcb95": pytest.approx(-1.6)},
        {"name": "d_R0", "observed": 3.0, "lcb95": pytest.approx(0.4)},
    ]


def training_summary(c0: bool) -> dict[str, object]:
    components = {name: (None if c0 else 0.1) for name in (
        "cosine_first", "cosine_last", "directional_first", "directional_last",
        "v_direction_min_first", "v_direction_min_last",
    )}
    return {
        "attempted_steps": 2000, "successful_steps": 2000, "optimizer_steps": 2000,
        "ema_updates": 0 if c0 else 2000, "first_100_mean_total": 1.0,
        "last_100_mean_total": 0.5, "component_scalars": components,
    }


def readout_summary() -> dict[str, object]:
    return {
        "preprocess_hash": DIGEST, "coefficient_hash": DIGEST,
        "constant_coordinate_mask_hash": DIGEST, "iterations": 4,
        "converged": True, "zero_short_circuit": False,
    }


def assay_summary() -> dict[str, object]:
    return {"nll_mean": 0.5, "balanced_accuracy": 0.6, "row_nll_hash": DIGEST, "logit_hash": DIGEST}


def success_fixture() -> dict[str, object]:
    provenance = {
        "build_provenance_sha256": DIGEST, "target_commit": one.TARGET_COMMIT,
        "implementation_commit": "a" * 40, "clean_tree": True,
        "source_digests": one.SOURCE_DIGESTS,
        "implementation_digests": {path: DIGEST for path in one.IMPLEMENTATION_PATHS},
        "python_version": "3.10.12", "numpy_version": "2", "torch_version": "2",
        "platform_machine": "x86_64", "platform_system": "Linux", "blas_fingerprint": "sha256:x",
    }
    seed_audit = {
        "approved_envelope_sha256": DIGEST, "manifest_sha256": DIGEST,
        "historical_inventory_sha256": DIGEST, "generated_audit_sha256": DIGEST,
        "production_path_count": 40, "historical_path_count": 1,
        "path_intersection_count": 0, "root_intersection_count": 0,
    }
    contrasts = [
        {"name": "d_C0", "observed": 0.2, "lcb95": 0.1},
        {"name": "d_R0", "observed": 0.1, "lcb95": 0.01},
    ]
    return {
        "schema": "BP011-J04C-V3-R0RESID-1M-RESULT-V1", "namespace": one.NAMESPACE,
        "contract_sha256": one.CONTRACT_SHA256,
        "claim_ceiling": "ONE_MODEL_SAFE_PUBLIC_INCREMENTAL_UTILITY_ONLY",
        "provenance": provenance, "seed_audit": seed_audit, "fixed": one._fixed_contract(),
        "checks": {name: True for name in (
            "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "successful_steps",
            "readout_deterministic", "baseline_frozen", "family_exact", "output_allowlist",
        )},
        "model": {
            "state_hashes": {name: DIGEST for name in (
                "e0", "candidate_adapter", "candidate_predictor", "candidate_ema_adapter", "c0_adapter",
            )},
            "training": {"RESID_CANDIDATE": training_summary(False), "C0_DIRECT": training_summary(True)},
            "readouts": {name: readout_summary() for name in (
                "R0_BASE", "RESID_CANDIDATE_ADDITIVE", "C0_DIRECT_ADDITIVE",
            )},
            "cal": {name: assay_summary() for name in (
                "R0_BASE", "RESID_CANDIDATE_ADDITIVE", "C0_DIRECT_ADDITIVE",
            )},
        },
        "bootstrap": {"replicates": 10000, "family_size": 2, "quantile_method": "linear",
                      "critical_value": 0.05, "contrasts": contrasts},
        "valid": True, "eligible": True, "scientific_gates": {"d_c0": True, "d_r0": True},
        "terminal_outcome": "SUPPORTED",
    }


def test_recursive_success_failure_schema_and_forbidden_nested_data():
    value = success_fixture()
    one.validate_success_schema(value)
    ineligible = json.loads(json.dumps(value))
    ineligible["bootstrap"]["contrasts"][0]["lcb95"] = 0.0
    ineligible["terminal_outcome"] = "INELIGIBLE"
    ineligible["eligible"] = False
    ineligible["scientific_gates"] = {"d_c0": False, "d_r0": True}
    one.validate_success_schema(ineligible)
    unsupported = json.loads(json.dumps(value))
    unsupported["bootstrap"]["contrasts"][1]["lcb95"] = 0.0
    unsupported["terminal_outcome"] = "NOT_SUPPORTED"
    unsupported["scientific_gates"] = {"d_c0": True, "d_r0": False}
    one.validate_success_schema(unsupported)
    bad = json.loads(json.dumps(value))
    bad["model"]["raw_hidden"] = [1, 2]
    with pytest.raises(one.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
        one.validate_success_schema(bad)
    bad = json.loads(json.dumps(value))
    bad["bootstrap"]["contrasts"].append({"name": "extra", "observed": 0.0, "lcb95": 0.0})
    with pytest.raises(one.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
        one.validate_success_schema(bad)
    for path in (("model", "state_hashes"), ("model", "training"), ("model", "readouts"),
                 ("model", "cal"), ("bootstrap", "contrasts")):
        malformed = json.loads(json.dumps(value))
        malformed[path[0]][path[1]] = ["wrong-container"]
        with pytest.raises(one.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
            one.validate_success_schema(malformed)
    malformed = json.loads(json.dumps(value))
    malformed["bootstrap"]["contrasts"][0] = ["wrong-item"]
    with pytest.raises(one.PrototypeInvariantError, match="SERIALIZATION_INVALID"):
        one.validate_success_schema(malformed)
    failure = one.failure_artifact("READOUT", "READOUT_INVALID")
    one.validate_failure_schema(failure)
    assert set(failure) == {"schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"}


def test_build_provenance_separates_source_base_and_implementation():
    value = {
        "schema": "BP011-J04C-V3-R0RESID-1M-BUILD-PROVENANCE-V1",
        "target_commit": one.TARGET_COMMIT, "implementation_commit": "b" * 40, "clean_tree": True,
        "source_digests": one.SOURCE_DIGESTS,
        "implementation_digests": {path: DIGEST for path in one.IMPLEMENTATION_PATHS},
        "python_version": "3.10", "numpy_version": "2", "torch_version": "2",
        "platform_machine": "x86_64", "platform_system": "Linux", "blas_fingerprint": "sha256:x",
    }
    parsed = one.build_provenance_from_dict(value)
    assert parsed.target_commit == one.TARGET_COMMIT and parsed.implementation_commit == "b" * 40
    changed = dict(value, target_commit="c" * 40)
    with pytest.raises(one.PrototypeInvariantError, match="PROVENANCE_CONTENT"):
        one.build_provenance_from_dict(changed)
    source = Path("clinical_jepa/eval/j04c_v3_r0resid.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == one.SOURCE_DIGESTS[str(source)]


def test_tiny_real_encoder_candidate_c0_readout_and_bootstrap_seam():
    generator_seed, model_seed = 101, 202
    train = independent_train_nuisance(generate_factor_split(generator_seed, TRAIN, 8), generator_seed)
    probe = generate_factor_split(generator_seed, PROBE_FIT, 64)
    cal = generate_factor_split(generator_seed, CAL_OOD, 64)
    transform = fit_stage0_time_transform(train)
    encoder = full.freeze_encoder(model_seed)
    before = full._state_dict_bytes(encoder)
    schedule = tiny_pretraining_indices(model_seed, n=8, batch_size=4, epochs=1)
    candidate = full._train_l1_arm(
        "RESID_CANDIDATE", train, transform, model_seed, encoder, schedule,
        train_adapter=True, expected_steps=2,
    )
    c0 = full._train_c0(
        train, transform, model_seed, encoder,
        tiny_pretraining_indices(model_seed, n=8, batch_size=4, epochs=1), expected_steps=2,
    )
    assert full._state_dict_bytes(encoder) == before
    candidate_probe = full.extract_feature_blocks(candidate, probe, transform)
    c0_probe = full.extract_feature_blocks(c0, probe, transform)
    candidate_cal = full.extract_feature_blocks(candidate, cal, transform)
    c0_cal = full.extract_feature_blocks(c0, cal, transform)
    assert candidate_probe.z0.tobytes() == c0_probe.z0.tobytes()
    assert candidate_cal.z0.tobytes() == c0_cal.z0.tobytes()
    y_probe = probe.S[:, 0].astype(float)
    y_cal = cal.S[:, 0].astype(float)
    r0_summary, r0_logits, _ = full._fit_base_bundle(
        candidate_probe.z0, y_probe, {"cal": candidate_cal.z0}, "tiny.R0",
    )
    candidate_summary, candidate_logits, _ = full._fit_additive_bundle(
        candidate_probe.delta_z, y_probe, r0_logits["probe"], {"cal": candidate_cal.delta_z},
        {"cal": r0_logits["cal"]}, "tiny.candidate",
    )
    c0_summary, c0_logits, _ = full._fit_additive_bundle(
        c0_probe.delta_z, y_probe, r0_logits["probe"], {"cal": c0_cal.delta_z},
        {"cal": r0_logits["cal"]}, "tiny.c0",
    )
    assert all(summary["converged"] for summary in (r0_summary, candidate_summary, c0_summary))
    nll_r0 = full.binary_row_nll(r0_logits["cal"], y_cal)
    rows = {
        "d_C0": nll_r0 - full.binary_row_nll(c0_logits["cal"], y_cal),
        "d_R0": nll_r0 - full.binary_row_nll(candidate_logits["cal"], y_cal),
    }
    indices = np.tile(np.arange(64), (20, 1))
    critical, contrasts, _ = one.bootstrap_two_lcbs(rows, LOW, replicates=20, supplied_indices=indices)
    assert np.isfinite(critical) and len(contrasts) == 2


def test_runner_fail_closed_schema_without_inputs(capfd):
    path = Path("scripts/bp_clinjepa_011_j04c_v3_r0resid_1m_beta.py")
    spec = importlib.util.spec_from_file_location("bp011_1m_runner_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.main([]) == 2
    captured = capfd.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    one.validate_failure_schema(value)
    assert value["phase"] == "INPUT" and value["error_code"] == "INPUT_SCHEMA"
    expected = {
        "GENERATION": "GENERATION_INVARIANT", "TRAINING": "TRAINING_INVARIANT",
        "READOUT": "READOUT_INVALID", "BOOTSTRAP": "BOOTSTRAP_INVALID",
        "SERIALIZATION": "SERIALIZATION_INVALID",
    }
    for phase, code in expected.items():
        assert runner._normalize_failure(RuntimeError("generic"), phase) == (phase, code)


def test_module_and_runner_static_guards():
    module_path = Path("clinical_jepa/eval/j04c_v3_r0resid_1m.py")
    runner_path = Path("scripts/bp_clinjepa_011_j04c_v3_r0resid_1m_beta.py")
    module_tree = ast.parse(module_path.read_text())
    runner_tree = ast.parse(runner_path.read_text())
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
    text = runner_path.read_text()
    assert "subprocess" not in text and "fallback" not in text and "glob(" not in text
    module_text = module_path.read_text()
    stage_markers = [f'notify("{phase}")' for phase in (
        "GENERATION", "TRAINING", "READOUT", "BOOTSTRAP", "SERIALIZATION",
    )]
    assert all(marker in module_text for marker in stage_markers)
    assert [module_text.index(marker) for marker in stage_markers] == sorted(
        module_text.index(marker) for marker in stage_markers
    )
