"""Smallest-genuine one-model frozen-R0 residual beta for BP011.

The scientific module is CPU-only and in-memory.  It selects no seeds, reads no
files, and writes no artifacts.  The guarded runner is the sole I/O boundary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from clinical_jepa.eval import j04c_v3_r0resid as full

NAMESPACE = "BP011-J04C-V3-R0RESID-1M-BETA-K0"
CONTRACT_SHA256 = "e0f1957c9f76a3430adb56410a429bd5ec9340f8a6c4086eec409ed7e1997a65"
TARGET_COMMIT = "a22850891a324d3d3e208bbeb3d1b0c9f047c90d"
PRODUCTION_PATH_COUNT = 40
IMPLEMENTATION_PATHS = (
    "clinical_jepa/eval/j04c_v3_r0resid_1m.py",
    "scripts/bp_clinjepa_011_j04c_v3_r0resid_1m_beta.py",
    "tests/test_bp_clinjepa_011_j04c_v3_r0resid_1m.py",
)
SOURCE_DIGESTS = {
    **full.SOURCE_DIGESTS,
    "clinical_jepa/eval/j04c_v3_r0resid.py":
        "3e38b75450bb50fd0043dee9c5e4759db2d00d07e31354954f3acb4b1de994c1",
}
CLOSED_LINEAGE_AUDIT_SHA256 = "843bf7bfde013ea7a117f2635dd40bcb309a21c6746cd69a3b47a8583e7f7066"
CLOSED_LINEAGE_SOURCE_KEY = "closed-lineage/production-generated-seed-audit.json"
CLOSED_LINEAGE_PURPOSE_PREFIX = "CLOSED_K0__"
CLOSED_LINEAGE_PURPOSE_COUNTS = {
    "GENERATOR_SPLIT": 3, "TRAIN_NUISANCE": 1, "E0_INIT": 3, "PREDICTOR_INIT": 3,
    "C0_HEAD_INIT": 3, "TRAIN_SCHEDULE": 96, "TARGET_SHUFFLE": 3,
    "CORRESPONDENCE_PROBE": 3, "CORRESPONDENCE_CAL_ORIGINAL": 3,
    "CORRESPONDENCE_CAL_INTERVENTION": 3, "NUISANCE_INTERVENTION": 1, "BOOTSTRAP": 1,
}

PrototypeInvariantError = full.PrototypeInvariantError
FAILURE_CODES = full.FAILURE_CODES
FAILURE_PHASES = full.FAILURE_PHASES


@dataclass(frozen=True)
class OneModelSeedManifest:
    schema: Literal["BP011-J04C-V3-R0RESID-1M-SEEDS-V1"]
    generator_seed: int
    model_seed: int
    bootstrap_root: int


@dataclass(frozen=True)
class OneModelApprovedSeedEnvelope:
    schema: Literal["BP011-J04C-V3-R0RESID-1M-SEED-APPROVAL-V1"]
    manifest_sha256: str
    historical_inventory_sha256: str
    expected_generated_audit_sha256: str
    production_path_count: Literal[40]


@dataclass(frozen=True)
class OneModelBuildProvenance:
    schema: Literal["BP011-J04C-V3-R0RESID-1M-BUILD-PROVENANCE-V1"]
    target_commit: str
    implementation_commit: str
    clean_tree: bool
    source_digests: dict[str, str]
    implementation_digests: dict[str, str]
    python_version: str
    numpy_version: str
    torch_version: str
    platform_machine: str
    platform_system: str
    blas_fingerprint: str


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def seed_manifest_from_dict(value: object) -> OneModelSeedManifest:
    keys = {"schema", "generator_seed", "model_seed", "bootstrap_root"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-1M-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    manifest = OneModelSeedManifest(**value)
    validate_seed_manifest(manifest)
    return manifest


def validate_seed_manifest(manifest: OneModelSeedManifest) -> None:
    if manifest.schema != "BP011-J04C-V3-R0RESID-1M-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    roots = (manifest.generator_seed, manifest.model_seed, manifest.bootstrap_root)
    if any(isinstance(x, bool) or not isinstance(x, int) or not 2**31 <= x < 2**32 for x in roots):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if len(set(roots)) != 3:
        raise PrototypeInvariantError("SEED_COLLISION")


def approved_envelope_from_dict(value: object) -> OneModelApprovedSeedEnvelope:
    keys = {"schema", "manifest_sha256", "historical_inventory_sha256",
            "expected_generated_audit_sha256", "production_path_count"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-1M-SEED-APPROVAL-V1" or \
       value.get("production_path_count") != PRODUCTION_PATH_COUNT or \
       isinstance(value.get("production_path_count"), bool):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if any(not _is_digest(value.get(name)) for name in (
        "manifest_sha256", "historical_inventory_sha256", "expected_generated_audit_sha256"
    )):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return OneModelApprovedSeedEnvelope(**value)


def build_provenance_from_dict(value: object) -> OneModelBuildProvenance:
    keys = {"schema", "target_commit", "implementation_commit", "clean_tree", "source_digests",
            "implementation_digests", "python_version", "numpy_version", "torch_version",
            "platform_machine", "platform_system", "blas_fingerprint"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-1M-BUILD-PROVENANCE-V1" or \
       value.get("target_commit") != TARGET_COMMIT or value.get("clean_tree") is not True or \
       value.get("source_digests") != SOURCE_DIGESTS:
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    commit = value.get("implementation_commit")
    implementation = value.get("implementation_digests")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit) or \
       not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS) or \
       any(not _is_digest(item) for item in implementation.values()):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    if any(not isinstance(value.get(name), str) or not value[name] for name in (
        "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system",
        "blas_fingerprint",
    )):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    return OneModelBuildProvenance(**value)


def _record(purpose: str, path: Sequence[int]) -> dict[str, object]:
    values = list(path)
    if not purpose.isascii() or not values or any(
        isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in values
    ):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return {"purpose": purpose, "path": values}


def generated_seed_audit(manifest: OneModelSeedManifest) -> list[dict[str, object]]:
    validate_seed_manifest(manifest)
    records = [_record("GENERATOR_SPLIT", (manifest.generator_seed, split)) for split in (1, 3, 6)]
    records.append(_record("TRAIN_NUISANCE", (manifest.generator_seed, 1, 7101)))
    records.extend((
        _record("E0_INIT", (manifest.model_seed, 1)),
        _record("PREDICTOR_INIT", (manifest.model_seed, 2)),
        _record("C0_HEAD_INIT", (manifest.model_seed, 40)),
    ))
    records.extend(_record("TRAIN_SCHEDULE", (manifest.model_seed, epoch, 6101)) for epoch in range(32))
    records.append(_record("BOOTSTRAP", (manifest.bootstrap_root, 7601)))
    records.sort(key=lambda item: (item["purpose"], item["path"]))
    if len(records) != PRODUCTION_PATH_COUNT or len({
        (item["purpose"], tuple(item["path"])) for item in records
    }) != PRODUCTION_PATH_COUNT:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return records


def validate_closed_lineage_inventory(
    inventory: object, *, expected_audit_sha256: str | None = None,
) -> None:
    if expected_audit_sha256 is None:
        expected_audit_sha256 = CLOSED_LINEAGE_AUDIT_SHA256
    if not isinstance(inventory, dict):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    source_digests = inventory.get("source_artifact_digests")
    records = inventory.get("records")
    if not isinstance(source_digests, dict) or \
       source_digests.get(CLOSED_LINEAGE_SOURCE_KEY) != expected_audit_sha256 or \
       not isinstance(records, list):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    extracted = []
    for item in records:
        if isinstance(item, dict) and isinstance(item.get("purpose"), str) and \
           item["purpose"].startswith(CLOSED_LINEAGE_PURPOSE_PREFIX):
            extracted.append({
                "purpose": item["purpose"][len(CLOSED_LINEAGE_PURPOSE_PREFIX):],
                "path": item.get("path"),
            })
    extracted.sort(key=lambda item: (item["purpose"], item["path"]))
    if len(extracted) != 123 or Counter(item["purpose"] for item in extracted) != CLOSED_LINEAGE_PURPOSE_COUNTS or \
       len({item["path"][0] for item in extracted}) != 18 or \
       full.sha256_hex(full.canonical_json_bytes(extracted)) != expected_audit_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")


def validate_seed_audit(
    manifest: OneModelSeedManifest, manifest_raw: bytes, envelope: OneModelApprovedSeedEnvelope,
    inventory: object, inventory_raw: bytes,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if full.sha256_hex(manifest_raw) != envelope.manifest_sha256 or \
       full.sha256_hex(inventory_raw) != envelope.historical_inventory_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    records = generated_seed_audit(manifest)
    audit_digest = full.sha256_hex(full.canonical_json_bytes(records))
    if audit_digest != envelope.expected_generated_audit_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    historical_roots, historical_paths = full.validate_historical_inventory(inventory)
    validate_closed_lineage_inventory(inventory)
    paths = {tuple(item["path"]) for item in records}
    roots = {path[0] for path in paths}
    if paths & historical_paths or roots & historical_roots:
        raise PrototypeInvariantError("SEED_COLLISION")
    return records, {
        "manifest_sha256": envelope.manifest_sha256,
        "historical_inventory_sha256": envelope.historical_inventory_sha256,
        "generated_audit_sha256": audit_digest,
        "production_path_count": PRODUCTION_PATH_COUNT,
        "historical_path_count": len(historical_paths),
        "path_intersection_count": 0,
        "root_intersection_count": 0,
    }


def bootstrap_two_lcbs(
    row_contrasts: Mapping[str, np.ndarray], bootstrap_root: int, *, replicates: int = 10000,
    supplied_indices: np.ndarray | None = None,
) -> tuple[float, list[dict[str, object]], np.ndarray]:
    names = ("d_C0", "d_R0")
    if list(row_contrasts) != list(names):
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    arrays = [np.ascontiguousarray(np.asarray(row_contrasts[name], dtype="<f8")) for name in names]
    n = arrays[0].size
    if n == 0 or any(array.shape != (n,) or not np.isfinite(array).all() for array in arrays):
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    observed = np.asarray([array.mean() for array in arrays], dtype=np.float64)
    if supplied_indices is None:
        if replicates != 10000 or n != 2048:
            raise PrototypeInvariantError("BOOTSTRAP_INVALID")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([bootstrap_root, 7601])))
        indices = rng.integers(0, n, size=(replicates, n), dtype=np.int64, endpoint=False)
    else:
        indices = np.ascontiguousarray(np.asarray(supplied_indices, dtype=np.int64))
        if indices.shape != (replicates, n) or np.any(indices < 0) or np.any(indices >= n):
            raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    stacked = np.stack(arrays)
    centered_max = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        selected = indices[start:start + 64]
        means = stacked[:, selected].mean(axis=2).T
        centered_max[start:start + selected.shape[0]] = np.max(means - observed, axis=1)
    critical = float(np.quantile(centered_max, 0.95, method="linear"))
    if not math.isfinite(critical):
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    contrasts = [
        {"name": name, "observed": float(observed[i]), "lcb95": float(observed[i] - critical)}
        for i, name in enumerate(names)
    ]
    return critical, contrasts, indices


def evaluate_terminal_outcome(contrasts: Sequence[Mapping[str, object]], *, valid: bool) -> tuple[str, bool, dict[str, bool]]:
    if not valid:
        return "INVALID", False, {}
    if [item.get("name") for item in contrasts] != ["d_C0", "d_R0"]:
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    d_c0 = float(contrasts[0]["lcb95"]) > 0.0
    d_r0 = float(contrasts[1]["lcb95"]) > 0.0
    gates = {"d_c0": d_c0, "d_r0": d_r0}
    if not d_c0:
        return "INELIGIBLE", False, gates
    return ("SUPPORTED" if d_r0 else "NOT_SUPPORTED"), True, gates


def failure_artifact(phase: str, error_code: str) -> dict[str, object]:
    if phase not in FAILURE_PHASES or error_code not in FAILURE_CODES:
        phase, error_code = "SERIALIZATION", "SERIALIZATION_INVALID"
    result = {
        "schema": "BP011-J04C-V3-R0RESID-1M-INVALID-V1",
        "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256,
        "terminal_outcome": "INVALID",
        "phase": phase,
        "error_code": error_code,
    }
    validate_failure_schema(result)
    return result


def validate_failure_schema(value: object) -> None:
    keys = {"schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-1M-INVALID-V1" or \
       value.get("namespace") != NAMESPACE or value.get("contract_sha256") != CONTRACT_SHA256 or \
       value.get("terminal_outcome") != "INVALID" or value.get("phase") not in FAILURE_PHASES or \
       value.get("error_code") not in FAILURE_CODES:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value, failure=True)


def _validate_digest_fields(value: object, key: str = "root") -> None:
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if (nested_key.endswith("_sha256") or nested_key.endswith("_hash")) and not _is_digest(nested_value):
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
            _validate_digest_fields(nested_value, nested_key)
    elif isinstance(value, list):
        for item in value:
            _validate_digest_fields(item, key)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_training_summary(value: object, *, c0: bool) -> None:
    keys = {"attempted_steps", "successful_steps", "optimizer_steps", "ema_updates",
            "first_100_mean_total", "last_100_mean_total", "component_scalars"}
    component_keys = {"cosine_first", "cosine_last", "directional_first", "directional_last",
                      "v_direction_min_first", "v_direction_min_last"}
    expected_counts = (2000, 2000, 2000, 0 if c0 else 2000)
    actual_counts = tuple(value.get(name) for name in (
        "attempted_steps", "successful_steps", "optimizer_steps", "ema_updates"
    )) if isinstance(value, dict) else ()
    if not isinstance(value, dict) or set(value) != keys or actual_counts != expected_counts or \
       any(isinstance(item, bool) or not isinstance(item, int) for item in actual_counts) or \
       not _finite_number(value["first_100_mean_total"]) or not _finite_number(value["last_100_mean_total"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    components = value["component_scalars"]
    if not isinstance(components, dict) or set(components) != component_keys:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if c0:
        if any(item is not None for item in components.values()):
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    elif any(not _finite_number(item) for item in components.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


def _validate_readout_summary(value: object) -> None:
    keys = {"preprocess_hash", "coefficient_hash", "constant_coordinate_mask_hash",
            "iterations", "converged", "zero_short_circuit"}
    if not isinstance(value, dict) or set(value) != keys or \
       any(not _is_digest(value[name]) for name in (
           "preprocess_hash", "coefficient_hash", "constant_coordinate_mask_hash"
       )) or isinstance(value["iterations"], bool) or not isinstance(value["iterations"], int) or \
       value["iterations"] < 0 or value["converged"] is not True or \
       not isinstance(value["zero_short_circuit"], bool):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


def _validate_assay_summary(value: object) -> None:
    keys = {"nll_mean", "balanced_accuracy", "row_nll_hash", "logit_hash"}
    if not isinstance(value, dict) or set(value) != keys or \
       not _finite_number(value["nll_mean"]) or float(value["nll_mean"]) < 0.0 or \
       not _finite_number(value["balanced_accuracy"]) or not 0.0 <= float(value["balanced_accuracy"]) <= 1.0 or \
       not _is_digest(value["row_nll_hash"]) or not _is_digest(value["logit_hash"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


SUCCESS_ROOT_KEYS = {
    "schema", "namespace", "contract_sha256", "claim_ceiling", "provenance", "seed_audit", "fixed",
    "checks", "model", "bootstrap", "valid", "eligible", "scientific_gates", "terminal_outcome",
}


def validate_success_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != SUCCESS_ROOT_KEYS or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-1M-RESULT-V1" or \
       value.get("namespace") != NAMESPACE or value.get("contract_sha256") != CONTRACT_SHA256 or \
       value.get("claim_ceiling") != "ONE_MODEL_SAFE_PUBLIC_INCREMENTAL_UTILITY_ONLY" or \
       value.get("valid") is not True or value.get("terminal_outcome") not in {
           "SUPPORTED", "NOT_SUPPORTED", "INELIGIBLE"
       }:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    provenance = value.get("provenance")
    seed_audit = value.get("seed_audit")
    fixed = value.get("fixed")
    checks = value.get("checks")
    model = value.get("model")
    bootstrap = value.get("bootstrap")
    gates = value.get("scientific_gates")
    provenance_keys = {"build_provenance_sha256", "target_commit", "implementation_commit", "clean_tree",
                       "source_digests", "implementation_digests", "python_version", "numpy_version",
                       "torch_version", "platform_machine", "platform_system", "blas_fingerprint"}
    if not isinstance(provenance, dict) or set(provenance) != provenance_keys or \
       not _is_digest(provenance["build_provenance_sha256"]) or provenance["target_commit"] != TARGET_COMMIT or \
       not isinstance(provenance["implementation_commit"], str) or len(provenance["implementation_commit"]) != 40 or \
       any(c not in "0123456789abcdef" for c in provenance["implementation_commit"]) or \
       provenance["clean_tree"] is not True or provenance["source_digests"] != SOURCE_DIGESTS or \
       not isinstance(provenance["implementation_digests"], dict) or \
       set(provenance["implementation_digests"]) != set(IMPLEMENTATION_PATHS) or \
       any(not _is_digest(item) for item in provenance["implementation_digests"].values()) or \
       any(not isinstance(provenance[name], str) or not provenance[name] for name in (
           "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system",
           "blas_fingerprint",
       )):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    seed_keys = {"approved_envelope_sha256", "manifest_sha256", "historical_inventory_sha256",
                 "generated_audit_sha256", "production_path_count", "historical_path_count",
                 "path_intersection_count", "root_intersection_count"}
    if not isinstance(seed_audit, dict) or set(seed_audit) != seed_keys or \
       any(not _is_digest(seed_audit[name]) for name in (
           "approved_envelope_sha256", "manifest_sha256", "historical_inventory_sha256",
           "generated_audit_sha256",
       )) or seed_audit["production_path_count"] != 40 or \
       any(isinstance(seed_audit[name], bool) or not isinstance(seed_audit[name], int) or seed_audit[name] < 0
           for name in ("historical_path_count", "path_intersection_count", "root_intersection_count")) or \
       seed_audit["path_intersection_count"] != 0 or seed_audit["root_intersection_count"] != 0:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if fixed != _fixed_contract():
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(checks, dict) or set(checks) != {
        "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "successful_steps",
        "readout_deterministic", "baseline_frozen", "family_exact", "output_allowlist",
    } or not all(item is True for item in checks.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(model, dict) or set(model) != {"state_hashes", "training", "readouts", "cal"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    state_hashes, training, readouts, cal = (
        model["state_hashes"], model["training"], model["readouts"], model["cal"],
    )
    if not all(isinstance(item, dict) for item in (state_hashes, training, readouts, cal)) or \
       set(state_hashes) != {"e0", "candidate_adapter", "candidate_predictor", "candidate_ema_adapter", "c0_adapter"} or \
       set(training) != {"RESID_CANDIDATE", "C0_DIRECT"} or \
       set(readouts) != {"R0_BASE", "RESID_CANDIDATE_ADDITIVE", "C0_DIRECT_ADDITIVE"} or \
       set(cal) != {"R0_BASE", "RESID_CANDIDATE_ADDITIVE", "C0_DIRECT_ADDITIVE"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    _validate_training_summary(training["RESID_CANDIDATE"], c0=False)
    _validate_training_summary(training["C0_DIRECT"], c0=True)
    for item in state_hashes.values():
        if not _is_digest(item):
            raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for item in readouts.values():
        _validate_readout_summary(item)
    for item in cal.values():
        _validate_assay_summary(item)
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "replicates", "family_size", "quantile_method", "critical_value", "contrasts"
    } or bootstrap.get("replicates") != 10000 or bootstrap.get("family_size") != 2 or \
       bootstrap.get("quantile_method") != "linear" or not _finite_number(bootstrap.get("critical_value")) or \
       not isinstance(bootstrap.get("contrasts"), list) or len(bootstrap["contrasts"]) != 2 or \
       any(not isinstance(item, dict) for item in bootstrap["contrasts"]) or \
       [item.get("name") for item in bootstrap["contrasts"]] != ["d_C0", "d_R0"] or \
       any(set(item) != {"name", "observed", "lcb95"} or not _finite_number(item["observed"]) or
           not _finite_number(item["lcb95"]) for item in bootstrap["contrasts"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(gates, dict) or set(gates) != {"d_c0", "d_r0"} or \
       any(not isinstance(item, bool) for item in gates.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    outcome, eligible, expected_gates = evaluate_terminal_outcome(bootstrap["contrasts"], valid=True)
    if value["terminal_outcome"] != outcome or value.get("eligible") is not eligible or gates != expected_gates:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value)
    _validate_digest_fields(value)


def _fixed_contract() -> dict[str, object]:
    return {
        "generator_count": 1, "model_count": 1, "train_n": 8192, "probe_n": 2048, "cal_n": 2048,
        "batch_size": 128, "updates": 2000, "ema_momentum": 0.996,
        "adapter": "W2_GELU_W1_PARAMETER_FREE_LN_TOKENWISE", "recipe": "L1_AVG",
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001,
                      "betas": [0.9, 0.999], "eps": 1e-8},
        "directional_weight": 5.0, "directional_floor": 0.010,
        "readout_ridge": 0.001, "bootstrap_replicates": 10000, "bootstrap_family_size": 2,
    }


def run_one_model_beta(
    manifest: OneModelSeedManifest, provenance: OneModelBuildProvenance, seed_audit: Mapping[str, object],
    *, build_provenance_sha256: str, approved_envelope_sha256: str,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    from clinical_jepa.eval.j04c_falsifier import (
        TRAIN, CAL_OOD, PROBE_FIT, fit_stage0_time_transform, generate_factor_split,
    )
    from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance

    validate_seed_manifest(manifest)
    if not _is_digest(build_provenance_sha256) or not _is_digest(approved_envelope_sha256):
        raise PrototypeInvariantError("PROVENANCE_DIGEST")
    notify = phase_callback if phase_callback is not None else lambda _: None
    notify("GENERATION")
    train = independent_train_nuisance(
        generate_factor_split(manifest.generator_seed, TRAIN, 8192), manifest.generator_seed,
    )
    probe = generate_factor_split(manifest.generator_seed, PROBE_FIT, 2048)
    cal = generate_factor_split(manifest.generator_seed, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    encoder = full.freeze_encoder(manifest.model_seed)
    e0_before = full._state_dict_bytes(encoder)
    notify("TRAINING")
    candidate = full.train_resid_candidate(train, transform, manifest.model_seed, encoder)
    c0 = full.train_c0_direct(train, transform, manifest.model_seed, encoder)
    if full._state_dict_bytes(encoder) != e0_before:
        raise PrototypeInvariantError("TRAINING_INVARIANT")

    notify("READOUT")
    candidate_probe = full.extract_feature_blocks(candidate, probe, transform)
    candidate_cal = full.extract_feature_blocks(candidate, cal, transform)
    c0_probe = full.extract_feature_blocks(c0, probe, transform)
    c0_cal = full.extract_feature_blocks(c0, cal, transform)
    if candidate_probe.z0.tobytes() != c0_probe.z0.tobytes() or \
       candidate_cal.z0.tobytes() != c0_cal.z0.tobytes():
        raise PrototypeInvariantError("GENERATION_INVARIANT")
    labels_probe = probe.S[:, 0].astype(np.float64)
    labels_cal = cal.S[:, 0].astype(np.float64)
    r0_summary, r0_logits, _ = full._fit_base_bundle(
        candidate_probe.z0, labels_probe, {"cal": candidate_cal.z0}, "R0_BASE",
    )
    candidate_summary, candidate_logits, _ = full._fit_additive_bundle(
        candidate_probe.delta_z, labels_probe, r0_logits["probe"], {"cal": candidate_cal.delta_z},
        {"cal": r0_logits["cal"]}, "RESID_CANDIDATE_ADDITIVE",
    )
    c0_summary, c0_logits, _ = full._fit_additive_bundle(
        c0_probe.delta_z, labels_probe, r0_logits["probe"], {"cal": c0_cal.delta_z},
        {"cal": r0_logits["cal"]}, "C0_DIRECT_ADDITIVE",
    )
    nll_r0 = full.binary_row_nll(r0_logits["cal"], labels_cal)
    nll_candidate = full.binary_row_nll(candidate_logits["cal"], labels_cal)
    nll_c0 = full.binary_row_nll(c0_logits["cal"], labels_cal)
    rows = {"d_C0": nll_r0 - nll_c0, "d_R0": nll_r0 - nll_candidate}
    notify("BOOTSTRAP")
    critical, contrasts, _ = bootstrap_two_lcbs(rows, manifest.bootstrap_root)
    outcome, eligible, gates = evaluate_terminal_outcome(contrasts, valid=True)
    notify("SERIALIZATION")
    provenance_output = {
        "build_provenance_sha256": build_provenance_sha256,
        "target_commit": provenance.target_commit,
        "implementation_commit": provenance.implementation_commit,
        "clean_tree": provenance.clean_tree,
        "source_digests": provenance.source_digests,
        "implementation_digests": provenance.implementation_digests,
        "python_version": provenance.python_version,
        "numpy_version": provenance.numpy_version,
        "torch_version": provenance.torch_version,
        "platform_machine": provenance.platform_machine,
        "platform_system": provenance.platform_system,
        "blas_fingerprint": provenance.blas_fingerprint,
    }
    result = {
        "schema": "BP011-J04C-V3-R0RESID-1M-RESULT-V1",
        "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256,
        "claim_ceiling": "ONE_MODEL_SAFE_PUBLIC_INCREMENTAL_UTILITY_ONLY",
        "provenance": provenance_output,
        "seed_audit": {"approved_envelope_sha256": approved_envelope_sha256, **dict(seed_audit)},
        "fixed": _fixed_contract(),
        "checks": {
            "source_pins": True, "seed_audit": True, "e0_immutable": True, "pooling_exact": True,
            "successful_steps": True, "readout_deterministic": True, "baseline_frozen": True,
            "family_exact": True, "output_allowlist": True,
        },
        "model": {
            "state_hashes": {
                "e0": full.canonical_state_sha256(encoder),
                "candidate_adapter": full.canonical_state_sha256(candidate.adapter),
                "candidate_predictor": full.canonical_state_sha256(candidate.predictor),
                "candidate_ema_adapter": full.canonical_state_sha256(candidate.teacher_adapter.adapter),
                "c0_adapter": full.canonical_state_sha256(c0.adapter),
            },
            "training": {"RESID_CANDIDATE": candidate.training, "C0_DIRECT": c0.training},
            "readouts": {
                "R0_BASE": r0_summary,
                "RESID_CANDIDATE_ADDITIVE": candidate_summary,
                "C0_DIRECT_ADDITIVE": c0_summary,
            },
            "cal": {
                "R0_BASE": full.assay_summary(r0_logits["cal"], labels_cal, "cal.R0_BASE"),
                "RESID_CANDIDATE_ADDITIVE": full.assay_summary(
                    candidate_logits["cal"], labels_cal, "cal.RESID_CANDIDATE_ADDITIVE",
                ),
                "C0_DIRECT_ADDITIVE": full.assay_summary(c0_logits["cal"], labels_cal, "cal.C0_DIRECT_ADDITIVE"),
            },
        },
        "bootstrap": {
            "replicates": 10000, "family_size": 2, "quantile_method": "linear",
            "critical_value": critical, "contrasts": contrasts,
        },
        "valid": True,
        "eligible": eligible,
        "scientific_gates": gates,
        "terminal_outcome": outcome,
    }
    return result
