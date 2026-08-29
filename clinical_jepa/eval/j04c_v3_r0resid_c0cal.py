"""Frozen-TRAIN-R0-conditioned C0 assay-calibration beta for BP011.

CPU-only and in-memory: this module selects no seeds and performs no file I/O.
The guarded runner is the sole input/output boundary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_v3_r0resid_1m as one

NAMESPACE = "BP011-J04C-V3-R0RESID-C0CAL-BETA-K0"
CONTRACT_SHA256 = "c5dd162747fe84d1af122d609be1635e0be15141d5897ee996b3694e0e855b67"
TARGET_COMMIT = "2cc18b6a336b1554f12b02345467852cdcbba6a7"
PRODUCTION_PATH_COUNT = 39
IMPLEMENTATION_PATHS = (
    "clinical_jepa/eval/j04c_v3_r0resid_c0cal.py",
    "scripts/bp_clinjepa_011_j04c_v3_r0resid_c0cal_beta.py",
    "tests/test_bp_clinjepa_011_j04c_v3_r0resid_c0cal.py",
)
SOURCE_DIGESTS = {
    **one.SOURCE_DIGESTS,
    "clinical_jepa/eval/j04c_v3_r0resid_1m.py":
        "293cd2c647f834de2b48a4e54774df105055c48a83dc39470a6c13dbf97d3568",
}
CLOSED_1M_AUDIT_SHA256 = "938d98f510c8913f7edb353a0790d151abdda0707c1ea6425b7ee10db7f9a2f5"
CLOSED_1M_SOURCE_KEY = "closed-one-model/production-generated-seed-audit.json"
CLOSED_1M_PURPOSE_PREFIX = "CLOSED_1M_K0__"
CLOSED_1M_PURPOSE_COUNTS = {
    "GENERATOR_SPLIT": 3, "TRAIN_NUISANCE": 1, "E0_INIT": 1,
    "PREDICTOR_INIT": 1, "C0_HEAD_INIT": 1, "TRAIN_SCHEDULE": 32, "BOOTSTRAP": 1,
}

PrototypeInvariantError = full.PrototypeInvariantError
FAILURE_CODES = full.FAILURE_CODES
FAILURE_PHASES = full.FAILURE_PHASES


@dataclass(frozen=True)
class C0CalSeedManifest:
    schema: Literal["BP011-J04C-V3-R0RESID-C0CAL-SEEDS-V1"]
    generator_seed: int
    model_seed: int
    bootstrap_root: int


@dataclass(frozen=True)
class C0CalApprovedSeedEnvelope:
    schema: Literal["BP011-J04C-V3-R0RESID-C0CAL-SEED-APPROVAL-V1"]
    manifest_sha256: str
    historical_inventory_sha256: str
    expected_generated_audit_sha256: str
    production_path_count: Literal[39]


@dataclass(frozen=True)
class C0CalBuildProvenance:
    schema: Literal["BP011-J04C-V3-R0RESID-C0CAL-BUILD-PROVENANCE-V1"]
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


@dataclass
class ConditionedC0Product:
    condition: full.FrozenResidualCondition
    training_head_hash: str
    train_r0_logit_hash: str


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def seed_manifest_from_dict(value: object) -> C0CalSeedManifest:
    keys = {"schema", "generator_seed", "model_seed", "bootstrap_root"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-C0CAL-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    manifest = C0CalSeedManifest(**value)
    validate_seed_manifest(manifest)
    return manifest


def validate_seed_manifest(manifest: C0CalSeedManifest) -> None:
    roots = (manifest.generator_seed, manifest.model_seed, manifest.bootstrap_root)
    if manifest.schema != "BP011-J04C-V3-R0RESID-C0CAL-SEEDS-V1" or \
       any(isinstance(x, bool) or not isinstance(x, int) or not 2**31 <= x < 2**32 for x in roots):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if len(set(roots)) != 3:
        raise PrototypeInvariantError("SEED_COLLISION")


def approved_envelope_from_dict(value: object) -> C0CalApprovedSeedEnvelope:
    keys = {"schema", "manifest_sha256", "historical_inventory_sha256",
            "expected_generated_audit_sha256", "production_path_count"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-C0CAL-SEED-APPROVAL-V1" or \
       not isinstance(value.get("production_path_count"), int) or \
       isinstance(value.get("production_path_count"), bool) or \
       value.get("production_path_count") != PRODUCTION_PATH_COUNT or \
       any(not _is_digest(value.get(name)) for name in (
           "manifest_sha256", "historical_inventory_sha256", "expected_generated_audit_sha256"
       )):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return C0CalApprovedSeedEnvelope(**value)


def build_provenance_from_dict(value: object) -> C0CalBuildProvenance:
    keys = {"schema", "target_commit", "implementation_commit", "clean_tree", "source_digests",
            "implementation_digests", "python_version", "numpy_version", "torch_version",
            "platform_machine", "platform_system", "blas_fingerprint"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-C0CAL-BUILD-PROVENANCE-V1" or \
       value.get("target_commit") != TARGET_COMMIT or value.get("clean_tree") is not True or \
       value.get("source_digests") != SOURCE_DIGESTS:
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    commit = value.get("implementation_commit")
    implementation = value.get("implementation_digests")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit) or \
       not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS) or \
       any(not _is_digest(item) for item in implementation.values()) or \
       any(not isinstance(value.get(name), str) or not value[name] for name in (
           "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system",
           "blas_fingerprint",
       )):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    return C0CalBuildProvenance(**value)


def _record(purpose: str, path: Sequence[int]) -> dict[str, object]:
    values = list(path)
    if not purpose.isascii() or not values or any(
        isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in values
    ):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return {"purpose": purpose, "path": values}


def generated_seed_audit(manifest: C0CalSeedManifest) -> list[dict[str, object]]:
    validate_seed_manifest(manifest)
    records = [_record("GENERATOR_SPLIT", (manifest.generator_seed, split)) for split in (1, 3, 6)]
    records.append(_record("TRAIN_NUISANCE", (manifest.generator_seed, 1, 7101)))
    records.extend((
        _record("E0_INIT", (manifest.model_seed, 1)),
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


def validate_closed_lineages_inventory(inventory: object) -> None:
    one.validate_closed_lineage_inventory(inventory)
    if not isinstance(inventory, dict):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    sources = inventory.get("source_artifact_digests")
    records = inventory.get("records")
    if not isinstance(sources, dict) or sources.get(CLOSED_1M_SOURCE_KEY) != CLOSED_1M_AUDIT_SHA256 or \
       not isinstance(records, list):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    extracted = []
    for item in records:
        if isinstance(item, dict) and isinstance(item.get("purpose"), str) and \
           item["purpose"].startswith(CLOSED_1M_PURPOSE_PREFIX):
            extracted.append({
                "purpose": item["purpose"][len(CLOSED_1M_PURPOSE_PREFIX):],
                "path": item.get("path"),
            })
    extracted.sort(key=lambda item: (item["purpose"], item["path"]))
    if len(extracted) != 40 or Counter(item["purpose"] for item in extracted) != CLOSED_1M_PURPOSE_COUNTS or \
       len({item["path"][0] for item in extracted}) != 3 or \
       full.sha256_hex(full.canonical_json_bytes(extracted)) != CLOSED_1M_AUDIT_SHA256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")


def validate_seed_audit(
    manifest: C0CalSeedManifest, manifest_raw: bytes, envelope: C0CalApprovedSeedEnvelope,
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
    validate_closed_lineages_inventory(inventory)
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


def make_bias_free_c0_head(model_seed: int) -> nn.Linear:
    def factory() -> nn.Module:
        head = nn.Linear(16, 1, bias=False)
        nn.init.xavier_uniform_(head.weight)
        return head
    head = full._with_preserved_cpu_rng(
        full.component_seed(model_seed, full.C0_HEAD_COMPONENT_CODE), factory,
    )
    if not isinstance(head, nn.Linear) or head.bias is not None or \
       not torch.isfinite(head.weight).all() or not bool(torch.count_nonzero(head.weight)):
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    return head


def train_c0_conditioned(
    train: object, transform: object, model_seed: int, encoder: object,
    train_r0_logits: np.ndarray, schedule: Iterable[np.ndarray], *, expected_steps: int,
) -> ConditionedC0Product:
    e0_before = full._validate_frozen_encoder(encoder)
    offsets = np.ascontiguousarray(np.asarray(train_r0_logits, dtype="<f8"))
    if offsets.shape != (train.S.shape[0],) or not np.isfinite(offsets).all():
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    offsets_before = offsets.tobytes()
    adapter = full.TokenwiseResidualAdapter().train()
    head = make_bias_free_c0_head(model_seed).train()
    optimizer = full._adamw(list(adapter.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4)
    if not full.optimizer_membership((adapter, head), optimizer):
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    totals: list[float] = []
    attempted = successful = 0
    for indices in schedule:
        attempted += 1
        prefix_ids, prefix_times = full._prefix_inputs(train, transform, "L1_AVG", indices)
        labels = full._tensor_rows(train.S[:, 0], indices, dtype=torch.float32)
        frozen_offsets = torch.as_tensor(offsets[indices], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        _, delta = full.encode_context(encoder, adapter, prefix_ids, prefix_times)
        residual_logits = head(delta).squeeze(-1)
        if attempted == 1 and (
            bool(torch.count_nonzero(residual_logits)) or
            not torch.equal(frozen_offsets + residual_logits, frozen_offsets)
        ):
            raise PrototypeInvariantError("TRAINING_INVARIANT")
        loss = F.binary_cross_entropy_with_logits(frozen_offsets + residual_logits, labels, reduction="mean")
        loss.backward()
        if attempted == 1 and not full.first_w2_gradient_is_finite_nonzero(adapter):
            raise PrototypeInvariantError("TRAINING_INVARIANT")
        full.successful_optimizer_update(loss, optimizer)
        successful += 1
        totals.append(float(loss.detach()))
    if attempted != expected_steps or successful != expected_steps or \
       full._state_dict_bytes(encoder) != e0_before or offsets.tobytes() != offsets_before:
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    adapter.eval()
    head.eval()
    condition = full.FrozenResidualCondition(
        "C0_CONDITIONED", encoder, adapter, None, None,
        full._training_summary(totals, (), (), (), successful, attempted, 0, c0=True),
    )
    return ConditionedC0Product(
        condition=condition,
        training_head_hash=full.canonical_state_sha256(head),
        train_r0_logit_hash=full.array_sha256("train_r0.logits", offsets),
    )


def bootstrap_one_lcb(
    row_contrast: np.ndarray, bootstrap_root: int, *, replicates: int = 10000,
    supplied_indices: np.ndarray | None = None,
) -> tuple[float, list[dict[str, object]], np.ndarray]:
    values = np.ascontiguousarray(np.asarray(row_contrast, dtype="<f8"))
    n = values.size
    if values.shape != (n,) or n == 0 or not np.isfinite(values).all():
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    observed = float(values.mean())
    if supplied_indices is None:
        if replicates != 10000 or n != 2048:
            raise PrototypeInvariantError("BOOTSTRAP_INVALID")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([bootstrap_root, 7601])))
        indices = rng.integers(0, n, size=(replicates, n), dtype=np.int64, endpoint=False)
    else:
        indices = np.ascontiguousarray(np.asarray(supplied_indices, dtype=np.int64))
        if indices.shape != (replicates, n) or np.any(indices < 0) or np.any(indices >= n):
            raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    centered = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        selected = indices[start:start + 64]
        centered[start:start + selected.shape[0]] = values[selected].mean(axis=1) - observed
    critical = float(np.quantile(centered, 0.95, method="linear"))
    if not math.isfinite(critical):
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    contrast = [{"name": "d_C0", "observed": observed, "lcb95": float(observed - critical)}]
    return critical, contrast, indices


def evaluate_terminal_outcome(contrasts: Sequence[Mapping[str, object]], *, valid: bool) -> tuple[str, bool, dict[str, bool]]:
    if not valid:
        return "INVALID", False, {}
    if len(contrasts) != 1 or contrasts[0].get("name") != "d_C0":
        raise PrototypeInvariantError("BOOTSTRAP_INVALID")
    gate = float(contrasts[0]["lcb95"]) > 0.0
    return ("ELIGIBLE" if gate else "INELIGIBLE"), gate, {"d_c0": gate}


def failure_artifact(phase: str, error_code: str) -> dict[str, object]:
    if phase not in FAILURE_PHASES or error_code not in FAILURE_CODES:
        phase, error_code = "SERIALIZATION", "SERIALIZATION_INVALID"
    result = {
        "schema": "BP011-J04C-V3-R0RESID-C0CAL-INVALID-V1", "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256, "terminal_outcome": "INVALID",
        "phase": phase, "error_code": error_code,
    }
    validate_failure_schema(result)
    return result


def validate_failure_schema(value: object) -> None:
    keys = {"schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-C0CAL-INVALID-V1" or \
       value.get("namespace") != NAMESPACE or value.get("contract_sha256") != CONTRACT_SHA256 or \
       value.get("terminal_outcome") != "INVALID" or value.get("phase") not in FAILURE_PHASES or \
       value.get("error_code") not in FAILURE_CODES:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value, failure=True)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_digest_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if (key.endswith("_sha256") or key.endswith("_hash")) and not _is_digest(nested):
                raise PrototypeInvariantError("SERIALIZATION_INVALID")
            _validate_digest_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_digest_fields(nested)


def _validate_training_summary(value: object) -> None:
    keys = {"attempted_steps", "successful_steps", "optimizer_steps", "ema_updates",
            "first_100_mean_total", "last_100_mean_total", "component_scalars"}
    counts = tuple(value.get(name) for name in (
        "attempted_steps", "successful_steps", "optimizer_steps", "ema_updates"
    )) if isinstance(value, dict) else ()
    component_keys = {"cosine_first", "cosine_last", "directional_first", "directional_last",
                      "v_direction_min_first", "v_direction_min_last"}
    if not isinstance(value, dict) or set(value) != keys or \
       any(isinstance(item, bool) or not isinstance(item, int) for item in counts) or \
       counts != (2000, 2000, 2000, 0) or \
       not _finite_number(value["first_100_mean_total"]) or not _finite_number(value["last_100_mean_total"]) or \
       not isinstance(value["component_scalars"], dict) or \
       set(value["component_scalars"]) != component_keys or \
       any(item is not None for item in value["component_scalars"].values()):
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


def _fixed_contract() -> dict[str, object]:
    return {
        "generator_count": 1, "model_count": 1, "train_n": 8192, "probe_n": 2048, "cal_n": 2048,
        "batch_size": 128, "updates": 2000,
        "adapter": "W2_GELU_W1_PARAMETER_FREE_LN_TOKENWISE",
        "training_objective": "FROZEN_TRAIN_R0_PLUS_BIAS_FREE_RESIDUAL",
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001,
                      "betas": [0.9, 0.999], "eps": 1e-8},
        "readout_ridge": 0.001, "bootstrap_replicates": 10000, "bootstrap_family_size": 1,
    }


SUCCESS_ROOT_KEYS = {
    "schema", "namespace", "contract_sha256", "claim_ceiling", "provenance", "seed_audit", "fixed",
    "checks", "model", "bootstrap", "valid", "eligible", "scientific_gates", "terminal_outcome",
}


def validate_success_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != SUCCESS_ROOT_KEYS or \
       value.get("schema") != "BP011-J04C-V3-R0RESID-C0CAL-RESULT-V1" or \
       value.get("namespace") != NAMESPACE or value.get("contract_sha256") != CONTRACT_SHA256 or \
       value.get("claim_ceiling") != "SAFE_PUBLIC_C0_ASSAY_CALIBRATION_ONLY" or \
       value.get("valid") is not True or value.get("terminal_outcome") not in {"ELIGIBLE", "INELIGIBLE"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    provenance = value.get("provenance")
    seed_audit = value.get("seed_audit")
    checks = value.get("checks")
    model = value.get("model")
    bootstrap = value.get("bootstrap")
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
       )) or any(
           isinstance(seed_audit[name], bool) or not isinstance(seed_audit[name], int) or seed_audit[name] < 0
           for name in ("production_path_count", "historical_path_count",
                        "path_intersection_count", "root_intersection_count")
       ) or seed_audit["production_path_count"] != 39 or seed_audit["path_intersection_count"] != 0 or \
       seed_audit["root_intersection_count"] != 0:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    expected_checks = {
        "source_pins", "seed_audit", "e0_immutable", "pooling_exact", "successful_steps",
        "train_r0_only", "train_r0_frozen", "c0_head_bias_free", "first_w2_gradient_nonzero",
        "readout_deterministic", "baseline_frozen", "family_exact", "output_allowlist",
    }
    if value.get("fixed") != _fixed_contract() or not isinstance(checks, dict) or \
       set(checks) != expected_checks or not all(item is True for item in checks.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    if not isinstance(model, dict) or set(model) != {"state_hashes", "training", "readouts", "cal"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    state_hashes, training, readouts, cal = (
        model["state_hashes"], model["training"], model["readouts"], model["cal"],
    )
    if not all(isinstance(item, dict) for item in (state_hashes, training, readouts, cal)) or \
       set(state_hashes) != {"e0", "c0_adapter", "c0_training_head", "train_r0_logit_hash"} or \
       set(training) != {"C0_CONDITIONED"} or \
       set(readouts) != {"TRAIN_R0_OPTIMIZATION_ONLY", "PROBE_R0_BASE", "C0_CONDITIONED_ADDITIVE"} or \
       set(cal) != {"R0_BASE", "C0_CONDITIONED_ADDITIVE"} or \
       any(not _is_digest(item) for item in state_hashes.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    _validate_training_summary(training["C0_CONDITIONED"])
    for item in readouts.values():
        _validate_readout_summary(item)
    for item in cal.values():
        _validate_assay_summary(item)
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "replicates", "family_size", "quantile_method", "critical_value", "contrasts"
    } or any(isinstance(bootstrap.get(name), bool) or not isinstance(bootstrap.get(name), int)
             for name in ("replicates", "family_size")) or \
       bootstrap.get("replicates") != 10000 or bootstrap.get("family_size") != 1 or \
       bootstrap.get("quantile_method") != "linear" or not _finite_number(bootstrap.get("critical_value")) or \
       not isinstance(bootstrap.get("contrasts"), list) or len(bootstrap["contrasts"]) != 1 or \
       not isinstance(bootstrap["contrasts"][0], dict) or \
       bootstrap["contrasts"][0].get("name") != "d_C0" or \
       set(bootstrap["contrasts"][0]) != {"name", "observed", "lcb95"} or \
       not _finite_number(bootstrap["contrasts"][0]["observed"]) or \
       not _finite_number(bootstrap["contrasts"][0]["lcb95"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    outcome, eligible, gates = evaluate_terminal_outcome(bootstrap["contrasts"], valid=True)
    if value["terminal_outcome"] != outcome or value.get("eligible") is not eligible or \
       value.get("scientific_gates") != gates:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value)
    _validate_digest_fields(value)


def run_c0cal_beta(
    manifest: C0CalSeedManifest, provenance: C0CalBuildProvenance, seed_audit: Mapping[str, object],
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
    base_condition = full.FrozenResidualCondition("R0_BASE", encoder, None, None, None, {})
    train_base = full.extract_feature_blocks(base_condition, train, transform)
    labels_train = train.S[:, 0].astype(np.float64)
    train_r0_summary, train_r0_logits, _ = full._fit_base_bundle(
        train_base.z0, labels_train, {}, "TRAIN_R0_OPTIMIZATION_ONLY",
    )
    c0_product = train_c0_conditioned(
        train, transform, manifest.model_seed, encoder, train_r0_logits["probe"],
        full.pretraining_indices(manifest.model_seed), expected_steps=2000,
    )
    if full._state_dict_bytes(encoder) != e0_before:
        raise PrototypeInvariantError("TRAINING_INVARIANT")

    notify("READOUT")
    base_probe = full.extract_feature_blocks(base_condition, probe, transform)
    base_cal = full.extract_feature_blocks(base_condition, cal, transform)
    c0_probe = full.extract_feature_blocks(c0_product.condition, probe, transform)
    c0_cal = full.extract_feature_blocks(c0_product.condition, cal, transform)
    if base_probe.z0.tobytes() != c0_probe.z0.tobytes() or base_cal.z0.tobytes() != c0_cal.z0.tobytes():
        raise PrototypeInvariantError("GENERATION_INVARIANT")
    labels_probe = probe.S[:, 0].astype(np.float64)
    labels_cal = cal.S[:, 0].astype(np.float64)
    r0_summary, r0_logits, _ = full._fit_base_bundle(
        base_probe.z0, labels_probe, {"cal": base_cal.z0}, "PROBE_R0_BASE",
    )
    c0_summary, c0_logits, _ = full._fit_additive_bundle(
        c0_probe.delta_z, labels_probe, r0_logits["probe"], {"cal": c0_cal.delta_z},
        {"cal": r0_logits["cal"]}, "C0_CONDITIONED_ADDITIVE",
    )
    nll_r0 = full.binary_row_nll(r0_logits["cal"], labels_cal)
    nll_c0 = full.binary_row_nll(c0_logits["cal"], labels_cal)
    notify("BOOTSTRAP")
    critical, contrasts, _ = bootstrap_one_lcb(nll_r0 - nll_c0, manifest.bootstrap_root)
    outcome, eligible, gates = evaluate_terminal_outcome(contrasts, valid=True)
    notify("SERIALIZATION")
    result = {
        "schema": "BP011-J04C-V3-R0RESID-C0CAL-RESULT-V1",
        "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256,
        "claim_ceiling": "SAFE_PUBLIC_C0_ASSAY_CALIBRATION_ONLY",
        "provenance": {
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
        },
        "seed_audit": {"approved_envelope_sha256": approved_envelope_sha256, **dict(seed_audit)},
        "fixed": _fixed_contract(),
        "checks": {
            "source_pins": True, "seed_audit": True, "e0_immutable": True, "pooling_exact": True,
            "successful_steps": True, "train_r0_only": True, "train_r0_frozen": True,
            "c0_head_bias_free": True, "first_w2_gradient_nonzero": True,
            "readout_deterministic": True, "baseline_frozen": True, "family_exact": True,
            "output_allowlist": True,
        },
        "model": {
            "state_hashes": {
                "e0": full.canonical_state_sha256(encoder),
                "c0_adapter": full.canonical_state_sha256(c0_product.condition.adapter),
                "c0_training_head": c0_product.training_head_hash,
                "train_r0_logit_hash": c0_product.train_r0_logit_hash,
            },
            "training": {"C0_CONDITIONED": c0_product.condition.training},
            "readouts": {
                "TRAIN_R0_OPTIMIZATION_ONLY": train_r0_summary,
                "PROBE_R0_BASE": r0_summary,
                "C0_CONDITIONED_ADDITIVE": c0_summary,
            },
            "cal": {
                "R0_BASE": full.assay_summary(r0_logits["cal"], labels_cal, "cal.R0_BASE"),
                "C0_CONDITIONED_ADDITIVE": full.assay_summary(
                    c0_logits["cal"], labels_cal, "cal.C0_CONDITIONED_ADDITIVE",
                ),
            },
        },
        "bootstrap": {
            "replicates": 10000, "family_size": 1, "quantile_method": "linear",
            "critical_value": critical, "contrasts": contrasts,
        },
        "valid": True,
        "eligible": eligible,
        "scientific_gates": gates,
        "terminal_outcome": outcome,
    }
    validate_success_schema(result)
    return result
