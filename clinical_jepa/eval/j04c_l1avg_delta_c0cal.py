"""Full-encoder pooled-delta C0 assay-calibration beta for BP011."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from clinical_jepa.eval import j04c_v3_r0resid as full
from clinical_jepa.eval import j04c_v3_r0resid_c0cal as prior
from clinical_jepa.eval import j04c_stage1 as stage1

NAMESPACE = "BP011-J04C-L1AVG-DELTA-C0CAL-1P1M-K0"
CONTRACT_SHA256 = "eca7c96f4a2ac32513231e85a30d980f065c02bb79670d10585bad6fe127dbec"
TARGET_COMMIT = "305d76a73b3d50ec9199c5b253e92decbe425b5f"
PRODUCTION_PATH_COUNT = 39
PAIR_S = (0.425, 0.525, 0.575)
PAIR_FLIP = (0.125, 0.185, 0.165)
IMPLEMENTATION_PATHS = (
    "clinical_jepa/eval/j04c_l1avg_delta_c0cal.py",
    "scripts/bp_clinjepa_011_j04c_l1avg_delta_c0cal_beta.py",
    "tests/test_bp_clinjepa_011_j04c_l1avg_delta_c0cal.py",
)
SOURCE_DIGESTS = {
    **prior.SOURCE_DIGESTS,
    "clinical_jepa/eval/j04c_v3_r0resid_c0cal.py":
        "0cba0bf223496293e174ce5461943a98dc5d6e037f9917610a2dbb4f3bda54eb",
    "scripts/bp_clinjepa_011_j04c_l1_generator_family.py":
        "f5c472768ecd2453f32455ac3c959b707bf8c9f025336e9ad5d92eae6fcb3950",
}
CLOSED_C0CAL_AUDIT_SHA256 = "85d63dfc7552a6f13d193cee49ce45b5ea025210544b2ac7187a1cdc1eb96f52"
CLOSED_C0CAL_SOURCE_KEY = "closed-c0cal/production-generated-seed-audit.json"
CLOSED_C0CAL_PREFIX = "CLOSED_C0CAL_K0__"
CLOSED_C0CAL_COUNTS = {
    "GENERATOR_SPLIT": 3, "TRAIN_NUISANCE": 1, "E0_INIT": 1,
    "C0_HEAD_INIT": 1, "TRAIN_SCHEDULE": 32, "BOOTSTRAP": 1,
}

PrototypeInvariantError = full.PrototypeInvariantError
FAILURE_CODES = full.FAILURE_CODES
FAILURE_PHASES = full.FAILURE_PHASES


@dataclass(frozen=True)
class SeedManifest:
    schema: Literal["BP011-J04C-L1AVG-DELTA-C0CAL-SEEDS-V1"]
    train_generator_seed: int
    heldout_generator_seed: int
    model_seed: int
    bootstrap_root: int


@dataclass(frozen=True)
class ApprovedSeedEnvelope:
    schema: Literal["BP011-J04C-L1AVG-DELTA-C0CAL-SEED-APPROVAL-V1"]
    manifest_sha256: str
    historical_inventory_sha256: str
    expected_generated_audit_sha256: str
    production_path_count: Literal[39]


@dataclass(frozen=True)
class BuildProvenance:
    schema: Literal["BP011-J04C-L1AVG-DELTA-C0CAL-BUILD-PROVENANCE-V1"]
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
class C0Product:
    encoder: nn.Module
    training: dict[str, object]
    head_hash: str
    train_r0_logit_hash: str


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def seed_manifest_from_dict(value: object) -> SeedManifest:
    keys = {"schema", "train_generator_seed", "heldout_generator_seed", "model_seed", "bootstrap_root"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-L1AVG-DELTA-C0CAL-SEEDS-V1":
        raise PrototypeInvariantError("INPUT_SCHEMA")
    manifest = SeedManifest(**value); validate_seed_manifest(manifest); return manifest


def validate_seed_manifest(manifest: SeedManifest) -> None:
    roots = (manifest.train_generator_seed, manifest.heldout_generator_seed,
             manifest.model_seed, manifest.bootstrap_root)
    if manifest.schema != "BP011-J04C-L1AVG-DELTA-C0CAL-SEEDS-V1" or any(
        isinstance(x, bool) or not isinstance(x, int) or not 2**31 <= x < 2**32 for x in roots
    ):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    if len(set(roots)) != 4: raise PrototypeInvariantError("SEED_COLLISION")


def approved_envelope_from_dict(value: object) -> ApprovedSeedEnvelope:
    keys = {"schema", "manifest_sha256", "historical_inventory_sha256",
            "expected_generated_audit_sha256", "production_path_count"}
    count = value.get("production_path_count") if isinstance(value, dict) else None
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-L1AVG-DELTA-C0CAL-SEED-APPROVAL-V1" or \
       not isinstance(count, int) or isinstance(count, bool) or count != 39 or any(
           not _is_digest(value.get(name)) for name in (
               "manifest_sha256", "historical_inventory_sha256", "expected_generated_audit_sha256"
           )
       ):
        raise PrototypeInvariantError("INPUT_SCHEMA")
    return ApprovedSeedEnvelope(**value)


def build_provenance_from_dict(value: object) -> BuildProvenance:
    keys = {"schema", "target_commit", "implementation_commit", "clean_tree", "source_digests",
            "implementation_digests", "python_version", "numpy_version", "torch_version",
            "platform_machine", "platform_system", "blas_fingerprint"}
    if not isinstance(value, dict) or set(value) != keys or \
       value.get("schema") != "BP011-J04C-L1AVG-DELTA-C0CAL-BUILD-PROVENANCE-V1" or \
       value.get("target_commit") != TARGET_COMMIT or value.get("clean_tree") is not True or \
       value.get("source_digests") != SOURCE_DIGESTS:
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    commit = value.get("implementation_commit"); implementation = value.get("implementation_digests")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit) or \
       not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_PATHS) or \
       any(not _is_digest(item) for item in implementation.values()) or any(
           not isinstance(value.get(name), str) or not value[name] for name in (
               "python_version", "numpy_version", "torch_version", "platform_machine", "platform_system",
               "blas_fingerprint",
           )
       ):
        raise PrototypeInvariantError("PROVENANCE_CONTENT")
    return BuildProvenance(**value)


def _record(purpose: str, path: Sequence[int]) -> dict[str, object]:
    values = list(path)
    if not values or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in values):
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return {"purpose": purpose, "path": values}


def generated_seed_audit(manifest: SeedManifest) -> list[dict[str, object]]:
    validate_seed_manifest(manifest)
    records = [
        _record("TRAIN_GENERATOR_SPLIT", (manifest.train_generator_seed, 1)),
        _record("TRAIN_NUISANCE", (manifest.train_generator_seed, 1, 7101)),
        _record("HELDOUT_PROBE", (manifest.heldout_generator_seed, 6)),
        _record("HELDOUT_CAL", (manifest.heldout_generator_seed, 3)),
        _record("E0_INIT", (manifest.model_seed, 1)),
        _record("C0_HEAD_INIT", (manifest.model_seed, 40)),
    ]
    records.extend(_record("TRAIN_SCHEDULE", (manifest.model_seed, epoch, 6101)) for epoch in range(32))
    records.append(_record("BOOTSTRAP", (manifest.bootstrap_root, 7601)))
    records.sort(key=lambda item: (item["purpose"], item["path"]))
    if len(records) != 39 or len({(item["purpose"], tuple(item["path"])) for item in records}) != 39:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    return records


def validate_closed_inventory(inventory: object) -> None:
    prior.validate_closed_lineages_inventory(inventory)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("records"), list) or \
       not isinstance(inventory.get("source_artifact_digests"), dict) or \
       inventory["source_artifact_digests"].get(CLOSED_C0CAL_SOURCE_KEY) != CLOSED_C0CAL_AUDIT_SHA256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    extracted = []
    for item in inventory["records"]:
        if isinstance(item, dict) and isinstance(item.get("purpose"), str) and item["purpose"].startswith(CLOSED_C0CAL_PREFIX):
            extracted.append({"purpose": item["purpose"][len(CLOSED_C0CAL_PREFIX):], "path": item.get("path")})
    extracted.sort(key=lambda item: (item["purpose"], item["path"]))
    if len(extracted) != 39 or Counter(item["purpose"] for item in extracted) != CLOSED_C0CAL_COUNTS or \
       len({item["path"][0] for item in extracted}) != 3 or \
       full.sha256_hex(full.canonical_json_bytes(extracted)) != CLOSED_C0CAL_AUDIT_SHA256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")


def validate_seed_audit(
    manifest: SeedManifest, manifest_raw: bytes, envelope: ApprovedSeedEnvelope,
    inventory: object, inventory_raw: bytes,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if full.sha256_hex(manifest_raw) != envelope.manifest_sha256 or \
       full.sha256_hex(inventory_raw) != envelope.historical_inventory_sha256:
        raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    records = generated_seed_audit(manifest); audit_digest = full.sha256_hex(full.canonical_json_bytes(records))
    if audit_digest != envelope.expected_generated_audit_sha256: raise PrototypeInvariantError("SEED_AUDIT_DIGEST")
    historical_roots, historical_paths = full.validate_historical_inventory(inventory); validate_closed_inventory(inventory)
    paths = {tuple(item["path"]) for item in records}; roots = {path[0] for path in paths}
    if paths & historical_paths or roots & historical_roots: raise PrototypeInvariantError("SEED_COLLISION")
    return records, {"manifest_sha256": envelope.manifest_sha256,
                     "historical_inventory_sha256": envelope.historical_inventory_sha256,
                     "generated_audit_sha256": audit_digest, "production_path_count": 39,
                     "historical_path_count": len(historical_paths), "path_intersection_count": 0,
                     "root_intersection_count": 0}


def make_head(model_seed: int) -> nn.Linear:
    def factory() -> nn.Module:
        head = nn.Linear(16, 1, bias=False); nn.init.xavier_uniform_(head.weight); return head
    head = full._with_preserved_cpu_rng(full.component_seed(model_seed, full.C0_HEAD_COMPONENT_CODE), factory)
    if not isinstance(head, nn.Linear) or head.bias is not None or not torch.isfinite(head.weight).all() or \
       not bool(torch.count_nonzero(head.weight)):
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    return head


def _pooled(encoder: nn.Module, split: object, transform: object, indices: np.ndarray | None = None,
            *, grad: bool) -> torch.Tensor:
    ids, times = stage1._prefix_inputs(split, transform, "L1_AVG", indices)
    if grad:
        _, _, z = encoder(ids, times, causal=False); return z
    with torch.no_grad():
        _, _, z = encoder(ids, times, causal=False); return z.detach()


def pooled_numpy(encoder: nn.Module, split: object, transform: object) -> np.ndarray:
    encoder.eval(); return np.ascontiguousarray(_pooled(encoder, split, transform, grad=False).numpy().astype("<f8"))


def _pooled_frozen_functional_reference(
    module: nn.Module, parameters: Mapping[str, torch.Tensor], buffers: Mapping[str, torch.Tensor],
    split: object, transform: object, indices: np.ndarray,
) -> torch.Tensor:
    ids, times = stage1._prefix_inputs(split, transform, "L1_AVG", indices)
    _, _, z = functional_call(module, (parameters, buffers), (ids, times), {"causal": False})
    return z.detach()


def train_conditioned_c0(
    train: object, transform: object, model_seed: int, e0: nn.Module, train_r0_logits: np.ndarray,
    schedule: Iterable[np.ndarray], *, expected_steps: int,
) -> C0Product:
    e0.eval()
    for parameter in e0.parameters(): parameter.requires_grad_(False); parameter.grad = None
    e0_before = full._state_dict_bytes(e0)
    c0 = copy.deepcopy(e0).train()
    for parameter in c0.parameters(): parameter.requires_grad_(True); parameter.grad = None
    if full._state_dict_bytes(c0) != e0_before: raise PrototypeInvariantError("TRAINING_INVARIANT")
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_(True) for name, parameter in e0.named_parameters()
    }
    reference_buffers = {name: buffer.detach().clone() for name, buffer in e0.named_buffers()}
    reference_before = {name: value.detach().clone() for name, value in reference_parameters.items()}
    offsets = np.ascontiguousarray(np.asarray(train_r0_logits, dtype="<f8")); offsets_before = offsets.tobytes()
    if offsets.shape != (train.S.shape[0],) or not np.isfinite(offsets).all():
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    head = make_head(model_seed).train(); optimizer = stage1._adamw(
        list(c0.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4,
    )
    if not stage1.optimizer_membership((c0, head), optimizer): raise PrototypeInvariantError("TRAINING_INVARIANT")
    totals: list[float] = []; attempted = successful = 0
    for indices in schedule:
        attempted += 1; optimizer.zero_grad(set_to_none=True)
        z0 = _pooled_frozen_functional_reference(
            c0, reference_parameters, reference_buffers, train, transform, indices,
        )
        zc = _pooled(c0, train, transform, indices, grad=True)
        delta = zc - z0; residual = head(delta).squeeze(-1)
        frozen_offsets = torch.as_tensor(offsets[indices], dtype=torch.float32)
        labels = stage1._tensor_rows(train.S[:, 0], indices, dtype=torch.float32)
        if attempted == 1 and (
            not torch.equal(zc.detach(), z0) or bool(torch.count_nonzero(delta.detach())) or
            bool(torch.count_nonzero(residual.detach())) or
            not torch.equal(frozen_offsets + residual.detach(), frozen_offsets)
        ):
            raise PrototypeInvariantError("TRAINING_INVARIANT")
        loss = F.binary_cross_entropy_with_logits(frozen_offsets + residual, labels, reduction="mean")
        loss.backward()
        head_grad = head.weight.grad
        encoder_grads = [p.grad for p in c0.parameters()]
        if attempted == 1 and (
            head_grad is None or bool(torch.count_nonzero(head_grad)) or
            any(g is None or not bool(torch.isfinite(g).all()) for g in encoder_grads) or
            not any(bool(torch.count_nonzero(g)) for g in encoder_grads) or
            any(p.grad is not None for p in e0.parameters()) or
            any(p.grad is not None for p in reference_parameters.values())
        ):
            raise PrototypeInvariantError("TRAINING_INVARIANT")
        full.successful_optimizer_update(loss, optimizer); successful += 1; totals.append(float(loss.detach()))
    if attempted != expected_steps or successful != expected_steps or full._state_dict_bytes(e0) != e0_before or \
       offsets.tobytes() != offsets_before or any(
           not torch.equal(reference_parameters[name].detach(), value)
           for name, value in reference_before.items()
       ):
        raise PrototypeInvariantError("TRAINING_INVARIANT")
    c0.eval(); head.eval()
    training = {"attempted_steps": attempted, "successful_steps": successful,
                "optimizer_steps": successful, "ema_updates": 0,
                "first_100_mean_total": float(np.mean(totals[:100])),
                "last_100_mean_total": float(np.mean(totals[-100:]))}
    return C0Product(c0, training, full.canonical_state_sha256(head),
                     full.array_sha256("train_r0.logits", offsets))


def failure_artifact(phase: str, error_code: str) -> dict[str, object]:
    if phase not in FAILURE_PHASES or error_code not in FAILURE_CODES:
        phase, error_code = "SERIALIZATION", "SERIALIZATION_INVALID"
    value = {"schema": "BP011-J04C-L1AVG-DELTA-C0CAL-INVALID-V1", "namespace": NAMESPACE,
             "contract_sha256": CONTRACT_SHA256, "terminal_outcome": "INVALID",
             "phase": phase, "error_code": error_code}
    validate_failure_schema(value); return value


def validate_failure_schema(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"schema", "namespace", "contract_sha256", "terminal_outcome", "phase", "error_code"} or \
       value.get("schema") != "BP011-J04C-L1AVG-DELTA-C0CAL-INVALID-V1" or value.get("namespace") != NAMESPACE or \
       value.get("contract_sha256") != CONTRACT_SHA256 or value.get("terminal_outcome") != "INVALID" or \
       value.get("phase") not in FAILURE_PHASES or value.get("error_code") not in FAILURE_CODES:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value, failure=True)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fixed() -> dict[str, object]:
    return {"train_n": 8192, "probe_n": 2048, "cal_n": 2048, "model_count": 1,
            "s_probabilities": list(PAIR_S), "signal_flip_probabilities": list(PAIR_FLIP),
            "target_factor": 0, "updates": 2000, "batch_size": 128,
            "feature": "POOLED_ONLINE_ENCODER_DELTA_Z_C0_MINUS_Z0",
            "training_objective": "FROZEN_TRAIN_R0_PLUS_BIAS_FREE_RESIDUAL",
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4,
                          "betas": [0.9, 0.999], "eps": 1e-8},
            "readout_ridge": 1e-3, "bootstrap_replicates": 10000, "bootstrap_family_size": 1}


def _validate_readout_summary(value: object) -> None:
    keys = {"preprocess_hash", "coefficient_hash", "constant_coordinate_mask_hash",
            "iterations", "converged", "zero_short_circuit"}
    if not isinstance(value, dict) or set(value) != keys or any(
        not _is_digest(value[name]) for name in (
            "preprocess_hash", "coefficient_hash", "constant_coordinate_mask_hash"
        )
    ) or isinstance(value["iterations"], bool) or not isinstance(value["iterations"], int) or \
       value["iterations"] < 0 or value["converged"] is not True or \
       not isinstance(value["zero_short_circuit"], bool):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


def _validate_assay_summary(value: object) -> None:
    keys = {"nll_mean", "balanced_accuracy", "row_nll_hash", "logit_hash"}
    if not isinstance(value, dict) or set(value) != keys or \
       not _finite(value["nll_mean"]) or float(value["nll_mean"]) < 0.0 or \
       not _finite(value["balanced_accuracy"]) or not 0.0 <= float(value["balanced_accuracy"]) <= 1.0 or \
       not _is_digest(value["row_nll_hash"]) or not _is_digest(value["logit_hash"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")


def validate_success_schema(value: object) -> None:
    roots = {"schema", "namespace", "contract_sha256", "claim_ceiling", "provenance", "seed_audit",
             "fixed", "checks", "model", "bootstrap", "valid", "eligible", "scientific_gates", "terminal_outcome"}
    if not isinstance(value, dict) or set(value) != roots or \
       value.get("schema") != "BP011-J04C-L1AVG-DELTA-C0CAL-RESULT-V1" or value.get("namespace") != NAMESPACE or \
       value.get("contract_sha256") != CONTRACT_SHA256 or \
       value.get("claim_ceiling") != "SAFE_PUBLIC_FULL_ENCODER_DELTA_C0_ASSAY_CALIBRATION_ONLY" or \
       value.get("valid") is not True or not isinstance(value.get("eligible"), bool) or \
       value.get("terminal_outcome") not in {"ELIGIBLE", "INELIGIBLE"} or value.get("fixed") != _fixed():
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    checks = value.get("checks")
    expected_checks = {"source_pins", "seed_audit", "e0_clone_exact", "e0_immutable", "train_r0_only",
                       "train_r0_frozen", "initial_delta_zero", "head_bias_free", "first_encoder_gradient_nonzero",
                       "first_head_gradient_zero", "successful_steps", "readout_deterministic", "baseline_frozen",
                       "family_exact", "output_allowlist", "pair_exact"}
    if not isinstance(checks, dict) or set(checks) != expected_checks or not all(v is True for v in checks.values()):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    provenance = value.get("provenance")
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
    seed = value.get("seed_audit")
    seed_keys = {"approved_envelope_sha256", "manifest_sha256", "historical_inventory_sha256",
                 "generated_audit_sha256", "production_path_count", "historical_path_count",
                 "path_intersection_count", "root_intersection_count"}
    count_names = ("production_path_count", "historical_path_count", "path_intersection_count", "root_intersection_count")
    if not isinstance(seed, dict) or set(seed) != seed_keys or any(
        not _is_digest(seed[name]) for name in (
            "approved_envelope_sha256", "manifest_sha256", "historical_inventory_sha256", "generated_audit_sha256"
        )
    ) or any(isinstance(seed[name], bool) or not isinstance(seed[name], int) or seed[name] < 0 for name in count_names) or \
       seed["production_path_count"] != 39 or seed["path_intersection_count"] != 0 or seed["root_intersection_count"] != 0:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    model = value.get("model")
    if not isinstance(model, dict) or set(model) != {"state_hashes", "training", "readouts", "cal"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    state_hashes, training_map, readouts, assays = (
        model["state_hashes"], model["training"], model["readouts"], model["cal"]
    )
    if not all(isinstance(item, dict) for item in (state_hashes, training_map, readouts, assays)) or \
       set(state_hashes) != {"e0", "c0_encoder", "c0_training_head", "train_r0_logit_hash"} or \
       any(not _is_digest(x) for x in state_hashes.values()) or \
       set(training_map) != {"C0_FULL_ENCODER_CONDITIONED"} or \
       set(readouts) != {"TRAIN_R0_OPTIMIZATION_ONLY", "PROBE_R0_BASE", "C0_DELTA_ADDITIVE"} or \
       set(assays) != {"R0_BASE", "C0_DELTA_ADDITIVE"}:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    training = training_map["C0_FULL_ENCODER_CONDITIONED"]
    training_keys = {"attempted_steps", "successful_steps", "optimizer_steps", "ema_updates",
                     "first_100_mean_total", "last_100_mean_total"}
    counts = tuple(training.get(name) for name in (
        "attempted_steps", "successful_steps", "optimizer_steps", "ema_updates"
    )) if isinstance(training, dict) else ()
    if not isinstance(training, dict) or set(training) != training_keys or \
       any(isinstance(x, bool) or not isinstance(x, int) for x in counts) or counts != (2000, 2000, 2000, 0) or \
       not _finite(training["first_100_mean_total"]) or not _finite(training["last_100_mean_total"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    for summary in readouts.values(): _validate_readout_summary(summary)
    for summary in assays.values(): _validate_assay_summary(summary)
    boot = value.get("bootstrap")
    boot_keys = {"replicates", "family_size", "quantile_method", "critical_value", "contrasts"}
    if not isinstance(boot, dict) or set(boot) != boot_keys or \
       any(isinstance(boot.get(name), bool) or not isinstance(boot.get(name), int) for name in ("replicates", "family_size")) or \
       boot["replicates"] != 10000 or boot["family_size"] != 1 or boot.get("quantile_method") != "linear" or \
       not _finite(boot.get("critical_value")) or not isinstance(boot.get("contrasts"), list) or len(boot["contrasts"]) != 1 or \
       not isinstance(boot["contrasts"][0], dict) or set(boot["contrasts"][0]) != {"name", "observed", "lcb95"} or \
       boot["contrasts"][0].get("name") != "d_C0" or not _finite(boot["contrasts"][0]["observed"]) or \
       not _finite(boot["contrasts"][0]["lcb95"]):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    gates = value.get("scientific_gates")
    if not isinstance(gates, dict) or set(gates) != {"d_c0"} or not isinstance(gates["d_c0"], bool):
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    outcome, eligible, expected_gates = prior.evaluate_terminal_outcome(boot["contrasts"], valid=True)
    if value["terminal_outcome"] != outcome or value["eligible"] is not eligible or gates != expected_gates:
        raise PrototypeInvariantError("SERIALIZATION_INVALID")
    full.validate_recursive_output(value)
    prior._validate_digest_fields(value)

def run_beta(
    manifest: SeedManifest, provenance: BuildProvenance, seed_audit: Mapping[str, object], *,
    build_provenance_sha256: str, approved_envelope_sha256: str,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    from clinical_jepa.eval.j04c_falsifier import TRAIN, PROBE_FIT, CAL_OOD, fit_stage0_time_transform, generate_factor_split
    from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
    from scripts.bp_clinjepa_011_j04c_l1_generator_family import parameterized_split

    notify = phase_callback if phase_callback is not None else lambda _: None
    notify("GENERATION")
    train = independent_train_nuisance(generate_factor_split(manifest.train_generator_seed, TRAIN, 8192), manifest.train_generator_seed)
    probe = parameterized_split(manifest.heldout_generator_seed, PROBE_FIT, 2048, PAIR_S, PAIR_FLIP)
    cal_split = parameterized_split(manifest.heldout_generator_seed, CAL_OOD, 2048, PAIR_S, PAIR_FLIP)
    transform = fit_stage0_time_transform(train)
    e0 = stage1._fresh_encoder(manifest.model_seed).eval(); e0_before = full._state_dict_bytes(e0)
    notify("TRAINING")
    z0_train = pooled_numpy(e0, train, transform); y_train = train.S[:, 0].astype(np.float64)
    train_r0_summary, train_logits, _ = full._fit_base_bundle(z0_train, y_train, {}, "TRAIN_R0_OPTIMIZATION_ONLY")
    product = train_conditioned_c0(train, transform, manifest.model_seed, e0, train_logits["probe"],
                                   stage1.pretraining_indices(manifest.model_seed), expected_steps=2000)
    if full._state_dict_bytes(e0) != e0_before: raise PrototypeInvariantError("TRAINING_INVARIANT")
    notify("READOUT")
    z0_probe = pooled_numpy(e0, probe, transform); z0_cal = pooled_numpy(e0, cal_split, transform)
    zc_probe = pooled_numpy(product.encoder, probe, transform); zc_cal = pooled_numpy(product.encoder, cal_split, transform)
    delta_probe = np.ascontiguousarray(zc_probe - z0_probe); delta_cal = np.ascontiguousarray(zc_cal - z0_cal)
    y_probe = probe.S[:, 0].astype(np.float64); y_cal = cal_split.S[:, 0].astype(np.float64)
    r0_summary, r0_logits, _ = full._fit_base_bundle(z0_probe, y_probe, {"cal": z0_cal}, "PROBE_R0_BASE")
    c0_summary, c0_logits, _ = full._fit_additive_bundle(delta_probe, y_probe, r0_logits["probe"],
                                                         {"cal": delta_cal}, {"cal": r0_logits["cal"]},
                                                         "C0_DELTA_ADDITIVE")
    rows = full.binary_row_nll(r0_logits["cal"], y_cal) - full.binary_row_nll(c0_logits["cal"], y_cal)
    notify("BOOTSTRAP")
    critical, contrasts, _ = prior.bootstrap_one_lcb(rows, manifest.bootstrap_root)
    outcome, eligible, gates = prior.evaluate_terminal_outcome(contrasts, valid=True)
    notify("SERIALIZATION")
    result = {
        "schema": "BP011-J04C-L1AVG-DELTA-C0CAL-RESULT-V1", "namespace": NAMESPACE,
        "contract_sha256": CONTRACT_SHA256,
        "claim_ceiling": "SAFE_PUBLIC_FULL_ENCODER_DELTA_C0_ASSAY_CALIBRATION_ONLY",
        "provenance": {"build_provenance_sha256": build_provenance_sha256,
            "target_commit": provenance.target_commit, "implementation_commit": provenance.implementation_commit,
            "clean_tree": provenance.clean_tree, "source_digests": provenance.source_digests,
            "implementation_digests": provenance.implementation_digests,
            "python_version": provenance.python_version, "numpy_version": provenance.numpy_version,
            "torch_version": provenance.torch_version, "platform_machine": provenance.platform_machine,
            "platform_system": provenance.platform_system, "blas_fingerprint": provenance.blas_fingerprint},
        "seed_audit": {"approved_envelope_sha256": approved_envelope_sha256, **dict(seed_audit)},
        "fixed": _fixed(),
        "checks": {name: True for name in ("source_pins", "seed_audit", "e0_clone_exact", "e0_immutable",
            "train_r0_only", "train_r0_frozen", "initial_delta_zero", "head_bias_free",
            "first_encoder_gradient_nonzero", "first_head_gradient_zero", "successful_steps",
            "readout_deterministic", "baseline_frozen", "family_exact", "output_allowlist", "pair_exact")},
        "model": {"state_hashes": {"e0": full.canonical_state_sha256(e0),
            "c0_encoder": full.canonical_state_sha256(product.encoder), "c0_training_head": product.head_hash,
            "train_r0_logit_hash": product.train_r0_logit_hash},
            "training": {"C0_FULL_ENCODER_CONDITIONED": product.training},
            "readouts": {"TRAIN_R0_OPTIMIZATION_ONLY": train_r0_summary, "PROBE_R0_BASE": r0_summary,
                         "C0_DELTA_ADDITIVE": c0_summary},
            "cal": {"R0_BASE": full.assay_summary(r0_logits["cal"], y_cal, "cal.R0_BASE"),
                    "C0_DELTA_ADDITIVE": full.assay_summary(c0_logits["cal"], y_cal, "cal.C0_DELTA_ADDITIVE")}},
        "bootstrap": {"replicates": 10000, "family_size": 1, "quantile_method": "linear",
                      "critical_value": critical, "contrasts": contrasts},
        "valid": True, "eligible": eligible, "scientific_gates": gates, "terminal_outcome": outcome,
    }
    validate_success_schema(result); return result
