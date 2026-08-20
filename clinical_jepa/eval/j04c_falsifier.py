"""Pure safe-public instrumentation for BP-CLINJEPA-011 J04c v2 Stage 0."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

import numpy as np
import torch
from torch import nn

from clinical_jepa.arms.v0f.own_latent import (
    EMATeacher,
    J04Encoder,
    NonAutoregressiveNextEventsHead,
    _initialize,
)
from clinical_jepa.targets.next_event_contract import (
    TimeTransform,
    construct_latent_targets,
    resolve_layer_sets,
)

TRAIN = 1
CAL_ID = 2
CAL_OOD = 3
ID_DEV = 4
OOD_DEV = 5
PROBE_FIT = 6

ANCHOR = 2
SIG_COMP_0, SIG_COMP_1 = 3, 4
SIG_ORDER_0, SIG_ORDER_1 = 5, 6
TIME_SIGNAL = 7
NUIS_COMP_0, NUIS_COMP_1 = 8, 9
NUIS_ORDER_0, NUIS_ORDER_1 = 10, 11
TIME_NUIS = 12
TYPE_A, TYPE_B, TYPE_C = 13, 14, 15
UNUSED_0, UNUSED_1 = 16, 17

_CORRELATED_SPLITS = frozenset((TRAIN, CAL_ID, ID_DEV))
_INDEPENDENT_SPLITS = frozenset((CAL_OOD, OOD_DEV, PROBE_FIT))


@dataclass(frozen=True)
class SyntheticFactorSplit:
    prefix_type_ids: np.ndarray
    prefix_intervals: np.ndarray
    target_type_ids: np.ndarray
    target_intervals: np.ndarray
    S: np.ndarray
    X: np.ndarray
    N: np.ndarray
    L_after: np.ndarray


def _validate_split(split: SyntheticFactorSplit) -> None:
    n = split.S.shape[0]
    expected = {
        "prefix_type_ids": ((n, 7), np.dtype("int64")),
        "prefix_intervals": ((n, 7), np.dtype("float64")),
        "target_type_ids": ((n, 4), np.dtype("int64")),
        "target_intervals": ((n, 4), np.dtype("float64")),
        "S": ((n, 3), np.dtype("uint8")),
        "X": ((n, 3), np.dtype("uint8")),
        "N": ((n, 3), np.dtype("uint8")),
        "L_after": ((n, 3), np.dtype("uint8")),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(split, name)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def generate_factor_split(generator_seed: int, split_code: int, n: int) -> SyntheticFactorSplit:
    """Generate the exact per-sequence S, then X, then N random stream."""
    if split_code not in _CORRELATED_SPLITS | _INDEPENDENT_SPLITS:
        raise ValueError("split_code must be in 1..6")
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([generator_seed, split_code])))
    s = np.empty((n, 3), dtype=np.uint8)
    x = np.empty((n, 3), dtype=np.uint8)
    nuisance = np.empty((n, 3), dtype=np.uint8)
    correlated = split_code in _CORRELATED_SPLITS
    for row in range(n):
        s_row = (rng.random(3) < 0.5).astype(np.uint8)
        x_flip = (rng.random(3) < 0.15).astype(np.uint8)
        n_draw = (rng.random(3) < (0.05 if correlated else 0.5)).astype(np.uint8)
        s[row] = s_row
        x[row] = np.bitwise_xor(s_row, x_flip)
        nuisance[row] = np.bitwise_xor(s_row, n_draw) if correlated else n_draw

    prefix_types = np.empty((n, 7), dtype=np.int64)
    prefix_types[:, 0] = ANCHOR
    prefix_types[:, 1] = np.where(x[:, 0] == 0, SIG_COMP_0, SIG_COMP_1)
    prefix_types[:, 2] = np.where(x[:, 1] == 0, SIG_ORDER_0, SIG_ORDER_1)
    prefix_types[:, 3] = TIME_SIGNAL
    prefix_types[:, 4] = np.where(nuisance[:, 0] == 0, NUIS_COMP_0, NUIS_COMP_1)
    prefix_types[:, 5] = np.where(nuisance[:, 1] == 0, NUIS_ORDER_0, NUIS_ORDER_1)
    prefix_types[:, 6] = TIME_NUIS
    prefix_times = np.ones((n, 7), dtype=np.float64)
    prefix_times[:, 0] = 0.0
    prefix_times[:, 3] = np.where(x[:, 2] == 0, 1.0, 4.0)
    prefix_times[:, 6] = np.where(nuisance[:, 2] == 0, 1.0, 4.0)

    target_types = np.empty((n, 4), dtype=np.int64)
    target_times = np.empty((n, 4), dtype=np.float64)
    for row in range(n):
        types = [TYPE_A, TYPE_A, TYPE_B, TYPE_C] if s[row, 0] == 0 else [TYPE_A, TYPE_B, TYPE_B, TYPE_C]
        if s[row, 1] == 1:
            types.reverse()
        target_types[row] = types
        target_times[row] = [1.0, 1.0, 4.0, 4.0] if s[row, 2] == 0 else [4.0, 4.0, 1.0, 1.0]

    result = SyntheticFactorSplit(prefix_types, prefix_times, target_types, target_times, s, x, nuisance, s.copy())
    _validate_split(result)
    return result


def nuisance_only(split: SyntheticFactorSplit) -> SyntheticFactorSplit:
    """Return a copy with all signal marker content removed, preserving nuisance."""
    _validate_split(split)
    types = split.prefix_type_ids.copy()
    types[:, 1:3] = UNUSED_0
    times = split.prefix_intervals.copy()
    times[:, 3] = 2.5
    return SyntheticFactorSplit(
        types, times, split.target_type_ids.copy(), split.target_intervals.copy(),
        split.S.copy(), split.X.copy(), split.N.copy(), split.L_after.copy(),
    )


def time_free_model_intervals(split: SyntheticFactorSplit) -> np.ndarray:
    """Return separate zero prefix model inputs without changing source or labels."""
    _validate_split(split)
    return np.zeros_like(split.prefix_intervals)


def fit_stage0_time_transform(train_split: SyntheticFactorSplit) -> TimeTransform:
    """Fit the Stage-0-frozen TRAIN prefix-then-target row-major population."""
    _validate_split(train_split)
    population = np.concatenate((train_split.prefix_intervals, train_split.target_intervals), axis=1).reshape(-1)
    return TimeTransform(unit=1.0).fit_train(population)


def component_seed(model_seed: int, component_code: int) -> int:
    raw = np.random.SeedSequence([model_seed, component_code]).generate_state(1, dtype=np.uint64)[0]
    return int(raw & np.uint64((1 << 63) - 1))


def _with_preserved_cpu_rng(seed: int, factory: Callable[[], nn.Module]) -> nn.Module:
    state = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(seed)
        return factory()
    finally:
        torch.random.set_rng_state(state)


def make_r0_encoder(model_seed: int) -> J04Encoder:
    encoder = _with_preserved_cpu_rng(component_seed(model_seed, 1), J04Encoder)
    assert isinstance(encoder, J04Encoder)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def make_initialized_teacher(model_seed: int) -> tuple[J04Encoder, EMATeacher]:
    online = _with_preserved_cpu_rng(component_seed(model_seed, 1), J04Encoder)
    assert isinstance(online, J04Encoder)
    online.eval()
    teacher = EMATeacher(online, momentum=0.996).eval()
    return online, teacher


def _probe_factory() -> nn.Linear:
    probe = nn.Linear(16, 1, bias=True)
    _initialize(probe)
    return probe


def make_readout_initializations(model_seed: int) -> tuple[tuple[nn.Linear, nn.Linear, nn.Linear], NonAutoregressiveNextEventsHead]:
    probes = tuple(_with_preserved_cpu_rng(component_seed(model_seed, code), _probe_factory) for code in (10, 11, 12))
    head = _with_preserved_cpu_rng(component_seed(model_seed, 3), NonAutoregressiveNextEventsHead)
    assert all(isinstance(probe, nn.Linear) for probe in probes)
    assert isinstance(head, NonAutoregressiveNextEventsHead)
    return probes, head


def readout_exposure_spec() -> dict[str, object]:
    return {
        "probe": {
            "batch_size": 256, "batches_per_epoch": 8, "total_batches": 250,
            "full_epochs": 31, "partial_epoch_batches": 2,
            "epoch_shuffle_seed_tuple": ["model_seed", "factor_code", "epoch", 6201],
        },
        "head": {
            "batch_size": 128, "batches_per_epoch": 16, "total_batches": 250,
            "full_epochs": 15, "partial_epoch_batches": 10,
            "epoch_shuffle_seed_tuple": ["model_seed", "epoch", 6301],
        },
    }


def nuisance_stratified_permutations(
    labels: np.ndarray, nuisance: np.ndarray, generator_seed: int, split_code: int, factor_code: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    nuisance = np.asarray(nuisance, dtype=np.uint8)
    if labels.ndim != 1 or nuisance.shape != (labels.size, 3) or factor_code not in (0, 1, 2):
        raise ValueError("labels [n], nuisance [n,3], and factor_code 0..2 required")
    output = np.empty((1000, labels.size), dtype=np.uint8)
    for k in range(1000):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3101, generator_seed, split_code, factor_code, k])))
        permuted = np.empty_like(labels)
        for n0 in (0, 1):
            for n1 in (0, 1):
                for n2 in (0, 1):
                    idx = np.flatnonzero(np.all(nuisance == (n0, n1, n2), axis=1))
                    perm = rng.permutation(len(idx))
                    permuted[idx] = labels[idx[perm]]
        output[k] = permuted
    return output


def sever_student_leak(l_after: np.ndarray, generator_seed: int) -> np.ndarray:
    values = np.asarray(l_after, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("L_after must be [n,3]")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7101, generator_seed, 5])))
    permutation = rng.permutation(values.shape[0])
    return values[permutation].copy()


def balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    if predictions.shape != labels.shape or not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("matching predictions/labels with both true classes required")
    return 0.5 * (float(np.mean(predictions[labels == 1] == 1)) + float(np.mean(predictions[labels == 0] == 0)))


def full_panel_negative_control(
    predictions: np.ndarray, labels: np.ndarray, permutations: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    predictions = np.asarray(predictions, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    permutations = np.asarray(permutations, dtype=np.uint8)
    if predictions.shape != (3, 3, 2048) or labels.shape != (3, 2048) or permutations.shape != (3, 1000, 2048):
        raise ValueError("exact [3,3,2048], [3,2048], [3,1000,2048] panel required")
    observed = np.mean([
        balanced_accuracy(predictions[g, m], labels[g]) for g in range(3) for m in range(3)
    ])
    null = np.empty(1000, dtype=np.float64)
    for k in range(1000):
        null[k] = np.mean([
            balanced_accuracy(predictions[g, m], permutations[g, k]) for g in range(3) for m in range(3)
        ])
    critical = float(np.quantile(null, 1.0 - 0.05 / 7.0, method="linear"))
    return float(observed), null, critical


def model_free_factor_calibration(generator_seeds: tuple[int, ...] = (1101, 1102, 1103)) -> list[dict[str, float | int]]:
    splits = {seed: generate_factor_split(seed, CAL_OOD, 2048) for seed in generator_seeds}
    output: list[dict[str, float | int]] = []
    for factor in range(3):
        medians = []
        for seed in generator_seeds:
            split = splits[seed]
            permutations = nuisance_stratified_permutations(split.S[:, factor], split.N, seed, CAL_OOD, factor)
            values = np.asarray([balanced_accuracy(split.S[:, factor], row) for row in permutations])
            medians.append(float(np.quantile(values, 0.5, method="linear")))
        m_null = float(np.mean(medians))
        gap = 1.0 - m_null
        delta = max(0.05, 0.25 * gap)
        delta_power = min(gap, delta + 0.10)
        if gap < 0.10 or not delta_power > delta:
            raise RuntimeError("model-free calibration gap invariant failed")
        output.append({"factor_code": factor, "M_null": m_null, "G": gap, "delta": delta, "delta_power": delta_power})
    return output


def analytic_llr_report() -> dict[str, float | bool | str]:
    signal = math.log(0.85 / 0.15)
    nuisance = math.log(0.95 / 0.05)
    dominance = nuisance > signal
    return {
        "signal_llr_magnitude": signal,
        "nuisance_llr_magnitude": nuisance,
        "nuisance_dominates_on_disagreement": dominance,
        "train_fit_ood_severed_hard_decision": "TRAIN-fit hard decisions follow nuisance on disagreements and lose that advantage after OOD nuisance severing",
    }


def split_fixture_hash(split: SyntheticFactorSplit) -> str:
    _validate_split(split)
    digest = hashlib.sha256()
    for name in ("prefix_type_ids", "prefix_intervals", "target_type_ids", "target_intervals", "S", "X", "N", "L_after"):
        value = np.ascontiguousarray(getattr(split, name))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _normalized_diagnostics(rows: np.ndarray) -> dict[str, float]:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or not np.all(np.isfinite(rows)):
        raise ValueError("finite [n,d] rows required")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = rows / np.maximum(norms, 1e-8)
    variance = float(np.var(normalized, axis=0, ddof=0).mean())
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / rows.shape[0]
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(eigenvalues.sum())
    if trace <= 1e-12:
        rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / trace
        rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    centered_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered_unit = centered / np.maximum(centered_norms, 1e-8)
    cosine_matrix = centered_unit @ centered_unit.T
    cosine = float((cosine_matrix.sum() - np.trace(cosine_matrix)) / (rows.shape[0] * (rows.shape[0] - 1)))
    return {"normalized_variance": variance, "effective_rank": rank, "centered_off_diagonal_cosine": cosine}


def _expanded_conditional_means(values: np.ndarray, factors: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.uint8)
    if values.ndim != 3 or factors.shape != (values.shape[0], 3):
        raise ValueError("values [n,identity,d] and factors [n,3] required")
    assigned = np.empty_like(values)
    for s0 in (0, 1):
        for s1 in (0, 1):
            for s2 in (0, 1):
                rows = np.all(factors == (s0, s1, s2), axis=1)
                if not np.any(rows):
                    raise RuntimeError("missing factor tuple")
                assigned[rows] = values[rows].mean(axis=0, keepdims=True)
    return assigned


def _state_dict_bytes(module: nn.Module) -> bytes:
    return b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in module.state_dict().values())


def initialized_teacher_reference_calibration(
    generator_seeds: tuple[int, ...] = (1101, 1102, 1103),
    model_seeds: tuple[int, ...] = (2101, 2102, 2103),
) -> dict[str, object]:
    """Run the exact 9-pair, target-only, ascending 16-batch reference audit."""
    recipes = ("L0_EMA_POOL", "L1_AVG", "L2_SEP")
    counts = {"L0_EMA_POOL": 1, "L1_AVG": 4, "L2_SEP": 16}
    _, spaced = resolve_layer_sets(4)
    if spaced != [1, 2, 3, 4]:
        raise RuntimeError("unexpected L2 layer identity order")
    runs: dict[str, list[list[dict[str, float]]]] = {
        recipe: [[] for _ in range(counts[recipe])] for recipe in recipes
    }
    batch_slices: list[list[int]] | None = None
    for generator_seed in generator_seeds:
        train = generate_factor_split(generator_seed, TRAIN, 8192)
        cal = generate_factor_split(generator_seed, CAL_OOD, 2048)
        transform = fit_stage0_time_transform(train)
        transformed_targets = transform.transform(cal.target_intervals).astype(np.float32, copy=False)
        for model_seed in model_seeds:
            _, teacher = make_initialized_teacher(model_seed)
            per_recipe: dict[str, list[np.ndarray]] = {recipe: [] for recipe in recipes}
            observed_slices: list[list[int]] = []
            with torch.no_grad():
                for start in range(0, 2048, 128):
                    stop = min(start + 128, 2048)
                    observed_slices.append([start, stop])
                    ids = torch.from_numpy(cal.target_type_ids[start:stop])
                    times = torch.from_numpy(transformed_targets[start:stop])
                    blocks, _, _ = teacher(ids, times, causal=True)
                    latent_mask = torch.ones((stop - start, 4), dtype=torch.bool)
                    for recipe in recipes:
                        target, mask, selected = construct_latent_targets(blocks, latent_mask, recipe)
                        if not bool(mask.all()):
                            raise RuntimeError("all-four-position latent mask was not retained")
                        if recipe == "L0_EMA_POOL":
                            flat = target[:, None, :]
                        else:
                            flat = target.reshape(stop - start, -1, 16)
                        if flat.shape[1] != counts[recipe]:
                            raise RuntimeError("wrong identity count")
                        if recipe == "L2_SEP" and selected != [1, 2, 3, 4]:
                            raise RuntimeError("wrong L2 layer order")
                        per_recipe[recipe].append(flat.detach().cpu().numpy().astype(np.float64))
            if len(observed_slices) != 16 or observed_slices != [[i, i + 128] for i in range(0, 2048, 128)]:
                raise RuntimeError("teacher traversal is not 16 ascending batch-128 slices")
            if batch_slices is None:
                batch_slices = observed_slices
            for recipe in recipes:
                raw = np.concatenate(per_recipe[recipe], axis=0)
                expanded = _expanded_conditional_means(raw, cal.S)
                for identity in range(counts[recipe]):
                    runs[recipe][identity].append(_normalized_diagnostics(expanded[:, identity]))

    collapse = np.zeros((2048, 16), dtype=np.float64)
    collapse[:, 0] = 1.0
    collapse_diag = _normalized_diagnostics(collapse)
    arm_reports: dict[str, object] = {}
    all_pass = True
    for recipe in recipes:
        identities = []
        for identity, values in enumerate(runs[recipe]):
            if len(values) != 9:
                raise RuntimeError("each identity requires nine seed-pair references")
            item: dict[str, object] = {"identity_index": identity}
            identity_pass = True
            for metric in ("normalized_variance", "effective_rank"):
                reference_values = np.asarray([value[metric] for value in values], dtype=np.float64)
                collapse_values = np.full(9, collapse_diag[metric], dtype=np.float64)
                q99_collapse = float(np.quantile(collapse_values, 0.99, method="linear"))
                q01_reference = float(np.quantile(reference_values, 0.01, method="linear"))
                separated = bool(np.isfinite(q01_reference) and q99_collapse < q01_reference)
                identity_pass = identity_pass and separated
                item[metric] = {
                    "collapse_q99": q99_collapse,
                    "reference_q01": q01_reference,
                    "threshold_midpoint": 0.5 * (q99_collapse + q01_reference),
                    "strict_separation": separated,
                }
            item["centered_off_diagonal_cosine_descriptive"] = {
                "collapse": collapse_diag["centered_off_diagonal_cosine"],
                "reference_values": [value["centered_off_diagonal_cosine"] for value in values],
            }
            item["strict_separation_pass"] = identity_pass
            all_pass = all_pass and identity_pass
            identities.append(item)
        arm_reports[recipe] = {"identity_count": counts[recipe], "identities": identities}
    if not all_pass:
        raise RuntimeError("initialized-teacher reference separation failed")
    return {
        "identity_order": {
            "L0_EMA_POOL": "single pooled identity",
            "L1_AVG": "j=0..3",
            "L2_SEP": "j*4+s, j=0..3, s=0..3, layers=[1,2,3,4]",
        },
        "teacher_batch_slices": batch_slices,
        "arms": arm_reports,
        "all_reference_separations_pass": all_pass,
    }


def position_specificity(
    prediction: torch.Tensor, target: torch.Tensor, latent_mask: torch.Tensor,
) -> dict[str, object]:
    """Algebraic L1/L2 position statistic; empty comparisons are omitted."""
    if prediction.shape != target.shape or prediction.ndim not in (3, 4):
        raise ValueError("matching L1 [B,J,D] or L2 [B,J,L,D] tensors required")
    if latent_mask.dtype != torch.bool or latent_mask.shape != prediction.shape[:2]:
        raise ValueError("latent_mask must be boolean [B,J]")
    per_examples: list[torch.Tensor] = []
    eligible_positions = 0
    eligible_identity_comparisons = 0
    for row in range(prediction.shape[0]):
        row_scores: list[torch.Tensor] = []
        valid_positions = torch.nonzero(latent_mask[row].reshape(-1), as_tuple=False).view(-1).tolist()
        for j in valid_positions:
            alternatives = [other for other in valid_positions if other != j]
            if not alternatives:
                continue
            if prediction.ndim == 3:
                same = torch.nn.functional.cosine_similarity(prediction[row, j], target[row, j], dim=0, eps=1e-8)
                other_scores = torch.stack([
                    torch.nn.functional.cosine_similarity(prediction[row, j], target[row, other], dim=0, eps=1e-8)
                    for other in alternatives
                ])
                row_scores.append(same - other_scores.mean())
                eligible_identity_comparisons += 1
            else:
                layer_scores = []
                for layer in range(prediction.shape[2]):
                    same = torch.nn.functional.cosine_similarity(prediction[row, j, layer], target[row, j, layer], dim=0, eps=1e-8)
                    other_scores = torch.stack([
                        torch.nn.functional.cosine_similarity(prediction[row, j, layer], target[row, other, layer], dim=0, eps=1e-8)
                        for other in alternatives
                    ])
                    layer_scores.append(same - other_scores.mean())
                row_scores.append(torch.stack(layer_scores).mean())
                eligible_identity_comparisons += prediction.shape[2]
            eligible_positions += 1
        if row_scores:
            per_examples.append(torch.stack(row_scores).mean())
    if per_examples:
        values = torch.stack(per_examples)
        aggregate: float | None = float(values.mean().item())
    else:
        values = torch.empty(0, dtype=prediction.dtype, device=prediction.device)
        aggregate = None
    return {
        "per_example": values,
        "aggregate": aggregate,
        "eligible_example_count": len(per_examples),
        "eligible_position_count": eligible_positions,
        "eligible_identity_comparison_count": eligible_identity_comparisons,
        "gated": False,
    }


def initialization_audit(model_seed: int) -> dict[str, bool]:
    ambient = torch.random.get_rng_state().clone()
    r0 = make_r0_encoder(model_seed)
    ambient_preserved_r0 = torch.equal(torch.random.get_rng_state(), ambient)
    online, teacher = make_initialized_teacher(model_seed)
    ambient_preserved_teacher = torch.equal(torch.random.get_rng_state(), ambient)
    paired_equal = _state_dict_bytes(r0) == _state_dict_bytes(online) == _state_dict_bytes(teacher.model)
    probes_a, head_a = make_readout_initializations(model_seed)
    ambient_preserved_readouts = torch.equal(torch.random.get_rng_state(), ambient)
    probes_b, head_b = make_readout_initializations(model_seed)
    deterministic = all(_state_dict_bytes(a) == _state_dict_bytes(b) for a, b in zip(probes_a, probes_b)) and _state_dict_bytes(head_a) == _state_dict_bytes(head_b)
    independent_storage = all(a.weight.data_ptr() != b.weight.data_ptr() for a, b in zip(probes_a, probes_b)) and head_a.queries.data_ptr() != head_b.queries.data_ptr()
    return {
        "ambient_cpu_rng_preserved": ambient_preserved_r0 and ambient_preserved_teacher and ambient_preserved_readouts,
        "r0_online_teacher_byte_identical": paired_equal,
        "readouts_deterministic": deterministic,
        "readouts_independent_storage": independent_storage,
    }
