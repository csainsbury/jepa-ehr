#!/usr/bin/env python3
"""Stdout-only BP-CLINJEPA-011 J04c v2 Stage 0 audit."""
from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import pytest
import torch

from clinical_jepa.eval.j04c_falsifier import (
    CAL_OOD,
    PROBE_FIT,
    TRAIN,
    analytic_llr_report,
    fit_stage0_time_transform,
    generate_factor_split,
    initialization_audit,
    initialized_teacher_reference_calibration,
    model_free_factor_calibration,
    readout_exposure_spec,
    split_fixture_hash,
)


def _compact_reference(report: dict[str, object]) -> dict[str, object]:
    arms = {}
    for arm_name, arm in report["arms"].items():
        identities = []
        for identity in arm["identities"]:
            identities.append({
                "identity_index": identity["identity_index"],
                "normalized_variance": identity["normalized_variance"],
                "effective_rank": identity["effective_rank"],
                "strict_separation_pass": identity["strict_separation_pass"],
            })
        arms[arm_name] = {"identity_count": arm["identity_count"], "identities": identities}
    return {
        "identity_order": report["identity_order"],
        "arms": arms,
        "all_reference_separations_pass": report["all_reference_separations_pass"],
    }


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    generator_seeds = (1101, 1102, 1103)
    model_seeds = (2101, 2102, 2103)
    split_sizes = {TRAIN: 8192, CAL_OOD: 2048, PROBE_FIT: 2048}

    fixture_hashes: dict[str, str] = {}
    transform_states = []
    aggregate = hashlib.sha256()
    for generator_seed in generator_seeds:
        generated = {}
        for split_code in (TRAIN, CAL_OOD, PROBE_FIT):
            split = generate_factor_split(generator_seed, split_code, split_sizes[split_code])
            generated[split_code] = split
            key = f"generator_{generator_seed}_split_{split_code}"
            fixture_hashes[key] = split_fixture_hash(split)
        transform = fit_stage0_time_transform(generated[TRAIN])
        state_hash = hashlib.sha256(transform.state_bytes()).hexdigest()
        aggregate.update(transform.state_bytes())
        transform_states.append({
            "generator_seed": generator_seed,
            "mu": transform.mu,
            "sigma": transform.sigma,
            "state_sha256": state_hash,
        })

    exposure = readout_exposure_spec()
    if exposure["probe"]["full_epochs"] * exposure["probe"]["batches_per_epoch"] + exposure["probe"]["partial_epoch_batches"] != 250:
        raise RuntimeError("probe exposure arithmetic failed")
    if exposure["head"]["full_epochs"] * exposure["head"]["batches_per_epoch"] + exposure["head"]["partial_epoch_batches"] != 250:
        raise RuntimeError("head exposure arithmetic failed")

    initialization = {str(seed): initialization_audit(seed) for seed in model_seeds}
    if not all(all(values.values()) for values in initialization.values()):
        raise RuntimeError("initialization invariant failed")

    references = initialized_teacher_reference_calibration(generator_seeds, model_seeds)
    compact_references = _compact_reference(references)
    identity_total = sum(arm["identity_count"] for arm in compact_references["arms"].values())
    if identity_total != 21 or not compact_references["all_reference_separations_pass"]:
        raise RuntimeError("reference identity invariant failed")

    llr = analytic_llr_report()
    if not llr["nuisance_dominates_on_disagreement"]:
        raise RuntimeError("analytic LLR dominance failed")

    output = {
        "schema_version": "bp-clinjepa-011-j04c-stage0-v1",
        "authority": "instrumentation/no-training calibration only",
        "seeds": {"generator": list(generator_seeds), "model": list(model_seeds)},
        "split_codes_used": {"TRAIN": TRAIN, "CAL_OOD": CAL_OOD, "PROBE_FIT": PROBE_FIT},
        "split_sizes": {"TRAIN": 8192, "CAL_OOD": 2048, "PROBE_FIT": 2048},
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytest": pytest.__version__,
            "torch": torch.__version__,
        },
        "generator_fixture_sha256": fixture_hashes,
        "train_time_transform": {
            "population": "TRAIN prefix intervals then target intervals, per row, row-major",
            "states": transform_states,
            "aggregate_state_sha256": aggregate.hexdigest(),
        },
        "model_free_factor_calibration": model_free_factor_calibration(generator_seeds),
        "analytic_llr": llr,
        "initialization": initialization,
        "readout_exposure_spec": exposure,
        "collapse_reference": compact_references,
        "all_reference_separations_pass": True,
        "reference_identity_count": identity_total,
        "candidate_c_a_computed": False,
        "training_performed": False,
        "ood_dev_generated": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
