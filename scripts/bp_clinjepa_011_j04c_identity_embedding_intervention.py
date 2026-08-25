#!/usr/bin/env python3
"""Test one label-free deterministic type-embedding initialization intervention."""
from __future__ import annotations

import json
import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import set_identity_type_embedding, train_l0_identity_embedding
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import (
    TrainedCondition, _fresh_encoder, evaluate_readouts, fit_condition_readouts,
    freeze_encoder, frozen_representations, threshold_reference_report,
    trained_collapse_diagnostics,
)

GENERATOR_SEED = 1102
MODEL_SEEDS = (2101, 2102, 2103)
COMMON_READOUT_SEED = 2102


def composition_geometry(condition, split, transform):
    z = frozen_representations(condition, split, transform).z.detach().cpu().numpy().astype(np.float64)
    labels = split.S[:, 0]
    c0, c1 = z[labels == 0], z[labels == 1]
    m0, m1 = c0.mean(axis=0), c1.mean(axis=0)
    within = np.concatenate((c0 - m0, c1 - m1), axis=0)
    correlations = [0.0 if np.std(z[:, d]) == 0 else float(np.corrcoef(z[:, d], labels)[0, 1]) for d in range(z.shape[1])]
    covariance = np.cov(z, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities[probabilities > 0]
    return {
        "centroid_distance": float(np.linalg.norm(m1 - m0)),
        "within_class_rms": float(np.sqrt(np.mean(np.sum(within * within, axis=1)))),
        "centroid_to_within_ratio": float(np.linalg.norm(m1 - m0) / np.sqrt(np.mean(np.sum(within * within, axis=1)))),
        "max_abs_dimension_correlation": max(abs(value) for value in correlations),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
    }


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train = independent_train_nuisance(generate_factor_split(GENERATOR_SEED, TRAIN, 8192), GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    reference, threshold_digest = threshold_reference_report()
    results = {}
    composition_values = []
    for model_seed in MODEL_SEEDS:
        initial_encoder = _fresh_encoder(model_seed)
        set_identity_type_embedding(initial_encoder)
        initial = TrainedCondition("R0_INIT", initial_encoder, None, None, {"losses": None})
        trained = train_l0_identity_embedding(train, transform, model_seed)
        freeze_encoder(initial); freeze_encoder(trained)
        probes, head, fit_report = fit_condition_readouts(trained, probe_fit, transform, COMMON_READOUT_SEED)
        evaluation, _ = evaluate_readouts(trained, probes, head, cal_ood, transform)
        factor_ba = [row["balanced_accuracy"] for row in evaluation["factors"]]
        composition_values.append(factor_ba[0])
        collapse_rows, _ = trained_collapse_diagnostics(trained, cal_ood, transform, reference)
        results[str(model_seed)] = {
            "factor_ba": factor_ba,
            "type_nll": evaluation["type_nll"],
            "interval_nll": evaluation["interval_nll"],
            "initial_composition_geometry": composition_geometry(initial, probe_fit, transform),
            "trained_composition_geometry": composition_geometry(trained, probe_fit, transform),
            "training_loss_summary": trained.training["losses"],
            "collapse_all_pass": all(row["both_metrics_pass"] for row in collapse_rows),
            "readout_fit": fit_report,
        }
    pass_checks = {
        "collapse_all_three": all(result["collapse_all_pass"] for result in results.values()),
        "composition_ba_at_least_0_80_all_three": min(composition_values) >= 0.80,
        "composition_ba_range_at_most_0_05": max(composition_values) - min(composition_values) <= 0.05,
    }
    print(json.dumps({
        "schema": "BP011-J04C-L0-IDENTITY-EMBEDDING-INTERVENTION-V1",
        "contract": {
            "axis": "replace random type-embedding initialization with label-free deterministic unit-norm identity codes",
            "pass": "collapse 3/3, composition BA >=0.80 in 3/3, and composition range <=0.05",
            "not_tested": "direct-supervision superiority or target-family changes",
        },
        "generator_seed": GENERATOR_SEED,
        "model_seeds": MODEL_SEEDS,
        "common_readout_seed": COMMON_READOUT_SEED,
        "results": results,
        "composition_ba": composition_values,
        "composition_ba_range": max(composition_values) - min(composition_values),
        "pass_checks": pass_checks,
        "contract_pass": all(pass_checks.values()),
        "threshold_digest": threshold_digest,
        "claim_ceiling": "PUBLIC_SYNTHETIC_INITIALIZATION_INTERVENTION_EVIDENCE",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
