#!/usr/bin/env python3
"""Paired L0 diagnostic separating encoder-training seed from frozen-readout seed."""
from __future__ import annotations

import json
import math
import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split, make_r0_encoder
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import (
    TrainedCondition, evaluate_readouts, fit_condition_readouts, freeze_encoder,
    frozen_representations, threshold_reference_report, train_latent_condition,
    trained_collapse_diagnostics,
)

GENERATOR_SEED = 1102
ENCODER_TRAINING_SEEDS = (2101, 2102)
READOUT_SEEDS = (2101, 2102)


def representation_geometry(condition, split, transform):
    z = frozen_representations(condition, split, transform).z.detach().cpu().numpy().astype(np.float64)
    covariance = np.cov(z, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if positive.size else 0.0
    participation_ratio = float(total * total / np.square(eigenvalues).sum()) if np.square(eigenvalues).sum() > 0 else 0.0
    factors = []
    for factor in range(3):
        labels = split.S[:, factor].astype(np.float64)
        class0 = z[labels == 0]
        class1 = z[labels == 1]
        centroid0 = class0.mean(axis=0)
        centroid1 = class1.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid1 - centroid0))
        within = np.concatenate((class0 - centroid0, class1 - centroid1), axis=0)
        within_rms = float(np.sqrt(np.mean(np.sum(within * within, axis=1))))
        correlations = []
        for dimension in range(z.shape[1]):
            if np.std(z[:, dimension]) == 0:
                correlations.append(0.0)
            else:
                correlations.append(float(np.corrcoef(z[:, dimension], labels)[0, 1]))
        factors.append({
            "factor": factor,
            "centroid_distance": centroid_distance,
            "within_class_rms": within_rms,
            "centroid_to_within_ratio": centroid_distance / within_rms if within_rms else None,
            "max_abs_dimension_correlation": max(abs(value) for value in correlations),
            "dimension_correlations": correlations,
        })
    return {
        "feature_std_mean": float(z.std(axis=0).mean()),
        "feature_std_min": float(z.std(axis=0).min()),
        "feature_std_max": float(z.std(axis=0).max()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "eigenvalues": eigenvalues.tolist(),
        "factors": factors,
    }


def evaluate_with_readout_seed(condition, probe_fit, cal_ood, transform, readout_seed):
    probes, head, fit_report = fit_condition_readouts(condition, probe_fit, transform, readout_seed)
    evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
    return {
        "factor_ba": [row["balanced_accuracy"] for row in evaluation["factors"]],
        "type_nll": evaluation["type_nll"],
        "interval_nll": evaluation["interval_nll"],
        "readout_fit": fit_report,
    }


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train = independent_train_nuisance(generate_factor_split(GENERATOR_SEED, TRAIN, 8192), GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    reference, threshold_digest = threshold_reference_report()

    result = {}
    for encoder_seed in ENCODER_TRAINING_SEEDS:
        initial = TrainedCondition("R0_INIT", make_r0_encoder(encoder_seed), None, None, {
            "attempted_steps": 0, "successful_steps": 0, "optimizer_steps": 0,
            "ema_updates": 0, "losses": None,
        })
        trained = train_latent_condition("L0_EMA_POOL", train, transform, encoder_seed)
        freeze_encoder(initial)
        freeze_encoder(trained)
        collapse_rows, _ = trained_collapse_diagnostics(trained, cal_ood, transform, reference)
        result[str(encoder_seed)] = {
            "training_loss_summary": trained.training["losses"],
            "initial_geometry": representation_geometry(initial, probe_fit, transform),
            "trained_geometry": representation_geometry(trained, probe_fit, transform),
            "l0_collapse": {"all_pass": all(row["both_metrics_pass"] for row in collapse_rows), "rows": collapse_rows},
            "initial_readout_cross": {str(readout_seed): evaluate_with_readout_seed(initial, probe_fit, cal_ood, transform, readout_seed) for readout_seed in READOUT_SEEDS},
            "trained_readout_cross": {str(readout_seed): evaluate_with_readout_seed(trained, probe_fit, cal_ood, transform, readout_seed) for readout_seed in READOUT_SEEDS},
        }

    composition_matrix = [[result[str(encoder_seed)]["trained_readout_cross"][str(readout_seed)]["factor_ba"][0]
                           for readout_seed in READOUT_SEEDS] for encoder_seed in ENCODER_TRAINING_SEEDS]
    initial_composition_matrix = [[result[str(encoder_seed)]["initial_readout_cross"][str(readout_seed)]["factor_ba"][0]
                                   for readout_seed in READOUT_SEEDS] for encoder_seed in ENCODER_TRAINING_SEEDS]
    print(json.dumps({
        "schema": "BP011-J04C-L0-PAIRED-INITIALIZATION-DIAGNOSTIC-V1",
        "contract": {
            "live_axis": "encoder-training seed versus frozen-readout seed contribution to composition instability",
            "pass_encoder_attribution": "composition follows encoder-training seed under both common readout seeds",
            "pass_readout_attribution": "composition follows readout seed across both encoders",
            "not_tested": "no initialization intervention or target-family change",
        },
        "generator_seed": GENERATOR_SEED,
        "encoder_training_seeds": ENCODER_TRAINING_SEEDS,
        "readout_seeds": READOUT_SEEDS,
        "factor_order": ["composition", "order", "time"],
        "threshold_digest": threshold_digest,
        "trained_composition_ba_matrix_rows_encoder_cols_readout": composition_matrix,
        "initial_composition_ba_matrix_rows_encoder_cols_readout": initial_composition_matrix,
        "conditions": result,
        "claim_ceiling": "PUBLIC_SYNTHETIC_PAIRED_SEED_DIAGNOSTIC",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
