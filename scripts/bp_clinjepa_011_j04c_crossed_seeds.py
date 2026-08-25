#!/usr/bin/env python3
"""3x3 crossed generator/model seed decomposition for L0, direct and R0 only."""
from __future__ import annotations

import json
import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split, make_r0_encoder
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import (
    TrainedCondition, evaluate_readouts, fit_condition_readouts, freeze_encoder,
    threshold_reference_report, train_direct_condition, train_latent_condition,
    trained_collapse_diagnostics,
)

GENERATOR_SEEDS = (1101, 1102, 1103)
MODEL_SEEDS = (2101, 2102, 2103)
ARMS = ("L0_EMA_POOL", "C0_DIRECT", "R0_INIT")


def run_pair(generator_seed, model_seed, reference):
    train = independent_train_nuisance(generate_factor_split(generator_seed, TRAIN, 8192), generator_seed)
    probe_fit = generate_factor_split(generator_seed, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(generator_seed, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    conditions = {
        "L0_EMA_POOL": train_latent_condition("L0_EMA_POOL", train, transform, model_seed),
        "C0_DIRECT": train_direct_condition("C0_DIRECT", train, transform, model_seed),
        "R0_INIT": TrainedCondition("R0_INIT", make_r0_encoder(model_seed), None, None, {
            "attempted_steps": 0, "successful_steps": 0, "optimizer_steps": 0,
            "ema_updates": 0, "losses": None,
        }),
    }
    for condition in conditions.values():
        freeze_encoder(condition)
    metrics = {}
    for name, condition in conditions.items():
        probes, head, _ = fit_condition_readouts(condition, probe_fit, transform, model_seed)
        evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
        metrics[name] = {
            "factor_ba": [row["balanced_accuracy"] for row in evaluation["factors"]],
            "type_nll": evaluation["type_nll"],
            "interval_nll": evaluation["interval_nll"],
        }
    collapse_rows, _ = trained_collapse_diagnostics(conditions["L0_EMA_POOL"], cal_ood, transform, reference)
    return {
        "generator_seed": generator_seed,
        "model_seed": model_seed,
        "metrics": metrics,
        "l0_collapse_all_pass": all(row["both_metrics_pass"] for row in collapse_rows),
        "l0_collapse_rows": collapse_rows,
    }


def crossed_decomposition(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    grand = float(matrix.mean())
    generator_means = matrix.mean(axis=1)
    model_means = matrix.mean(axis=0)
    additive = generator_means[:, None] + model_means[None, :] - grand
    residual = matrix - additive
    total_ss = float(((matrix - grand) ** 2).sum())
    generator_ss = float(matrix.shape[1] * ((generator_means - grand) ** 2).sum())
    model_ss = float(matrix.shape[0] * ((model_means - grand) ** 2).sum())
    interaction_ss = float((residual ** 2).sum())
    return {
        "grand_mean": grand,
        "generator_means": generator_means.tolist(),
        "model_means": model_means.tolist(),
        "total_sum_squares": total_ss,
        "generator_sum_squares": generator_ss,
        "model_sum_squares": model_ss,
        "interaction_sum_squares": interaction_ss,
        "generator_fraction_total_ss": None if total_ss == 0 else generator_ss / total_ss,
        "model_fraction_total_ss": None if total_ss == 0 else model_ss / total_ss,
        "interaction_fraction_total_ss": None if total_ss == 0 else interaction_ss / total_ss,
        "range": [float(matrix.min()), float(matrix.max())],
    }


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    reference, threshold_digest = threshold_reference_report()
    runs = [run_pair(g, m, reference) for g in GENERATOR_SEEDS for m in MODEL_SEEDS]
    matrices = {}
    decompositions = {}
    for arm in ARMS:
        matrices[arm] = []
        decompositions[arm] = []
        for factor in range(3):
            matrix = [[next(run for run in runs if run["generator_seed"] == g and run["model_seed"] == m)["metrics"][arm]["factor_ba"][factor]
                       for m in MODEL_SEEDS] for g in GENERATOR_SEEDS]
            matrices[arm].append(matrix)
            decompositions[arm].append(crossed_decomposition(matrix))
    comparisons = {
        "L0_composition_beats_R0_cell_count": sum(run["metrics"]["L0_EMA_POOL"]["factor_ba"][0] > run["metrics"]["R0_INIT"]["factor_ba"][0] for run in runs),
        "L0_composition_equals_R0_cell_count": sum(run["metrics"]["L0_EMA_POOL"]["factor_ba"][0] == run["metrics"]["R0_INIT"]["factor_ba"][0] for run in runs),
        "L0_composition_beats_direct_cell_count": sum(run["metrics"]["L0_EMA_POOL"]["factor_ba"][0] > run["metrics"]["C0_DIRECT"]["factor_ba"][0] for run in runs),
        "L0_collapse_pass_cell_count": sum(run["l0_collapse_all_pass"] for run in runs),
    }
    print(json.dumps({
        "schema": "BP011-J04C-NUISANCE-SEVERED-CROSSED-SEEDS-3X3-V1",
        "generator_seeds": GENERATOR_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "arms": ARMS,
        "factor_order": ["composition", "order", "time"],
        "threshold_digest": threshold_digest,
        "runs": runs,
        "factor_ba_matrices": matrices,
        "decompositions": decompositions,
        "comparisons": comparisons,
        "claim_ceiling": "PUBLIC_SYNTHETIC_CROSSED_SEED_SENSITIVITY_EVIDENCE",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
