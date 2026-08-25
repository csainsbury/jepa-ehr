#!/usr/bin/env python3
"""2x2 localization of encoder-init versus predictor/order seed in standard L0."""
from __future__ import annotations

import json
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts, fit_condition_readouts, freeze_encoder, threshold_reference_report, trained_collapse_diagnostics

GENERATOR_SEED = 1102
ENCODER_SEEDS = (2101, 2102)
TRAINING_SEEDS = (2101, 2102)
COMMON_READOUT_SEED = 2102


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    train = independent_train_nuisance(generate_factor_split(GENERATOR_SEED, TRAIN, 8192), GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    reference, threshold_digest = threshold_reference_report()
    results = {}
    matrix = []
    for encoder_seed in ENCODER_SEEDS:
        row = []
        for training_seed in TRAINING_SEEDS:
            condition = train_l0_decoupled_seeds(train, transform, encoder_seed, training_seed)
            freeze_encoder(condition)
            probes, head, _ = fit_condition_readouts(condition, probe_fit, transform, COMMON_READOUT_SEED)
            evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
            factor_ba = [factor["balanced_accuracy"] for factor in evaluation["factors"]]
            collapse_rows, _ = trained_collapse_diagnostics(condition, cal_ood, transform, reference)
            key = f"encoder_{encoder_seed}__training_{training_seed}"
            results[key] = {"factor_ba": factor_ba, "training_loss_summary": condition.training["losses"],
                            "collapse_all_pass": all(row_["both_metrics_pass"] for row_ in collapse_rows)}
            row.append(factor_ba[0])
        matrix.append(row)
    encoder_effect = 0.5 * ((matrix[1][0] + matrix[1][1]) - (matrix[0][0] + matrix[0][1]))
    training_effect = 0.5 * ((matrix[0][1] + matrix[1][1]) - (matrix[0][0] + matrix[1][0]))
    interaction = (matrix[1][1] - matrix[1][0]) - (matrix[0][1] - matrix[0][0])
    checks = {
        "collapse_all_four": all(value["collapse_all_pass"] for value in results.values()),
        "encoder_main_effect_at_least_0_05": abs(encoder_effect) >= 0.05,
        "encoder_effect_exceeds_training_effect": abs(encoder_effect) > abs(training_effect),
        "encoder_effect_exceeds_interaction": abs(encoder_effect) > abs(interaction),
    }
    print(json.dumps({
        "schema": "BP011-J04C-L0-DECOUPLED-SEED-DIAGNOSTIC-V1",
        "contract": {"axis": "encoder/teacher initialization seed versus predictor initialization plus batch-order seed",
                     "encoder_attribution": "collapse 4/4 and encoder effect >=0.05 larger than training and interaction effects",
                     "not_tested": "individual encoder submodules or an intervention"},
        "generator_seed": GENERATOR_SEED, "encoder_seeds": ENCODER_SEEDS,
        "training_seeds": TRAINING_SEEDS, "common_readout_seed": COMMON_READOUT_SEED,
        "composition_ba_matrix_rows_encoder_cols_training": matrix,
        "encoder_main_effect": encoder_effect, "training_main_effect": training_effect,
        "interaction": interaction, "checks": checks,
        "encoder_attribution_pass": all(checks.values()), "results": results,
        "threshold_digest": threshold_digest,
        "claim_ceiling": "PUBLIC_SYNTHETIC_SEED_COMPONENT_LOCALIZATION",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
