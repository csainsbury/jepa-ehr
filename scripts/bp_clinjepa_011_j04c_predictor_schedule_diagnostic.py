#!/usr/bin/env python3
"""2x2 split of predictor initialization and batch-order seed for failing encoder 2101."""
from __future__ import annotations

import json
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts, fit_condition_readouts, freeze_encoder, threshold_reference_report, trained_collapse_diagnostics

GENERATOR_SEED = 1102
ENCODER_SEED = 2101
PREDICTOR_SEEDS = (2101, 2102)
SCHEDULE_SEEDS = (2101, 2102)
COMMON_READOUT_SEED = 2102


def main():
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    train = independent_train_nuisance(generate_factor_split(GENERATOR_SEED, TRAIN, 8192), GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    reference, threshold_digest = threshold_reference_report()
    matrix, results = [], {}
    for predictor_seed in PREDICTOR_SEEDS:
        row = []
        for schedule_seed in SCHEDULE_SEEDS:
            condition = train_l0_decoupled_seeds(train, transform, ENCODER_SEED, predictor_seed, schedule_seed=schedule_seed)
            freeze_encoder(condition)
            probes, head, _ = fit_condition_readouts(condition, probe_fit, transform, COMMON_READOUT_SEED)
            evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
            factor_ba = [factor["balanced_accuracy"] for factor in evaluation["factors"]]
            collapse_rows, _ = trained_collapse_diagnostics(condition, cal_ood, transform, reference)
            key = f"predictor_{predictor_seed}__schedule_{schedule_seed}"
            results[key] = {"factor_ba": factor_ba, "collapse_all_pass": all(item["both_metrics_pass"] for item in collapse_rows),
                            "training_loss_summary": condition.training["losses"]}
            row.append(factor_ba[0])
        matrix.append(row)
    predictor_effect = 0.5 * ((matrix[1][0] + matrix[1][1]) - (matrix[0][0] + matrix[0][1]))
    schedule_effect = 0.5 * ((matrix[0][1] + matrix[1][1]) - (matrix[0][0] + matrix[1][0]))
    interaction = (matrix[1][1] - matrix[1][0]) - (matrix[0][1] - matrix[0][0])
    magnitudes = {"predictor": abs(predictor_effect), "schedule": abs(schedule_effect), "interaction": abs(interaction)}
    dominant = max(magnitudes, key=magnitudes.get)
    print(json.dumps({
        "schema": "BP011-J04C-L0-PREDICTOR-SCHEDULE-DIAGNOSTIC-V1",
        "contract": {"axis": "predictor initialization versus pretraining batch-order seed under failing encoder seed 2101",
                     "resolved": "largest absolute effect is at least 0.05 and exceeds each alternative",
                     "not_tested": "encoder submodules or an intervention"},
        "generator_seed": GENERATOR_SEED, "encoder_seed": ENCODER_SEED,
        "predictor_seeds": PREDICTOR_SEEDS, "schedule_seeds": SCHEDULE_SEEDS,
        "common_readout_seed": COMMON_READOUT_SEED,
        "composition_ba_matrix_rows_predictor_cols_schedule": matrix,
        "predictor_main_effect": predictor_effect, "schedule_main_effect": schedule_effect,
        "interaction": interaction, "absolute_effects": magnitudes,
        "dominant_term": dominant,
        "resolved": magnitudes[dominant] >= 0.05 and sum(value == magnitudes[dominant] for value in magnitudes.values()) == 1,
        "results": results, "threshold_digest": threshold_digest,
        "claim_ceiling": "PUBLIC_SYNTHETIC_SEED_COMPONENT_LOCALIZATION",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__": main()
