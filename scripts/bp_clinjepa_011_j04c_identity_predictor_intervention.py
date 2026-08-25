#!/usr/bin/env python3
"""Test identity-through-GELU initialization of the existing L0 predictor MLP."""
from __future__ import annotations

import json
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l0_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts, fit_condition_readouts, freeze_encoder, threshold_reference_report, trained_collapse_diagnostics

GENERATOR_SEED = 1102
MODEL_SEEDS = (2101, 2102, 2103)
COMMON_READOUT_SEED = 2102


def main():
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    train = independent_train_nuisance(generate_factor_split(GENERATOR_SEED, TRAIN, 8192), GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    reference, threshold_digest = threshold_reference_report()
    results, composition = {}, []
    for model_seed in MODEL_SEEDS:
        condition = train_l0_decoupled_seeds(train, transform, model_seed, model_seed,
                                              schedule_seed=model_seed, identity_predictor=True)
        freeze_encoder(condition)
        probes, head, _ = fit_condition_readouts(condition, probe_fit, transform, COMMON_READOUT_SEED)
        evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
        factor_ba = [factor["balanced_accuracy"] for factor in evaluation["factors"]]
        collapse_rows, _ = trained_collapse_diagnostics(condition, cal_ood, transform, reference)
        results[str(model_seed)] = {"factor_ba": factor_ba,
                                    "collapse_all_pass": all(row["both_metrics_pass"] for row in collapse_rows),
                                    "training_loss_summary": condition.training["losses"]}
        composition.append(factor_ba[0])
    checks = {"collapse_all_three": all(value["collapse_all_pass"] for value in results.values()),
              "composition_ba_at_least_0_80_all_three": min(composition) >= 0.80,
              "composition_ba_range_at_most_0_05": max(composition) - min(composition) <= 0.05}
    print(json.dumps({
        "schema": "BP011-J04C-L0-IDENTITY-PREDICTOR-INTERVENTION-V1",
        "contract": {"axis": "initialize the existing predictor MLP as an exact identity-through-GELU map after LayerNorm",
                     "pass": "collapse 3/3, composition BA >=0.80 in 3/3, and composition range <=0.05",
                     "not_tested": "direct-supervision superiority or target-family changes"},
        "generator_seed": GENERATOR_SEED, "model_seeds": MODEL_SEEDS,
        "common_readout_seed": COMMON_READOUT_SEED, "composition_ba": composition,
        "composition_ba_range": max(composition) - min(composition),
        "checks": checks, "contract_pass": all(checks.values()), "results": results,
        "threshold_digest": threshold_digest,
        "claim_ceiling": "PUBLIC_SYNTHETIC_PREDICTOR_INITIALIZATION_INTERVENTION",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__": main()
