#!/usr/bin/env python3
"""Direct public-synthetic J04c run with TRAIN nuisance independently regenerated."""
from __future__ import annotations

import json
import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split, make_r0_encoder
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import (
    GENERATOR_SEED, LATENT_NAMES, MODEL_SEED, TrainedCondition,
    evaluate_readouts, fit_condition_readouts, freeze_encoder,
    threshold_reference_report, train_direct_condition, train_latent_condition,
    trained_collapse_diagnostics,
)


def phi(left, right):
    left = left.astype(np.float64); right = right.astype(np.float64)
    return float(np.corrcoef(left, right)[0, 1])


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    raw_train = generate_factor_split(GENERATOR_SEED, TRAIN, 8192)
    train = independent_train_nuisance(raw_train, GENERATOR_SEED)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)

    conditions = {name: train_latent_condition(name, train, transform) for name in LATENT_NAMES}
    conditions["C0_DIRECT"] = train_direct_condition("C0_DIRECT", train, transform)
    conditions["R0_INIT"] = TrainedCondition("R0_INIT", make_r0_encoder(MODEL_SEED), None, None, {
        "attempted_steps": 0, "successful_steps": 0, "optimizer_steps": 0,
        "ema_updates": 0, "losses": None,
    })
    for condition in conditions.values():
        freeze_encoder(condition)

    metrics = {}
    for name, condition in conditions.items():
        probes, head, readout_report = fit_condition_readouts(condition, probe_fit, transform)
        evaluation, _ = evaluate_readouts(condition, probes, head, cal_ood, transform)
        metrics[name] = {"cal_ood": evaluation, "training": condition.training,
                         "readout_fit": readout_report}

    reference, threshold_digest = threshold_reference_report()
    collapse = {}
    for name in LATENT_NAMES:
        rows, _ = trained_collapse_diagnostics(conditions[name], cal_ood, transform, reference)
        collapse[name] = {
            "identity_count": len(rows),
            "pass_count": sum(row["both_metrics_pass"] for row in rows),
            "all_pass": all(row["both_metrics_pass"] for row in rows),
            "rows": rows,
        }

    train_correlations = {
        f"S{signal}_N{nuisance}": phi(train.S[:, signal], train.N[:, nuisance])
        for signal in range(3) for nuisance in range(3)
    }
    raw_diagonal = [phi(raw_train.S[:, index], raw_train.N[:, index]) for index in range(3)]
    result = {
        "schema": "BP011-J04C-NUISANCE-SEVERED-BRIDGE-V1",
        "scientific_change": "TRAIN nuisance regenerated independently; S, X, targets, model, objectives, schedules and CAL-OOD held fixed",
        "seeds": {"generator": GENERATOR_SEED, "model": MODEL_SEED},
        "train_size": 8192,
        "probe_fit_size": 2048,
        "cal_ood_size": 2048,
        "raw_train_diagonal_S_N_correlations": raw_diagonal,
        "bridged_train_S_N_correlations": train_correlations,
        "max_abs_bridged_train_S_N_correlation": max(abs(value) for value in train_correlations.values()),
        "metrics": metrics,
        "collapse": {"threshold_digest": threshold_digest, "arms": collapse},
        "claim_ceiling": "PUBLIC_SYNTHETIC_SINGLE_SEED_J04C_BRIDGE_EVIDENCE",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
