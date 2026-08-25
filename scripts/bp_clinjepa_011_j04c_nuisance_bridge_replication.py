#!/usr/bin/env python3
"""Three-pair replication of the direct J04c independently-severed TRAIN nuisance bridge."""
from __future__ import annotations

import json
import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import CAL_OOD, PROBE_FIT, TRAIN, fit_stage0_time_transform, generate_factor_split, make_r0_encoder
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import (
    LATENT_NAMES, TrainedCondition, evaluate_readouts, fit_condition_readouts,
    freeze_encoder, threshold_reference_report, train_direct_condition,
    train_latent_condition, trained_collapse_diagnostics,
)

PAIRS = ((1101, 2101), (1102, 2102), (1103, 2103))


def phi(left, right):
    return float(np.corrcoef(left.astype(np.float64), right.astype(np.float64))[0, 1])


def run_pair(generator_seed, model_seed, reference):
    train = independent_train_nuisance(generate_factor_split(generator_seed, TRAIN, 8192), generator_seed)
    probe_fit = generate_factor_split(generator_seed, PROBE_FIT, 2048)
    cal_ood = generate_factor_split(generator_seed, CAL_OOD, 2048)
    transform = fit_stage0_time_transform(train)
    conditions = {name: train_latent_condition(name, train, transform, model_seed) for name in LATENT_NAMES}
    conditions["C0_DIRECT"] = train_direct_condition("C0_DIRECT", train, transform, model_seed)
    conditions["R0_INIT"] = TrainedCondition("R0_INIT", make_r0_encoder(model_seed), None, None, {
        "attempted_steps": 0, "successful_steps": 0, "optimizer_steps": 0,
        "ema_updates": 0, "losses": None,
    })
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
    collapse = {}
    for name in LATENT_NAMES:
        rows, _ = trained_collapse_diagnostics(conditions[name], cal_ood, transform, reference)
        collapse[name] = {"pass_count": sum(row["both_metrics_pass"] for row in rows),
                          "identity_count": len(rows), "all_pass": all(row["both_metrics_pass"] for row in rows)}
    correlations = [phi(train.S[:, i], train.N[:, j]) for i in range(3) for j in range(3)]
    return {"generator_seed": generator_seed, "model_seed": model_seed,
            "max_abs_train_S_N_correlation": max(abs(value) for value in correlations),
            "metrics": metrics, "collapse": collapse}


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    reference, threshold_digest = threshold_reference_report()
    runs = [run_pair(generator_seed, model_seed, reference) for generator_seed, model_seed in PAIRS]
    names = ("L0_EMA_POOL", "L1_AVG", "L2_SEP", "C0_DIRECT", "R0_INIT")
    summary = {}
    for name in names:
        summary[name] = {
            "mean_factor_ba": [float(np.mean([run["metrics"][name]["factor_ba"][f] for run in runs])) for f in range(3)],
            "min_factor_ba": [float(np.min([run["metrics"][name]["factor_ba"][f] for run in runs])) for f in range(3)],
            "mean_type_nll": float(np.mean([run["metrics"][name]["type_nll"] for run in runs])),
            "mean_interval_nll": float(np.mean([run["metrics"][name]["interval_nll"] for run in runs])),
        }
    comparisons = {
        "L0_composition_beats_R0_run_count": sum(run["metrics"]["L0_EMA_POOL"]["factor_ba"][0] > run["metrics"]["R0_INIT"]["factor_ba"][0] for run in runs),
        "L0_composition_beats_direct_run_count": sum(run["metrics"]["L0_EMA_POOL"]["factor_ba"][0] > run["metrics"]["C0_DIRECT"]["factor_ba"][0] for run in runs),
        "L0_all_collapse_pass_run_count": sum(run["collapse"]["L0_EMA_POOL"]["all_pass"] for run in runs),
        "L1_all_collapse_pass_run_count": sum(run["collapse"]["L1_AVG"]["all_pass"] for run in runs),
        "L2_all_collapse_pass_run_count": sum(run["collapse"]["L2_SEP"]["all_pass"] for run in runs),
    }
    print(json.dumps({
        "schema": "BP011-J04C-NUISANCE-SEVERED-BRIDGE-REPLICATION-V1",
        "scientific_change": "TRAIN nuisance independently regenerated; three disjoint generator/model pairs",
        "pairs": PAIRS,
        "threshold_digest": threshold_digest,
        "runs": runs,
        "summary": summary,
        "comparisons": comparisons,
        "claim_ceiling": "PUBLIC_SYNTHETIC_THREE_PAIR_J04C_BRIDGE_EVIDENCE",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
