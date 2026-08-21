#!/usr/bin/env python3
"""No-argument stdout-only BP-CLINJEPA-011 J04c v2 Stage 1 CAL smoke."""
from __future__ import annotations

import json

import numpy as np
import torch

from clinical_jepa.eval.j04c_falsifier import (
    CAL_ID, CAL_OOD, PROBE_FIT, TRAIN,
    fit_stage0_time_transform, generate_factor_split, model_free_factor_calibration,
)
from clinical_jepa.eval.j04c_stage1 import (
    CONDITION_NAMES, FROZEN_CONDITION_NAMES, GENERATOR_SEED, LATENT_NAMES, MODEL_SEED,
    THRESHOLD_DIGEST, TrainedCondition, evaluate_readouts, fit_condition_readouts,
    freeze_encoder, one_run_negative_control, reduced_complete_rule_dry,
    threshold_reference_report, train_direct_condition, train_latent_condition,
    trained_collapse_diagnostics,
)
from clinical_jepa.eval.j04c_falsifier import make_r0_encoder, sever_student_leak

CONTRACT_BASE_COMMIT = "e683a7d339e3c06010f724ee8d72d4ec45cee999"
DEPENDENCY_SHA256 = {
    "clinical_jepa/arms/v0f/__init__.py": "9cdf70f66385b0c93b425abba13b9c6798cda445e01fdd7dddf5fc80588b40b3",
    "clinical_jepa/arms/v0f/own_latent.py": "f3cd838225f8099c79d604036961cc64170c98a87b925fd960550846b7c50dfb",
    "clinical_jepa/eval/next_event_metrics.py": "471d3e8b3f37a64ad26ff79f011b12da326f925188ae0c249547dfe628755b7d",
    "clinical_jepa/targets/next_event_contract.py": "7903f587996a2fd82fde5b13316f853cd06b2c1c8c13eca642dedcad7f8c755d",
    "tests/test_bp_clinjepa_011_metrics.py": "6aec5975f7bd3126af7b2f209cc5f5b85c20e3c980b4de9f8076b8a8a709c62d",
    "tests/test_bp_clinjepa_011_model.py": "e3ce77e077a266454600dd0ef6e7eba13dee4fbdb7d6f237f5c2eb98f635e749",
    "tests/test_bp_clinjepa_011_targets.py": "23149cb9b2770219d92e071fdc758f35997fb6c6c8799a2e0e6f197c97effe8f",
    "clinical_jepa/eval/j04c_falsifier.py": "206bf1d59d36a180168b6bb0954d68db50f46b3a0828c6c4c9bbc2e19c843a0a",
    "scripts/bp_clinjepa_011_j04c_stage0.py": "729e3c19481d7f3abf554c5a4b784f0aaf2d4175dba1b96bf4ce3c1720e4bc5f",
    "tests/test_bp_clinjepa_011_j04c.py": "66763b4e093063e49bd15b50db09577dfae251566699d6232a728c2946d6ac5e",
}
PACKAGE_VERSIONS = {"python": "3.10.12", "numpy": "2.2.6", "pytest": "9.0.2", "torch": "2.9.1+cu128"}


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    train = generate_factor_split(GENERATOR_SEED, TRAIN, 8192)
    probe_fit = generate_factor_split(GENERATOR_SEED, PROBE_FIT, 2048)
    cal_id = generate_factor_split(GENERATOR_SEED, CAL_ID, 2048)
    cal_ood = generate_factor_split(GENERATOR_SEED, CAL_OOD, 2048)
    power_cal_ood = tuple(
        generate_factor_split(seed, CAL_OOD, 2048)
        for seed in (1101, 1102, 1103)
    )
    transform = fit_stage0_time_transform(train)

    conditions: dict[str, TrainedCondition] = {}
    for name in LATENT_NAMES:
        conditions[name] = train_latent_condition(name, train, transform)
    for name in CONDITION_NAMES[3:]:
        conditions[name] = train_direct_condition(name, train, transform)
    r0 = make_r0_encoder(MODEL_SEED)
    conditions["R0_INIT"] = TrainedCondition("R0_INIT", r0, None, None, {
        "attempted_steps": 0, "successful_steps": 0, "optimizer_steps": 0,
        "ema_updates": 0, "losses": None,
    })
    for condition in conditions.values():
        freeze_encoder(condition)

    readouts = {}
    readout_reports = {}
    cal_metrics = {"CAL_ID": {}, "CAL_OOD": {}}
    cal_ood_predictions = {}
    for name in FROZEN_CONDITION_NAMES:
        probes, head, report = fit_condition_readouts(conditions[name], probe_fit, transform)
        readouts[name] = (probes, head)
        readout_reports[name] = report
        cal_metrics["CAL_ID"][name], _ = evaluate_readouts(conditions[name], probes, head, cal_id, transform)
        cal_metrics["CAL_OOD"][name], predictions = evaluate_readouts(conditions[name], probes, head, cal_ood, transform)
        cal_ood_predictions[name] = predictions

    negative_controls = []
    for factor in range(3):
        row = one_run_negative_control(cal_ood_predictions["C0_NUISANCE_ONLY"][factor], cal_ood.S[:, factor], cal_ood.N, factor)
        row["assertion"] = f"nuisance_only_{('comp','order','time')[factor]}"
        negative_controls.append(row)
    row = one_run_negative_control(cal_ood_predictions["C0_TIME_FREE"][2], cal_ood.S[:, 2], cal_ood.N, 2)
    row["assertion"] = "time_free_time"; negative_controls.append(row)

    pre_sever_ba = [cal_metrics["CAL_OOD"]["C0_STUDENT_LEAK"]["factors"][f]["balanced_accuracy"] for f in range(3)]
    if any(value < 0.95 for value in pre_sever_ba):
        # This is a substantive smoke result, not a structural failure; still compute severed controls.
        pass
    truth_before = cal_ood.S.copy()
    severed = sever_student_leak(cal_ood.L_after, GENERATOR_SEED)
    severed_metrics, severed_predictions = evaluate_readouts(
        conditions["C0_STUDENT_LEAK"], *readouts["C0_STUDENT_LEAK"], cal_ood, transform,
        leak_override=severed,
    )
    if not np.array_equal(cal_ood.S, truth_before):
        raise RuntimeError("student-leak severing changed S")
    for factor in range(3):
        row = one_run_negative_control(severed_predictions[factor], cal_ood.S[:, factor], cal_ood.N, factor)
        row["assertion"] = f"student_leak_severed_{('comp','order','time')[factor]}"
        negative_controls.append(row)

    reference, threshold_digest = threshold_reference_report()
    collapse_rows = []
    position_diagnostics = {}
    for name in LATENT_NAMES:
        rows, position = trained_collapse_diagnostics(conditions[name], cal_ood, transform, reference)
        collapse_rows.extend(rows)
        if position is not None:
            position_diagnostics[name] = position

    r0_metrics = cal_metrics["CAL_OOD"]["R0_INIT"]
    r0_deltas = []
    factor_names = ("comp", "order", "time")
    for arm, factors in (("L0_EMA_POOL", (0,)), ("L1_AVG", (0, 1, 2)), ("L2_SEP", (0, 1, 2))):
        for factor in factors:
            trained_ba = cal_metrics["CAL_OOD"][arm]["factors"][factor]["balanced_accuracy"]
            r0_ba = r0_metrics["factors"][factor]["balanced_accuracy"]
            r0_deltas.append({"contrast": f"{arm}-{factor_names[factor]}_trained_minus_R0_BA", "difference": trained_ba - r0_ba})
    for arm in LATENT_NAMES:
        r0_deltas.append({"contrast": f"{arm}_R0_minus_trained_type_NLL",
                          "difference": r0_metrics["type_nll"] - cal_metrics["CAL_OOD"][arm]["type_nll"]})
    if len(r0_deltas) != 10:
        raise RuntimeError("R0 contrast count mismatch")

    calibrations = model_free_factor_calibration((1101, 1102, 1103))
    reduced = reduced_complete_rule_dry(power_cal_ood, calibrations)

    training_counts_pass = all(
        conditions[name].training["successful_steps"] == 2000 and
        conditions[name].training["optimizer_steps"] == 2000 and
        conditions[name].training["ema_updates"] == (2000 if name in LATENT_NAMES else 0)
        for name in CONDITION_NAMES
    )
    readout_counts_pass = all(
        all(row["successful_steps"] == 250 for row in readout_reports[name]["probes"]) and
        readout_reports[name]["head"]["successful_steps"] == 250
        for name in FROZEN_CONDITION_NAMES
    )
    smoke_gate = {
        "exact_training_step_counts": training_counts_pass,
        "exact_readout_step_counts": readout_counts_pass,
        "pre_sever_student_leak_all_factors_ge_0_95": all(value >= 0.95 for value in pre_sever_ba),
        "seven_negative_controls_pass": len(negative_controls) == 7 and all(row["passes"] for row in negative_controls),
        "all_21_collapse_flags_pass": len(collapse_rows) == 21 and all(row["both_metrics_pass"] for row in collapse_rows),
        "threshold_digest_match": threshold_digest == THRESHOLD_DIGEST,
    }

    output = {
        "schema_version": "bp-clinjepa-011-j04c-stage1-v1",
        "authority": "public-synthetic CAL-only plumbing smoke",
        "contract_base_commit": CONTRACT_BASE_COMMIT,
        "accepted_dependency_sha256": DEPENDENCY_SHA256,
        "package_versions": PACKAGE_VERSIONS,
        "seeds": {"smoke_generator": GENERATOR_SEED, "smoke_model": MODEL_SEED,
                  "model_free_calibration_generators": [1101, 1102, 1103],
                  "smoke_pair_disjoint_from_candidate_panel": True},
        "split_codes_used": {"TRAIN": TRAIN, "CAL_ID": CAL_ID, "CAL_OOD": CAL_OOD, "PROBE_FIT": PROBE_FIT},
        "split_sizes": {"TRAIN": 8192, "CAL_ID": 2048, "CAL_OOD": 2048, "PROBE_FIT": 2048,
                        "model_free_calibration_CAL_OOD": [2048, 2048, 2048]},
        "materialization_declaration": {"only_split_codes": [1, 2, 3, 6], "all_arrays_in_memory": True,
                                        "model_free_panels": "CAL-OOD calibration/dry-machinery-only",
                                        "id_dev_generated": False, "ood_dev_generated": False},
        "train_time_transform": {"population": "TRAIN prefix intervals then target intervals, per row, row-major",
                                 "mu": transform.mu, "sigma": transform.sigma},
        "training": {name: conditions[name].training for name in CONDITION_NAMES},
        "readout_fits": readout_reports,
        "cal_metrics": cal_metrics,
        "negative_control_smoke": {"assertions": negative_controls,
                                   "student_leak_pre_sever_balanced_accuracy": dict(zip(factor_names, pre_sever_ba)),
                                   "student_leak_severed_descriptive_metrics": severed_metrics},
        "collapse": {"accepted_threshold_row_sha256": threshold_digest, "rows": collapse_rows,
                     "all_21_pass": len(collapse_rows) == 21 and all(row["both_metrics_pass"] for row in collapse_rows)},
        "position_specificity_cal_ood_descriptive": position_diagnostics,
        "r0_cal_ood_descriptive": {"contrasts": r0_deltas,
                                   "L0_order_BA": r0_metrics["factors"][1]["balanced_accuracy"],
                                   "L0_time_BA": r0_metrics["factors"][2]["balanced_accuracy"]},
        "reduced_dry_mode_not_complete_rule_calibration": {
            "results": reduced, "production_specification": {"simulations": 200, "bootstrap_replicates": 10000, "minimum_passes": 160},
            "excluded_from_stage1_smoke_gate": True,
        },
        "smoke_gate": smoke_gate,
        "training_performed": True,
        "candidate_c_a_computed": False,
        "cal_ood_smoke_c_a_computed": True,
        "id_dev_generated": False,
        "ood_dev_generated": False,
        "thresholds_modified": False,
        "seed_selected": False,
        "stage1_smoke_pass": all(smoke_gate.values()),
        "interpretation_ceiling": "Public-synthetic CAL-only plumbing evidence; not J04c PASS, arm superiority, real-EHR robustness, governed-comparison eligibility, or Stage 2/3 authority.",
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
