#!/usr/bin/env python3
"""Rung-2 sub-gate 1 END-TO-END DRY RUN on synthetic rollouts — prove the harness emits real numbers.

Sub-gate 1 answers the question that actually matters for autoregressive futures: when a rollout degrades,
is it the PREDICTOR or the TARGET that is the bottleneck? Before spending governed data and GPU time to find
out, this exercises the whole scoring path end to end on synthetic latents, with NO substrate, NO checkpoint,
NO governed read and NO TEST.

It checks four things:

  1. PATH GATING — the recursive-transition path must be NOT_EVALUABLE for a horizon_count=1 checkpoint (every
     existing v0B / encode-empty checkpoint), and evaluable for a genuine fixed-width non-overlapping
     transition config, whether the flag is stamped or DERIVED from the recorded training config.
  2. DIRECT-HORIZON PATH — runs `direct_horizon_metrics` on rollouts whose prediction error is a controlled
     function of the horizon, and checks the reported drift actually TRACKS the injected degradation. A
     metric that cannot see a planted effect cannot be trusted to report its absence.
  3. FAIL-HARD SEPARATION — a direct-path row that carries a recursive-only metric must raise, not warn.
  4. RECURSIVE PATH — with a qualifying checkpoint, the exposure gap and the frozen-margin signature are
     emitted, and the signature responds correctly to planted HEALTHY / DRIFT / COLLAPSE regimes.

Synthetic only. This validates the INSTRUMENT, not the model: no claim about real rollout behaviour follows
from it. Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_subgate1_synthetic_dryrun.py
"""
from __future__ import annotations

import json

import numpy as np

from clinical_jepa.eval import rung2_rollout_diag as RD
from clinical_jepa.eval import rung2_rollout_export as RE
from clinical_jepa.eval.rung2_contract import (
    NOT_EVALUABLE, SIG_COLLAPSE_DSELF_OVER_DNN, TRANSITION_META_KEY, validate_direct_path_row,
)
from clinical_jepa.eval.rung2_transition_regime import is_fixed_width_transition_training

SEED = 20260725
N_PATIENTS, D = 240, 64
HORIZONS = (0.25, 0.5, 1.0, 2.0)          # MIMIC-scale wall-clock horizons (days)
# injected prediction noise per horizon: a PLANTED horizon-decay effect the metric must recover
NOISE_BY_W = {0.25: 0.10, 0.5: 0.25, 1.0: 0.50, 2.0: 0.90}


def synth_rollouts(rng):
    """True target latents per patient per horizon, plus predictions degraded by a known amount."""
    pred_by_W, true_by_W, pats_by_W = {}, {}, {}
    patients = np.arange(N_PATIENTS)
    for W in HORIZONS:
        ztrue = rng.normal(size=(N_PATIENTS, D))
        noise = rng.normal(size=(N_PATIENTS, D)) * NOISE_BY_W[W]
        pred_by_W[W], true_by_W[W], pats_by_W[W] = ztrue + noise, ztrue, patients
    return pred_by_W, true_by_W, pats_by_W


def check_path_gating():
    """Existing checkpoints must NOT unlock the recursive path; a real transition config must."""
    # TWO separate questions, deliberately kept apart:
    #  * does the DERIVATION qualify this training config?  (rung2_transition_regime, at write time)
    #  * does the frozen CONTRACT gate open?                (flag-only, at read time)
    configs = {
        "v0B default (horizon_count=1)": (
            dict(autoregression_mode="recursive", horizon_count=1,
                 horizon_stride_tokens=32, max_target_tokens=32), False),
        "encode-empty (pinned to 1)": (
            dict(autoregression_mode="recursive", horizon_count=1, horizon_stride_tokens=32,
                 max_target_tokens=32, encode_empty=True), False),
        "horizon-conditioned mode": (
            dict(autoregression_mode="horizon_conditioned", horizon_count=4,
                 horizon_stride_tokens=32, max_target_tokens=32), False),
        "OVERLAPPING windows (stride<width)": (
            dict(autoregression_mode="recursive", horizon_count=4,
                 horizon_stride_tokens=16, max_target_tokens=32), False),
        "GAPPED windows (stride>width)": (
            dict(autoregression_mode="recursive", horizon_count=4,
                 horizon_stride_tokens=64, max_target_tokens=32), False),
        "QUALIFYING recursive transition": (
            dict(autoregression_mode="recursive", horizon_count=4,
                 horizon_stride_tokens=32, max_target_tokens=32), True),
    }
    rows, ok = [], True
    for label, (cfg, want) in configs.items():
        got = is_fixed_width_transition_training(**cfg)
        ok &= (got == want)
        rows.append((f"derive: {label}", want, got, got == want))
    # the frozen contract gate is FLAG-ONLY and fails closed without it
    for label, meta, want in (("gate: flag stamped True", {TRANSITION_META_KEY: True}, True),
                              ("gate: flag stamped False", {TRANSITION_META_KEY: False}, False),
                              ("gate: no flag at all", {}, False)):
        got = RE.plan_paths(meta)["recursive"]
        ok &= (got == want)
        rows.append((label, want, got, got == want))
    return rows, ok


def check_direct_path(pred_by_W, true_by_W, pats_by_W):
    rows = RE.direct_horizon_metrics(pred_by_W, true_by_W, pats_by_W, source="synthetic")
    drift = [r["d_self_over_ambient_nn"] for r in rows]
    monotone = all(b > a for a, b in zip(drift, drift[1:]))          # planted decay must be recovered
    return rows, monotone


def check_fail_hard(rows):
    """A direct row may NEVER carry a recursive-only metric — this must RAISE, not warn."""
    bad = {**rows[0], "exposure_gap": 0.01}
    try:
        validate_direct_path_row(bad)
        return False
    except Exception:
        return True


def check_recursive(rng):
    """The recursive path emits the exposure gap + signature, and the signature tracks planted regimes."""
    # flag-only gate: a qualifying checkpoint carries the flag the TRAINER derived at write time
    meta = {TRANSITION_META_KEY: True, "autoregression_mode": "recursive", "horizon_count_trained": 4,
            "horizon_stride_tokens": 32, "max_target_tokens": 32}
    n = N_PATIENTS
    regimes = {
        # free-running ~ teacher-forced, own-truth well inside the ambient scale
        "HEALTHY": dict(free=rng.normal(0.20, 0.02, n), tf=rng.normal(0.19, 0.02, n),
                        dself_over_nn=0.30, gap_lo=0.0, slope_hi=0.0),
        # free-running drifts away from teacher-forced, and drift grows with step
        "DRIFT_DOMINANT": dict(free=rng.normal(0.55, 0.03, n), tf=rng.normal(0.20, 0.02, n),
                               dself_over_nn=0.40, gap_lo=0.30, slope_hi=0.10),
        # own truth no better than the nearest WRONG instance
        "COLLAPSE_DOMINANT": dict(free=rng.normal(0.60, 0.03, n), tf=rng.normal(0.58, 0.03, n),
                                  dself_over_nn=SIG_COLLAPSE_DSELF_OVER_DNN + 0.05,
                                  gap_lo=0.0, slope_hi=0.0),
    }
    out, ok = [], True
    for want, r in regimes.items():
        res = RE.recursive_transition_metrics(
            meta, dself_free=r["free"], dself_tf=r["tf"], dself_over_nn_point=r["dself_over_nn"],
            exposure_gap_slope_lo=r["gap_lo"], dself_slope_hi=r["slope_hi"],
            source="synthetic", window_days=1.0)
        got = res.get("signature")
        ok &= (got == want)
        out.append((want, got, float(np.mean(RD.exposure_gap(r["free"], r["tf"]))), got == want))
    # and a NON-qualifying checkpoint must return NOT_EVALUABLE with a reason, never a signature
    ne = RE.recursive_transition_metrics({TRANSITION_META_KEY: False, "autoregression_mode": "recursive",
                                          "horizon_count_trained": 1, "horizon_stride_tokens": 32,
                                          "max_target_tokens": 32},
                                         source="synthetic", window_days=1.0)
    ne_ok = ne.get("status") == NOT_EVALUABLE and "signature" not in ne
    return out, ok, ne_ok


def main():
    rng = np.random.default_rng(SEED)
    pred_by_W, true_by_W, pats_by_W = synth_rollouts(rng)

    gate_rows, gate_ok = check_path_gating()
    print("1. PATH GATING — derivation (write time) and the frozen flag-only gate (read time)")
    for label, want, got, good in gate_rows:
        print(f"   {label:<45} expect={str(want):<5} got={str(got):<5} {'ok' if good else 'MISMATCH'}")

    direct_rows, monotone = check_direct_path(pred_by_W, true_by_W, pats_by_W)
    print("\n2. DIRECT-HORIZON PATH — real numbers, planted decay must be recovered")
    print(f"   {'W (days)':>9} {'injected noise':>15} {'d_self_mean':>12} {'d_self/ambient_NN':>18} {'n':>5}")
    for r in direct_rows:
        print(f"   {r['window_days']:>9.2f} {NOISE_BY_W[r['window_days']]:>15.2f} "
              f"{r['d_self_mean']:>12.4f} {r['d_self_over_ambient_nn']:>18.4f} {r['n']:>5}")
    print(f"   drift strictly increasing with horizon: {monotone}")

    fail_hard_ok = check_fail_hard(direct_rows)
    print(f"\n3. FAIL-HARD SEPARATION — direct row carrying a recursive metric raises: {fail_hard_ok}")

    rec_rows, rec_ok, ne_ok = check_recursive(rng)
    print("\n4. RECURSIVE PATH — exposure gap + frozen-margin signature")
    for want, got, gap, good in rec_rows:
        print(f"   planted={want:<18} signature={str(got):<18} mean exposure gap={gap:>7.4f} "
              f"{'ok' if good else 'MISMATCH'}")
    print(f"   non-qualifying checkpoint -> NOT_EVALUABLE with no signature: {ne_ok}")

    ok = gate_ok and monotone and fail_hard_ok and rec_ok and ne_ok
    print("\nRESULT:", "HARNESS GREEN — sub-gate 1 emits real, correctly-gated numbers end to end."
          if ok else "HARNESS PROBLEM — see mismatches above.")
    print("SCOPE: synthetic latents only. This validates the INSTRUMENT, not the model. No substrate, no "
          "checkpoint, no governed read, no TEST. Nothing here is a Rung-2 result.")
    print("\n" + json.dumps({"harness_green": bool(ok), "path_gating_ok": bool(gate_ok),
                             "direct_recovers_planted_decay": bool(monotone),
                             "fail_hard_ok": bool(fail_hard_ok),
                             "recursive_signatures_ok": bool(rec_ok),
                             "not_evaluable_ok": bool(ne_ok)}, indent=2))
    return ok


if __name__ == "__main__":
    main()
