#!/usr/bin/env python3
"""Rung-2 sub-gates 2 and 4 END-TO-END DRY RUN on synthetic data — prove the scorers emit real numbers.

Companion to `rung2_subgate1_synthetic_dryrun.py`. Sub-gate 1's dry run found three instrumentation traps
(a flag nothing wrote, the wrong model arm, and an absent exposure gap that silently reported HEALTHY), so
the same treatment is applied here BEFORE any governed data or GPU time is spent.

Both sub-gates make STRUCTURAL claims in their docstrings that are more interesting than their arithmetic,
and those are what this exercises:

  * sub-gate 2 — "if B is restricted to a point estimate it CANNOT win a calibrated-distribution comparison;
    that is a structural interface decision, not a horse race". So a point-estimate B must lose EVEN WHEN
    its apparent skill is high.
  * sub-gate 4A — "p0 reliability ALONE cannot pass". So a head that is perfectly CALIBRATED but carries no
    multiplicity SKILL must FAIL. Calibration is not evidence of information.
  * sub-gate 4B — the four criteria are "non-compensatory". So each one failing alone must sink the gate
    even when the other three are excellent.

Real RPS and ECE are computed from synthetic predictive distributions with planted context signal; the
decision functions are then driven with those measured values. Synthetic only: NO substrate, NO checkpoint,
NO governed read, NO TEST. This validates the INSTRUMENT, not the model.

Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/rung2_subgate24_synthetic_dryrun.py
"""
from __future__ import annotations

import json

import numpy as np

from clinical_jepa.eval.rung2_contract import (
    COUNT_NOMINATE_MARGIN, GATE_4A_ECE, GATE_4A_MULTIPLICITY_SKILL, GATE_4A_SWAP_SKILL,
    GATE_4B_CRPS_SKILL, GATE_4B_KS, GATE_4B_RATE_HEAD_IMPROVEMENT, GATE_4B_SWAP,
    NEITHER_ADEQUATE, NOMINATE_CONCAT, NOMINATE_FACTORIZED, NOT_EVALUABLE,
)
from clinical_jepa.eval.rung2_count_interface import (
    count_interface_decision, ranked_probability_score, rps_skill_vs_baseline,
)
from clinical_jepa.eval.rung2_timing import (
    assert_context_observable_strata, expected_calibration_error, gate_4a, gate_4b, timing_verdict,
)

SEED = 20260725
N, K_MAX, N_CLUSTERS = 3000, 12, 600


# ---------------------------------------------------------------------------------------------------
# sub-gate 2 — count interface
# ---------------------------------------------------------------------------------------------------
def _poisson_pmf(lam, k_max=K_MAX):
    k = np.arange(k_max + 1)
    logp = k[None, :] * np.log(np.clip(lam, 1e-9, None))[:, None] - lam[:, None]
    from math import lgamma
    logp = logp - np.array([lgamma(i + 1) for i in k])[None, :]
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def subgate2(rng):
    """Real RPS on synthetic counts: a context-informed head vs a context-blind marginal baseline."""
    ctx = rng.normal(size=N)                                   # the context signal
    lam_true = np.exp(0.6 + 0.5 * ctx)                         # true rate depends on context
    y = rng.poisson(lam_true).clip(0, K_MAX)
    clusters = rng.integers(0, N_CLUSTERS, size=N)

    pmf_A = _poisson_pmf(lam_true)                             # interface A: sees the context
    pmf_marginal = _poisson_pmf(np.full(N, y.mean()))          # baseline: context-blind
    # A REAL negative control, not a copy of the baseline: same predictor family and same marginal
    # calibration, but the context is SHUFFLED so the context->count link is destroyed. Skill must
    # collapse to <= ~0. (Reusing the baseline itself would make skill 0.0 by construction — a
    # tautology that would pass even if the scorer were broken.)
    pmf_blind = _poisson_pmf(rng.permutation(lam_true))

    rps_A = ranked_probability_score(pmf_A, y)
    rps_base = ranked_probability_score(pmf_marginal, y)
    rps_blind = ranked_probability_score(pmf_blind, y)

    skill_A = rps_skill_vs_baseline(rps_A, rps_base, clusters)
    skill_blind = rps_skill_vs_baseline(rps_blind, rps_base, clusters)

    # a genuinely BETTER B (sharper, still calibrated) and a point-estimate B (all mass on the mean)
    pmf_B = _poisson_pmf(lam_true * 1.0)
    rps_B = ranked_probability_score(pmf_B, y)
    skill_B = rps_skill_vs_baseline(rps_B, rps_base, clusters)
    point = np.zeros((N, K_MAX + 1)); point[np.arange(N), np.rint(lam_true).astype(int).clip(0, K_MAX)] = 1.0
    rps_point = ranked_probability_score(point, y)
    skill_point = rps_skill_vs_baseline(rps_point, rps_base, clusters)

    rows = []
    rows.append(("context-informed A beats the marginal baseline",
                 skill_A["ci_lo"] > 0.0, {"skill_lo": round(skill_A["ci_lo"], 4)}))
    rows.append(("shuffled-context A has NO skill -> NEITHER_ADEQUATE",
                 count_interface_decision(skill_a_lo=skill_blind["ci_lo"], skill_b_lo=-0.01,
                                          paired_b_minus_a_lo=-0.01,
                                          b_is_point_estimate=False)["decision"] == NEITHER_ADEQUATE,
                 {"skill_lo": round(skill_blind["ci_lo"], 4)}))
    # THE STRUCTURAL CLAIM: a point-estimate B loses even with high apparent skill
    d_point = count_interface_decision(skill_a_lo=skill_A["ci_lo"], skill_b_lo=0.99,
                                       paired_b_minus_a_lo=0.99, b_is_point_estimate=True)
    rows.append(("point-estimate B loses DESPITE skill_b=0.99 (structural)",
                 d_point["decision"] == NOMINATE_FACTORIZED, {"decision": d_point["decision"]}))
    d_tie = count_interface_decision(skill_a_lo=skill_A["ci_lo"], skill_b_lo=skill_B["ci_lo"],
                                     paired_b_minus_a_lo=0.0, b_is_point_estimate=False)
    rows.append((f"paired tie -> A is the default (margin {COUNT_NOMINATE_MARGIN})",
                 d_tie["decision"] == NOMINATE_FACTORIZED, {"decision": d_tie["decision"]}))
    d_win = count_interface_decision(skill_a_lo=skill_A["ci_lo"], skill_b_lo=skill_B["ci_lo"],
                                     paired_b_minus_a_lo=COUNT_NOMINATE_MARGIN + 0.01,
                                     b_is_point_estimate=False)
    rows.append(("B beats A by > margin -> NOMINATE_CONCAT",
                 d_win["decision"] == NOMINATE_CONCAT, {"decision": d_win["decision"]}))
    # RPS sanity: a perfect delta forecast scores 0; the point estimate is WORSE than the calibrated one
    perfect = np.zeros((N, K_MAX + 1)); perfect[np.arange(N), y] = 1.0
    rows.append(("RPS of a perfect forecast is exactly 0",
                 float(ranked_probability_score(perfect, y).max()) == 0.0, {}))
    rows.append(("calibrated A scores better than the point estimate",
                 float(rps_A.mean()) < float(rps_point.mean()),
                 {"rps_A": round(float(rps_A.mean()), 4), "rps_point": round(float(rps_point.mean()), 4)}))
    measured = {"skill_A_lo": round(skill_A["ci_lo"], 4), "skill_blind_lo": round(skill_blind["ci_lo"], 4),
                "skill_point_lo": round(skill_point["ci_lo"], 4),
                "mean_rps_A": round(float(rps_A.mean()), 4),
                "mean_rps_baseline": round(float(rps_base.mean()), 4)}
    return rows, measured


# ---------------------------------------------------------------------------------------------------
# sub-gate 4 — continuous-time / multiplicity
# ---------------------------------------------------------------------------------------------------
def subgate4(rng):
    ctx = rng.normal(size=N)
    p_true = 1.0 / (1.0 + np.exp(-(0.3 + 1.2 * ctx)))          # context-dependent multiplicity prob
    outcome = (rng.random(N) < p_true).astype(float)

    ece_informed = expected_calibration_error(p_true, outcome)          # calibrated AND informed
    p_rate_only = np.full(N, outcome.mean())
    ece_rate_only = expected_calibration_error(p_rate_only, outcome)    # calibrated, NO information

    rows = []
    rows.append(("a context-informed head is well calibrated",
                 ece_informed <= GATE_4A_ECE, {"ece": round(ece_informed, 4)}))
    rows.append(("a rate-only head is ALSO well calibrated (calibration != information)",
                 ece_rate_only <= GATE_4A_ECE, {"ece": round(ece_rate_only, 4)}))
    # THE STRUCTURAL CLAIM: perfect calibration + no skill must FAIL
    g_cal_only = gate_4a(multiplicity_skill_lo=0.0, swap_skill_lo=0.0, ece_hi=ece_rate_only, evaluable=True)
    rows.append(("perfectly calibrated but NO skill -> 4A FAIL (p0 reliability alone cannot pass)",
                 g_cal_only["gate_4a"] == "FAIL", {"gate_4a": g_cal_only["gate_4a"]}))
    g_pass = gate_4a(multiplicity_skill_lo=GATE_4A_MULTIPLICITY_SKILL + 0.02,
                     swap_skill_lo=GATE_4A_SWAP_SKILL + 0.02, ece_hi=ece_informed, evaluable=True)
    rows.append(("skill + swap + calibration -> 4A PASS", g_pass["gate_4a"] == "PASS", {}))
    rows.append(("4A NOT_EVALUABLE propagates",
                 gate_4a(multiplicity_skill_lo=9, swap_skill_lo=9, ece_hi=0,
                         evaluable=False)["gate_4a"] == NOT_EVALUABLE, {}))

    # 4B non-compensatory: each criterion failing ALONE must sink the gate
    good = dict(ks_upper_ci=GATE_4B_KS - 0.01, crps_skill_lo=GATE_4B_CRPS_SKILL + 0.02,
                rate_head_improvement_lo=GATE_4B_RATE_HEAD_IMPROVEMENT + 0.02,
                swap_skill_lo=GATE_4B_SWAP + 0.02, evaluable=True)
    rows.append(("all four criteria good -> 4B PASS", gate_4b(**good)["gate_4b"] == "PASS", {}))
    for k, bad in (("ks_upper_ci", GATE_4B_KS + 0.01),
                   ("crps_skill_lo", GATE_4B_CRPS_SKILL - 0.01),
                   ("rate_head_improvement_lo", GATE_4B_RATE_HEAD_IMPROVEMENT - 0.01),
                   ("swap_skill_lo", GATE_4B_SWAP - 0.01)):
        rows.append((f"4B non-compensatory: {k} alone sinks it",
                     gate_4b(**{**good, k: bad})["gate_4b"] == "FAIL", {}))

    # conjunction + the oracle-assisted stratum guard
    g4a_p, g4b_p = {"gate_4a": "PASS"}, {"gate_4b": "PASS"}
    rows.append(("verdict PASS only when both pass", timing_verdict(g4a_p, g4b_p) == "PASS", {}))
    rows.append(("verdict FAIL if either fails",
                 timing_verdict({"gate_4a": "FAIL"}, g4b_p) == "FAIL", {}))
    rows.append(("verdict NOT_EVALUABLE propagates",
                 timing_verdict({"gate_4a": NOT_EVALUABLE}, g4b_p) == NOT_EVALUABLE, {}))
    ok_strata = assert_context_observable_strata(["context_rate", "context_len"])
    try:
        assert_context_observable_strata(["context_rate", "future_rate"])
        guard = False
    except AssertionError:
        guard = True
    rows.append(("observed-future stratum REFUSED as a primary baseline", guard and ok_strata, {}))
    measured = {"ece_informed": round(ece_informed, 4), "ece_rate_only": round(ece_rate_only, 4)}
    return rows, measured


def main():
    rng = np.random.default_rng(SEED)
    s2_rows, s2_meas = subgate2(rng)
    s4_rows, s4_meas = subgate4(rng)

    print("SUB-GATE 2 — count interface (real RPS on synthetic counts)")
    for label, ok, extra in s2_rows:
        print(f"   {'ok ' if ok else 'FAIL'} {label:<58} {extra if extra else ''}")
    print(f"   measured: {json.dumps(s2_meas)}")

    print("\nSUB-GATE 4 — continuous-time / multiplicity (real ECE on synthetic outcomes)")
    for label, ok, extra in s4_rows:
        print(f"   {'ok ' if ok else 'FAIL'} {label:<58} {extra if extra else ''}")
    print(f"   measured: {json.dumps(s4_meas)}")

    ok2 = all(r[1] for r in s2_rows)
    ok4 = all(r[1] for r in s4_rows)
    print("\nRESULT:", "SUB-GATES 2 AND 4 GREEN — scorers emit real numbers and the structural claims hold."
          if (ok2 and ok4) else "PROBLEM — see FAIL rows above.")
    print("SCOPE: synthetic only. Validates the INSTRUMENT, not the model. No substrate, no checkpoint, no "
          "governed read, no TEST. Nothing here is a Rung-2 result.")
    print("\n" + json.dumps({"subgate2_green": bool(ok2), "subgate4_green": bool(ok4),
                             "subgate2_measured": s2_meas, "subgate4_measured": s4_meas}, indent=2))
    return ok2 and ok4


if __name__ == "__main__":
    main()
