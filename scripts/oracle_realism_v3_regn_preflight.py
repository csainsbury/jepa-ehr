#!/usr/bin/env python3
"""Oracle realism v3 — REGISTERED-N=8000 map/support preflight for the boundary exemptions (Pi rev-6 #2).

The dev pilot (N=3000) could not decide the S3 boundary exemptions: a map refusal at N=3000 is a DEV-SCALE floor
refusal (registered floor is 500), not proof the estimand is structurally unavailable. v2 already showed S3_loggap
evaluable on the N=8000 bounded control. This preflight therefore evaluates, at the REGISTERED N=8000, per exact
regime {bounded-short, SCID-scale, MIMIC-scale, structural-zero}:

  * S3_loggap: does the reference-owned frozen map ISSUE (status OK) at N=8000? i.e. is the conditional estimand
    structurally EVALUABLE at registered scale? Plus Δ-aligned detection P[d>Δ | @0.5] where evaluable;
  * S3_tau: is the pooled-tau estimand eligible (floor met) at N=8000? Plus Δ-aligned detection.

Decision rule (per regime, boundary-support is the exemption locus): a subcheck is EXEMPT on bounded support only
if it is structurally un-calibratable there at registered N — the map does NOT issue (S3_loggap) or the estimand
is ineligible (S3_tau) OR its registered-N detection of a real @0.5 alternative is < 0.5. Both exemptions stay
PROVISIONAL until the reserved calibration draw. Development-only, aggregate-hashed; NO reserved map-design draw,
no calibration/eval seed. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_regn_preflight.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_verifier import _TAU, _LOGGAP
from scripts.oracle_realism_v3_map import build_frozen_map, apply_frozen_map, map_identity
from scripts.oracle_realism_v3_phase0_pilot import T_pool

NS = "v3-regnpreflight-dev"
N = 8000                              # REGISTERED sample size
SEEDS = list(range(95000, 95010))     # 10 dev seeds
REGIMES = [("bounded_short", "boundary_short", "bounded"),
           ("scid_scale", "scid_scale_control", "full"),
           ("mimic_scale", "mimic_scale_control", "full"),
           ("structural_zero", "structural_zero_control", "full")]


def dseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (NS, *p))).encode()).digest()[:6], "big")


def draw(profile, tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, tag))


def _tau_d(A, B):
    a, b = T_pool(A), T_pool(B)
    return None if (a is None or b is None) else abs(a - b)


def regime_preflight(name, profile, regime):
    # S3_loggap map issuance at registered N
    map_ref = draw(profile, "mapref")
    fm = build_frozen_map(map_ref, "S3_loggap", profile=profile, regime=regime,
                          seed=int(dseed("mapref", name)), N=N)
    loggap_issues = fm["status"] == "OK"

    lg_pow, lg_null, lg_ne = [], [], 0
    tau_pow, tau_null, tau_ne = [], [], 0
    for k in SEEDS:
        A = draw(profile, ("A", k)); B = draw(profile, ("B", k))
        Bc = apply_coupling(list(B), "burst_timing", 0.5, seed=dseed("cpl", name, k))
        # S3_tau
        dn, dp = _tau_d(A, B), _tau_d(A, Bc)
        if dn is None or dp is None:
            tau_ne += 1
        else:
            tau_null.append(dn > _TAU); tau_pow.append(dp > _TAU)
        # S3_loggap (only if the map issues)
        if loggap_issues:
            ln = apply_frozen_map(B, A, "S3_loggap", fm, expect_profile=profile, expect_regime=regime)
            lp = apply_frozen_map(Bc, A, "S3_loggap", fm, expect_profile=profile, expect_regime=regime)
            if ln is None or lp is None:
                lg_ne += 1
            else:
                lg_null.append(ln > _LOGGAP); lg_pow.append(lp > _LOGGAP)
    rate = lambda v: round(float(np.mean(v)), 3) if v else None
    return {
        "regime": regime, "profile": profile,
        "S3_loggap": {"map_status": fm["status"], "map_issues_at_registered_N": loggap_issues,
                      "map_identity": map_identity(fm),
                      "detect_P[d>delta]@0.5": rate(lg_pow), "false_P[d>delta]_null": rate(lg_null),
                      "ne_rate": round(lg_ne / len(SEEDS), 3), "n_pow": len(lg_pow)},
        "S3_tau": {"eligible_at_registered_N": tau_ne < len(SEEDS),
                   "detect_P[d>delta]@0.5": rate(tau_pow), "false_P[d>delta]_null": rate(tau_null),
                   "ne_rate": round(tau_ne / len(SEEDS), 3), "n_pow": len(tau_pow)},
    }


def main():
    results = {name: regime_preflight(name, profile, regime) for name, profile, regime in REGIMES}

    # boundary exemption decision at REGISTERED N, per subcheck (locus = bounded support)
    b = results["bounded_short"]
    def _exempt(sub):
        s = b[sub]
        if sub == "S3_loggap":
            evaluable = s["map_issues_at_registered_N"] and s["n_pow"] > 0
        else:
            evaluable = s["eligible_at_registered_N"] and s["n_pow"] > 0
        detect = s["detect_P[d>delta]@0.5"]
        # exempt iff NOT structurally evaluable at registered N, OR registered-N detection < 0.5
        return (not evaluable) or ((detect or 0.0) < 0.5), evaluable, detect
    decision = {}
    for sub in ("S3_tau", "S3_loggap"):
        ex, ev, det = _exempt(sub)
        decision[sub] = {"exempt_at_registered_N": ex, "structurally_evaluable_bounded_N8000": ev,
                         "bounded_registered_detect": det}

    out = {"namespace": NS, "N": N, "n_seeds": len(SEEDS), "delta_tau": _TAU, "delta_loggap": round(_LOGGAP, 8),
           "per_regime": results,
           "boundary_exemption_decision_registered_N": decision,
           "reporting_rule": "final group design MUST be reported with AND without each exemption INDEPENDENTLY "
                             "(4 combinations), not only both-together; both exemptions remain PROVISIONAL until "
                             "the reserved calibration draw (blocked).",
           "note": "Corrects Pi rev-6 #2: dev-scale (N=3000) map refusal is NOT structural. This decides "
                   "evaluability + Δ-aligned detection at REGISTERED N=8000 per regime. No reserved map draw.",
           "authorization": "dev-only preflight; no reserved map-design namespace, no calibration/eval seed, no policy."}
    print(json.dumps(out, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(out))
    return out


if __name__ == "__main__":
    main()
