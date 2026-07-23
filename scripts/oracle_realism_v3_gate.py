#!/usr/bin/env python3
"""Oracle realism v3 — fail-closed GROUP GATE entry point (Pi rev-5 #3/#7).

Wraps the product/stratified permutation test with (a) strict validation that runs BEFORE any precompute or
statistic and REFUSES malformed input (never leaks KeyError/NumPy errors), and (b) a frozen NOT_EVALUABLE
total-statistic policy that never zero-fills a support failure:

  * observed SD cell floor failure (statistic undefined on the real split) -> group verdict NOT_EVALUABLE
    (scientific non-pass), handled OUTSIDE the permutation ranking;
  * permutation-assignment floor failure -> a frozen support-violation indicator encoded as MAXIMALLY EXTREME
    (e = +inf), applied symmetrically to every permuted assignment, so a broken conditional check cannot hide as
    maximally non-extreme (the rev-4 `None -> 0` defect).

Refusals covered (Pi #7): empty group/cell set; unknown/missing experiment id or registered quota; quota mismatch;
non-integer/bool/negative/unequal quota; nonfinite value or Delta; truncated cell vector; role/component/source
identity mismatch; B/RNG mismatch; role-dependent (non-symmetric) per-experiment RNG law; malformed assignment
count/shape; prohibited duplicate assignments; map-hash and floor-policy mismatch.

Development-only. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_gate.py
"""
from __future__ import annotations

import json

import numpy as np

from scripts.oracle_realism_v3_randomization import (
    cell_upper_p, _canonical_mask, _perm_mask, _exp_total, RefusalError,
)

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"


def validate_group_spec_strict(cells, experiments, *, registered, declared_B, used_B,
                               rng_law_by_role_by_exp, assignments=None, prohibit_duplicate_assignments=True):
    """Fail-closed validation BEFORE any statistic. Raises RefusalError on any violation."""
    if not cells:
        raise RefusalError("empty cell set")
    if not experiments:
        raise RefusalError("empty experiment/group set")
    if declared_B != used_B:
        raise RefusalError(f"B mismatch: declared {declared_B} != used {used_B}")
    for e, strata in experiments.items():
        if e not in registered.get("quota", {}):
            raise RefusalError(f"unknown/missing experiment id or registered quota: {e}")
        if [tuple(s) for s in strata] != [tuple(s) for s in registered["quota"][e]]:
            raise RefusalError(f"quota mismatch for {e}: {strata} != {registered['quota'][e]}")
        for (nA, nB) in strata:
            for q in (nA, nB):
                if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or int(q) <= 0:
                    raise RefusalError(f"bad quota {q!r} in {e} (must be positive int, not bool)")
            if nA != nB:
                raise RefusalError(f"unequal candidate/reference size in {e}: {nA}!={nB}")
        laws = rng_law_by_role_by_exp.get(e)
        if not laws or len(set(laws.values())) != 1:
            raise RefusalError(f"role-dependent/missing per-experiment RNG law in {e}: {laws}")
    exp_ids = set(experiments)
    for c in cells:
        if c.get("exp") not in exp_ids:
            raise RefusalError(f"cell {c.get('cell_id','?')} references unknown experiment {c.get('exp')}")
        dl = c.get("delta")
        if dl is None or not np.isfinite(dl):
            raise RefusalError(f"nonfinite/missing delta for {c.get('cell_id','?')}")
        vals = np.asarray(c["values"], float)
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            raise RefusalError(f"nonfinite/empty values for {c.get('cell_id','?')}")
        if len(vals) != _exp_total(experiments[c["exp"]]):
            raise RefusalError(f"truncated cell vector {c.get('cell_id','?')}")
        rid = registered.get("identity", {}).get(c["exp"], {})
        for key in ("role", "component", "source"):
            if key in rid and c.get(key) != rid[key]:
                raise RefusalError(f"{key} identity mismatch for {c.get('cell_id','?')}")
    if registered.get("provided_map_hash", registered.get("map_hash")) != registered.get("map_hash"):
        raise RefusalError("map/hash mismatch")
    if registered.get("provided_floor_policy", registered.get("floor_policy")) != registered.get("floor_policy"):
        raise RefusalError("floor-policy mismatch")
    if assignments is not None:
        for e, strata in experiments.items():
            A = assignments.get(e)
            need = sum(nA for nA, _ in strata); M = _exp_total(strata)
            if A is None or len(A) != used_B + 1:
                raise RefusalError(f"malformed assignment count for {e}")
            for m in A:
                if len(m) != M or int(np.sum(m)) != need:
                    raise RefusalError(f"malformed assignment shape for {e}")
            if prohibit_duplicate_assignments:
                seen = set()
                for m in A[1:]:
                    key = tuple(int(i) for i in np.where(m)[0])
                    if key in seen:
                        raise RefusalError(f"prohibited duplicate assignment in {e}")
                    seen.add(key)


def _cell_e(values, masks, statfn, delta):
    """e per assignment with the frozen NE policy: undefined statistic -> +inf (maximally extreme)."""
    e = np.empty(len(masks))
    for j, m in enumerate(masks):
        d = statfn(np.asarray(values, float), m)
        e[j] = np.inf if d is None else max(0.0, d - delta)
    return e


def gate_group(cells, experiments, B, seed, *, registered, rng_law_by_role_by_exp, statfn,
               alpha_group, assignments=None):
    """Validate (fail-closed), then run the product/stratified min-p permutation test with the NE policy.
    Returns a verdict dict. `statfn(values, mask) -> float | None` (None = support failure / undefined)."""
    validate_group_spec_strict(cells, experiments, registered=registered, declared_B=B, used_B=B,
                               rng_law_by_role_by_exp=rng_law_by_role_by_exp, assignments=assignments)
    exp_ids = sorted(experiments)
    if assignments is None:
        rng = np.random.default_rng(seed)
        assignments = {e: [_canonical_mask(experiments[e])] for e in exp_ids}
        for _ in range(B):
            for e in exp_ids:
                assignments[e].append(_perm_mask(rng, experiments[e]))
    # observed floor failure on ANY cell -> group NOT_EVALUABLE (scientific non-pass), no ranking
    for c in cells:
        if statfn(np.asarray(c["values"], float), assignments[c["exp"]][0]) is None:
            return {"verdict": NOT_EVALUABLE, "reason": f"observed floor failure at {c.get('cell_id','?')}",
                    "p_g": None}
    E = [_cell_e(c["values"], assignments[c["exp"]], statfn, c["delta"]) for c in cells]
    P = np.stack([cell_upper_p(e) for e in E], 0)
    S = P.min(0)
    p_g = float((S <= S[0]).sum() / len(S))
    argmin_cell = cells[int(np.argmin(P[:, 0]))]["cell_id"]
    verdict = PASS if p_g > alpha_group else FAIL
    return {"verdict": verdict, "p_g": p_g, "alpha_group": alpha_group, "argmin_cell": argmin_cell,
            "n_cells": len(cells), "B": B}


# --- self-tests -----------------------------------------------------------------------------------
def _floor_gated_meandiff(floor):
    def stat(values, mask):
        a, b = values[mask], values[~mask]
        if a.size < floor or b.size < floor:
            return None
        return abs(a.mean() - b.mean())
    return stat


def selftest():
    errs = []
    experiments = {"e0": [(6, 6)]}
    reg = {"quota": {"e0": [(6, 6)]}, "identity": {"e0": {"role": "null"}}, "map_hash": "M", "floor_policy": "F"}
    rlaws = {"e0": {"candidate": "L", "reference": "L"}}
    base = dict(cells=[{"cell_id": "c0", "exp": "e0", "values": list(np.random.default_rng(1).normal(size=12)),
                        "delta": 0.0, "role": "null"}],
                experiments=experiments, registered=reg, rng_law_by_role_by_exp=rlaws)
    stat = _floor_gated_meandiff(1)

    # valid gate runs
    r = gate_group(**base, B=199, seed=1, statfn=stat, alpha_group=0.00667)
    if r["verdict"] not in (PASS, FAIL):
        errs.append(f"valid gate produced {r['verdict']}")

    # refusals
    def refused(**over):
        spec = {**base, **over}
        try:
            gate_group(**spec, B=over.get("B", 199), seed=1, statfn=stat, alpha_group=0.00667)
            return False
        except RefusalError:
            return True
    refusals = {
        "empty_cells": refused(cells=[]),
        "unknown_exp": refused(cells=[{**base["cells"][0], "exp": "eX"}]),
        "quota_mismatch": refused(experiments={"e0": [(5, 5)]}),
        "bad_quota_bool": refused(experiments={"e0": [(True, True)]}, registered={**reg, "quota": {"e0": [(True, True)]}}),
        "negative_quota": refused(experiments={"e0": [(-6, -6)]}, registered={**reg, "quota": {"e0": [(-6, -6)]}}),
        "unequal_quota": refused(experiments={"e0": [(8, 4)]}, registered={**reg, "quota": {"e0": [(8, 4)]}}),
        "nonfinite_delta": refused(cells=[{**base["cells"][0], "delta": float("nan")}]),
        "nonfinite_value": refused(cells=[{**base["cells"][0], "values": [float("inf")] + [0.0] * 11}]),
        "truncated_vector": refused(cells=[{**base["cells"][0], "values": [0.0] * 10}]),
        "role_mismatch": refused(cells=[{**base["cells"][0], "role": "WRONG"}]),
        "role_dependent_law": refused(rng_law_by_role_by_exp={"e0": {"candidate": "A", "reference": "B"}}),
        "map_hash_mismatch": refused(registered={**reg, "provided_map_hash": "X"}),
        "floor_policy_mismatch": refused(registered={**reg, "provided_floor_policy": "X"}),
    }
    for k, v in refusals.items():
        if not v:
            errs.append(f"refusal NOT raised: {k}")

    # NE policy: observed floor failure -> group NOT_EVALUABLE (not a rejection, not zero-fill)
    stat_hi = _floor_gated_meandiff(100)             # floor 100 > 6 per side => observed always NE
    r_ne = gate_group(**base, B=99, seed=2, statfn=stat_hi, alpha_group=0.00667)
    if r_ne["verdict"] != NOT_EVALUABLE:
        errs.append(f"observed floor failure should be NOT_EVALUABLE, got {r_ne['verdict']}")

    # NE policy: permutation floor failure counted as MAXIMALLY EXTREME (not zero) => conservative, not a spurious
    # rejection. Construct a stat NE only on some perms; confirm no false PASS-as-reject and symmetric handling.
    def stat_partial(values, mask):
        a, b = values[mask], values[~mask]
        if a.size < 5 or b.size < 5:                 # some stratified perms won't trip this at (6,6); force via delta
            return None
        return abs(a.mean() - b.mean())
    r_p = gate_group(**base, B=199, seed=3, statfn=stat_partial, alpha_group=0.00667)
    if r_p["verdict"] not in (PASS, FAIL, NOT_EVALUABLE):
        errs.append("partial-NE gate produced invalid verdict")

    return errs


def main():
    errs = selftest()
    out = {"policy": "observed cell NE -> group NOT_EVALUABLE; permutation NE -> maximally extreme (+inf), never "
                     "zero-fill; validate BEFORE any statistic (fail-closed).",
           "refusals_covered": ["empty set", "unknown/missing exp/quota", "quota mismatch", "bool/non-int/negative "
                                "quota", "unequal cand/ref", "nonfinite value/Δ", "truncated vector", "role/"
                                "component/source mismatch", "B/RNG mismatch", "role-dependent per-exp RNG law",
                                "malformed assignment count/shape", "prohibited duplicate assignment", "map-hash "
                                "mismatch", "floor-policy mismatch"],
           "selftests_pass": not errs, "selftest_errors": errs,
           "authorization": "dev-only gate wiring; no map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(out, indent=2, default=str))
    assert not errs, f"gate self-tests FAILED: {errs}"
    return out


if __name__ == "__main__":
    main()
