#!/usr/bin/env python3
"""Oracle realism v3 — SD conditional-randomization p-value algorithm + EXHAUSTIVE validation (Pi rev-3 §1).

The SD gate's group p-value has OPPOSITE tail directions at the cell and group levels:
  - per cell, upper tail:  e_c large => extreme;
  - group min-p, lower tail: S_g = min_c p_c small => extreme.
This module implements the exact conservative randomization p-value and validates the direction / nested-rank /
tie handling by EXHAUSTIVE enumeration over ALL balanced candidate/reference label assignments (no Monte Carlo) —
proving finite-sample exactness (P[reject | null] <= alpha) and correct behaviour under a planted effect.

Run:  PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_randomization.py
"""
from __future__ import annotations

import itertools
import numpy as np


# --------------------------------------------------------------------------------------------------
# exact p-value construction (all A assignments treated symmetrically; each includes itself)
# --------------------------------------------------------------------------------------------------
def cell_upper_p(e):
    """e: (A,) discrepancies over all A assignments (index 0 = observed). Upper-tail p per assignment:
    p^(j) = #{b : e[b] >= e[j]} / A   (includes self => conservative (1 + #{others>=})/A)."""
    e = np.asarray(e, float)
    ge = (e[:, None] <= e[None, :])            # ge[j,b] = e[b] >= e[j]
    return ge.sum(1) / len(e)


def group_p(cell_e_list):
    """cell_e_list: list over cells, each (A,) discrepancies (SYNCHRONIZED assignments; index 0 = observed).
    Returns (p_g_obs, S_all) where S_g^(j)=min_c p_c^(j) and the group p (lower tail):
    p_g = #{j : S[j] <= S[obs]} / A."""
    P = np.stack([cell_upper_p(e) for e in cell_e_list], 0)   # (ncells, A)
    S = P.min(0)                                              # (A,) group min-p per assignment
    p_g_obs = float((S <= S[0]).sum() / len(S))               # lower tail; obs = index 0
    return p_g_obs, S


# --------------------------------------------------------------------------------------------------
# a tiny two-sample world: nA + nB scalar items; a "cell" = |mean_A - mean_B| (deadband Delta applied first)
# --------------------------------------------------------------------------------------------------
def all_balanced_assignments(nA, nB):
    """All C(nA+nB, nA) label assignments; assignment = frozenset of indices assigned to group A."""
    n = nA + nB
    return [set(c) for c in itertools.combinations(range(n), nA)]


def cell_e_over_assignments(values, assignments, nA, delta=0.0):
    """e_c^(j) = (|mean_A - mean_B| - delta)_+ for each label assignment j."""
    v = np.asarray(values, float)
    out = []
    for A in assignments:
        mask = np.zeros(len(v), bool); mask[list(A)] = True
        d = abs(v[mask].mean() - v[~mask].mean())
        out.append(max(0.0, d - delta))
    return np.asarray(out)


# ==================================================================================================
# the ACTUAL evaluator path (Pi rev-4 defect #7): product/stratified FINITE-B Monte-Carlo group test,
# with within-stratum permutation, product permutation across independent experiments, nested cell/group
# ranks, deadbands + ties, deterministic replay, and malformed-input REFUSAL.
# ==================================================================================================
class RefusalError(ValueError):
    """Raised when the permutation spec violates an exchangeability / integrity precondition (fail-closed)."""


def _canonical_mask(strata):
    """Observed split: first n_candidate items of each stratum are candidate."""
    parts = []
    for nA, nB in strata:
        parts.append(np.array([True] * nA + [False] * nB))
    return np.concatenate(parts)


def _perm_mask(rng, strata):
    """WITHIN-stratum label permutation preserving each stratum's (n_candidate, n_reference) quota."""
    parts = []
    for nA, nB in strata:
        idx = rng.permutation(nA + nB)
        m = np.zeros(nA + nB, bool); m[idx[:nA]] = True
        parts.append(m)
    return np.concatenate(parts)


def _exp_total(strata):
    return sum(nA + nB for nA, nB in strata)


def validate_group_spec(cells, experiments, *, registered_quota, declared_B, used_B,
                        rng_law_by_role, item_index_by_exp=None):
    """Fail-closed refusal of the seven malformed-input classes (Pi #7). Raises RefusalError."""
    for e, strata in experiments.items():
        rq = registered_quota[e]
        # (2) wrong stratum quota
        if [tuple(s) for s in strata] != [tuple(s) for s in rq]:
            raise RefusalError(f"wrong stratum quota for {e}: {strata} != registered {rq}")
        for nA, nB in strata:
            # (1) unequal candidate/reference size (SD design is balanced within stratum)
            if nA != nB:
                raise RefusalError(f"unequal candidate/reference size in {e}: nA={nA} nB={nB}")
        # (3)(4) duplicate / missing pooled index (must be a bijection onto 0..M-1)
        if item_index_by_exp is not None:
            idx = list(item_index_by_exp[e]); M = _exp_total(strata)
            if len(idx) != len(set(idx)):
                raise RefusalError(f"duplicate pooled index in {e}")
            if sorted(idx) != list(range(M)):
                raise RefusalError(f"missing/non-bijective pooled index in {e}: not a permutation of 0..{M-1}")
    # (6) B / RNG mismatch (replay integrity)
    if declared_B != used_B:
        raise RefusalError(f"B mismatch: declared {declared_B} != used {used_B}")
    # (5) role-dependent coupling/RNG law (candidate and reference must share the law under H0)
    laws = set(rng_law_by_role.values())
    if len(laws) != 1:
        raise RefusalError(f"role-dependent law: {rng_law_by_role}")
    # (7) truncated cell vector
    for c in cells:
        need = _exp_total(experiments[c["exp_id"]])
        if len(c["values"]) != need:
            raise RefusalError(f"truncated cell vector {c.get('cell_id','?')}: {len(c['values'])} != {need}")


def _cell_e_for_masks(values, masks, delta):
    v = np.asarray(values, float)
    out = np.empty(len(masks))
    for j, m in enumerate(masks):
        out[j] = max(0.0, abs(v[m].mean() - v[~m].mean()) - delta)
    return out


def group_p_mc(cells, experiments, B, seed, *, assignments=None):
    """Finite-B product/stratified group p-value. Index 0 = observed (canonical split); 1..B = product
    permutations (each experiment permuted INDEPENDENTLY within its strata under one synchronized MC index).
    Nested: cell upper-tail p_c, group S_g = min_c p_c, group lower-tail p_g. Deterministic in `seed`."""
    exp_ids = sorted(experiments)
    if assignments is None:
        rng = np.random.default_rng(seed)
        assignments = {e: [_canonical_mask(experiments[e])] for e in exp_ids}
        for _ in range(B):
            for e in exp_ids:                                  # independent per experiment, shared index
                assignments[e].append(_perm_mask(rng, experiments[e]))
    E = [_cell_e_for_masks(c["values"], assignments[c["exp_id"]], c["delta"]) for c in cells]
    P = np.stack([cell_upper_p(e) for e in E], 0)              # (ncells, B+1) upper-tail cell ranks
    S = P.min(0)                                              # group min-p per assignment
    p_g = float((S <= S[0]).sum() / len(S))                   # lower tail; observed = index 0
    return p_g, S


# --------------------------------------------------------------------------------------------------
# exhaustive validation
# --------------------------------------------------------------------------------------------------
def _exact_null_rejection(nA, nB, ncells, seed, deltas, alphas):
    """Under the exact randomization null (values exchangeable), every balanced assignment is equally likely.
    Treat EACH assignment as 'observed' (index 0) with the full set as reference; verify P[p_g <= alpha] <= alpha."""
    rng = np.random.default_rng(seed)
    asg = all_balanced_assignments(nA, nB)
    A = len(asg)
    # exchangeable null values per cell (shared item set; ncells independent scalars per item)
    vals = [rng.normal(size=nA + nB) for _ in range(ncells)]
    e_full = [cell_e_over_assignments(vals[c], asg, nA, deltas[c]) for c in range(ncells)]  # (A,) each
    reject = {a: 0 for a in alphas}
    for j in range(A):                         # each assignment as the observed
        order = [j] + [k for k in range(A) if k != j]     # put observed first (index 0)
        cell_e_list = [e[order] for e in e_full]
        pg, _ = group_p(cell_e_list)
        for a in alphas:
            if pg <= a:
                reject[a] += 1
    return {a: reject[a] / A for a in alphas}, A


def _planted_effect_detected(nA, nB, reps=25):
    """A clearly-separated cell yields a small group p (near 1/A); a matched cell's p is ~Uniform (mean ~0.5).
    Averaged over reps to avoid asserting on a single random matched draw."""
    asg = all_balanced_assignments(nA, nB)
    obs = set(range(nA)); order = [asg.index(obs)] + [k for k in range(len(asg)) if k != asg.index(obs)]
    sep_ps, mat_ps = [], []
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        sep = np.concatenate([rng.normal(5, 0.3, nA), rng.normal(-5, 0.3, nB)])   # strong A/B separation
        mat = rng.normal(size=nA + nB)                                            # matched
        sep_ps.append(group_p([cell_e_over_assignments(sep, asg, nA, 0.0)[order]])[0])
        mat_ps.append(group_p([cell_e_over_assignments(mat, asg, nA, 0.0)[order]])[0])
    return float(np.mean(sep_ps)), float(np.mean(mat_ps)), 1.0 / len(asg)


def main():
    alphas = [0.05, 0.1, 0.2, 0.5]
    print("=== EXACT-NULL rejection rate P[p_g <= alpha] must be <= alpha (finite-sample exactness) ===")
    ok = True
    for (nA, nB, ncells) in [(4, 4, 1), (4, 4, 3), (5, 4, 2), (3, 5, 2)]:
        deltas = [0.0] * ncells
        rates, A = _exact_null_rejection(nA, nB, ncells, seed=7, deltas=deltas, alphas=alphas)
        line = {a: round(rates[a], 4) for a in alphas}
        valid = all(rates[a] <= a + 1e-9 for a in alphas)
        ok = ok and valid
        print(f"  nA={nA} nB={nB} cells={ncells} (A={A} assignments): {line}  exact<=alpha: {valid}")

    print("\n=== DIRECTION: planted effect => small group p (~1/A); matched => ~Uniform (mean ~0.5) ===")
    pg_sep, pg_mat, floor = _planted_effect_detected(5, 5, reps=25)
    print(f"  mean separated p_g={pg_sep:.4f} (min possible {floor:.4f}); mean matched p_g={pg_mat:.4f}")
    dir_ok = pg_sep <= 2 * floor and pg_mat >= 0.3
    ok = ok and dir_ok
    print(f"  direction correct: {dir_ok}")

    print("\n=== TIE handling: exchangeable ties still exact ===")
    rng = np.random.default_rng(11)
    asg = all_balanced_assignments(4, 4)
    vals = rng.integers(0, 3, 8).astype(float)      # heavy ties
    e = cell_e_over_assignments(vals, asg, 4, 0.0)
    rej = 0
    for j in range(len(asg)):
        order = [j] + [k for k in range(len(asg)) if k != j]
        pg, _ = group_p([e[order]])
        rej += pg <= 0.2
    tie_ok = rej / len(asg) <= 0.2 + 1e-9
    ok = ok and tie_ok
    print(f"  tie exact P[p<=0.2]={rej/len(asg):.4f} <= 0.2: {tie_ok}")

    # ==============================================================================================
    # the ACTUAL evaluator path: product/stratified finite-B MC + refusals (Pi rev-4 #7)
    # ==============================================================================================
    print("\n=== MC-vs-EXHAUSTIVE: finite-B engine with FULL enumeration reproduces the exhaustive p_g ===")
    rng = np.random.default_rng(3)
    values = rng.normal(size=8) + np.array([0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4])   # mild planted split
    asg_sets = all_balanced_assignments(4, 4)                                   # canonical {0,1,2,3} is first
    exhaustive_pg, _ = group_p([cell_e_over_assignments(values, asg_sets, 4, 0.02)])
    masks = [np.isin(np.arange(8), list(s)) for s in asg_sets]
    mc_pg, _ = group_p_mc([{"cell_id": "c", "exp_id": "e0", "values": values, "delta": 0.02}],
                          {"e0": [(4, 4)]}, B=len(masks) - 1, seed=0, assignments={"e0": masks})
    eq_ok = abs(exhaustive_pg - mc_pg) < 1e-12
    ok = ok and eq_ok
    print(f"  exhaustive p_g={exhaustive_pg:.6f}  MC(full-enum) p_g={mc_pg:.6f}  match: {eq_ok}")

    print("\n=== STRATIFIED quota preservation: every within-stratum permutation keeps (nA,nB) exactly ===")
    strata = [(3, 3), (2, 2), (4, 4)]
    rng = np.random.default_rng(5)
    quota_ok = True
    for _ in range(200):
        m = _perm_mask(rng, strata); off = 0
        for nA, nB in strata:
            if int(m[off:off + nA + nB].sum()) != nA:
                quota_ok = False
            off += nA + nB
    ok = ok and quota_ok
    print(f"  per-stratum candidate count preserved over 200 draws: {quota_ok}")

    print("\n=== PRODUCT + finite-B CONSERVATIVE SIZE: 2 independent experiments, null => P[p_g<=a] <= a ===")
    experiments = {"e0": [(3, 3)], "e1": [(3, 3)]}
    T, Bfin, a = 400, 199, 0.1
    rej = 0
    for s in range(T):
        rg = np.random.default_rng(20000 + s)
        cells = [{"cell_id": "c0", "exp_id": "e0", "values": rg.normal(size=6), "delta": 0.0},
                 {"cell_id": "c1", "exp_id": "e1", "values": rg.normal(size=6), "delta": 0.0}]
        pg, _ = group_p_mc(cells, experiments, B=Bfin, seed=70000 + s)
        rej += pg <= a
    size = rej / T
    size_ok = size <= a + 0.045                                                 # MC noise margin (valid test)
    ok = ok and size_ok
    print(f"  finite-B (B={Bfin}) product null size P[p_g<={a}]={size:.4f} <= {a} (+margin): {size_ok}")

    print("\n=== DETERMINISTIC REPLAY: same seed => identical p_g and S ===")
    cells = [{"cell_id": "c0", "exp_id": "e0", "values": np.random.default_rng(9).normal(size=6), "delta": 0.0},
             {"cell_id": "c1", "exp_id": "e1", "values": np.random.default_rng(10).normal(size=6), "delta": 0.0}]
    p1, S1 = group_p_mc(cells, experiments, B=300, seed=123)
    p2, S2 = group_p_mc(cells, experiments, B=300, seed=123)
    replay_ok = (p1 == p2) and np.array_equal(S1, S2)
    ok = ok and replay_ok
    print(f"  replay identical: {replay_ok}")

    print("\n=== TIE handling under finite-B: heavy integer ties still conservative ===")
    rej = 0
    for s in range(400):
        rg = np.random.default_rng(30000 + s)
        cells = [{"cell_id": "c0", "exp_id": "e0", "values": rg.integers(0, 3, 6).astype(float), "delta": 0.0}]
        pg, _ = group_p_mc(cells, {"e0": [(3, 3)]}, B=199, seed=40000 + s)
        rej += pg <= 0.2
    tie_mc_ok = rej / 400 <= 0.2 + 0.05
    ok = ok and tie_mc_ok
    print(f"  finite-B tie size P[p_g<=0.2]={rej/400:.4f} <= 0.2 (+margin): {tie_mc_ok}")

    print("\n=== REFUSAL: the seven malformed-input classes each fail closed ===")
    base = dict(experiments={"e0": [(3, 3)]}, registered_quota={"e0": [(3, 3)]},
                cells=[{"cell_id": "c0", "exp_id": "e0", "values": [0.] * 6, "delta": 0.0}],
                declared_B=100, used_B=100, rng_law_by_role={"candidate": "L1", "reference": "L1"},
                item_index_by_exp={"e0": [0, 1, 2, 3, 4, 5]})
    def _refused(**over):
        spec = {**base, **over}
        try:
            validate_group_spec(spec["cells"], spec["experiments"], registered_quota=spec["registered_quota"],
                                declared_B=spec["declared_B"], used_B=spec["used_B"],
                                rng_law_by_role=spec["rng_law_by_role"],
                                item_index_by_exp=spec["item_index_by_exp"])
            return False
        except RefusalError:
            return True
    refusals = {
        "unequal_cand_ref": _refused(experiments={"e0": [(4, 2)]}, registered_quota={"e0": [(4, 2)]},
                                     cells=[{"cell_id": "c0", "exp_id": "e0", "values": [0.] * 6, "delta": 0.0}]),
        "wrong_stratum_quota": _refused(experiments={"e0": [(2, 2)]}),
        "duplicate_index": _refused(item_index_by_exp={"e0": [0, 1, 2, 2, 4, 5]}),
        "missing_index": _refused(item_index_by_exp={"e0": [0, 1, 2, 3, 4, 9]}),
        "role_dependent_law": _refused(rng_law_by_role={"candidate": "LA", "reference": "LB"}),
        "B_rng_mismatch": _refused(used_B=200),
        "truncated_cell_vector": _refused(cells=[{"cell_id": "c0", "exp_id": "e0", "values": [0.] * 5, "delta": 0.0}]),
    }
    refuse_ok = all(refusals.values())
    ok = ok and refuse_ok
    for k, v in refusals.items():
        print(f"  {k:22s} refused: {v}")
    # a fully-valid spec must NOT refuse
    valid_ok = not _refused()
    ok = ok and valid_ok
    print(f"  valid spec accepted (not refused): {valid_ok}")

    print(f"\nALL CHECKS PASS: {ok}")
    assert ok, "randomization p-value validation FAILED"
    return ok


if __name__ == "__main__":
    main()
