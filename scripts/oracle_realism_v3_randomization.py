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

    print(f"\nALL EXHAUSTIVE CHECKS PASS: {ok}")
    assert ok, "randomization p-value validation FAILED"
    return ok


if __name__ == "__main__":
    main()
