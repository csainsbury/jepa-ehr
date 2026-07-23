#!/usr/bin/env python3
"""Oracle realism v3 — dev-seed GROUP critical-value + POWER demonstration under the joint permutation null.

Pi rev-4 authorized dev-only item 5 (also Pi rev-3 §4: "grouping != power gain until shown"). Demonstrates, on
DEVELOPMENT seeds only (disjoint namespace; NO calibration/audit/evaluation seed, NO map draw), that the grouped
min-p permutation gate:

  (A) has a CALIBRATED critical value — under the same-distribution null, P[p_g <= alpha] <= alpha, so the
      alpha_group-quantile of the group min-p null IS a usable critical value (no transported constants);
  (B) has POWER — with a single burst-timing cell coupled at 0.5, the group rejects at alpha_group far more often
      than alpha_group (grouping does not dissolve a real single-cell effect);
  (C) behaves under a PRODUCT permutation across two independent experiments (synchronized MC index); and
  (D) is reported for BOTH boundary-exemption variants (registry group sizes with/without the provisional S3
      exemption) so an exemption cannot manufacture favourable group size/power.

Uses REAL estimators (pooled phase-spanning tau-b, positive-gap KS, delta-t-zero fraction) recomputed per
permutation via O(N) precompute. Aggregate-hashed. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_group_demo.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_verifier import _positive_gaps_and_prev_size, _TAU, _KS, _DT0
from scripts.oracle_realism_v3_randomization import cell_upper_p
from scripts.oracle_realism_v3_phase0_pilot import _seq_components
import scripts.oracle_realism_v3_registry as REG

DEV_NS = "v3-groupdemo-dev"                      # disjoint dev namespace
DEV_SEEDS = list(range(91000, 91030))            # 30 dev seeds for the power demonstration
N = 3000
ALPHA_SD = 0.04
G = 6                                            # from the exact registry
ALPHA_GROUP = ALPHA_SD / G


def dseed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *parts))).encode()).digest()[:6], "big")


def draw(profile, tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, tag))


# --- REAL per-cell statistics recomputed per permutation via O(N) precompute ------------------------
def tau_pre(seqs):
    return np.array([_seq_components(r) for r in seqs])                    # (n,4): cmd,n0,n1,n2


def _tau(C):
    s = C.sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
    return None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)


def d_tau(pre, mask):
    a, b = _tau(pre[mask]), _tau(pre[~mask])
    return None if (a is None or b is None) else abs(a - b)


def dt0_pre(seqs):
    nz = np.array([max(0, r.L_total - r.K) for r in seqs], float)          # within-cluster (zero-gap) adjacencies
    na = np.array([max(0, r.L_total - 1) for r in seqs], float)           # total adjacencies
    return np.stack([nz, na], 1)


def d_dt0(pre, mask):
    a, b = pre[mask].sum(0), pre[~mask].sum(0)
    return None if (a[1] == 0 or b[1] == 0) else abs(a[0] / a[1] - b[0] / b[1])


def gap_pre(seqs):
    gaps, owner = [], []
    for i, r in enumerate(seqs):
        y, _ = _positive_gaps_and_prev_size(r)
        for v in y:
            if v > 0:
                gaps.append(float(v)); owner.append(i)
    gaps = np.asarray(gaps); owner = np.asarray(owner, int)
    order = np.argsort(gaps, kind="mergesort")
    return gaps[order], owner[order]


def d_gap_ks(pre, mask):
    _, owner = pre
    inA = mask[owner]
    nA = int(inA.sum()); nB = len(inA) - nA
    if nA == 0 or nB == 0:
        return None
    return float(np.max(np.abs(np.cumsum(inA) / nA - np.cumsum(~inA) / nB)))


CELL_STATS = {"S3_tau": (tau_pre, d_tau, _TAU), "delta_t_zero_abs": (dt0_pre, d_dt0, _DT0),
              "positive_gap_ks": (gap_pre, d_gap_ks, _KS)}


# --- group permutation test with REAL sequence-level estimators -------------------------------------
def _masks(rng, nA, nB, B):
    M = nA + nB
    canon = np.zeros(M, bool); canon[:nA] = True
    out = [canon]
    for _ in range(B):
        idx = rng.permutation(M); m = np.zeros(M, bool); m[idx[:nA]] = True
        out.append(m)
    return out


def group_perm_test(cells, experiments, B, seed):
    """cells: list of {exp, check, pre, statfn, delta}. experiments: {exp: (nA,nB)}. Product permutation across
    experiments (independent within experiment) under one synchronized MC index; nested cell/group ranks."""
    rng = np.random.default_rng(seed)
    masks = {e: _masks(rng, nA, nB, B) for e, (nA, nB) in sorted(experiments.items())}
    E = []
    for c in cells:
        ej = np.empty(B + 1)
        for j, m in enumerate(masks[c["exp"]]):
            d = c["statfn"](c["pre"], m)
            ej[j] = 0.0 if d is None else max(0.0, d - c["delta"])
        E.append(ej)
    P = np.stack([cell_upper_p(e) for e in E], 0)
    S = P.min(0)
    return float((S <= S[0]).sum() / len(S)), S


def _build_cells(seqs_by_exp, checks):
    cells, experiments = [], {}
    for exp, (cand, ref) in seqs_by_exp.items():
        experiments[exp] = (len(cand), len(ref))
        pooled = list(cand) + list(ref)
        for ck in checks:
            pre_fn, stat_fn, delta = CELL_STATS[ck]
            cells.append({"exp": exp, "check": ck, "pre": pre_fn(pooled), "statfn": stat_fn, "delta": delta})
    return cells, experiments


def null_calibration(checks, T=30, B=999, alphas=(0.05, ALPHA_GROUP)):
    """(A) same-distribution null: P[p_g <= alpha] <= alpha (calibrated critical value)."""
    rej = {a: 0 for a in alphas}
    for t in range(T):
        cand = draw("mimic_scale_control", ("nc_A", t)); ref = draw("mimic_scale_control", ("nc_B", t))
        cells, experiments = _build_cells({"e0": (cand, ref)}, checks)
        pg, _ = group_perm_test(cells, experiments, B, seed=dseed("nullcal", t))
        for a in alphas:
            rej[a] += pg <= a
    return {f"size@{a}": round(rej[a] / T, 4) for a in alphas} | {"T": T, "B": B,
            "calibrated": all(rej[a] / T <= a + 0.04 for a in alphas)}


def power(checks, coupled="burst_timing", B=999, alpha=ALPHA_GROUP):
    """(B) plant a single burst-timing cell at 0.5: group should reject at alpha_group (grouping keeps power)."""
    hits = 0; n = 0
    for k in DEV_SEEDS:
        cand = draw("mimic_scale_control", ("pw_A", k))
        ref0 = draw("mimic_scale_control", ("pw_B", k))
        ref = apply_coupling(list(ref0), coupled, 0.5, seed=dseed("pw_cpl", k))   # coupled reference
        cells, experiments = _build_cells({"e0": (cand, ref)}, checks)
        pg, _ = group_perm_test(cells, experiments, B, seed=dseed("power", k))
        hits += pg <= alpha; n += 1
    return {"detect_rate@alpha_group": round(hits / n, 3), "alpha_group": round(alpha, 6), "n": n, "B": B}


def product_two_experiment(checks, T=20, B=999, alpha=ALPHA_GROUP):
    """(C) product permutation across two INDEPENDENT experiments (mimic + scid), null => calibrated."""
    rej = 0
    for t in range(T):
        e0 = (draw("mimic_scale_control", ("p_A", t)), draw("mimic_scale_control", ("p_B", t)))
        e1 = (draw("scid_scale_control", ("p_C", t)), draw("scid_scale_control", ("p_D", t)))
        cells, experiments = _build_cells({"e0": e0, "e1": e1}, checks)
        pg, _ = group_perm_test(cells, experiments, B, seed=dseed("prod", t))
        rej += pg <= alpha
    return {f"size@alpha_group": round(rej / T, 4), "alpha_group": round(alpha, 6), "T": T, "B": B,
            "calibrated": rej / T <= alpha + 0.03}


def main():
    checks = ["S3_tau", "delta_t_zero_abs", "positive_gap_ks"]      # real burst-timing routes (fast precompute)
    # registry group sizes for BOTH variants (defect #6 — exemption cannot manufacture size/power)
    variants = {}
    for name, uncal in (("with_provisional_exemption", True), ("without_provisional_exemption", False)):
        groups = REG.build_groups(REG.build_sd_cells(apply_uncalibratable_exemption=uncal))
        variants[name] = {g: len(v["cells"]) for g, v in sorted(groups.items())}

    agg = {
        "dev_namespace": DEV_NS, "N": N, "alpha_sd": ALPHA_SD, "G": G, "alpha_group": round(ALPHA_GROUP, 6),
        "demo_group_checks": checks,
        "A_null_calibration": null_calibration(checks),
        "B_power_single_cell_coupled": power(checks),
        "C_product_two_experiment_null": product_two_experiment(checks),
        "D_registry_group_sizes_both_variants": variants,
        "notes": ("REAL estimators recomputed per permutation (O(N) precompute). Group crit value = the "
                  "alpha_group-quantile of the min-p null on THIS draw (no transported constants). Degenerate "
                  "per-perm d (None) counted as 0 discrepancy (conservative). Dev-only; no map draw/calibration."),
    }
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
