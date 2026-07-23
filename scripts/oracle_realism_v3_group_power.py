#!/usr/bin/env python3
"""Oracle realism v3 — EXACT-GROUP product/min-p power demonstration (Pi rev-5 #4).

The rev-5 group demo was a 3-cell/1-experiment MECHANICAL SMOKE. This exercises the EXACT registry burst-timing
group: the four burst-timing checks {S3_tau, S3_loggap, delta_t_zero_abs, positive_gap_ks} across all NINE full
SD experiments = 36 cells, product-permuted (independent per-experiment within-stratum permutation under one
synchronized MC index), at a B meeting the group resolution `B >= K_g/alpha_group`. It uses REAL estimators
(pooled phase-spanning tau-b; the CORRECTED frozen-map S3_loggap via the registered estimand; delta-t-zero; a
TIED-KS on positive gaps evaluated at UNIQUE support values per Pi #4), the frozen NOT_EVALUABLE policy (no
zero-fill), and reports:

  (A) that the exact 36-cell group EVALUATES and its observed null verdict;
  (B) POWER after K=36 nested min-p multiplicity — a single burst-timing cell's candidate arm perturbed at 0.5 is
      still detected at alpha_group — with a Wilson confidence interval (NOT an empirical size claim at 0.00667);
  (C) the BOUNDED group under BOTH exemption variants (with / without the provisional S3 cells).

Development-only, aggregate-hashed. A dev floor + dev N are LABELLED (registered floor 500 at N=8000); exact size
control comes from randomization theory + the exhaustive tests, not this run. NO map draw / calibration / eval seed
/ policy. Run: PYTHONPATH=<repo> python3 scripts/oracle_realism_v3_group_power.py
"""
from __future__ import annotations

import hashlib
import json
from math import sqrt

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2_fixture import sample_fixture
from clinical_jepa.eval.oracle_realism_v2_verifier_design import PROFILES
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_verifier import (
    _positive_gaps_and_prev_size, _bin_index, CLUSTER_BINS, coarsen_reference, _TAU, _KS, _DT0, _LOGGAP,
)
from scripts.oracle_realism_v3_randomization import cell_upper_p, _canonical_mask, _perm_mask
from scripts.oracle_realism_v3_phase0_pilot import _seq_components
import scripts.oracle_realism_v3_registry as REG

DEV_NS = "v3-grouppower-dev"
N = 1500                        # dev per-side sample (LABELLED dev scale; registered N=8000)
DEV_FLOOR = 90                  # LABELLED dev conditional floor (registered ORACLE_ENV_MIN_DENOM=500 at N=8000)

# vectorized cluster-size -> bin lookup (avoids per-item python _bin_index in the S3_loggap precompute)
_MAXSZ = 200
_CLUSTER_LUT = np.array([(_bin_index(s, CLUSTER_BINS) if _bin_index(s, CLUSTER_BINS) is not None else -1)
                         for s in range(_MAXSZ)], int)
ALPHA_SD, G = 0.04, 6
ALPHA_GROUP = ALPHA_SD / G
BURST = ["S3_tau", "S3_loggap", "delta_t_zero_abs", "positive_gap_ks"]
DELTA = {"S3_tau": _TAU, "S3_loggap": _LOGGAP, "delta_t_zero_abs": _DT0, "positive_gap_ks": _KS}


def dseed(*p):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (DEV_NS, *p))).encode()).digest()[:6], "big")


def _draw(profile, tag):
    sk = "SCID" if "scid" in profile else "MIMIC"
    return sample_fixture(sk, PROFILES[profile], N, seed=dseed(profile, tag))


def draw_experiment(exp_id, cond, src, comp, trial, perturb_cell=None):
    """(cand, ref) for one full SD experiment. Repeatability: both arms carry `comp`@0.5 (same-distribution).
    `perturb_cell`=(exp_id) injects an EXTRA burst_timing@0.5 coupling into THIS experiment's candidate only
    (a planted single-cell alternative for the power demonstration)."""
    ref = _draw(src, ("ref", trial)); cand = _draw(src, ("cand", trial))
    if cond == "repeatability":
        ref = apply_coupling(list(ref), comp, 0.5, seed=dseed("cpl_ref", exp_id, trial))
        cand = apply_coupling(list(cand), comp, 0.5, seed=dseed("cpl_cand", exp_id, trial))
    if perturb_cell == exp_id:
        cand = apply_coupling(list(cand), "burst_timing", 0.5, seed=dseed("perturb", exp_id, trial))
    return cand, ref


# --- REAL estimators via O(N) precompute ----------------------------------------------------------
def _tau_pre(pool):
    return np.array([_seq_components(r) for r in pool])


def _d_tau(pre, mask):
    def t(C):
        s = C.sum(0); dA, dB = s[1] - s[2], s[1] - s[3]
        return None if (dA <= 0 or dB <= 0) else s[0] / np.sqrt(dA * dB)
    a, b = t(pre[mask]), t(pre[~mask])
    return None if (a is None or b is None) else abs(a - b)


def _dt0_pre(pool):
    nz = np.array([max(0, r.L_total - r.K) for r in pool], float)
    na = np.array([max(0, r.L_total - 1) for r in pool], float)
    return np.stack([nz, na], 1)


def _d_dt0(pre, mask):
    a, b = pre[mask].sum(0), pre[~mask].sum(0)
    return None if (a[1] == 0 or b[1] == 0) else abs(a[0] / a[1] - b[0] / b[1])


def _gap_pre(pool):
    gaps, owner = [], []
    for i, r in enumerate(pool):
        y, _ = _positive_gaps_and_prev_size(r)
        for v in y:
            if v > 0:
                gaps.append(float(v)); owner.append(i)
    gaps = np.asarray(gaps); owner = np.asarray(owner, int)
    uniq, inv = np.unique(gaps, return_inverse=True)      # tied-KS: evaluate ECDF at UNIQUE support values (Pi #4)
    return {"owner": owner, "inv": inv, "nu": len(uniq)}


def _d_gap_ks(pre, mask):
    inA = mask[pre["owner"]]; nA = int(inA.sum()); nB = len(inA) - nA
    if nA < DEV_FLOOR or nB < DEV_FLOOR:
        return None
    ca = np.bincount(pre["inv"][inA], minlength=pre["nu"]); cb = np.bincount(pre["inv"][~inA], minlength=pre["nu"])
    return float(np.max(np.abs(np.cumsum(ca) / nA - np.cumsum(cb) / nB)))


def _loggap_pre(pool):
    nb = len(CLUSTER_BINS); nseq = len(pool)
    sm = [np.full(nseq, np.nan) for _ in range(nb)]; sp = [np.zeros(nseq) for _ in range(nb)]
    for i, r in enumerate(pool):
        g, ps = _positive_gaps_and_prev_size(r)
        if g.shape[0] == 0:
            continue
        lg = np.log(g)
        bins = _CLUSTER_LUT[np.clip(ps.astype(int), 0, _MAXSZ - 1)]     # vectorized bin assignment
        for b in range(nb):
            m = bins == b
            if m.any():
                sm[b][i] = float(lg[m].mean()); sp[b][i] = int(m.sum())
    return {"sm": sm, "sp": sp}


def _frozen_loggap_groups(reference_pool):
    """Reference-owned frozen grouping of ORIGINAL CLUSTER_BINS at the dev floor (registered estimand, Pi #1)."""
    pre = _loggap_pre(reference_pool)
    counts = np.asarray([int(np.sum(~np.isnan(pre["sm"][b]))) for b in range(len(CLUSTER_BINS))])
    return coarsen_reference(counts, floor=DEV_FLOOR)


def _d_loggap(pre, mask, groups):
    if groups is None:
        return None
    d = 0.0
    for grp in groups:
        cv, rv, cp, rp = [], [], 0.0, 0.0
        for b in grp:
            sm, sp = pre["sm"][b], pre["sp"][b]
            pres = ~np.isnan(sm)
            cv.append(sm[pres & mask]); rv.append(sm[pres & ~mask])
            cp += sp[pres & mask].sum(); rp += sp[pres & ~mask].sum()
        cvv, rvv = np.concatenate(cv), np.concatenate(rv)
        if cvv.size < DEV_FLOOR or rvv.size < DEV_FLOOR or cp < DEV_FLOOR or rp < DEV_FLOOR:
            return None
        d = max(d, abs(cvv.mean() - rvv.mean()))
    return d


def _build_cell(check, cand, ref, ref_for_map=None):
    pool = list(cand) + list(ref)
    if check == "S3_tau":
        pre = _tau_pre(pool); fn = _d_tau
    elif check == "delta_t_zero_abs":
        pre = _dt0_pre(pool); fn = _d_dt0
    elif check == "positive_gap_ks":
        pre = _gap_pre(pool); fn = _d_gap_ks
    else:  # S3_loggap with the reference-owned frozen grouping
        groups = _frozen_loggap_groups(ref_for_map if ref_for_map is not None else ref)
        pre = _loggap_pre(pool); fn = lambda p, m: _d_loggap(p, m, groups)
    return {"check": check, "n_cand": len(cand), "pre": pre, "fn": fn, "delta": DELTA[check]}


def _e_vec(cell, masks):
    e = np.empty(len(masks))
    for j, m in enumerate(masks):
        d = cell["fn"](cell["pre"], m)
        e[j] = np.inf if d is None else max(0.0, d - cell["delta"])   # NE policy: perm undefined -> maximally extreme
    return e


def group_gate_ne(cells, experiments, B, seed):
    """Product/stratified min-p permutation gate with the frozen NE policy. `experiments`: {exp: (nA,nB)}.
    observed NE on ANY cell -> group NOT_EVALUABLE; else nested ranks -> p_g -> PASS/FAIL."""
    rng = np.random.default_rng(seed)
    masks = {}
    for e, (nA, nB) in sorted(experiments.items()):
        strata = [(nA, nB)]
        ms = [_canonical_mask(strata)]
        for _ in range(B):
            ms.append(_perm_mask(rng, strata))
        masks[e] = ms
    for c in cells:                                          # observed floor failure -> group NE
        if c["fn"](c["pre"], masks[c["exp"]][0]) is None:
            return {"verdict": "NOT_EVALUABLE", "p_g": None, "reason": f"observed NE at {c['exp']}/{c['check']}"}
    E = [_e_vec(c, masks[c["exp"]]) for c in cells]
    P = np.stack([cell_upper_p(e) for e in E], 0); S = P.min(0)
    p_g = float((S <= S[0]).sum() / len(S))
    return {"verdict": "PASS" if p_g > ALPHA_GROUP else "FAIL", "p_g": p_g}


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def build_burst_group(trial, perturb_cell=None):
    """The exact 36-cell burst-timing group across the 9 full experiments, product strata."""
    full = [e for e in REG.SD_EXPERIMENTS if e[4] == "full"]
    cells, experiments = [], {}
    for exp_id, cond, src, comp, _ in full:
        cand, ref = draw_experiment(exp_id, cond, src, comp, trial, perturb_cell)
        experiments[exp_id] = (len(cand), len(ref))
        for chk in BURST:
            cell = _build_cell(chk, cand, ref)
            cell["exp"] = exp_id
            cells.append(cell)
    return cells, experiments


def main():
    K_g = len(BURST) * sum(1 for e in REG.SD_EXPERIMENTS if e[4] == "full")   # 36
    B = int(np.ceil(K_g / ALPHA_GROUP / 100.0)) * 100                          # >= K_g/alpha_group (resolution floor)

    # (A) exact group evaluates + observed null verdict
    cells, experiments = build_burst_group(trial=0)
    null_run = group_gate_ne(cells, experiments, B, seed=dseed("null", 0))

    # (B) POWER: perturb ONE experiment's candidate (single-cell alternative) -> detect at alpha_group, Wilson CI
    T = 8
    hits, evald = 0, 0
    for t in range(T):
        cells_t, exps_t = build_burst_group(trial=t, perturb_cell="null_mimic")
        r = group_gate_ne(cells_t, exps_t, B, seed=dseed("pow", t))
        if r["verdict"] != "NOT_EVALUABLE":
            evald += 1
            hits += (r["verdict"] == "FAIL")
    power = round(hits / evald, 3) if evald else None

    # (C) both exemption variants (registry group sizes; the exemption only moves the BOUNDED group)
    variants = {}
    for name, u in (("with_exemption", True), ("without_exemption", False)):
        sd = REG.build_sd_cells(apply_uncalibratable_exemption=u)
        groups = REG.build_groups(sd)
        variants[name] = {"M0": len([c for c in sd if c["scope"] == "in"]),
                          "G_bounded_support": len(groups["G_bounded_support"]["cells"])}

    agg = {"dev_namespace": DEV_NS, "N": N, "dev_floor": DEV_FLOOR,
           "note_scale": f"dev floor={DEV_FLOOR}/N={N} are LABELLED dev scale (registered floor 500 at N=8000). "
                         "Exact SD size comes from randomization theory + exhaustive tests, NOT this run.",
           "group": "burst_timing (exact registry group)", "K_g": K_g, "B_resolution_min": int(np.ceil(K_g / ALPHA_GROUP)),
           "B": B, "alpha_group": round(ALPHA_GROUP, 6), "n_experiments": len(experiments),
           "A_exact_group_evaluates": {"verdict": null_run["verdict"], "p_g": null_run.get("p_g")},
           "B_power_single_cell_perturbed": {"T": T, "evaluated": evald, "detections": hits, "power": power,
                                             "wilson95": _wilson(hits, evald), "perturbed_experiment": "null_mimic"},
           "C_both_exemption_variants": variants,
           "estimators": "pooled phase-spanning tau-b; CORRECTED frozen-map S3_loggap (registered estimand); "
                         "delta-t-zero; tied-KS on unique support values. NE policy: observed NE->group NE, "
                         "perm NE->maximally extreme (no zero-fill).",
           "authorization": "dev-only; no map draw, no calibration/eval seed, no policy, no launch."}
    print(json.dumps(agg, indent=2, default=str))
    print("\nAGGREGATE_HASH:", canonical_hash(agg))
    return agg


if __name__ == "__main__":
    main()
