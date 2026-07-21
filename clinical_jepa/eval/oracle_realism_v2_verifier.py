"""Executable realism-v2 verifier — S1–S9 sequence-route statistics (rebuild step 3).

Implements the frozen design (`oracle_realism_v2_verifier_design`, m3a_design_dev_hash 60c85b64) as pure,
fail-closed functions over `SequenceRecord` samples. Every statistic reduces per-sequence first and
equal-weights ELIGIBLE sequences; conditional checks bin on the frozen overflow bins with a REFERENCE-ONLY
coarsening map; every threshold is on CANDIDATE − REFERENCE. Any floor breach after coarsening returns
NOT_EVALUABLE (never zero-filled). S8/S9 are terminal adequacy checks (no D route). Synthetic-only.

This is the sequence route. The six registered marginals keep their exact v1 estimands and live on the
separate AggregateStats route (see `oracle_realism_v2_fixture.reg_*` + `marginal_route_checks` here).
"""
from __future__ import annotations

import dataclasses
from math import log

import numpy as np
from scipy.stats import kendalltau, ks_2samp

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_FROZEN_BINS
from clinical_jepa.eval.oracle_realism_v2_fixture import (
    C, reg_lengths, reg_cluster_counts, reg_class_tv_proportions, reg_occupancy_mean, reg_dt0_pooled,
    reg_positive_gaps,
)
from clinical_jepa.eval.rung2_contract import (
    ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS, ORACLE_ENV_MIN_DENOM,
)

_KS, _TV, _OCC, _DT0 = ORACLE_ENV_KS, ORACLE_ENV_TV, ORACLE_ENV_OCCUPANCY_ABS, ORACLE_ENV_DT0_ABS
_TAU = 0.05
_LOGGAP = log(1.10)
FLOOR = ORACLE_ENV_MIN_DENOM        # 500
MIN_BINS = 3
LENGTH_BINS = V2_FROZEN_BINS["length"]
CLUSTER_BINS = V2_FROZEN_BINS["cluster_size"]

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    status: str                     # PASS | FAIL | NOT_EVALUABLE
    value: float | None             # candidate-reference statistic (None if NOT_EVALUABLE)
    threshold: float
    detail: dict = dataclasses.field(default_factory=dict)


def _bin_index(v: int, bins) -> int | None:
    for i, (lo, hi) in enumerate(bins):
        if v >= lo and (hi is None or v <= hi):
            return i
    return None


def coarsen_reference(ref_counts: np.ndarray, *, floor: int = FLOOR, min_bins: int = MIN_BINS):
    """6-step reference-only coarsening. Returns a list of index-groups (each a list of original bin indices)
    or None (refuse: cannot reach `floor` in every group while keeping >= min_bins bins)."""
    groups = [[i] for i in range(len(ref_counts))]
    while True:
        gc = np.asarray([sum(int(ref_counts[i]) for i in g) for g in groups])
        sparse = [gi for gi, c in enumerate(gc) if c < floor]
        if not sparse:
            return groups
        if len(groups) <= min_bins:
            return None                       # still sparse but cannot merge below min_bins
        j = sparse[-1]                        # highest-index sparse group
        if j > 0:
            groups[j - 1].extend(groups[j]); del groups[j]
        else:
            groups[1] = groups[0] + groups[1]; del groups[0]


def _grouped(per_bin_values: list[list[float]], groups) -> list[np.ndarray]:
    """Collect per-original-bin value lists into coarsened groups (concatenated), equal-weight preserved."""
    out = []
    for g in groups:
        vals: list[float] = []
        for i in g:
            vals.extend(per_bin_values[i])
        out.append(np.asarray(vals, dtype=float))
    return out


def _conditional_maxbin(name, cand_per_bin, ref_per_bin, bins, threshold, *, reducer):
    """Generic conditional check: coarsen on reference sequence-counts, refuse on candidate floor breach, then
    max-group |reducer(cand) - reducer(ref)| vs threshold. `*_per_bin` are lists (len=#bins) of per-sequence
    value arrays."""
    ref_counts = np.asarray([len(v) for v in ref_per_bin])
    groups = coarsen_reference(ref_counts)
    if groups is None:
        return CheckResult(name, NOT_EVALUABLE, None, threshold, {"reason": "reference coarsening refused"})
    cand_g = _grouped(cand_per_bin, groups)
    ref_g = _grouped(ref_per_bin, groups)
    if any(len(v) < FLOOR for v in cand_g) or any(len(v) < FLOOR for v in ref_g):
        return CheckResult(name, NOT_EVALUABLE, None, threshold, {"reason": "floor breach under reference map"})
    diffs = [abs(reducer(c) - reducer(r)) for c, r in zip(cand_g, ref_g)]
    val = float(max(diffs))
    return CheckResult(name, PASS if val <= threshold else FAIL, val, threshold,
                       {"n_groups": len(groups), "per_group": [round(d, 5) for d in diffs]})


# --------------------------------------------------------------------------------------------------
# per-record primitives
# --------------------------------------------------------------------------------------------------
def _runs(rec) -> np.ndarray:
    return np.bincount(rec.cluster_ids).astype(int)         # size of each maximal run


def _positive_gaps_and_prev_size(rec):
    """(gaps, preceding-run-size) for each inter-cluster boundary of a record."""
    if rec.L_total < 2:
        return np.array([]), np.array([])
    runs = _runs(rec)
    # boundary timestamps: first item time of each run
    first_idx = np.concatenate([[0], np.where(np.diff(rec.cluster_ids) == 1)[0] + 1])
    run_times = rec.timestamps[first_idx]
    gaps = np.diff(run_times)
    prev_size = runs[:-1]
    pos = gaps > 0.0
    return gaps[pos], prev_size[pos]


# --------------------------------------------------------------------------------------------------
# S1 — cluster density + tau(L,K)
# --------------------------------------------------------------------------------------------------
def s1(cand, ref) -> dict:
    def per_bin_density(sample):
        pb = [[] for _ in LENGTH_BINS]
        for r in sample:
            b = _bin_index(r.L_total, LENGTH_BINS)
            if b is not None:
                pb[b].append(r.K / r.L_total)
        return pb
    dens = _conditional_maxbin("S1_density", per_bin_density(cand), per_bin_density(ref), LENGTH_BINS,
                               _OCC, reducer=lambda v: float(np.mean(v)))
    # S1_tau: one source-level tau-b(L,K) over independent sequences
    def tau(sample):
        L = np.asarray([r.L_total for r in sample], float); K = np.asarray([r.K for r in sample], float)
        t, _ = kendalltau(L, K)
        return t
    tc, tr = tau(cand), tau(ref)
    if not (np.isfinite(tc) and np.isfinite(tr)):
        tau_res = CheckResult("S1_tau", NOT_EVALUABLE, None, _TAU, {"reason": "undefined tau"})
    else:
        v = abs(float(tc) - float(tr))
        tau_res = CheckResult("S1_tau", PASS if v <= _TAU else FAIL, v, _TAU, {})
    return {"S1_density": dens, "S1_tau": tau_res}


# --------------------------------------------------------------------------------------------------
# S2 — run-size ECDF (sequence-equal), KS
# --------------------------------------------------------------------------------------------------
def s2(cand, ref) -> dict:
    def seq_ecdfs(sample):
        return [np.sort(_runs(r)) for r in sample if r.L_total >= 1]
    ec, er = seq_ecdfs(cand), seq_ecdfs(ref)
    if len(ec) < FLOOR or len(er) < FLOOR:
        return {"S2_ks": CheckResult("S2_ks", NOT_EVALUABLE, None, _KS, {"reason": "sequence floor"})}
    support = np.unique(np.concatenate([np.concatenate(ec), np.concatenate(er)]))

    def meanF(ecdfs, x):
        # mean over sequences of (fraction of that sequence's runs <= x)
        return np.mean([np.searchsorted(s, x, side="right") / s.shape[0] for s in ecdfs])
    Fc = np.asarray([meanF(ec, x) for x in support])
    Fr = np.asarray([meanF(er, x) for x in support])
    v = float(np.max(np.abs(Fc - Fr)))
    return {"S2_ks": CheckResult("S2_ks", PASS if v <= _KS else FAIL, v, _KS,
                                 {"n_clusters_cand": int(sum(s.shape[0] for s in ec))})}


# --------------------------------------------------------------------------------------------------
# S3 — gap by preceding cluster-size bin (loggap + tau)
# --------------------------------------------------------------------------------------------------
def s3(cand, ref) -> dict:
    def per_bin_logmean(sample):
        pb = [[] for _ in CLUSTER_BINS]
        for r in sample:
            g, ps = _positive_gaps_and_prev_size(r)
            if g.shape[0] == 0:
                continue
            lg = np.log(g)
            for b in range(len(CLUSTER_BINS)):
                mask = np.asarray([_bin_index(int(s), CLUSTER_BINS) == b for s in ps])
                if mask.any():
                    pb[b].append(float(np.mean(lg[mask])))
        return pb
    loggap = _conditional_maxbin("S3_loggap", per_bin_logmean(cand), per_bin_logmean(ref), CLUSTER_BINS,
                                 _LOGGAP, reducer=lambda v: float(np.mean(v)))

    def per_seq_tau(sample):
        taus = []
        for r in sample:
            g, ps = _positive_gaps_and_prev_size(r)
            if g.shape[0] >= 2 and np.unique(ps).shape[0] >= 2 and np.unique(g).shape[0] >= 2:
                t, _ = kendalltau(ps, g)
                if np.isfinite(t):
                    taus.append(float(t))
        return np.asarray(taus)
    tc, tr = per_seq_tau(cand), per_seq_tau(ref)
    if tc.shape[0] < FLOOR or tr.shape[0] < FLOOR:
        tau_res = CheckResult("S3_tau", NOT_EVALUABLE, None, _TAU, {"reason": "eligible-sequence floor"})
    else:
        v = abs(float(np.mean(tc)) - float(np.mean(tr)))
        tau_res = CheckResult("S3_tau", PASS if v <= _TAU else FAIL, v, _TAU, {})
    return {"S3_loggap": loggap, "S3_tau": tau_res}


# --------------------------------------------------------------------------------------------------
# S4 — same-class same-cluster vs adjacent-cluster (combinatorics)
# --------------------------------------------------------------------------------------------------
def _s4_contrast(rec):
    runs_idx = rec.cluster_ids
    K = rec.K
    same_pairs = same_same = 0
    for c in range(K):
        cnt = np.bincount(rec.class_ids[runs_idx == c], minlength=C)
        s = int(cnt.sum())
        same_pairs += s * (s - 1) // 2
        same_same += int(np.sum(cnt * (cnt - 1) // 2))
    adj_pairs = adj_same = 0
    for c in range(K - 1):
        a = np.bincount(rec.class_ids[runs_idx == c], minlength=C)
        b = np.bincount(rec.class_ids[runs_idx == c + 1], minlength=C)
        adj_pairs += int(a.sum() * b.sum())
        adj_same += int(np.sum(a * b))
    if same_pairs == 0 or adj_pairs == 0:
        return None
    return (same_same / same_pairs) - (adj_same / adj_pairs)


def s4(cand, ref) -> dict:
    def vals(sample):
        return np.asarray([x for x in (_s4_contrast(r) for r in sample) if x is not None])
    vc, vr = vals(cand), vals(ref)
    if vc.shape[0] < FLOOR or vr.shape[0] < FLOOR:
        return {"S4_abs": CheckResult("S4_abs", NOT_EVALUABLE, None, _OCC, {"reason": "eligible-sequence floor"})}
    v = abs(float(np.mean(vc)) - float(np.mean(vr)))
    return {"S4_abs": CheckResult("S4_abs", PASS if v <= _OCC else FAIL, v, _OCC, {})}


# --------------------------------------------------------------------------------------------------
# S5 — occupancy by length bin
# --------------------------------------------------------------------------------------------------
def s5(cand, ref) -> dict:
    def per_bin_occ(sample):
        pb = [[] for _ in LENGTH_BINS]
        for r in sample:
            b = _bin_index(r.L_total, LENGTH_BINS)
            if b is not None:
                pb[b].append(len(np.unique(r.class_ids)) / C)
        return pb
    return {"S5_abs": _conditional_maxbin("S5_abs", per_bin_occ(cand), per_bin_occ(ref), LENGTH_BINS,
                                          _OCC, reducer=lambda v: float(np.mean(v)))}


# --------------------------------------------------------------------------------------------------
# S6 — length-dependent class mix (TV)
# --------------------------------------------------------------------------------------------------
def s6(cand, ref) -> dict:
    def per_bin_vecs(sample):
        pb = [[] for _ in LENGTH_BINS]
        for r in sample:
            b = _bin_index(r.L_total, LENGTH_BINS)
            if b is not None:
                pb[b].append(np.bincount(r.class_ids, minlength=C) / r.L_total)
        return pb
    cand_pb, ref_pb = per_bin_vecs(cand), per_bin_vecs(ref)
    ref_counts = np.asarray([len(v) for v in ref_pb])
    groups = coarsen_reference(ref_counts)
    if groups is None:
        return {"S6_tv": CheckResult("S6_tv", NOT_EVALUABLE, None, _TV, {"reason": "coarsening refused"})}

    def group_meanvec(pb, g):
        vs = [v for i in g for v in pb[i]]
        return np.mean(np.asarray(vs), axis=0), len(vs)
    diffs = []
    for g in groups:
        cv, cn = group_meanvec(cand_pb, g); rv, rn = group_meanvec(ref_pb, g)
        if cn < FLOOR or rn < FLOOR:
            return {"S6_tv": CheckResult("S6_tv", NOT_EVALUABLE, None, _TV, {"reason": "floor breach"})}
        diffs.append(0.5 * float(np.sum(np.abs(cv - rv))))
    v = float(max(diffs))
    return {"S6_tv": CheckResult("S6_tv", PASS if v <= _TV else FAIL, v, _TV, {"per_group": [round(d, 5) for d in diffs]})}


# --------------------------------------------------------------------------------------------------
# S7 — class diversity by cluster-size bin
# --------------------------------------------------------------------------------------------------
def s7(cand, ref) -> dict:
    def per_bin_div(sample):
        pb = [[] for _ in CLUSTER_BINS]
        for r in sample:
            by_bin = [[] for _ in CLUSTER_BINS]
            for c in range(r.K):
                cls = r.class_ids[r.cluster_ids == c]
                b = _bin_index(int(cls.shape[0]), CLUSTER_BINS)
                if b is not None:
                    by_bin[b].append(len(np.unique(cls)) / C)
            for b in range(len(CLUSTER_BINS)):
                if by_bin[b]:
                    pb[b].append(float(np.mean(by_bin[b])))     # within-seq average clusters in bin
        return pb
    return {"S7_abs": _conditional_maxbin("S7_abs", per_bin_div(cand), per_bin_div(ref), CLUSTER_BINS,
                                          _OCC, reducer=lambda v: float(np.mean(v)))}


# --------------------------------------------------------------------------------------------------
# S8 — position nonstationarity (terminal); density + class over 4 fixed quartiles
# --------------------------------------------------------------------------------------------------
def _quartile(pos):
    return np.minimum(3, (pos * 4).astype(int))


def s8(cand, ref) -> dict:
    def per_q(sample):
        dens = [[] for _ in range(4)]; vecs = [[] for _ in range(4)]; items = [0, 0, 0, 0]
        for r in sample:
            q = _quartile(r.positions)
            starts = np.concatenate([[True], np.diff(r.cluster_ids) == 1]) if r.L_total > 1 else np.array([True])
            for qi in range(4):
                m = q == qi
                n = int(m.sum())
                if n > 0:
                    items[qi] += n
                    dens[qi].append(float(starts[m].sum()) / n)
                    vecs[qi].append(np.bincount(r.class_ids[m], minlength=C) / n)
        return dens, vecs, items
    dc, vc, ic = per_q(cand); dr, vr, ir = per_q(ref)
    ddiffs = []; cdiffs = []
    for qi in range(4):
        if (len(dc[qi]) < FLOOR or len(dr[qi]) < FLOOR or ic[qi] < FLOOR or ir[qi] < FLOOR):
            res = CheckResult("S8", NOT_EVALUABLE, None, _OCC, {"reason": f"quartile {qi} floor"})
            return {"S8_density": res, "S8_class": dataclasses.replace(res, name="S8_class", threshold=_TV)}
        ddiffs.append(abs(float(np.mean(dc[qi])) - float(np.mean(dr[qi]))))
        cdiffs.append(0.5 * float(np.sum(np.abs(np.mean(vc[qi], axis=0) - np.mean(vr[qi], axis=0)))))
    dv, cv = float(max(ddiffs)), float(max(cdiffs))
    return {"S8_density": CheckResult("S8_density", PASS if dv <= _OCC else FAIL, dv, _OCC, {"terminal": True}),
            "S8_class": CheckResult("S8_class", PASS if cv <= _TV else FAIL, cv, _TV, {"terminal": True})}


# --------------------------------------------------------------------------------------------------
# S9 — block-seam invisibility (terminal)
# --------------------------------------------------------------------------------------------------
def _seam_mask(L):
    # adjacency i (between item i, i+1) is a seam iff (i+1) % 8 == 0
    idx = np.arange(L - 1)
    return (idx + 1) % 8 == 0


def s9(cand, ref) -> dict:
    def collect(sample):
        zc, cc, sa, na = [], [], 0, 0
        seam_gaps, non_gaps = [], []
        for r in sample:
            if r.L_total < 2:
                continue
            dt = np.diff(r.timestamps)
            same = (r.class_ids[:-1] == r.class_ids[1:])
            seam = _seam_mask(r.L_total); non = ~seam
            if seam.sum() == 0 or non.sum() == 0:
                continue
            sa += int(seam.sum()); na += int(non.sum())
            zc.append(float((dt[seam] == 0).mean()) - float((dt[non] == 0).mean()))
            cc.append(float(same[seam].mean()) - float(same[non].mean()))
            seam_gaps.extend(dt[seam][dt[seam] > 0].tolist()); non_gaps.extend(dt[non][dt[non] > 0].tolist())
        return (np.asarray(zc), np.asarray(cc), sa, na,
                np.asarray(seam_gaps), np.asarray(non_gaps))
    zc, cc, sa, na, sgc, ngc = collect(cand)
    zr, cr, sar, nar, sgr, ngr = collect(ref)
    floors_ok = (min(zc.shape[0], zr.shape[0]) >= FLOOR and min(sa, sar) >= FLOOR and min(na, nar) >= FLOOR
                 and min(sgc.shape[0], sgr.shape[0]) >= FLOOR and min(ngc.shape[0], ngr.shape[0]) >= FLOOR)
    if not floors_ok:
        r0 = CheckResult("S9", NOT_EVALUABLE, None, _OCC, {"reason": "S9 floor (per-sample)"})
        return {"S9_zero": dataclasses.replace(r0, name="S9_zero"),
                "S9_class": dataclasses.replace(r0, name="S9_class"),
                "S9_gap": dataclasses.replace(r0, name="S9_gap", threshold=_KS)}
    zv = abs(float(np.mean(zc)) - float(np.mean(zr)))
    cv = abs(float(np.mean(cc)) - float(np.mean(cr)))
    gap_ok = (ks_2samp(sgc, ngc).statistic <= _KS and ks_2samp(sgr, ngr).statistic <= _KS
              and max(ks_2samp(sgc, sgr).statistic, ks_2samp(ngc, ngr).statistic) <= _KS)
    gv = float(max(ks_2samp(sgc, sgr).statistic, ks_2samp(ngc, ngr).statistic))
    return {"S9_zero": CheckResult("S9_zero", PASS if zv <= _OCC else FAIL, zv, _OCC, {"terminal": True}),
            "S9_class": CheckResult("S9_class", PASS if cv <= _OCC else FAIL, cv, _OCC, {"terminal": True}),
            "S9_gap": CheckResult("S9_gap", PASS if gap_ok else FAIL, gv, _KS, {"terminal": True})}


# --------------------------------------------------------------------------------------------------
# six registered marginals (AggregateStats route) — exact v1 estimands
# --------------------------------------------------------------------------------------------------
def marginal_route_checks(cand, ref) -> dict:
    out = {}
    out["length_ks"] = _ks_check("length_ks", reg_lengths(cand), reg_lengths(ref), _KS)
    out["count_ks"] = _ks_check("count_ks", reg_cluster_counts(cand), reg_cluster_counts(ref), _KS)
    out["positive_gap_ks"] = _ks_check("positive_gap_ks", reg_positive_gaps(cand), reg_positive_gaps(ref), _KS)
    tv = 0.5 * float(np.sum(np.abs(reg_class_tv_proportions(cand) - reg_class_tv_proportions(ref))))
    out["class_tv"] = CheckResult("class_tv", PASS if tv <= _TV else FAIL, tv, _TV, {})
    occ = abs(reg_occupancy_mean(cand) - reg_occupancy_mean(ref))
    out["occupancy_abs"] = CheckResult("occupancy_abs", PASS if occ <= _OCC else FAIL, occ, _OCC, {})
    dc, dr = reg_dt0_pooled(cand), reg_dt0_pooled(ref)
    if not (np.isfinite(dc) and np.isfinite(dr)):
        out["delta_t_zero_abs"] = CheckResult("delta_t_zero_abs", NOT_EVALUABLE, None, _DT0, {"reason": "no adjacencies"})
    else:
        d = abs(float(dc) - float(dr))
        out["delta_t_zero_abs"] = CheckResult("delta_t_zero_abs", PASS if d <= _DT0 else FAIL, d, _DT0, {})
    return out


def _ks_check(name, a, b, thr):
    if a.shape[0] < FLOOR or b.shape[0] < FLOOR:
        return CheckResult(name, NOT_EVALUABLE, None, thr, {"reason": "floor"})
    v = float(ks_2samp(a, b).statistic)
    return CheckResult(name, PASS if v <= thr else FAIL, v, thr, {})


def sequence_route_checks(cand, ref) -> dict:
    out = {}
    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9):
        out.update(fn(cand, ref))
    return out


VERIFIER_IMPL = {
    "name": "realism_v2_verifier_dev",
    "subchecks": ["length_ks", "class_tv", "count_ks", "occupancy_abs", "delta_t_zero_abs", "positive_gap_ks",
                  "S1_density", "S1_tau", "S2_ks", "S3_loggap", "S3_tau", "S4_abs", "S5_abs", "S6_tv", "S7_abs",
                  "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap"],
    "thresholds": {"ks": _KS, "tv": _TV, "abs": _OCC, "dt0": _DT0, "tau": _TAU, "loggap": round(_LOGGAP, 6)},
    "floor": FLOOR, "min_bins": MIN_BINS, "scored_on": "candidate - reference; per-sequence equal-weight",
    "coarsening": "reference-only 6-step; candidate floor breach => NOT_EVALUABLE",
    "terminal_no_D": ["S2_ks", "S8_density", "S8_class", "S9_zero", "S9_class", "S9_gap"],
}


def verifier_impl_identity() -> str:
    return canonical_hash(VERIFIER_IMPL)
