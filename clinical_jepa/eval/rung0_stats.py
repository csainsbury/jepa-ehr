"""Rung-0 paired inference + decision (Pi R5 C5/C7).

Two marginal non-overlapping CIs are NOT a substitute for a paired test (Pi R5 C5):
patients are resampled SYNCHRONOUSLY across channels (coarse/fine) and horizons, and
the paired coarse−fine gap and the paired slope difference are inferred directly. A
predeclared PRACTICAL effect (not merely a detectable sign) is required. The decision
separates from epistemic status (C7): BUILD / NO-BUILD_INCONCLUSIVE /
NO-BUILD_EFFECT-RULED-OUT. Aggregate-only: patient ids stay inside; only counts/CIs out.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

LEVEL_GATE = 0.10          # coarse − fine R@10 >= 0.10 (legacy pre-registered level)


def _r_at_k(ranks: list[int], k: int = 10) -> float:
    return float((np.asarray(ranks, dtype=np.float64) <= k).mean()) if ranks else float("nan")


def _slope(ws: np.ndarray, ys: np.ndarray) -> float:
    """OLS slope of y vs log(W); nan if <2 finite points."""
    m = np.isfinite(ys)
    if m.sum() < 2:
        return float("nan")
    x = np.log(ws[m])
    return float(np.polyfit(x, ys[m], 1)[0])


def paired_bootstrap_gap(records: list[dict[str, Any]], *, k: int = 10, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """records = paired per-query {patient, coarse_rank, fine_rank} at ONE horizon.
    Returns the point coarse−fine R@k gap + a patient-bootstrap paired CI."""
    by_patient: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_patient[r["patient"]].append(r)
    patients = list(by_patient)
    point = _r_at_k([r["coarse_rank"] for r in records], k) - _r_at_k([r["fine_rank"] for r in records], k)
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    P = len(patients)
    for b in range(n_boot):
        samp = rng.integers(0, P, size=P)                       # synchronous patient resample
        recs = [r for i in samp for r in by_patient[patients[i]]]
        gaps[b] = _r_at_k([r["coarse_rank"] for r in recs], k) - _r_at_k([r["fine_rank"] for r in recs], k)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"gap": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_patients": P, "n_queries": len(records)}


def paired_bootstrap_slope(records_by_W: dict[float, list[dict[str, Any]]], *, k: int = 10,
                           n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Synchronous patient resample across ALL horizons; per resample fit R@k vs log W
    per channel and take β_fine − β_coarse. Positive => coarse decays slower per unit
    time (§3.3). Returns point + CI + the implied range-scale widening (Pi R5 C5)."""
    ws = np.array(sorted(records_by_W), dtype=np.float64)
    all_patients = sorted({r["patient"] for W in records_by_W for r in records_by_W[W]})
    idx = {p: i for i, p in enumerate(all_patients)}
    # index records by (patient, W) for fast resampling
    by_pat_W: dict[float, dict[int, list[dict[str, Any]]]] = {W: defaultdict(list) for W in records_by_W}
    for W, recs in records_by_W.items():
        for r in recs:
            by_pat_W[W][idx[r["patient"]]].append(r)

    def _betas(patient_sample: np.ndarray) -> tuple[float, float]:
        # β = DECAY RATE = -(OLS slope of R@k vs log W): positive = R@k drops with W.
        # "coarse decays slower" <=> β_fine > β_coarse <=> β_fine - β_coarse > 0.
        rc, rf = [], []
        for W in ws:
            recs = [r for i in patient_sample for r in by_pat_W[W].get(i, ())]
            rc.append(_r_at_k([r["coarse_rank"] for r in recs], k))
            rf.append(_r_at_k([r["fine_rank"] for r in recs], k))
        return -_slope(ws, np.array(rc)), -_slope(ws, np.array(rf))

    P = len(all_patients)
    bc, bf = _betas(np.arange(P))
    point_diff = bf - bc
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, P, size=P)
        cbc, cbf = _betas(s)
        diffs[b] = cbf - cbc
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    log_range = float(np.log(ws.max() / ws.min())) if len(ws) >= 2 and ws.min() > 0 else 0.0
    return {
        "beta_coarse": bc, "beta_fine": bf, "slope_diff_fine_minus_coarse": point_diff,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "implied_range_widening": point_diff * log_range, "log_range": log_range,
    }


def _by_patient(records: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    by: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["patient"]].append(r)
    return by


def paired_gap_streams(coarse_records: list[dict[str, Any]], fine_records: list[dict[str, Any]],
                       *, k: int = 10, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paired coarse−fine R@k gap when the two channels have DIFFERENT row counts
    (coarse = per-block, fine = per-block×sub-window). Records = [{patient, rank}];
    patients resampled SYNCHRONOUSLY, R@k recomputed per channel over the resample."""
    cby, fby = _by_patient(coarse_records), _by_patient(fine_records)
    patients = sorted(set(cby) | set(fby))
    r10 = lambda recs: _r_at_k([r["rank"] for r in recs], k)  # noqa: E731
    point = r10(coarse_records) - r10(fine_records)
    rng = np.random.default_rng(seed)
    P = len(patients)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.integers(0, P, size=P)
        cr = [r for i in samp for r in cby.get(patients[i], ())]
        fr = [r for i in samp for r in fby.get(patients[i], ())]
        gaps[b] = r10(cr) - r10(fr)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"gap": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_patients": P,
            "n_coarse": len(coarse_records), "n_fine": len(fine_records)}


def paired_slope_streams(coarse_by_W: dict[float, list[dict[str, Any]]],
                         fine_by_W: dict[float, list[dict[str, Any]]],
                         *, k: int = 10, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Stream variant of paired_bootstrap_slope: coarse/fine records per horizon keyed
    by patient; synchronous patient resample; β = decay rate (−slope of R@k vs log W);
    β_fine − β_coarse > 0 ⇒ coarse decays slower per unit time."""
    ws = np.array(sorted(set(coarse_by_W) | set(fine_by_W)), dtype=np.float64)
    patients = sorted({r["patient"] for W in coarse_by_W for r in coarse_by_W[W]}
                      | {r["patient"] for W in fine_by_W for r in fine_by_W[W]})
    ix = {p: i for i, p in enumerate(patients)}

    def _pw(by_W: dict[float, list[dict[str, Any]]]) -> dict[float, dict[int, list[dict[str, Any]]]]:
        out: dict[float, dict[int, list[dict[str, Any]]]] = {}
        for W in ws:
            d: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in by_W.get(W, ()):
                d[ix[r["patient"]]].append(r)
            out[W] = d
        return out

    c_pw, f_pw = _pw(coarse_by_W), _pw(fine_by_W)

    def betas(samp: np.ndarray) -> tuple[float, float]:
        rc, rf = [], []
        for W in ws:
            rc.append(_r_at_k([r["rank"] for i in samp for r in c_pw[W].get(i, ())], k))
            rf.append(_r_at_k([r["rank"] for i in samp for r in f_pw[W].get(i, ())], k))
        return -_slope(ws, np.array(rc)), -_slope(ws, np.array(rf))

    P = len(patients)
    bc, bf = betas(np.arange(P))
    point = bf - bc
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        cbc, cbf = betas(rng.integers(0, P, size=P))
        diffs[b] = cbf - cbc
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    log_range = float(np.log(ws.max() / ws.min())) if len(ws) >= 2 and ws.min() > 0 else 0.0
    return {"beta_coarse": bc, "beta_fine": bf, "slope_diff_fine_minus_coarse": point,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "implied_range_widening": point * log_range, "log_range": log_range}


def decision(*, level_gap: dict[str, Any], coarse_b_gap: dict[str, Any], slope: dict[str, Any],
             raw_count_ok: bool, veto: bool, sufficiency_ok: bool, adequate: bool,
             practical_level: float = LEVEL_GATE, practical_widening: float = 0.05) -> dict[str, Any]:
    """Three-way decision (Pi R5 C7). BUILD requires ALL of: literal level gate AND
    budget-matched coarse_B gate (both paired-CI clear of the practical level) AND
    slope separation (paired CI clear of the practical widening) AND raw-count
    corroboration AND no veto AND sufficiency. A floor-passing but non-clearing cell
    is INCONCLUSIVE unless the CI upper bound excludes the meaningful effect."""
    if not adequate:
        return {"decision": "NO-BUILD_INCONCLUSIVE", "reason": "cell fails adequacy floor (degeneracy screen)"}

    def _clears(g: dict[str, Any], thr: float) -> bool:
        return bool(np.isfinite(g.get("ci_lo", np.nan)) and g["ci_lo"] > thr)

    level_ok = _clears(level_gap, practical_level)
    coarse_b_ok = _clears(coarse_b_gap, practical_level)
    slope_ok = bool(np.isfinite(slope.get("ci_lo", np.nan)) and slope["ci_lo"] > practical_widening)
    build = level_ok and coarse_b_ok and slope_ok and raw_count_ok and (not veto) and sufficiency_ok
    if build:
        return {"decision": "BUILD", "level_ok": True, "coarse_b_ok": True, "slope_ok": True,
                "raw_count_ok": True, "sufficiency_ok": True, "veto": False}

    # Effect ruled out ONLY when the CI upper bound excludes the meaningful effect on
    # the corrected (coarse_B) level AND the slope direction is uninformative/negative.
    level_ruled_out = bool(np.isfinite(coarse_b_gap.get("ci_hi", np.nan)) and coarse_b_gap["ci_hi"] < practical_level)
    slope_ruled_out = bool(np.isfinite(slope.get("ci_hi", np.nan)) and slope["ci_hi"] < practical_widening)
    if level_ruled_out and slope_ruled_out and not veto:
        status = "NO-BUILD_EFFECT-RULED-OUT"
    else:
        status = "NO-BUILD_INCONCLUSIVE"
    return {"decision": status, "level_ok": level_ok, "coarse_b_ok": coarse_b_ok, "slope_ok": slope_ok,
            "raw_count_ok": raw_count_ok, "sufficiency_ok": sufficiency_ok, "veto": veto}


def assert_k1_null(gap: dict[str, Any], *, tol: float = 1e-9) -> None:
    """K=1 is a harness assertion, not a significance test (Pi R5 C5): coarse ≡ fine
    ⇒ the point gap must be ~0. A non-zero gap is a harness bug."""
    if abs(float(gap.get("gap", 0.0))) > tol:
        raise AssertionError(f"K=1 null gap {gap.get('gap')} != 0 within {tol}: coarse≡fine harness bug")


def target_geometry(embeddings: np.ndarray) -> dict[str, Any]:
    """C4 target-geometry diagnostics — equal N gives equal chance, not equal
    difficulty. Duplicate/tie rate + effective rank of the candidate target set."""
    x = np.asarray(embeddings, dtype=np.float64)
    n = len(x)
    if n < 2:
        return {"n": int(n), "duplicate_rate": 0.0, "effective_rank": 0.0}
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    uniq = np.unique(np.round(xn, 6), axis=0).shape[0]
    s = np.linalg.svd(xn - xn.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {"n": int(n), "duplicate_rate": float(1.0 - uniq / n), "effective_rank": eff_rank}
