"""Rung-2 sub-gate 1 rollout-diagnosis stats core (Pi v2: authorized to build).

numpy-only (operates on precomputed latents; the torch rollout export is a separate driver) so
the DIAGNOSTIC contract is cheap to synthetic-test. Two estimands are kept strictly apart:

  * DIRECT-horizon path — horizon-conditioned; horizon decay only; MUST NOT emit exposure-gap or
    recursive metrics (enforced by rung2_contract.validate_direct_path_row).
  * RECURSIVE-transition path — fixed-width non-overlapping δ states; emits d_self (free/teacher-
    forced), the exposure gap, and drift; NOT_EVALUABLE unless the checkpoint metadata proves
    fixed-width-transition training.

ρ_t is DESCRIPTIVE ONLY (never a signature discriminator). The load-bearing collapse signal is
d_self normalised by the ambient true-true NN distance. Categorical HEALTHY/DRIFT/COLLAPSE labels
are emitted ONLY with the frozen margins; otherwise continuous diagnostics only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_probes import cluster_bootstrap_ci
from clinical_jepa.eval.rung2_contract import (
    DRIFT_SLOPE_TAU, EXPOSURE_GAP_MARGIN, SIG_COLLAPSE_DSELF_OVER_DNN,
)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _norm(a), _norm(b)
    return 1.0 - np.sum(a * b, axis=-1)


def ambient_true_nn_distance(true_latents: np.ndarray, patients: Any) -> float:
    """Mean nearest patient-disjoint true-true cosine distance — the ambient scale that
    normalises d_self (collapse compresses ALL distances, so raw d_NN is misleading)."""
    t = _norm(np.asarray(true_latents)); pats = np.asarray(patients)
    sims = t @ t.T
    n = len(t)
    same = pats[:, None] == pats[None, :]
    np.fill_diagonal(same, True)
    sims = np.where(same, -np.inf, sims)
    nn = sims.max(axis=1)
    return float(np.mean(1.0 - nn[np.isfinite(nn)]))


def drift_self_over_nn(zhat: np.ndarray, ztrue: np.ndarray, true_pool: np.ndarray, patients: Any,
                       pool_patients: Any) -> dict[str, np.ndarray]:
    """Per-row d_self = 1-cos(ẑ, own true) and d_NN = distance to the nearest patient-disjoint true
    latent; the collapse ratio d_self/d_NN (≥ SIG_COLLAPSE ⇒ own truth no better than nearest wrong
    instance)."""
    d_self = cos_dist(zhat, ztrue)
    q = _norm(np.asarray(zhat)); pool = _norm(np.asarray(true_pool))
    pats = np.asarray(patients); ppats = np.asarray(pool_patients)
    sims = q @ pool.T
    disjoint = pats[:, None] != ppats[None, :]
    sims = np.where(disjoint, sims, -np.inf)
    d_nn = 1.0 - sims.max(axis=1)
    return {"d_self": d_self, "d_nn": d_nn, "d_self_over_nn": d_self / np.clip(d_nn, 1e-9, None)}


def exposure_gap(d_self_free: np.ndarray, d_self_tf: np.ndarray) -> np.ndarray:
    """g_t = free-running − teacher-forced same-instance drift (RECURSIVE path only)."""
    return np.asarray(d_self_free, dtype=np.float64) - np.asarray(d_self_tf, dtype=np.float64)


def population_dispersion(cloud: np.ndarray) -> dict[str, float]:
    """v_t^pop — total variance + effective rank of a predicted/true cloud at a step."""
    x = _norm(np.asarray(cloud))
    xc = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return {"total_var": float((s ** 2).sum() / max(len(x), 1)),
            "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}


def collapse_ratio_descriptive(pred_cloud: np.ndarray, true_cloud: np.ndarray) -> float:
    """ρ_t = v_pred / v_true — DESCRIPTIVE ONLY (Pi): a deterministic mean-regressor has ρ<1 by
    Jensen; never used to assign a signature."""
    vp = population_dispersion(pred_cloud)["total_var"]
    vt = population_dispersion(true_cloud)["total_var"]
    return float(vp / max(vt, 1e-12))


def perturbation_top_singular(context_latents: np.ndarray, rolled: np.ndarray) -> float:
    """Finite-difference local-contraction estimate: top singular value of the map from an
    ε-perturbed context ensemble to the rolled output (>1 expansive, <1 contractive)."""
    dc = _norm(context_latents) - _norm(context_latents).mean(0, keepdims=True)
    dr = _norm(rolled) - _norm(rolled).mean(0, keepdims=True)
    sc = np.linalg.svd(dc, compute_uv=False)[0] + 1e-12
    sr = np.linalg.svd(dr, compute_uv=False)[0]
    return float(sr / sc)


def classify_signature(*, dself_over_nn_point: float, exposure_gap_slope_lo: float,
                       dself_slope_hi: float, transition_evaluable: bool) -> str:
    """Frozen-margin categorical signature (Pi #2: only emitted WITH the frozen numbers).
    Returns a continuous-only sentinel if the recursive path is NOT_EVALUABLE."""
    from clinical_jepa.eval.rung2_contract import NOT_EVALUABLE
    if not transition_evaluable:
        return NOT_EVALUABLE                                     # no recursive semantics -> no signature
    if dself_over_nn_point >= SIG_COLLAPSE_DSELF_OVER_DNN:
        return "COLLAPSE_DOMINANT"                               # own truth no better than nearest wrong
    if exposure_gap_slope_lo > EXPOSURE_GAP_MARGIN and dself_slope_hi > DRIFT_SLOPE_TAU:
        return "DRIFT_DOMINANT"
    return "HEALTHY"


def bootstrap_curve(values: Any, clusters: Any, **kw) -> dict[str, float]:
    """Cluster-bootstrap CI for a per-row diagnostic (reuse the Rung-1 patient-cluster bootstrap)."""
    return cluster_bootstrap_ci(values, clusters, **kw)
