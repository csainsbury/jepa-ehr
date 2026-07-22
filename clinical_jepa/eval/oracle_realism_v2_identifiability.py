"""Identifiability battery for the realism-v2 D components (rebuild step 3; Pi step-3 re-gate §6).

Executable standardized-Jacobian rank + grid-recovery + collision search over the four-scalar D vector
`S3_tau, S3_loggap, S4_abs, S7_abs` as a function of the three D-component strengths
`theta = (burst_timing, mark_burst_tie, cluster_size_mark_diversity)` (length_class_mix / S6_tv dropped at the
step-4 result gate). COMPONENTS and D_VECTOR derive from the design constants, so the grid is now 3^3.

Frozen numerics (design): param range [0, 0.6]; grid 0.10/0.35/0.55; central CRN finite differences (step 0.02),
forward one-sided at 0.0, backward one-sided at 0.60; null covariance whitening with ridge
`Sigma + 1e-3*trace(Sigma)/d*I`; standardized-Jacobian rank criterion `sigma_min/sigma_max >= 1e-3`;
deterministic nearest-grid recovery; recovery tol `<= 0.05 * range` and `<= half a grid step`; collision search.
Synthetic-only; uses the independent fixture + coupling constructions; no governed read.

Cost: the full `3^4 x nuisance-profiles x seeds x long-sequence` grid may exceed the cap — the runner returns
PARTIAL/non-pass and re-gates rather than silently reducing. This module implements the machinery; the full
grid runs only under the reviewed step-4 job.
"""
from __future__ import annotations

from math import log

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash
from clinical_jepa.eval.oracle_realism_v2 import V2_D_COMPONENT_MENU
from clinical_jepa.eval.oracle_realism_v2_coupling import apply_coupling
from clinical_jepa.eval.oracle_realism_v2_verifier import sequence_route_checks, NOT_EVALUABLE
from clinical_jepa.eval.oracle_realism_v2_verifier_design import IDENTIFIABILITY_VECTOR, IDENTIFIABILITY_NUISANCE

COMPONENTS = tuple(V2_D_COMPONENT_MENU)                    # 4 D params, composition order
D_VECTOR = tuple(IDENTIFIABILITY_VECTOR)                   # 5 sensitive scalars
PARAM_RANGE = (0.0, 0.6)
GRID = (0.10, 0.35, 0.55)
FD_STEP = 0.02
RIDGE = 1e-3
RANK_MIN = 1e-3                                            # sigma_min / sigma_max
RECOVER_TOL = 0.05 * (PARAM_RANGE[1] - PARAM_RANGE[0])    # 0.03 of the raw range
# 3-profile Jacobian nuisance (Pi §5); boundary-short is a terminal support control, not a grid nuisance.
NUISANCE_PROFILES = tuple(IDENTIFIABILITY_NUISANCE)
# FIXED raw collision tolerance vector aligned to D_VECTOR (S3_tau,S3_loggap,S4_abs,S7_abs), Pi re-gate.
ACCEPT_TOL = np.asarray([0.05, log(1.10), 0.03, 0.03])


def _cseed(component, tag) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"ident|{component}|{tag}".encode()).digest()[:8], "big")


def apply_theta(sample, theta, *, tag="theta"):
    """Compose the active D couplings at strengths ``theta`` (a dict component->strength) in menu order."""
    out = list(sample)
    for comp in COMPONENTS:
        s = float(theta.get(comp, 0.0))
        if s > 0.0:
            out = apply_coupling(out, comp, s, seed=_cseed(comp, tag))
    return out


def cross_stat_vector(coupled, reference):
    """The 5 D-sensitive scalar VALUES of ``coupled`` vs ``reference``. Returns None if any is NOT_EVALUABLE."""
    checks = sequence_route_checks(coupled, reference)
    vals = []
    for k in D_VECTOR:
        r = checks[k]
        if r.status == NOT_EVALUABLE or r.value is None:
            return None
        vals.append(float(r.value))
    return np.asarray(vals)


def f_theta(theta, *, base_sampler, source_profile, seed):
    """f(theta) = the 5-vector of (null base coupled at theta) vs an INDEPENDENT null reference (CRN base)."""
    coupled = apply_theta(base_sampler(source_profile, seed, "ident_coupled"), theta,
                          tag=f"{source_profile}|{seed}")
    reference = base_sampler(source_profile, seed, "ident_ref")
    return cross_stat_vector(coupled, reference)


def covariance_from_rows(rows):
    """Ridge-regularised covariance from pre-computed NULL rows. Shared by null_covariance and the CHECKPOINTED
    runner (which collects rows one cov-seed at a time) so both produce the IDENTICAL Sigma (no divergence)."""
    if rows is None or len(rows) < 2:
        return None
    X = np.asarray(rows)
    Sigma = np.cov(X, rowvar=False)
    d = Sigma.shape[0]
    return Sigma + RIDGE * (np.trace(Sigma) / d) * np.eye(d)


def null_covariance(base_sampler, seeds, *, source_profile):
    """STRICT ridge-regularised covariance of the 5-vector under NULL (theta=0) replicates (Pi §5): EVERY seed
    must evaluate — any NOT_EVALUABLE row => None (the profile is non-pass, never silently dropped)."""
    zero = {c: 0.0 for c in COMPONENTS}
    rows = [f_theta(zero, base_sampler=base_sampler, source_profile=source_profile, seed=s) for s in seeds]
    if any(r is None for r in rows):
        return None
    return covariance_from_rows(rows)


def _fd_pair(theta0, comp):
    """Central CRN offsets, forward one-sided at 0.0, backward one-sided at 0.60."""
    v = theta0[comp]
    if v <= PARAM_RANGE[0]:
        return ({**theta0, comp: v + FD_STEP}, dict(theta0), FD_STEP)         # forward
    if v >= PARAM_RANGE[1]:
        return (dict(theta0), {**theta0, comp: v - FD_STEP}, FD_STEP)         # backward
    return ({**theta0, comp: v + FD_STEP}, {**theta0, comp: v - FD_STEP}, 2 * FD_STEP)   # central


def jacobian(theta0, *, base_sampler, source_profile, seed):
    """5x4 CRN finite-difference Jacobian d(cross-stats)/d(theta) at theta0. Returns None if any eval refuses."""
    cols = []
    for comp in COMPONENTS:
        hi_t, lo_t, denom = _fd_pair(theta0, comp)
        hi = f_theta(hi_t, base_sampler=base_sampler, source_profile=source_profile, seed=seed)
        lo = f_theta(lo_t, base_sampler=base_sampler, source_profile=source_profile, seed=seed)
        if hi is None or lo is None:
            return None
        cols.append((hi - lo) / denom)
    return np.stack(cols, axis=1)                         # (5, 4)


def standardized_rank(J, Sigma_lambda):
    """Whiten the Jacobian rows by Sigma_lambda^{-1/2} and report sigma_min/sigma_max (Pi rank criterion)."""
    w, V = np.linalg.eigh(Sigma_lambda)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T      # Sigma_lambda^{-1/2}
    Js = W @ J
    sv = np.linalg.svd(Js, compute_uv=False)
    ratio = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    return {"singular_values": [round(float(x), 6) for x in sv], "sigma_min_over_max": round(ratio, 6),
            "rank_ok": ratio >= RANK_MIN}


def whitening_matrix(Sigma_lambda):
    """Sigma_lambda^{-1/2} (symmetric) — the whitening applied to BOTH the query and grid vectors in recovery."""
    w, V = np.linalg.eigh(Sigma_lambda)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def nearest_grid_recovery(vec, grid_thetas, grid_vectors, *, W=None):
    """Deterministic nearest-grid recovery under the WHITENED-L2 objective (W = whitening_matrix; identity if
    None). Lexicographic menu-order tie-break. The whitening is APPLIED to the query and grid vectors (Pi §5)."""
    qv = vec if W is None else W @ vec
    best = None
    for gt, gv in zip(grid_thetas, grid_vectors):
        d = float(np.sum((qv - (gv if W is None else W @ gv)) ** 2))
        key = (d,) + tuple(gt[c] for c in COMPONENTS)      # tie-break by lowest whitened-L2 then menu-order theta
        if best is None or key < best[0]:
            best = (key, gt)
    return best[1]


def recovered_within_tol(true_theta, rec_theta) -> bool:
    """A recovery succeeds iff every component is within RECOVER_TOL of the true setting (Pi recovery verdict)."""
    return all(abs(true_theta[c] - rec_theta[c]) <= RECOVER_TOL for c in COMPONENTS)


def collision_search(grid_thetas, grid_vectors, *, accept_tol=None):
    """Two settings separated beyond RECOVER_TOL COLLIDE iff EVERY cross-stat diff stays within the FIXED
    ACCEPT_TOL vector (aligned to D_VECTOR). Returns the colliding pairs."""
    tol = ACCEPT_TOL if accept_tol is None else np.asarray(accept_tol)
    collisions = []
    for i in range(len(grid_thetas)):
        for j in range(i + 1, len(grid_thetas)):
            sep = max(abs(grid_thetas[i][c] - grid_thetas[j][c]) for c in COMPONENTS)
            if sep > RECOVER_TOL and np.all(np.abs(grid_vectors[i] - grid_vectors[j]) <= tol):
                collisions.append((grid_thetas[i], grid_thetas[j]))
    return collisions


def cost_forecast(n=4000, *, cov_seeds=25, ref_seeds=1, heldout_seeds=1, rank_points=3,
                  seconds_per_eval=None) -> dict:
    """Structural forecast matching the runner (seeds replicate where they actually multiply, Pi §5): null
    covariance = cov_seeds x nuisance; grid vectors = grid_points x (ref+heldout) x nuisance; Jacobian rank =
    rank_points x (2*k) x nuisance. A NAIVE seeds-at-every-grid-point cross would be far larger and over cap —
    the runner never does that; if the structural forecast still exceeds the cap it returns PARTIAL/re-gate."""
    grid_points = len(GRID) ** len(COMPONENTS)            # 81
    k, nuis = len(COMPONENTS), len(NUISANCE_PROFILES)
    cov = cov_seeds * nuis
    grid = grid_points * (ref_seeds + heldout_seeds) * nuis
    jac = rank_points * (2 * k) * nuis
    evals = cov + grid + jac
    naive = grid_points * nuis * (1 + 2 * k) * cov_seeds  # the infeasible naive cross, for contrast
    return {"grid_points": grid_points, "nuisance_profiles": nuis, "cov_seeds": cov_seeds,
            "f_evals": {"null_covariance": cov, "grid_vectors": grid, "jacobian_rank": jac, "total": evals},
            "naive_cross_evals": naive, "n_per_eval": n,
            "est_wall_hours": None if seconds_per_eval is None else round(evals * seconds_per_eval / 3600, 1),
            "note": "structural runner cost; if total > cap => PARTIAL / non-pass / re-gate (no silent reduction)"}


IDENTIFIABILITY_IMPL = {
    "name": "realism_v2_identifiability_dev",
    "d_vector": list(D_VECTOR),
    "components": list(COMPONENTS),
    "param_range": list(PARAM_RANGE),
    "grid": list(GRID),
    "fd_step": FD_STEP,
    "ridge": RIDGE,
    "rank_min": RANK_MIN,
    "recover_tol": RECOVER_TOL,
    "nuisance_profiles": list(NUISANCE_PROFILES),          # 3 (SCID/MIMIC/structural-zero); no boundary-short
    "accept_tol": [round(float(x), 6) for x in ACCEPT_TOL],
    "fd_rule": "central CRN interior; forward one-sided at 0.0; backward one-sided at 0.60",
    "rank_rule": "standardized (null-covariance-whitened) Jacobian sigma_min/sigma_max >= 1e-3",
    "null_covariance": "STRICT: every seed must evaluate; any NOT_EVALUABLE row => profile non-pass",
    "recovery": "deterministic nearest-grid, whitening APPLIED to query+grid vectors, menu-order tie-break; "
                "recovered iff every component within recover_tol",
    "collision": "sep > recover_tol AND every cross-stat diff <= FIXED accept_tol vector",
    "cost_forecast": "structural: null_cov(cov_seeds x nuis) + grid((ref+heldout) x 81 x nuis) + "
                     "jac(rank_points x 2*4 x nuis); naive seeds-at-every-grid cross is over cap",
    "cap_behaviour": "grid may exceed cap => PARTIAL / non-pass / re-gate; never silent reduction",
}


def identifiability_impl_identity() -> str:
    return canonical_hash(IDENTIFIABILITY_IMPL)
