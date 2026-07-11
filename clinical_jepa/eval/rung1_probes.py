"""Rung-1 decoder-free probe library + statistical core (Pi R7/R8).

numpy-only (the expressive M2 decoder lives in rung1_decode.py and feeds arrays here) so
the statistical contract is cheap to test. Provides:
  * cluster (patient/sequence) bootstrap — mean CI, paired swap-excess CI, ratio-skill CI;
  * M1a analytic multiset inverse; M1b closed-form ridge;
  * exact-count / exact-order / tie-aware Kendall-τ / marginal F1·Jaccard metrics;
  * hurdle/mixed timing marginal + RANDOMIZED PIT (Pi R8 #7 zero/tied Δt), KS-D UPPER-CI,
    and normalized-CRPS skill (Pi R8 Q5).
All CIs are cluster-level (never row-level). Wrong-instance swaps are the deterministic
patient-disjoint derangement from the contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_contract import N_BOOT, SEED, deterministic_derangement


# --------------------------------------------------------------------------- bootstrap
def _cluster_sums(row_vals: np.ndarray, clusters: Any) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    C = len(uniq)
    csum = np.bincount(inv, weights=np.asarray(row_vals, dtype=np.float64), minlength=C)
    ccnt = np.bincount(inv, minlength=C).astype(np.float64)
    return csum, ccnt


def cluster_bootstrap_ci(row_vals: Any, clusters: Any, *, n_boot: int = N_BOOT,
                         seed: int = SEED, alpha: float = 0.05) -> dict[str, float]:
    """Percentile CI for a MEAN-over-rows statistic under cluster resampling."""
    csum, ccnt = _cluster_sums(np.asarray(row_vals, dtype=np.float64), clusters)
    C = len(ccnt)
    point = float(csum.sum() / max(ccnt.sum(), 1e-12))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, C, size=C)
        boot[b] = csum[s].sum() / max(ccnt[s].sum(), 1e-12)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_excess_ci(true_vals: Any, swap_vals: Any, clusters: Any, **kw) -> dict[str, float]:
    """Paired cluster-bootstrap CI of the readout's floor-adjusted EXCESS = score(M(z)) −
    score(M(z_swap)) (Pi R7 #1). Feed per-row scores for the true and wrong-instance latent."""
    d = np.asarray(true_vals, dtype=np.float64) - np.asarray(swap_vals, dtype=np.float64)
    return cluster_bootstrap_ci(d, clusters, **kw)


def ratio_skill_ci(num_rows: Any, den_rows: Any, clusters: Any, *, n_boot: int = N_BOOT,
                   seed: int = SEED, alpha: float = 0.05) -> dict[str, float]:
    """Cluster-bootstrap CI of a normalized skill 1 − E[num]/E[den] (Pi R8 Q5: CRPS skill).
    Lower CI is the gated quantity."""
    nsum, _ = _cluster_sums(np.asarray(num_rows, dtype=np.float64), clusters)
    dsum, _ = _cluster_sums(np.asarray(den_rows, dtype=np.float64), clusters)
    C = len(nsum)
    point = 1.0 - float(nsum.sum() / max(dsum.sum(), 1e-12))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, C, size=C)
        boot[b] = 1.0 - nsum[s].sum() / max(dsum[s].sum(), 1e-12)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}


# --------------------------------------------------------------------------- readouts
def analytic_multiset(E: np.ndarray, z: np.ndarray) -> np.ndarray:
    """M1a (untrained): min-norm p solving Eᵀp = z (mean_embed = Eᵀp). Returns the raw
    least-squares vector over the vocabulary; downstream normalises to the simplex."""
    return np.linalg.lstsq(np.asarray(E, dtype=np.float64).T, np.asarray(z, dtype=np.float64), rcond=None)[0]


def multiset_reconstruction_residual(E: np.ndarray, z: np.ndarray, p_hat: np.ndarray) -> float:
    """‖Eᵀp̂ − z‖ / ‖z‖ — how well the analytic inverse explains z (M1a sanity)."""
    z = np.asarray(z, dtype=np.float64)
    rec = np.asarray(E, dtype=np.float64).T @ np.asarray(p_hat, dtype=np.float64)
    return float(np.linalg.norm(rec - z) / max(np.linalg.norm(z), 1e-12))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge weights (with bias column). Fit on TRAIN only."""
    X = np.asarray(X, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ np.asarray(y, dtype=np.float64))


def ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ np.asarray(W, dtype=np.float64)


# --------------------------------------------------------------------------- metrics
def exact_count_hits(pred: Any, true: Any) -> np.ndarray:
    return (np.rint(np.asarray(pred, dtype=np.float64)) == np.rint(np.asarray(true, dtype=np.float64))).astype(float)


def exact_order_hits(pred_seqs: list[Any], true_seqs: list[Any]) -> np.ndarray:
    """UNCONDITIONAL exact ordered-sequence reconstruction (Pi R8 #2 headline), per row."""
    out = np.zeros(len(true_seqs))
    for i, (p, t) in enumerate(zip(pred_seqs, true_seqs)):
        p = list(p); t = list(t)
        out[i] = float(len(p) == len(t) and all(a == b for a, b in zip(p, t)))
    return out


def kendall_tau_tie_aware(pred_rank: Any, true_rank: Any) -> float:
    """Tie-aware Kendall τ-b between predicted and true orderings (companion metric)."""
    p = np.asarray(pred_rank, dtype=np.float64); t = np.asarray(true_rank, dtype=np.float64)
    n = len(p)
    if n < 2:
        return float("nan")
    c = d = tp = tt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = np.sign(p[i] - p[j]); dt = np.sign(t[i] - t[j])
            if dp == 0 and dt == 0:
                continue
            if dp == 0:
                tp += 1
            elif dt == 0:
                tt += 1
            elif dp == dt:
                c += 1
            else:
                d += 1
    denom = np.sqrt((c + d + tp) * (c + d + tt))
    return float((c - d) / denom) if denom > 0 else float("nan")


def marginal_f1_jaccard(pred_sets: list[set], true_sets: list[set]) -> dict[str, float]:
    f1s, jacs = [], []
    for ps, ts in zip(pred_sets, true_sets):
        ps, ts = set(ps), set(ts)
        inter = len(ps & ts)
        prec = inter / len(ps) if ps else 0.0
        rec = inter / len(ts) if ts else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        union = len(ps | ts)
        jacs.append(inter / union if union else 1.0)
    return {"macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "set_jaccard": float(np.mean(jacs)) if jacs else float("nan")}


# --------------------------------------------------------------------------- timing
def fit_marginal_hurdle(dt_train: Any) -> dict[str, Any]:
    """Marginal hurdle/mixed Δt model (Pi R8 #7): point mass at Δt=0 + empirical positive
    tail. Fit on TRAIN only. Returns {p0, pos_samples}."""
    dt = np.asarray(dt_train, dtype=np.float64)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return {"p0": 0.0, "pos_samples": np.asarray([1.0])}
    p0 = float(np.mean(dt <= 0.0))
    pos = dt[dt > 0.0]
    return {"p0": p0, "pos_samples": pos if len(pos) else np.asarray([1.0])}


def _hurdle_cdf_bounds(model: dict[str, Any], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F(y⁻), F(y) for the hurdle model — the discrete mass at 0 gives a jump used by the
    randomized PIT."""
    p0 = float(model["p0"]); pos = np.asarray(model["pos_samples"], dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = np.where(y <= 0.0, 0.0, p0 + (1 - p0) * _ecdf(pos, y, side="left"))
    hi = np.where(y <= 0.0, p0, p0 + (1 - p0) * _ecdf(pos, y, side="right"))
    return lo, hi


def _ecdf(samples: np.ndarray, y: np.ndarray, *, side: str) -> np.ndarray:
    s = np.sort(np.asarray(samples, dtype=np.float64))
    idx = np.searchsorted(s, np.asarray(y, dtype=np.float64), side=("left" if side == "left" else "right"))
    return idx / len(s)


def randomized_pit(y: Any, model: dict[str, Any], *, seed: int = SEED) -> np.ndarray:
    """Randomized PIT (Pi R8 #7): pit = F(y⁻) + u·(F(y) − F(y⁻)), u~U(0,1). Handles the
    Δt=0 point mass and ties so a well-specified model yields Uniform(0,1) PIT."""
    lo, hi = _hurdle_cdf_bounds(model, np.asarray(y, dtype=np.float64))
    u = np.random.default_rng(seed).uniform(size=len(lo))
    return np.clip(lo + u * (hi - lo), 0.0, 1.0)


def ks_d_uniform(pit: Any) -> float:
    x = np.sort(np.asarray(pit, dtype=np.float64))
    n = len(x)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))


def ks_d_upper_ci(pit: Any, clusters: Any, *, n_boot: int = N_BOOT, seed: int = SEED,
                  alpha: float = 0.05) -> dict[str, float]:
    """Cluster-bootstrap UPPER-95%-CI of KS-D vs Uniform (Pi R8 Q3 gates on the upper CI)."""
    pit = np.asarray(pit, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    by = [np.where(inv == c)[0] for c in range(len(uniq))]
    C = len(by)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, C, size=C)
        boot[b] = ks_d_uniform(np.concatenate([pit[by[c]] for c in s]))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"point": ks_d_uniform(pit), "ci_hi": hi}


def crps_from_samples(samples: Any, y: float) -> float:
    """Sample-based CRPS estimator: E|X−y| − ½E|X−X'|."""
    s = np.asarray(samples, dtype=np.float64)
    if len(s) == 0:
        return float("nan")
    t1 = float(np.mean(np.abs(s - float(y))))
    t2 = 0.5 * float(np.mean(np.abs(s[:, None] - s[None, :])))
    return t1 - t2


def crps_rows(pred_samples: list[Any], y: Any) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.asarray([crps_from_samples(pred_samples[i], y[i]) for i in range(len(y))])


def crps_marginal_rows(marg_samples: Any, y: Any, *, cap: int = 256, seed: int = SEED) -> np.ndarray:
    """Vectorized CRPS of a SHARED marginal predictive against each y. The marginal self-term
    E|X−X'| is a constant (same distribution for all rows), so this is O(n·cap), not O(n·m²)."""
    m = np.asarray(marg_samples, dtype=np.float64)
    if len(m) > cap:
        m = np.random.default_rng(seed).choice(m, size=cap, replace=False)
    y = np.asarray(y, dtype=np.float64)
    t1 = np.abs(m[None, :] - y[:, None]).mean(axis=1)
    t2 = 0.5 * float(np.abs(m[:, None] - m[None, :]).mean())
    return t1 - t2


def crps_quantile_rows(quantiles: Any, y: Any) -> np.ndarray:
    """Vectorized CRPS of per-row quantile predictive samples. quantiles = [n, Q]."""
    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t1 = np.abs(q - y[:, None]).mean(axis=1)
    t2 = 0.5 * np.abs(q[:, :, None] - q[:, None, :]).mean(axis=(1, 2))
    return t1 - t2


def hurdle_randomized_pit(p0: Any, pos_quantiles: Any, levels: Any, y: Any, *, seed: int = SEED) -> np.ndarray:
    """Randomized PIT for a CONDITIONAL hurdle predictive (Pi R8 #7): a point mass p0 at
    Δt=0 plus a positive continuous tail from per-row quantiles. y=0 -> PIT~U(0,p0);
    y>0 -> p0 + (1-p0)·F_pos(y), randomized within the enclosing quantile step."""
    p0 = np.asarray(p0, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    q = np.asarray(pos_quantiles, dtype=np.float64); lv = np.asarray(levels, dtype=np.float64)
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=len(y))
    out = np.empty(len(y))
    for i in range(len(y)):
        if y[i] <= 0.0:
            out[i] = u[i] * p0[i]
        else:
            below = np.searchsorted(q[i], y[i], side="left")
            above = np.searchsorted(q[i], y[i], side="right")
            lo = lv[below - 1] if below > 0 else 0.0
            hi = lv[min(above, len(lv) - 1)] if above < len(lv) else 1.0
            fpos = lo + u[i] * max(hi - lo, 0.0)
            out[i] = p0[i] + (1.0 - p0[i]) * fpos
    return np.clip(out, 0.0, 1.0)


def tie_aware_exact_order_hits(pred_index_orders: list[Any], token_seqs: list[Any]) -> np.ndarray:
    """Tie-aware exact-order hit per window (Pi Rung-1 result gate #1): a hit iff the decoded
    TOKEN sequence equals the true token sequence, so swapping identical-token occurrences is
    not counted wrong. token_seqs[i] is the true ordered token ids; pred_index_orders[i] the
    predicted event index permutation."""
    out = np.zeros(len(token_seqs))
    for i, (perm, toks) in enumerate(zip(pred_index_orders, token_seqs)):
        toks = np.asarray(toks)
        out[i] = float(np.array_equal(toks[np.asarray(perm, dtype=int)], toks))
    return out


def analytic_pinv(E: np.ndarray) -> np.ndarray:
    """Pseudo-inverse of Eᵀ ([V,D]) — precompute ONCE so per-slot decoding is a matmul, not a
    per-window least-squares solve (governed-scale speed)."""
    return np.linalg.pinv(np.asarray(E, dtype=np.float64).T)


def slot_pvals(Epinv: np.ndarray, slot_means_list: list[Any]) -> list[np.ndarray]:
    """Per-window per-slot vocabulary p-values [M,V] via the precomputed pseudo-inverse."""
    return [np.asarray(sm, dtype=np.float64) @ Epinv.T for sm in slot_means_list]


def slot_exact_from_pvals(pvals_list: list[np.ndarray], true_slot_sets: list[Any],
                          thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """EXACT all-slot token-SET match + slot-wise micro-F1 per window, given cached p-values."""
    exact = np.zeros(len(true_slot_sets)); f1 = np.zeros(len(true_slot_sets))
    for i, (Pmat, truesets) in enumerate(zip(pvals_list, true_slot_sets)):
        ok = True; tp = fp = fn = 0
        for row, ts in zip(Pmat, truesets):
            ps = set(np.where(row > thresh)[0].tolist()); ts = set(int(x) for x in ts)
            if ps != ts:
                ok = False
            tp += len(ps & ts); fp += len(ps - ts); fn += len(ts - ps)
        exact[i] = float(ok)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[i] = 2 * prec * rec / (prec + rec) if (prec + rec) else (1.0 if not (tp + fp + fn) else 0.0)
    return exact, f1


def slot_tokenset_exact_hits(E: np.ndarray, slot_means: list[Any], true_slot_sets: list[Any],
                             thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: temporal-slot TOKEN-SET fidelity (Pi amended #3) — decode each
    slot's token SET (presence, not multiset/counts) from its slot-mean via the analytic
    inverse, then EXACT all-slot match. An EASIER upper bound than exact multiset recovery.
    Returns (exact_all_slot_token_set_hits, slot_wise_micro_F1) per window."""
    return slot_exact_from_pvals(slot_pvals(analytic_pinv(E), slot_means), true_slot_sets, thresh)


slot_multiset_exact_hits = slot_tokenset_exact_hits   # deprecated alias (misnomer; see above)


def marginal_hurdle_quantiles(dt_train: Any, *, n_q: int = 9) -> tuple[float, np.ndarray, np.ndarray]:
    """Marginal hurdle as (p0, positive-Δt quantiles, quantile levels) so the marginal CRPS
    uses the SAME hurdle-CRPS estimator (and sample count) as the conditional — removing the
    finite-sample-bias asymmetry Pi flagged (result gate #2.2)."""
    dt = np.asarray(dt_train, dtype=np.float64)
    dt = dt[np.isfinite(dt)]
    p0 = float(np.mean(dt <= 0.0)) if len(dt) else 0.0
    pos = dt[dt > 0.0]
    levels = np.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
    q = np.quantile(pos, levels) if len(pos) else np.ones(n_q)
    return p0, q, levels


def hurdle_crps_rows(p0: Any, pos_quantiles: Any, y: Any, *, n_samp: int = 64, seed: int = SEED) -> np.ndarray:
    """CRPS of the conditional hurdle predictive via a small per-row Monte-Carlo sample
    (mass p0 at 0, else a positive quantile), vectorized through crps_quantile_rows."""
    p0 = np.asarray(p0, dtype=np.float64); q = np.asarray(pos_quantiles, dtype=np.float64)
    n, Q = q.shape
    rng = np.random.default_rng(seed)
    is_zero = rng.uniform(size=(n, n_samp)) < p0[:, None]
    pick = rng.integers(0, Q, size=(n, n_samp))
    samp = np.take_along_axis(q, pick, axis=1)
    samp = np.where(is_zero, 0.0, samp)
    return crps_quantile_rows(samp, y)


# --------------------------------------------------------------------------- swap index
def swap_partner_index(patients: list[str], seed: int) -> np.ndarray:
    """Deterministic on-manifold wrong-instance partner per row within one matching cell
    (patient-disjoint derangement). -1 rows have no partner and are dropped from the floor."""
    return deterministic_derangement(list(patients), seed=seed)
