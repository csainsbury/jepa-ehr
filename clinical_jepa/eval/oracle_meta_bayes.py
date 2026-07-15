"""Exact conditional R0 + context-Bayes π* (Pi whole-pass-rebuild #2/#3).

With EXACT item content (item features ARE ``class_embed[class]``), the content the recipe observes is
exactly the class, and the positive-regime order-score for a pair (a, b) is

    Δs = (cm[a] − cm[b]) + κ·(g_ab · driver) + (noise_a − noise_b),   g_ab = M·(embed[a] − embed[b]),

with ``noise ~ N(0, ORDER_NOISE²)``. Integrating a driver with mean μ and covariance Σ gives the exact
pairwise probability

    P(a≺b) = Φ( (−(cm[a]−cm[b]) − κ·(g_ab·μ)) / sqrt(2·ORDER_NOISE² + κ²·(g_abᵀ Σ g_ab)) ).

  * R0  — content prior: integrate the driver over the family PRIOR (μ, Σ ignore context).
  * π*  — context-Bayes: integrate the driver over the POSTERIOR given the observed context.

Gaussian-driver families are analytic; the Markov family is an exact categorical mixture over states;
the Student-t family uses frozen MC (importance-reweighted for the posterior). Every family is validated
against an independent high-precision MC (``reference_mc_error``). All synthetic / safe-public.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtr        # fast vectorized standard-normal CDF

from clinical_jepa.eval.oracle_meta_gen import (
    COUPLING_SCALE, CTX_NOISE, GLOBAL_INVARIANT_SEED, INVARIANT, K_STATES, N_CLASSES, ORDER_NOISE,
    STUDENT_T_DF,
)

_GAUSS_FAMILIES = ("T_latent_factor", "T_realized_history", "E_no_h_exogenous")


def _phi(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


@lru_cache(maxsize=1)
def _pair_tables():
    """Class-pair coupling vectors g_ab = M·(embed[a]−embed[b]) and content diffs cm[a]−cm[b]."""
    inv = INVARIANT
    emb = inv.class_embed                                  # (C, D_ITEM)
    g = np.einsum("de,abe->abd", inv.M, emb[:, None, :] - emb[None, :, :]) * (COUPLING_SCALE / inv.coupling_norm)
    cm = inv.class_means[:, None] - inv.class_means[None, :]   # (C, C)
    return g, cm                                          # (C, C, D_H), (C, C)


def _posterior_cov() -> np.ndarray:
    """Gaussian driver posterior covariance given context = W·driver + N(0, CTX_NOISE²): (WᵀW/σ² + I)^-1."""
    W = INVARIANT.W_ctx
    return np.linalg.inv((W.T @ W) / (CTX_NOISE ** 2) + np.eye(W.shape[1]))


@lru_cache(maxsize=1)
def _markov_stationary() -> np.ndarray:
    mrng = np.random.default_rng(GLOBAL_INVARIANT_SEED ^ 0xA11)
    P = mrng.random((K_STATES, K_STATES)) + 0.1
    P /= P.sum(1, keepdims=True)
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return pi / pi.sum()


def _gauss_pairprob(cm_pair, g_pair, kappa, gmu, ggg):
    """Analytic Φ for a Gaussian driver moment. gmu = g·μ (…), ggg = gᵀΣg (…)."""
    denom = np.sqrt(2.0 * ORDER_NOISE ** 2 + (kappa ** 2) * ggg)
    return _phi((-cm_pair - kappa * gmu) / np.maximum(denom, 1e-9))


@lru_cache(maxsize=64)
def _r0_table(family_id: str, kappa: float) -> tuple:
    """EXACT content-prior class-pair table P(a≺b | class_a, class_b) — R0 depends ONLY on class pairs,
    so it is a (C, C) table (cached), indexed per sequence. Gaussian analytic / Markov mixture /
    Student-t MC over the family prior."""
    g, cm = _pair_tables()                                             # g (C,C,D_H), cm (C,C)
    if family_id in _GAUSS_FAMILIES:                                   # μ_prior=0, Σ_prior=I
        ggg = np.einsum("abd,abd->ab", g, g)
        tab = _gauss_pairprob(cm, g, kappa, np.zeros_like(cm), ggg)
    elif family_id == "T_hmm_markov":
        pi = _markov_stationary()
        tab = np.einsum("abk,k->ab", _phi((-cm[..., None] - kappa * g) / (np.sqrt(2.0) * ORDER_NOISE)), pi)
    elif family_id == "E_offgrid_heavytail":
        rng = np.random.default_rng(12345)
        drv = rng.standard_t(STUDENT_T_DF, size=(20000, g.shape[-1]))
        gd = np.einsum("abd,md->abm", g, drv)
        tab = _phi((-cm[..., None] - kappa * gd) / (np.sqrt(2.0) * ORDER_NOISE)).mean(-1)
    else:
        raise KeyError(family_id)
    return tuple(map(tuple, tab))


def _pistar_gaussian(cell, kappa):
    g, cm = _pair_tables()
    mu = cell.context_features @ INVARIANT.A.T                         # (N, D_H) posterior mean
    Sig = _posterior_cov()
    cp = cm[cell.item_classes[:, :, None], cell.item_classes[:, None, :]]
    gp = g[cell.item_classes[:, :, None], cell.item_classes[:, None, :]]
    gmu = np.einsum("nijd,nd->nij", gp, mu)
    ggg = np.einsum("nijd,de,nije->nij", gp, Sig, gp)
    return _gauss_pairprob(cp, gp, kappa, gmu, ggg)


def _markov_weights_prior(n):
    return np.broadcast_to(_markov_stationary(), (n, K_STATES))


def _markov_weights_posterior(context):
    """p(state=k | context) ∝ N(context; W·e_k, CTX_NOISE²I)·π_k over the K one-hot states."""
    W = INVARIANT.W_ctx                                               # (D_CTX, D_H), e_k picks column k
    means = W.T                                                       # (D_H? no) -> W[:,k] is the mean for e_k
    ll = -((context[:, None, :] - W.T[None, :, :]) ** 2).sum(-1) / (2 * CTX_NOISE ** 2)   # (N, K)
    logw = ll + np.log(_markov_stationary())[None, :]
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); return w / w.sum(1, keepdims=True)


def _markov_mixture(classes, kappa, weights):
    """Σ_k w_k·Φ((−cm − κ·g[…,k])/(√2·σ)) — driver = one-hot state e_k => g·e_k = g[…,k]."""
    g, cm = _pair_tables()
    cp = cm[classes[:, :, None], classes[:, None, :]]                 # (N, L, L)
    gp = g[classes[:, :, None], classes[:, None, :]]                  # (N, L, L, D_H==K)
    denom = np.sqrt(2.0) * ORDER_NOISE
    phi_k = _phi((-cp[..., None] - kappa * gp) / denom)              # (N, L, L, K)
    return np.einsum("nijk,nk->nij", phi_k, weights)


def _student_prior_logpdf(drv):
    """log density of the iid Student-t(df) driver PRIOR, summed over coordinates (…, D) → (…)."""
    from scipy.special import gammaln
    df = STUDENT_T_DF
    c = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi)
    return (c - (df + 1.0) / 2.0 * np.log1p(drv ** 2 / df)).sum(-1)


def _gauss_std_logpdf(drv):
    """log density of the standard-normal driver prior N(0, I), summed over coordinates."""
    return (-0.5 * drv ** 2 - 0.5 * np.log(2.0 * np.pi)).sum(-1)


# π* MC budget for the ONLY non-analytic posterior (Student-t held-out); frozen for the reference.
STUDENT_PISTAR_NMC = 3072
STUDENT_PISTAR_SEED = 0x5717


def _pistar_no_h(cell, kappa):
    """E_no_h_exogenous exposes the driver EXACTLY as ``observed_covariates``; the Bayes-optimal
    reference conditions on it, so the posterior is a POINT MASS at the observed z (Σ_post = 0)."""
    g, cm = _pair_tables()
    z = np.asarray(cell.observed_covariates, dtype=float)             # (N, D_H) — the exact driver
    cls = cell.item_classes
    cp = cm[cls[:, :, None], cls[:, None, :]]
    gp = g[cls[:, :, None], cls[:, None, :]]
    gmu = np.einsum("nijd,nd->nij", gp, z)
    return _gauss_pairprob(cp, gp, kappa, gmu, np.zeros_like(cp))     # ggg = 0 → point mass at z


def _pistar_student(cell, kappa, n_mc=STUDENT_PISTAR_NMC, seed=STUDENT_PISTAR_SEED, return_ess=False):
    """Exact per-sequence Student-t posterior π*. The Gaussian-posterior proposal q(drv) ∝
    N(ctx; W·drv, σ²)·N(drv;0,I) is the target's likelihood times the WRONG (Gaussian) prior, so the
    self-normalized importance weight collapses to the prior-density ratio t_prior/N(0,I) — no likelihood
    evaluation, tight proposal, high ESS. Vectorized per-sequence in memory-bounded chunks."""
    g, cm = _pair_tables()
    ctx = cell.context_features
    N = ctx.shape[0]
    L = cell.item_classes.shape[1]
    mu = ctx @ INVARIANT.A.T                                          # (N, D) posterior mean
    Sig = _posterior_cov(); D = Sig.shape[0]
    Lc = np.linalg.cholesky(Sig + 1e-12 * np.eye(D))
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_mc, D)) @ Lc.T                      # shared zero-mean proposal block
    cls = cell.item_classes
    out = np.empty((N, L, L))
    ess = np.empty(N)
    denom = np.sqrt(2.0) * ORDER_NOISE
    chunk = max(1, 4_000_000 // (L * L * n_mc))
    for s in range(0, N, chunk):
        e = slice(s, min(N, s + chunk))
        drv = mu[e][:, None, :] + base[None, :, :]                   # (c, n_mc, D)
        logw = _student_prior_logpdf(drv) - _gauss_std_logpdf(drv)   # (c, n_mc)
        logw -= logw.max(1, keepdims=True)
        w = np.exp(logw); w /= w.sum(1, keepdims=True)
        ess[e] = 1.0 / (w ** 2).sum(1)
        cp = cm[cls[e][:, :, None], cls[e][:, None, :]]              # (c, L, L)
        gp = g[cls[e][:, :, None], cls[e][:, None, :]]              # (c, L, L, D)
        Y = np.einsum("cijd,cmd->cijm", gp, drv)                     # (c, L, L, n_mc)
        out[e] = np.einsum("cijm,cm->cij", _phi((-cp[..., None] - kappa * Y) / denom), w)
    if return_ess:
        return out, ess
    return out


def r0_pairwise(family_id: str, kappa: float, class_ids: np.ndarray) -> np.ndarray:
    """Exact content-prior P(a≺b | classes) for the positive regime — indexed from the class-pair table."""
    tab = np.array(_r0_table(family_id, float(kappa)))
    return tab[class_ids[:, :, None], class_ids[:, None, :]]


def pi_star_pairwise(cell, kappa: float) -> np.ndarray:
    """Exact context-Bayes P(a≺b | context, content) for the positive regime, per family."""
    kappa = float(kappa)
    fam = cell.family_id
    if fam == "E_no_h_exogenous":
        return _pistar_no_h(cell, kappa)                              # condition on the observed driver
    if fam in _GAUSS_FAMILIES:
        return _pistar_gaussian(cell, kappa)                         # latent-Gaussian posterior (analytic)
    if fam == "T_hmm_markov":
        return _markov_mixture(cell.item_classes, kappa, _markov_weights_posterior(cell.context_features))
    if fam == "E_offgrid_heavytail":
        return _pistar_student(cell, kappa)                          # true per-sequence Student-t posterior
    raise KeyError(fam)


def _student_pistar_validate(cell, kappa, n_mc=24000, seed=11):
    """INDEPENDENT high-precision Student-t π* by a GENUINELY DIFFERENT scheme: a heavy-tailed
    multivariate-t proposal centred at μ, with the FULL unnormalized posterior weight (explicit
    likelihood N(ctx;W·drv,σ²) × Student-t prior ÷ mv-t proposal density). Shares no arithmetic with
    ``_pistar_student`` (which cancels the likelihood analytically), so agreement is non-circular."""
    g, cm = _pair_tables()
    ctx = cell.context_features
    mu = ctx @ INVARIANT.A.T
    W = INVARIANT.W_ctx; sig2 = CTX_NOISE ** 2
    Sig = _posterior_cov(); D = Sig.shape[0]
    Lc = np.linalg.cholesky(Sig + 1e-12 * np.eye(D)) * 1.3            # inflated to cover the tails
    rng = np.random.default_rng(seed)
    df_p = 3.0
    cls = cell.item_classes
    out = np.empty((ctx.shape[0], cls.shape[1], cls.shape[1]))
    denom = np.sqrt(2.0) * ORDER_NOISE
    for i in range(ctx.shape[0]):
        z = rng.standard_normal((n_mc, D))
        chi = rng.chisquare(df_p, n_mc) / df_p
        t = z / np.sqrt(chi)[:, None]                                 # standard multivariate-t(df_p)
        drv = mu[i] + t @ Lc.T                                        # proposal draws (heavy-tailed)
        resid = ctx[i][None, :] - drv @ W.T
        loglik = -0.5 * (resid ** 2).sum(1) / sig2                    # explicit Gaussian likelihood
        logtarget = loglik + _student_prior_logpdf(drv)              # unnormalized posterior
        u = np.linalg.solve(Lc, (drv - mu[i]).T).T                    # whiten for the mv-t density
        logq = -(df_p + D) / 2.0 * np.log1p((u ** 2).sum(1) / df_p)  # ∝ mv-t proposal log-density
        logw = logtarget - logq
        logw -= logw.max(); w = np.exp(logw); w /= w.sum()
        cp = cm[cls[i][:, None], cls[i][None, :]]; gp = g[cls[i][:, None], cls[i][None, :]]
        Y = np.einsum("ijd,md->ijm", gp, drv)
        out[i] = np.einsum("ijm,m->ij", _phi((-cp[..., None] - kappa * Y) / denom), w)
    return out


def reference_mc_error(family_id: str, kappa: float, which: str = "r0", seed: int = 7) -> float:
    """Validate the analytic/MC reference against an INDEPENDENT high-precision method. R0 is checked
    over the family PRIOR; π* over the PER-SEQUENCE posterior — each family against a genuinely different
    integrator (no-h point mass vs noise-only MC; Gaussian vs brute MC; Markov vs categorical MC;
    Student-t vs the mv-t importance validator above)."""
    from clinical_jepa.eval.oracle_meta_gen import _driver, generate_meta_cell
    g, cm = _pair_tables()
    rng = np.random.default_rng(seed + 1)
    if which == "r0":
        cell = generate_meta_cell(family_id, kappa, "orthogonal", 300, seed=seed)
        ref = r0_pairwise(family_id, kappa, cell.item_classes)
        cp = cm[cell.item_classes[:, :, None], cell.item_classes[:, None, :]]
        gp = g[cell.item_classes[:, :, None], cell.item_classes[:, None, :]]
        drv, _ = _driver(family_id, 20000, rng)
        gd = np.einsum("nijd,md->nijm", gp, drv)
        noise = rng.standard_normal(20000) * (np.sqrt(2.0) * ORDER_NOISE)
        hit = (cp[..., None] + kappa * gd + noise[None, None, None, :] < 0).mean(-1)
        return float(np.abs(ref - hit).max())
    # π*: per-sequence posterior brute MC, sampling EACH family's ACTUAL posterior. Few sequences × a
    # large sample so the validator's own MC noise floor (max over all pairs) stays well under 0.01.
    cell = generate_meta_cell(family_id, kappa, "orthogonal", 24, seed=seed)
    ref = pi_star_pairwise(cell, kappa)
    if family_id == "E_offgrid_heavytail":
        return float(np.abs(ref - _student_pistar_validate(cell, kappa)).max())
    Sig = _posterior_cov(); D = Sig.shape[0]
    Lc = np.linalg.cholesky(Sig + 1e-9 * np.eye(D))
    nmc = 80000
    markov_w = _markov_weights_posterior(cell.context_features) if family_id == "T_hmm_markov" else None
    errs = []
    for i in range(cell.context_features.shape[0]):
        cls = cell.item_classes[i]
        if family_id == "E_no_h_exogenous":                               # posterior = point mass at observed z
            drv = np.broadcast_to(cell.observed_covariates[i], (nmc, D))
        elif family_id == "T_hmm_markov":                                 # discrete categorical posterior
            states = rng.choice(K_STATES, size=nmc, p=markov_w[i])
            drv = np.eye(K_STATES)[states]
        else:                                                             # Gaussian (latent) posterior
            drv = (INVARIANT.A @ cell.context_features[i]) + rng.standard_normal((nmc, D)) @ Lc.T
        cp = cm[cls[:, None], cls[None, :]]; gp = g[cls[:, None], cls[None, :]]
        gd = np.einsum("ijd,md->ijm", gp, drv)
        noise = rng.standard_normal(nmc) * (np.sqrt(2.0) * ORDER_NOISE)
        hit = (cp[..., None] + kappa * gd + noise[None, None, :] < 0).mean(-1)
        errs.append(np.abs(ref[i] - hit).max())
    return float(np.max(errs))


def student_pistar_ess(family_id: str = "E_offgrid_heavytail", kappa: float = 0.60, n: int = 200,
                       seed: int = 3) -> float:
    """Mean per-sequence importance ESS for the Student-t π* (reported at signal-freeze time)."""
    from clinical_jepa.eval.oracle_meta_gen import generate_meta_cell
    cell = generate_meta_cell(family_id, float(kappa), "orthogonal", int(n), seed=seed)
    _, ess = _pistar_student(cell, float(kappa), return_ess=True)
    return float(np.mean(ess))
