"""Shared-invariant META-FAMILY generator (Pi 2nd-pass REVISE #1/#2 — CORRECTED DESIGN).

The primary REVISE blocker: refitting the recipe on each held-out family voids structural held-out
evaluation. The fix is a SHARED, learnable cross-family invariant so ONE recipe fitted on the TRAIN
families transfers UNCHANGED to the held-out families.

Design:
  * GLOBAL FROZEN invariant, shared by EVERY family (seeded once, hashed): the driver→context map
    ``W_ctx``, the driver×item order coupling ``M``, a shared nonlinearity, per-class content-prior
    offsets ``class_means`` (so the exact content prior π0 is NON-uniform and computable), and a fixed
    class→item-feature embedding (so item class is observable to the recipe).
  * Families differ ONLY in the DRIVER LAW (how the D_H driver is produced) + null/nuisance:
      T_latent_factor    : driver ~ N(0, I)                         [train]
      T_hmm_markov       : driver = one-hot terminal Markov state   [train]
      T_realized_history : driver = realized-prefix summary         [train]
      E_no_h_exogenous   : driver = EXOGENOUS observed z (no hidden common cause)   [held-out]
      E_offgrid_nonlinear: driver ~ Student-t, off-grid κ           [held-out]
  * Order-score law (SHARED form): s_k = class_means[c_k] + κ·tanh(hᵀ M f_k) + noise. Null sequences
    drop the κ term (content-prior only). The exact π0 P(a≺b | classes) follows from class_means.
  * Repeated FIXED class multisets from a small frozen bank (so exact-multiset support clusters exist);
    support = eligible repeated-multiset cluster count, not raw N.

A recipe that learns the shared context→order map from TRAIN families transfers to held-out; a
family-specific / memorizing recipe cannot (it has no held-out access and no shared law to exploit).
Fully synthetic / safe-public.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash, context_view, future_view

D_H = 6
D_CTX = 12
D_ITEM = 5
L_ITEMS = 8
N_CLASSES = 6
CTX_NOISE = 0.5
ORDER_NOISE = 0.5          # irreducible realized-order noise (nulls retain this)
COUPLING_SCALE = 1.0       # scale of the (unit-normalized) context coupling before κ
CLASS_MEAN_SPAN = 0.6      # content-prior offset span (present but NOT dominant)
STUDENT_T_DF = 4.0
K_STATES = 6              # == D_H so a one-hot terminal Markov state IS the driver
HAWKES_BRANCHING = 0.5
GLOBAL_INVARIANT_SEED = 0x0180AC1E

TRAIN_FAMILIES = ("T_latent_factor", "T_hmm_markov", "T_realized_history")
HELDOUT_FAMILIES = ("E_no_h_exogenous", "E_offgrid_nonlinear")
NO_H_FAMILIES = ("E_no_h_exogenous",)

# a small frozen bank of FIXED class multisets (length L_ITEMS) so multisets REPEAT across sequences.
_MULTISET_BANK: tuple[tuple[int, ...], ...] = (
    (0, 0, 1, 2, 3, 3, 4, 5),
    (1, 1, 1, 2, 2, 4, 5, 5),
    (0, 1, 2, 3, 4, 5, 0, 2),
    (2, 2, 3, 3, 3, 4, 4, 5),
    (0, 0, 0, 1, 4, 4, 5, 5),
)


@dataclass(frozen=True)
class _Invariant:
    W_ctx: np.ndarray          # (D_CTX, D_H)
    M: np.ndarray              # (D_H, D_ITEM)
    class_means: np.ndarray    # (N_CLASSES,)
    class_embed: np.ndarray    # (N_CLASSES, D_ITEM)
    A: np.ndarray              # (D_H, D_CTX) linear driver-recovery from context (Bayes for the linear part)


def _shared_invariant() -> _Invariant:
    rng = np.random.default_rng(GLOBAL_INVARIANT_SEED)          # ONE global seed -> shared by all families
    W = rng.standard_normal((D_CTX, D_H))
    M = rng.standard_normal((D_H, D_ITEM))
    class_means = np.linspace(-CLASS_MEAN_SPAN, CLASS_MEAN_SPAN, N_CLASSES)   # non-uniform content prior
    class_embed = rng.standard_normal((N_CLASSES, D_ITEM))
    prec = (W.T @ W) / (CTX_NOISE ** 2) + np.eye(D_H)
    A = np.linalg.solve(prec, W.T) / (CTX_NOISE ** 2)
    return _Invariant(W, M, class_means, class_embed, A)


INVARIANT = _shared_invariant()


def invariant_hash() -> str:
    """Hash the SHARED invariant + the executable literal constants (Pi #3 — bind the real mechanism)."""
    inv = INVARIANT
    return canonical_hash({
        "seed": GLOBAL_INVARIANT_SEED, "D_H": D_H, "D_CTX": D_CTX, "D_ITEM": D_ITEM, "L": L_ITEMS,
        "N_CLASSES": N_CLASSES, "ctx_noise": CTX_NOISE, "order_noise": ORDER_NOISE,
        "student_t_df": STUDENT_T_DF, "k_states": K_STATES, "hawkes_branching": HAWKES_BRANCHING,
        "class_means": inv.class_means.round(6).tolist(),
        "W_ctx_sig": round(float(np.abs(inv.W_ctx).sum()), 6),
        "M_sig": round(float(np.abs(inv.M).sum()), 6),
        "class_embed_sig": round(float(np.abs(inv.class_embed).sum()), 6),
        "multiset_bank": _MULTISET_BANK,
    })


def _phi(x: np.ndarray) -> np.ndarray:               # standard normal CDF
    from math import erf, sqrt
    vec = np.vectorize(lambda t: 0.5 * (1.0 + erf(t / sqrt(2.0))))
    return vec(x)


def exact_pi0(class_ids: np.ndarray) -> np.ndarray:
    """Exact content prior P(item a precedes item b | classes) under class_means + irreducible order
    noise: P = Φ((cm[c_b] - cm[c_a]) / (sqrt(2)·ORDER_NOISE)). (N, L, L)."""
    cm = INVARIANT.class_means[class_ids]                       # (N, L)
    diff = cm[:, None, :] - cm[:, :, None]                     # cm[b] - cm[a]
    return _phi(diff / (np.sqrt(2.0) * ORDER_NOISE))


# ------------------------------------------------------------------------------------------------
# per-family DRIVER laws (the ONLY thing that differs; the context/order maps are shared)
# ------------------------------------------------------------------------------------------------
def _driver(family_id: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (driver (n, D_H), observed_covariate or None). Only E_no_h_exogenous observes its driver."""
    if family_id == "T_latent_factor":
        return rng.standard_normal((n, D_H)), None
    if family_id == "E_offgrid_nonlinear":
        return rng.standard_t(STUDENT_T_DF, size=(n, D_H)), None
    if family_id == "T_hmm_markov":
        mrng = np.random.default_rng(GLOBAL_INVARIANT_SEED ^ 0xA11)
        P = mrng.random((K_STATES, K_STATES)) + 0.1
        P /= P.sum(1, keepdims=True)
        state = rng.integers(0, K_STATES, size=n)
        for _ in range(8):
            state = np.array([rng.choice(K_STATES, p=P[s]) for s in state])
        return np.eye(D_H)[state], None                        # one-hot terminal state as the driver
    if family_id == "T_realized_history":
        prefix = rng.standard_normal((n, D_H))                 # realized-prefix summary
        return prefix, None
    if family_id == "E_no_h_exogenous":
        z = rng.standard_normal((n, D_H))                      # EXOGENOUS + OBSERVED (no hidden cause)
        return z, z
    raise KeyError(family_id)


@dataclass(frozen=True)
class MetaCell:
    family_id: str
    kappa: float
    nuisance_cell: str
    is_null: np.ndarray
    true_order: np.ndarray            # (N, L)
    nuisance_u: np.ndarray            # (N, L)
    driver: np.ndarray               # (N, D_H) EVAL-ONLY
    context_features: np.ndarray     # (N, D_CTX)
    item_features: np.ndarray        # (N, L, D_ITEM)
    item_classes: np.ndarray         # (N, L) class id per item (observable via item_features)
    observed_covariates: np.ndarray | None
    multiset_id: np.ndarray          # (N,) which fixed multiset each sequence uses
    future_events: np.ndarray        # (N, L) realized rank
    observable_allowlist: tuple[str, ...] = ()
    support_status: str = "SUPPORTED"

    def context_data(self) -> dict[str, Any]:
        full = {"context_features": self.context_features, "item_features": self.item_features,
                "context_timestamps": None, "observed_covariates": self.observed_covariates}
        return {k: full.get(k) for k in self.observable_allowlist}

    def context_view(self):
        return context_view(self.context_data())

    def future_view(self):
        return future_view({"future_multiset": self.item_classes, "future_events": self.future_events,
                            "future_timestamps": None})


def _repeated_multiset_support_ok(multiset_id: np.ndarray, floor: int) -> bool:
    """Support = every USED fixed multiset has >= floor eligible sequences (repeated-multiset clusters),
    not raw N (Pi #2)."""
    counts = np.bincount(multiset_id, minlength=len(_MULTISET_BANK))
    used = counts[counts > 0]
    return used.size > 0 and int(used.min()) >= floor


def generate_meta_cell(family_id: str, kappa: float, nuisance_cell: str, n_sequences: int, *,
                       seed: int, null_weight: float = 0.5, support_floor: int = 0) -> MetaCell:
    if family_id not in (*TRAIN_FAMILIES, *HELDOUT_FAMILIES):
        raise KeyError(family_id)
    if nuisance_cell not in ("orthogonal", "correlated_leak"):
        raise ValueError(nuisance_cell)
    inv = INVARIANT
    rng = np.random.default_rng(seed)
    n, L = n_sequences, L_ITEMS
    driver, covar = _driver(family_id, n, rng)
    x_ctx = driver @ inv.W_ctx.T + CTX_NOISE * rng.standard_normal((n, D_CTX))   # SHARED linear map (clean recovery)
    # fixed class multisets from the bank (repeated across sequences), permuted per sequence
    ms_id = rng.integers(0, len(_MULTISET_BANK), size=n)
    bank = np.array(_MULTISET_BANK)                                # (B, L)
    classes = bank[ms_id]
    perm = np.argsort(rng.standard_normal((n, L)), axis=1)
    classes = np.take_along_axis(classes, perm, axis=1)           # (N, L)
    item_feats = inv.class_embed[classes] + 0.3 * rng.standard_normal((n, L, D_ITEM))  # class observable
    raw = np.einsum("nd,de,nle->nl", driver, inv.M, item_feats)   # SHARED bilinear coupling
    coupling = COUPLING_SCALE * raw / (raw.std() + 1e-9)          # unit-normalized so κ controls signal
    cm = inv.class_means[classes]
    is_null = rng.random(n) < null_weight
    ctx_term = np.where(is_null[:, None], 0.0, kappa * coupling)
    s = cm + ctx_term + ORDER_NOISE * rng.standard_normal((n, L))
    if nuisance_cell == "orthogonal":
        u = rng.standard_normal((n, L))
    else:
        sz = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        u = 0.6 * sz + np.sqrt(1 - 0.36) * rng.standard_normal((n, L))
    allow = ("context_features", "item_features", "observed_covariates") if covar is not None \
        else ("context_features", "item_features")
    status = "SUPPORTED" if (support_floor <= 0 or _repeated_multiset_support_ok(ms_id, support_floor)) \
        else "SUPPORT_STARVED"
    return MetaCell(family_id, float(kappa), nuisance_cell, is_null, s, u, driver, x_ctx, item_feats,
                    classes, covar, ms_id, np.argsort(np.argsort(s, 1), 1), allow, status)
