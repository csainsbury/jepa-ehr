"""LITERAL per-family generator (Pi 2nd-pass Phase 2).

Each family GENUINELY instantiates its declared structural difference with frozen, hashed numeric
parameters — not one convenience mechanism relabelled (the compact linear-Gaussian stays in
``oracle_generator`` as the named ``smoke_linear_gaussian`` fixture and is NOT used here). Families:

  * ``T_hmm_markov``       — row-stochastic K-state Markov chain + categorical emissions; order from an
                             h-conditioned precedence kernel.
  * ``T_realized_history`` — autoregressive realized-prefix state; stable Hawkes-like timing (branching
                             ratio < 1, frozen horizon/truncation).
  * ``T_latent_factor``    — linear-Gaussian latent factor.
  * ``E_no_h_exogenous``   — observable exogenous clock, NO hidden common cause.
  * ``E_offgrid_nonlinear``— Student-t driver + genuinely nonlinear frozen context/order maps.

Every family emits ONE shared discriminated ``LiteralCell``: a common observable+future+timing+label
core, tagged family-specific optional channels, and an explicit per-family observable-channel allowlist
used to build the recipe's ``ContextView``. Missingness is explicit (``None``), never zero-filled in a
way that would leak family identity. Family ID / kappa / null flag / hidden state / true mechanism
params are EVAL-ONLY. Timing carries Δt=0 multiplicity clusters and strictly-positive inter-cluster
gaps; token/class banks and exact repeated fixed multisets are materialized; a support-starved cell is
emitted on request. Fully synthetic, safe-public, seeded/reproducible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash, context_view, future_view
from clinical_jepa.eval.oracle_spec import NUISANCE_LEAK_RHO, get_family
from clinical_jepa.eval.rung2_contract import ORDER_SUPPORT_FLOOR

# frozen structural constants (hashed into every cell's mechanism_params_hash)
N_CLASSES = 6                 # six-class token/event bank (matches the calibration occupancy C=6)
L_ITEMS = 8                   # future items whose order is certified
D_CTX = 10
D_ITEM = 4
K_STATES = 6                  # HMM latent cardinality
HAWKES_BRANCHING = 0.5        # < 1 => stable self-excitation
HAWKES_HORIZON = 16           # frozen truncation horizon
STUDENT_T_DF = 4.0
ZERO_GAP_RATE = 0.35          # fraction of adjacent items sharing a timestamp (Δt=0 multiplicity)


def _family_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class LiteralCell:
    family_id: str
    kappa: float
    nuisance_cell: str
    # --- eval-only labels ---
    is_null: np.ndarray                 # (N,)
    true_order: np.ndarray              # (N, L) order-scores (higher => later)
    nuisance_u: np.ndarray              # (N, L)
    hidden_state: np.ndarray | None     # (N, d_h) or None (no-h family) — EVAL-ONLY
    # --- common observable core (context) ---
    context_features: np.ndarray        # (N, D_CTX)
    item_features: np.ndarray           # (N, L, D_ITEM)
    observed_covariates: np.ndarray | None   # exogenous z for no-h; else None
    # --- future (target-side; ceiling / target construction only, never a certification input) ---
    future_multiset: np.ndarray         # (N, L) class ids in [0, N_CLASSES)
    future_events: np.ndarray           # (N, L) realized-order RANK of each item (0=first) — target-side
    future_timestamps: np.ndarray       # (N, L) event times, nondecreasing, Δt=0 within a cluster
    cluster_ids: np.ndarray             # (N, L) timestamp-cluster index per item
    multiplicity: np.ndarray            # (N, L) cluster size for each item's cluster
    # --- family-specific tagged optional channels (observable) ---
    family_channels: dict[str, Any] = field(default_factory=dict)
    # --- provenance ---
    observable_allowlist: tuple[str, ...] = ()
    mechanism_params_hash: str = ""
    support_status: str = "SUPPORTED"

    def context_data(self) -> dict[str, Any]:
        """Assemble the ContextView payload from ONLY this family's allowlisted observable channels
        (missing channels are explicit None, never zero-filled to hide family identity)."""
        full = {
            "context_features": self.context_features,
            "item_features": self.item_features,
            "context_timestamps": self.future_timestamps if "context_timestamps" in self.observable_allowlist else None,
            "observed_covariates": self.observed_covariates,
        }
        return {k: full.get(k) for k in self.observable_allowlist}

    def context_view(self):
        return context_view(self.context_data())

    def future_view(self):
        """Target-side view (fit / ceiling only). Exposes the realized future ordering + multiset +
        timestamps — NOT the eval-only true_order label object and NOT family identity."""
        return future_view({"future_multiset": self.future_multiset,
                            "future_events": self.future_events,
                            "future_timestamps": self.future_timestamps})


# ----------------------------------------------------------------------------------------------
# timing: build Δt=0 multiplicity clusters with strictly-positive inter-cluster gaps.
# ----------------------------------------------------------------------------------------------
def _marked_cluster_timing(rng: np.random.Generator, n: int, rate_scale: np.ndarray) -> tuple:
    """Return (timestamps, cluster_ids, multiplicity) for n sequences of L items.
    Adjacent items share a timestamp (Δt=0 multiplicity) with prob ZERO_GAP_RATE; otherwise a
    strictly-positive gap (Exp) scaled per sequence by rate_scale (>0)."""
    ts = np.zeros((n, L_ITEMS)); cid = np.zeros((n, L_ITEMS), dtype=int)
    same = rng.random((n, L_ITEMS - 1)) < ZERO_GAP_RATE          # Δt=0 (same cluster)
    gaps = rng.exponential(1.0, size=(n, L_ITEMS - 1)) * rate_scale[:, None] + 1e-3  # strictly positive
    for j in range(1, L_ITEMS):
        step = np.where(same[:, j - 1], 0.0, gaps[:, j - 1])
        ts[:, j] = ts[:, j - 1] + step
        cid[:, j] = cid[:, j - 1] + (step > 0).astype(int)
    mult = np.zeros((n, L_ITEMS), dtype=int)
    for i in range(n):
        counts = np.bincount(cid[i])
        mult[i] = counts[cid[i]]
    return ts, cid, mult


def _nuisance(rng: np.random.Generator, s_true: np.ndarray, nuisance_cell: str) -> np.ndarray:
    if nuisance_cell == "orthogonal":
        return rng.standard_normal(s_true.shape)                 # ⟂ order
    sz = (s_true - s_true.mean(1, keepdims=True)) / (s_true.std(1, keepdims=True) + 1e-9)
    rho = NUISANCE_LEAK_RHO
    return rho * sz + float(np.sqrt(1.0 - rho ** 2)) * rng.standard_normal(s_true.shape)


def _finish(family_id, kappa, nuisance_cell, *, is_null, s_true, hidden, ctx, item_feats,
            covar, rng, allowlist, params, fam_channels, support_status) -> LiteralCell:
    future_multiset = rng.integers(0, N_CLASSES, size=(s_true.shape[0], L_ITEMS))
    future_events = np.argsort(np.argsort(s_true, axis=1), axis=1)      # realized-order rank per item
    rate_scale = np.exp(0.3 * (ctx[:, 0] - ctx[:, 0].mean())) + 0.2     # per-seq positive timing scale
    ts, cid, mult = _marked_cluster_timing(rng, s_true.shape[0], rate_scale)
    u = _nuisance(rng, s_true, nuisance_cell)
    phash = canonical_hash({"family": family_id, "params": params, "kappa": float(kappa),
                            "nuisance": nuisance_cell})
    return LiteralCell(
        family_id=family_id, kappa=float(kappa), nuisance_cell=nuisance_cell,
        is_null=is_null, true_order=s_true, nuisance_u=u, hidden_state=hidden,
        context_features=ctx, item_features=item_feats, observed_covariates=covar,
        future_multiset=future_multiset, future_events=future_events,
        future_timestamps=ts, cluster_ids=cid, multiplicity=mult,
        family_channels=fam_channels, observable_allowlist=allowlist,
        mechanism_params_hash=phash, support_status=support_status,
    )


# ----------------------------------------------------------------------------------------------
# per-family literal mechanisms
# ----------------------------------------------------------------------------------------------
def _hmm_markov(rng, mrng, n, kappa, is_null):
    """Discrete K-state Markov chain; categorical emissions; order from an h-conditioned kernel."""
    P = mrng.random((K_STATES, K_STATES)) + 0.1
    P /= P.sum(1, keepdims=True)                                 # row-stochastic
    W_emit = mrng.standard_normal((K_STATES, D_CTX))
    Q = mrng.standard_normal((K_STATES, D_ITEM))                 # per-state precedence kernel
    h = np.zeros((n, HAWKES_HORIZON), dtype=int)
    h[:, 0] = mrng.integers(0, K_STATES, size=n)
    for t in range(1, HAWKES_HORIZON):
        for k in range(K_STATES):
            m = h[:, t - 1] == k
            if m.any():
                h[m, t] = rng.choice(K_STATES, size=int(m.sum()), p=P[k])
    hstate = np.eye(K_STATES)[h[:, -1]]                          # terminal-state one-hot (d_h = K)
    ctx = hstate @ W_emit + 0.5 * rng.standard_normal((n, D_CTX))
    item = rng.standard_normal((n, L_ITEMS, D_ITEM))
    coupling = np.einsum("nk,kd,nld->nl", hstate, Q, item)
    s = np.where(is_null[:, None], rng.standard_normal((n, L_ITEMS)),
                 kappa * coupling + np.sqrt(max(1e-6, 1 - kappa ** 2)) * rng.standard_normal((n, L_ITEMS)))
    return s, hstate, ctx, item, None, ("context_features", "item_features"), {"transition_rows": K_STATES}, {}


def _realized_history(rng, mrng, n, kappa, is_null):
    """State = running summary of the realized prefix; Hawkes-like stable timing (branching<1)."""
    W = mrng.standard_normal((D_CTX, D_ITEM))
    prefix = rng.standard_normal((n, D_ITEM))                    # realized-prefix summary (observable)
    ctx = prefix @ W.T + 0.5 * rng.standard_normal((n, D_CTX))
    item = rng.standard_normal((n, L_ITEMS, D_ITEM))
    coupling = np.einsum("nd,nld->nl", prefix, item)            # order depends on realized prefix
    s = np.where(is_null[:, None], rng.standard_normal((n, L_ITEMS)),
                 kappa * coupling + np.sqrt(max(1e-6, 1 - kappa ** 2)) * rng.standard_normal((n, L_ITEMS)))
    # Hawkes-like excitation is expressed via the timing rate scale (branching<1) tagged as observable
    excit = HAWKES_BRANCHING * np.abs(prefix).sum(1)
    return (s, None, ctx, item, prefix, ("context_features", "item_features", "observed_covariates"),
            {"branching": HAWKES_BRANCHING, "horizon": HAWKES_HORIZON}, {"excitation": excit})


def _latent_factor(rng, mrng, n, kappa, is_null):
    """Linear-Gaussian latent factor (literal, but simple)."""
    d = 8
    W = mrng.standard_normal((D_CTX, d)); M = mrng.standard_normal((d, D_ITEM))
    h = rng.standard_normal((n, d))
    ctx = h @ W.T + 0.5 * rng.standard_normal((n, D_CTX))
    item = rng.standard_normal((n, L_ITEMS, D_ITEM))
    coupling = np.einsum("nd,de,nle->nl", h, M, item)
    s = np.where(is_null[:, None], rng.standard_normal((n, L_ITEMS)),
                 kappa * coupling + np.sqrt(max(1e-6, 1 - kappa ** 2)) * rng.standard_normal((n, L_ITEMS)))
    return s, h, ctx, item, None, ("context_features", "item_features"), {"d_factor": d}, {}


def _no_h_exogenous(rng, mrng, n, kappa, is_null):
    """Observable exogenous clock z; NO hidden common cause. Order is a function of z (in context)."""
    d = 6
    W = mrng.standard_normal((D_CTX, d)); M = mrng.standard_normal((d, D_ITEM))
    z = rng.standard_normal((n, d))                             # EXOGENOUS + OBSERVED (covariate)
    ctx = z @ W.T + 0.5 * rng.standard_normal((n, D_CTX))
    item = rng.standard_normal((n, L_ITEMS, D_ITEM))
    coupling = np.einsum("nd,de,nle->nl", z, M, item)
    s = np.where(is_null[:, None], rng.standard_normal((n, L_ITEMS)),
                 kappa * coupling + np.sqrt(max(1e-6, 1 - kappa ** 2)) * rng.standard_normal((n, L_ITEMS)))
    # hidden_state is None (no h); z is exposed as observed_covariates (part of context)
    return (s, None, ctx, item, z, ("context_features", "item_features", "observed_covariates"),
            {"exogenous_dim": d}, {})


def _offgrid_nonlinear(rng, mrng, n, kappa, is_null):
    """Student-t driver + genuinely NONLINEAR frozen context/order maps."""
    d = 6
    W1 = mrng.standard_normal((16, d)); W2 = mrng.standard_normal((D_CTX, 16))
    M = mrng.standard_normal((d, D_ITEM))
    h = rng.standard_t(STUDENT_T_DF, size=(n, d))              # heavier tails, unseen in train
    ctx = np.tanh(h @ W1.T) @ W2.T + 0.5 * rng.standard_normal((n, D_CTX))   # nonlinear context map
    item = rng.standard_normal((n, L_ITEMS, D_ITEM))
    coupling = np.tanh(np.einsum("nd,de,nle->nl", h, M, item))              # nonlinear order map
    s = np.where(is_null[:, None], rng.standard_normal((n, L_ITEMS)),
                 kappa * coupling + np.sqrt(max(1e-6, 1 - kappa ** 2)) * rng.standard_normal((n, L_ITEMS)))
    return s, h, ctx, item, None, ("context_features", "item_features"), {"df": STUDENT_T_DF}, {}


_MECHANISMS = {
    "T_hmm_markov": _hmm_markov,
    "T_realized_history": _realized_history,
    "T_latent_factor": _latent_factor,
    "E_no_h_exogenous": _no_h_exogenous,
    "E_offgrid_nonlinear": _offgrid_nonlinear,
}


def generate_literal_cell(family_id: str, kappa: float, nuisance_cell: str, n_sequences: int,
                          *, seed: int, null_weight: float | None = None,
                          support_starved: bool = False) -> LiteralCell:
    """Generate one literal (family, kappa, nuisance-cell) cell. ``support_starved`` emits fewer than
    ORDER_SUPPORT_FLOOR sequences and tags SUPPORT_STARVED so downstream scoring returns NOT_EVALUABLE."""
    if family_id not in _MECHANISMS:
        raise KeyError(f"no literal mechanism for {family_id!r}")
    if nuisance_cell not in ("orthogonal", "correlated_leak"):
        raise ValueError(f"unknown nuisance cell: {nuisance_cell!r}")
    fam = get_family(family_id)
    nw = fam.null_mixture_weight if null_weight is None else null_weight
    n = max(1, min(ORDER_SUPPORT_FLOOR // 4, n_sequences)) if support_starved else n_sequences
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng(_family_seed("literal_mechanism", family_id))   # frozen per-family params
    is_null = rng.random(n) < nw
    s, hidden, ctx, item, covar, allowlist, params, fam_channels = _MECHANISMS[family_id](
        rng, mrng, n, float(kappa), is_null)
    status = "SUPPORT_STARVED" if (support_starved or n < ORDER_SUPPORT_FLOOR) else "SUPPORTED"
    return _finish(family_id, kappa, nuisance_cell, is_null=is_null, s_true=s, hidden=hidden,
                   ctx=ctx, item_feats=item, covar=covar, rng=rng, allowlist=allowlist,
                   params=params, fam_channels=fam_channels, support_status=status)
