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
N_CLASSES = 5                 # five-class token/event bank (matches the calibration occupancy C=5 —
                              # the substrate's natural clinical content families)
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
    calibration_adapter_hash: str | None = None   # None => uncalibrated default cell (Pi REVISE#4 #3)

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
def _clusters_from_gaps(same: np.ndarray, gaps: np.ndarray) -> tuple:
    """Assemble (timestamps, cluster_ids, multiplicity) from a per-adjacency same-cluster mask + gaps."""
    n = same.shape[0]
    ts = np.zeros((n, L_ITEMS)); cid = np.zeros((n, L_ITEMS), dtype=int)
    for j in range(1, L_ITEMS):
        step = np.where(same[:, j - 1], 0.0, gaps[:, j - 1])
        ts[:, j] = ts[:, j - 1] + step
        cid[:, j] = cid[:, j - 1] + (step > 0).astype(int)
    mult = np.zeros((n, L_ITEMS), dtype=int)
    for i in range(n):
        counts = np.bincount(cid[i])
        mult[i] = counts[cid[i]]
    return ts, cid, mult


def _marked_cluster_timing(rng: np.random.Generator, n: int, rate_scale: np.ndarray) -> tuple:
    """Return (timestamps, cluster_ids, multiplicity) for n sequences of L items. Adjacent items share a
    timestamp (Δt=0 multiplicity) with prob ZERO_GAP_RATE; otherwise a strictly-positive gap (Exp) scaled
    per sequence by rate_scale (>0). FROZEN mechanism — the calibration timing layer is applied SEPARATELY
    and post-hoc (see ``_apply_calibration_layer``), never interleaved with this RNG stream."""
    same = rng.random((n, L_ITEMS - 1)) < ZERO_GAP_RATE         # Δt=0 (same cluster)
    gaps = rng.exponential(1.0, size=(n, L_ITEMS - 1)) * rate_scale[:, None] + 1e-3  # strictly positive
    return _clusters_from_gaps(same, gaps)


CALIB_ADAPTER_VERSION = "calibration_layer_v2"     # v2: newly-positive gaps drawn from the pooled ECDF


def _normalize_knobs(knobs: dict) -> dict:
    """Fill defaults ONCE and clip to hard bounds — the runner executes exactly this normalized payload, so
    the adapter identity binds the executed values (Pi REVISE#4->5 #3). Default zero_gap_bias is the native
    ZERO_GAP_RATE, matching execution."""
    return {"token_freq_temperature": float(np.clip(knobs.get("token_freq_temperature", 1.0), 0.5, 2.0)),
            "zero_gap_bias": float(np.clip(knobs.get("zero_gap_bias", ZERO_GAP_RATE), 0.0, 0.9)),
            "timing_rate_scale": float(np.clip(knobs.get("timing_rate_scale", 1.0), 0.5, 2.0)),
            "gap_dispersion": float(np.clip(knobs.get("gap_dispersion", 1.0), 0.5, 2.0))}


def _context_hash(context: dict) -> str:
    return canonical_hash({"pooled_class_prior": [float(x) for x in context["pooled_class_prior"]],
                           "global_gap_median": float(context["global_gap_median"]),
                           "pooled_positive_gap_ecdf": [[float(g), float(c)]
                                                        for g, c in context["pooled_positive_gap_ecdf"]]})


def calibration_adapter_hash(knobs: dict, context: dict, *, family_id: str = "", kappa: float = 0.0,
                             nuisance_cell: str = "", cell_seed: int = 0, source_profile: str = "") -> str:
    """Exact identity of an executed adapter transform: the NORMALIZED knobs (full canonical precision),
    the full pooled-context hash, and the cell/source identity — so different executed transforms never
    share a hash and the domain-separated RNG cannot correlate across cell identities (Pi #3)."""
    return canonical_hash({"adapter": CALIB_ADAPTER_VERSION, "knobs": _normalize_knobs(knobs),
                           "context_hash": _context_hash(context), "family_id": family_id,
                           "kappa": float(kappa), "nuisance_cell": nuisance_cell,
                           "cell_seed": int(cell_seed), "source_profile": source_profile})


def _sample_positive_gaps(ecdf, size, arng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draw of strictly-positive gaps from the bound pooled positive-gap ECDF (Pi #2): every
    newly-positive adjacency gets a REAL positive gap from the declared distribution, not a median-clip of a
    within-cluster zero."""
    supp = np.asarray([g for g, _ in ecdf], float); cdf = np.asarray([c for _, c in ecdf], float)
    if supp.size == 0:
        return np.full(size, 1e-3)
    idx = np.clip(np.searchsorted(cdf, arng.random(size), side="left"), 0, supp.size - 1)
    return supp[idx]


def _apply_calibration_layer(future_multiset: np.ndarray, ts: np.ndarray, knobs: dict, context: dict,
                             arng: np.random.Generator) -> tuple:
    """POST-HOC calibration layer on an ALREADY-GENERATED default cell with a DEDICATED domain-separated
    RNG — touches ONLY class labels + timestamps (nuisance/order/context/item unchanged, Pi #3). Newly
    positive adjacencies draw a REAL positive gap from the pooled positive-gap ECDF, then apply the
    median-anchored dispersion/rate transform; at the native profile (zero_gap_bias=ZERO_GAP_RATE,
    rate=disp=1) it reproduces the BASE marginals so self-recovery passes (Pi #2)."""
    k = _normalize_knobs(knobs)
    prior = np.asarray(context["pooled_class_prior"], float)
    p = prior ** (1.0 / k["token_freq_temperature"]); p = p / p.sum() if p.sum() > 0 else prior
    fm2 = arng.choice(N_CLASSES, size=future_multiset.shape, p=p)      # tempered POOLED-prior class relabel
    med = float(context["global_gap_median"])
    n = ts.shape[0]
    same = arng.random((n, L_ITEMS - 1)) < k["zero_gap_bias"]
    raw = _sample_positive_gaps(context["pooled_positive_gap_ecdf"], (n, L_ITEMS - 1), arng)
    gaps2 = np.maximum((med + (raw - med) * k["gap_dispersion"]) / max(1e-6, k["timing_rate_scale"]), 1e-6)
    ts2, cid2, mult2 = _clusters_from_gaps(same, gaps2)
    return fm2, ts2, cid2, mult2


def _nuisance(rng: np.random.Generator, s_true: np.ndarray, nuisance_cell: str) -> np.ndarray:
    if nuisance_cell == "orthogonal":
        return rng.standard_normal(s_true.shape)                 # ⟂ order
    sz = (s_true - s_true.mean(1, keepdims=True)) / (s_true.std(1, keepdims=True) + 1e-9)
    rho = NUISANCE_LEAK_RHO
    return rho * sz + float(np.sqrt(1.0 - rho ** 2)) * rng.standard_normal(s_true.shape)


def _finish(family_id, kappa, nuisance_cell, *, is_null, s_true, hidden, ctx, item_feats,
            covar, rng, allowlist, params, fam_channels, support_status,
            calib_knobs: dict | None = None, calib_context: dict | None = None,
            cell_seed: int | None = None, calib_source_profile: str = "") -> LiteralCell:
    # --- FROZEN default cell (RNG stream identical to the None path; never perturbed by calibration) ---
    future_multiset = rng.integers(0, N_CLASSES, size=(s_true.shape[0], L_ITEMS))
    future_events = np.argsort(np.argsort(s_true, axis=1), axis=1)      # realized-order rank per item
    rate_scale = np.exp(0.3 * (ctx[:, 0] - ctx[:, 0].mean())) + 0.2     # per-seq positive timing scale
    ts, cid, mult = _marked_cluster_timing(rng, s_true.shape[0], rate_scale)
    u = _nuisance(rng, s_true, nuisance_cell)                           # nuisance from the DEFAULT stream
    # --- POST-HOC calibration layer (Pi REVISE#4 #3/#4): dedicated domain-separated RNG; touches only
    #     class labels + timestamps; order/nuisance/context/item/covariates stay the default cell's. ---
    adapter_hash = None
    if calib_knobs is not None:
        if calib_context is None or "pooled_positive_gap_ecdf" not in calib_context:
            raise ValueError("calib_knobs requires calib_context with the pooled positive-gap ECDF")
        adapter_hash = calibration_adapter_hash(calib_knobs, calib_context, family_id=family_id,
                                                kappa=kappa, nuisance_cell=nuisance_cell,
                                                cell_seed=int(cell_seed or 0),
                                                source_profile=calib_source_profile)
        # domain-separated RNG keyed by the FULL adapter identity — cannot correlate across cells (Pi #3)
        arng = np.random.default_rng(int.from_bytes(hashlib.sha256(adapter_hash.encode()).digest()[:8], "big"))
        future_multiset, ts, cid, mult = _apply_calibration_layer(
            future_multiset, ts, calib_knobs, calib_context, arng)
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
        calibration_adapter_hash=adapter_hash,
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
                          support_starved: bool = False, calib_knobs: dict | None = None,
                          calib_context: dict | None = None, calib_source_profile: str = "") -> LiteralCell:
    """Generate one literal (family, kappa, nuisance-cell) cell. ``support_starved`` emits fewer than
    ORDER_SUPPORT_FLOOR sequences and tags SUPPORT_STARVED so downstream scoring returns NOT_EVALUABLE.
    ``calib_knobs`` (default None → frozen mechanism, byte-identical) applies the FROZEN post-hoc
    calibration adapter — the single route by which fitted knobs alter generated cells. The default cell is
    generated first (RNG stream untouched) and the adapter runs with a DEDICATED domain-separated RNG,
    touching ONLY class labels + timestamps; it needs ``calib_context`` (pooled class prior + global gap
    median). The order mechanism / nuisance / invariant identity are unchanged (Pi REVISE#4 #3/#4)."""
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
                   params=params, fam_channels=fam_channels, support_status=status,
                   calib_knobs=calib_knobs, calib_context=calib_context, cell_seed=seed,
                   calib_source_profile=calib_source_profile)
