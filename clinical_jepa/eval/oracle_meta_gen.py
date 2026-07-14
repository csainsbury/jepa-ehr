"""Shared-invariant META-FAMILY generator — CANONICAL (Pi keystone GO-WITH-CHANGES).

ONE global frozen invariant shared by every family; families differ ONLY in the DRIVER law. A recipe
trained on the TRAIN families transfers unchanged to the held-out families.

Pi keystone corrections folded in:
  * FROZEN normalization (correction #2): the coupling scale is a TRAIN-derived population constant
    included in the hash — a held-out cell can NOT self-normalize away its own driver-law shift.
  * FULL executable hash (#4): the canonical arrays + every literal constant + code version are hashed,
    not lossy signatures.
  * Heavy-tail naming (#2-rename): the off-grid held-out family is ``E_offgrid_heavytail`` (unseen
    driver DISTRIBUTION + off-grid κ), NOT a nonlinear map. The shared maps are LINEAR.
  * Support floor (#5): certification uses ``ORDER_SUPPORT_FLOOR=500`` and per-multiset counts.
  * Planted SHORTCUT channel (#3): a per-item item-feature channel that leaks the item's order on the
    TRAIN families but is pure noise on the held-out families — a memorizer that leans on it fails to
    transfer, while the invariant learner ignores it (uses only the legitimate item content).

Exact content prior π0 conditioning on the content the recipe receives (positive-cell residual includes
the integrated coupling variance) lives in ``oracle_meta_refs`` and is MC-validated there. Fully
synthetic / safe-public.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clinical_jepa.eval.oracle_contracts import canonical_hash, context_view, future_view

GENERATOR_VERSION = "clinical-jepa-oracle-meta-gen-v4"

D_H = 6
D_CTX = 12
D_ITEM = 5                # 0..D_ITEM-1 legitimate item content; item dim D_ITEM is a shortcut channel
L_ITEMS = 8
N_CLASSES = 6
CTX_NOISE = 0.2
ORDER_NOISE = 0.35
COUPLING_SCALE = 1.6
CLASS_MEAN_SPAN = 0.6
STUDENT_T_DF = 4.0
K_STATES = 6
GLOBAL_INVARIANT_SEED = 0x0180AC1E
SHORTCUT_STRENGTH = 2.0   # per-item train-family order leak strength on the item shortcut channel
ZERO_GAP_RATE = 0.35      # base fraction of adjacent items sharing a timestamp (Δt=0 multiplicity)
HISTORY_GAP_DECAY = 0.5   # T_realized_history accumulated-prior-gap dependence (bounded, <1) —
                          # a history-dependent gap process, NOT a self-exciting event intensity

# certification κ discipline: fit/dev-select on the TRAIN grid ONLY; held-out endpoints + κmid are OC-only.
KAPPA_TRAIN_GRID = (0.0, 0.10, 0.30, 0.50, 0.75)
KAPPA_HELDOUT_ENDPOINTS = (0.15, 0.60)
KAPPA_MID = 0.35

TRAIN_FAMILIES = ("T_latent_factor", "T_hmm_markov", "T_realized_history")
HELDOUT_FAMILIES = ("E_no_h_exogenous", "E_offgrid_heavytail")
NO_H_FAMILIES = ("E_no_h_exogenous",)

_MULTISET_BANK: tuple[tuple[int, ...], ...] = (
    (0, 0, 1, 2, 3, 3, 4, 5),
    (1, 1, 1, 2, 2, 4, 5, 5),
    (0, 1, 2, 3, 4, 5, 0, 2),
    (2, 2, 3, 3, 3, 4, 4, 5),
    (0, 0, 0, 1, 4, 4, 5, 5),
)


@dataclass(frozen=True)
class _Invariant:
    W_ctx: np.ndarray
    M: np.ndarray
    class_means: np.ndarray
    class_embed: np.ndarray
    A: np.ndarray                 # linear driver-recovery from context
    coupling_norm: float          # FROZEN train-derived coupling scale (NOT per-cell)


def _raw_coupling(driver, M, item):
    return np.einsum("nd,de,nle->nl", driver, M, item)


def _shared_invariant() -> _Invariant:
    rng = np.random.default_rng(GLOBAL_INVARIANT_SEED)
    W = rng.standard_normal((D_CTX, D_H))
    M = rng.standard_normal((D_H, D_ITEM))
    class_means = np.linspace(-CLASS_MEAN_SPAN, CLASS_MEAN_SPAN, N_CLASSES)
    class_embed = rng.standard_normal((N_CLASSES, D_ITEM))
    prec = (W.T @ W) / (CTX_NOISE ** 2) + np.eye(D_H)
    A = np.linalg.solve(prec, W.T) / (CTX_NOISE ** 2)
    # FROZEN coupling normalization: population std of the raw coupling under a REFERENCE Gaussian
    # train driver (fixed seed), NOT the sealed cell being scored (Pi #2).
    ref = np.random.default_rng(GLOBAL_INVARIANT_SEED + 7)       # FIXED reference draw (not a scored cell)
    d_ref = ref.standard_normal((20000, D_H))
    cls = np.array(_MULTISET_BANK)[ref.integers(0, len(_MULTISET_BANK), 20000)]
    it_ref = class_embed[cls]                                    # EXACT item content (no residual)
    coupling_norm = float(_raw_coupling(d_ref, M, it_ref).std())
    return _Invariant(W, M, class_means, class_embed, A, coupling_norm)


INVARIANT = _shared_invariant()


def _array_id(a: np.ndarray) -> dict:
    """Full-byte array identity: exact contiguous bytes + dtype + shape (Pi #8 — not rounded)."""
    ac = np.ascontiguousarray(a)
    return {"bytes": ac.tobytes().hex(), "dtype": str(ac.dtype), "shape": list(ac.shape)}


def _scalar_id(x: float) -> str:
    return np.float64(x).tobytes().hex()                     # exact scalar encoding


def invariant_hash() -> str:
    """Hash the FULL executable invariant + every literal constant + code version (Pi #4/#8) — exact
    array/scalar BYTES (dtype, shape, contiguous bytes), so any sub-rounding change moves the hash."""
    inv = INVARIANT
    return canonical_hash({
        "version": GENERATOR_VERSION, "seed": GLOBAL_INVARIANT_SEED,
        "D_H": D_H, "D_CTX": D_CTX, "D_ITEM": D_ITEM, "L": L_ITEMS, "N_CLASSES": N_CLASSES,
        "ctx_noise": _scalar_id(CTX_NOISE), "order_noise": _scalar_id(ORDER_NOISE),
        "coupling_scale": _scalar_id(COUPLING_SCALE), "class_mean_span": _scalar_id(CLASS_MEAN_SPAN),
        "student_t_df": _scalar_id(STUDENT_T_DF), "k_states": K_STATES,
        "shortcut_strength": _scalar_id(SHORTCUT_STRENGTH), "coupling_norm": _scalar_id(inv.coupling_norm),
        "zero_gap_rate": _scalar_id(ZERO_GAP_RATE), "history_gap_decay": _scalar_id(HISTORY_GAP_DECAY),
        "kappa_train_grid": [_scalar_id(k) for k in KAPPA_TRAIN_GRID],
        "kappa_heldout": [_scalar_id(k) for k in KAPPA_HELDOUT_ENDPOINTS], "kappa_mid": _scalar_id(KAPPA_MID),
        "train_families": TRAIN_FAMILIES, "heldout_families": HELDOUT_FAMILIES,
        "multiset_bank": _MULTISET_BANK,
        "W_ctx": _array_id(inv.W_ctx), "M": _array_id(inv.M),
        "class_means": _array_id(inv.class_means), "class_embed": _array_id(inv.class_embed),
    })


def _driver(family_id: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    if family_id == "T_latent_factor":
        return rng.standard_normal((n, D_H)), None
    if family_id == "E_offgrid_heavytail":
        return rng.standard_t(STUDENT_T_DF, size=(n, D_H)), None      # heavier-tailed driver DISTRIBUTION
    if family_id == "T_hmm_markov":
        mrng = np.random.default_rng(GLOBAL_INVARIANT_SEED ^ 0xA11)
        P = mrng.random((K_STATES, K_STATES)) + 0.1
        P /= P.sum(1, keepdims=True)
        state = rng.integers(0, K_STATES, size=n)
        for _ in range(8):
            u = rng.random(n)
            cdf = np.cumsum(P[state], axis=1)
            state = (u[:, None] < cdf).argmax(1)
        return np.eye(D_H)[state], None
    if family_id == "T_realized_history":
        return rng.standard_normal((n, D_H)), None                   # realized-prefix summary
    if family_id == "E_no_h_exogenous":
        z = rng.standard_normal((n, D_H))
        return z, z                                                  # EXOGENOUS + OBSERVED
    raise KeyError(family_id)


@dataclass(frozen=True)
class MetaCell:
    family_id: str
    kappa: float
    nuisance_cell: str
    is_null: np.ndarray
    true_order: np.ndarray
    nuisance_u: np.ndarray
    driver: np.ndarray
    context_features: np.ndarray     # (N, D_CTX)
    item_features: np.ndarray        # (N, L, D_ITEM+1): legitimate class content + a per-item shortcut
    item_classes: np.ndarray
    observed_covariates: np.ndarray | None
    multiset_id: np.ndarray
    future_events: np.ndarray
    future_timestamps: np.ndarray | None = None    # (N, L) nondecreasing; Δt=0 within a cluster
    cluster_ids: np.ndarray | None = None          # (N, L) timestamp-cluster index per item
    multiplicity: np.ndarray | None = None         # (N, L) size of each item's timestamp cluster
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
                            "future_timestamps": self.future_timestamps})


def _marked_timing(family_id: str, driver: np.ndarray, rng: np.random.Generator):
    """Mechanism-driven marked-cluster timing: adjacent items share a timestamp (Δt=0 multiplicity)
    with a driver-modulated probability; otherwise a strictly-positive inter-cluster gap whose rate is
    driven by the family's driver summary (so the driver law measurably shapes timing). T_realized_
    history uses a bounded HISTORY-DEPENDENT gap process (accumulated prior gaps), NOT a self-
    exciting event intensity. Returns (timestamps, cluster_ids, multiplicity)."""
    n = driver.shape[0]
    summ = driver.mean(axis=1)                                    # family-specific driver summary
    rate_scale = np.exp(0.4 * (summ - summ.mean())) + 0.2         # >0, driver-modulated inter-cluster rate
    zero_p = np.clip(ZERO_GAP_RATE + 0.1 * np.tanh(summ), 0.02, 0.9)
    ts = np.zeros((n, L_ITEMS)); cid = np.zeros((n, L_ITEMS), dtype=int)
    hist = np.zeros(n)
    for j in range(1, L_ITEMS):
        same = rng.random(n) < zero_p                            # Δt=0 (same cluster)
        gap = rng.exponential(1.0, size=n) * (rate_scale + hist) + 1e-3
        if family_id == "T_realized_history":
            hist = HISTORY_GAP_DECAY * (hist + gap)               # bounded history-dependent gap (<1)
        step = np.where(same, 0.0, gap)
        ts[:, j] = ts[:, j - 1] + step
        cid[:, j] = cid[:, j - 1] + (step > 0).astype(int)
    mult = np.zeros((n, L_ITEMS), dtype=int)
    for i in range(n):
        counts = np.bincount(cid[i])
        mult[i] = counts[cid[i]]
    return ts, cid, mult


def _repeated_multiset_support_ok(multiset_id: np.ndarray, floor: int) -> bool:
    counts = np.bincount(multiset_id, minlength=len(_MULTISET_BANK))
    used = counts[counts > 0]
    return used.size > 0 and int(used.min()) >= floor


def multiset_cluster_counts(cell: MetaCell) -> dict[int, int]:
    counts = np.bincount(cell.multiset_id, minlength=len(_MULTISET_BANK))
    return {i: int(c) for i, c in enumerate(counts) if c > 0}


def generate_meta_cell(family_id: str, kappa: float, nuisance_cell: str, n_sequences: int, *,
                       seed: int, null_weight: float = 0.5, support_floor: int = 0) -> MetaCell:
    if family_id not in (*TRAIN_FAMILIES, *HELDOUT_FAMILIES):
        raise KeyError(family_id)
    if nuisance_cell not in ("orthogonal", "correlated_leak"):
        raise ValueError(nuisance_cell)
    inv = INVARIANT
    is_train_family = family_id in TRAIN_FAMILIES
    rng = np.random.default_rng(seed)
    n, L = n_sequences, L_ITEMS
    driver, covar = _driver(family_id, n, rng)
    x_ctx = driver @ inv.W_ctx.T + CTX_NOISE * rng.standard_normal((n, D_CTX))    # SHARED LINEAR map
    ms_id = rng.integers(0, len(_MULTISET_BANK), size=n)
    classes = np.array(_MULTISET_BANK)[ms_id]
    perm = np.argsort(rng.standard_normal((n, L)), axis=1)
    classes = np.take_along_axis(classes, perm, axis=1)
    # EXACT item content: item features ARE the class embedding (no per-item residual). The content the
    # recipe sees is therefore EXACTLY the class, so R0/π* condition on the complete content channel and
    # the class-pair reference tables are Bayes-exact (Pi #2).
    legit_item = inv.class_embed[classes]
    coupling = COUPLING_SCALE * _raw_coupling(driver, inv.M, legit_item) / inv.coupling_norm  # FROZEN scale
    cm = inv.class_means[classes]
    is_null = rng.random(n) < null_weight
    ctx_term = np.where(is_null[:, None], 0.0, kappa * coupling)
    s = cm + ctx_term + ORDER_NOISE * rng.standard_normal((n, L))
    # PER-ITEM PRE-FUTURE SHORTCUT (item dim D_ITEM): a train-stable / held-out-broken spurious channel
    # built from the (pre-future) coupling — NOT from realized order/rank (Pi #2, label isolation). Train
    # families leak the coupling; held-out families put pure noise there.
    if is_train_family:
        cz = (coupling - coupling.mean(1, keepdims=True)) / (coupling.std(1, keepdims=True) + 1e-9)
        shortcut = SHORTCUT_STRENGTH * cz + 0.3 * rng.standard_normal((n, L))       # pre-future coupling leak
    else:
        shortcut = rng.standard_normal((n, L))                                      # pure noise on held-out
    item_feats = np.concatenate([legit_item, shortcut[:, :, None]], axis=2)         # (N, L, D_ITEM+1)
    if nuisance_cell == "orthogonal":
        u = rng.standard_normal((n, L))
    else:
        sz = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        u = 0.6 * sz + np.sqrt(1 - 0.36) * rng.standard_normal((n, L))
    allow = ("context_features", "item_features", "observed_covariates") if covar is not None \
        else ("context_features", "item_features")
    status = "SUPPORTED" if (support_floor <= 0 or _repeated_multiset_support_ok(ms_id, support_floor)) \
        else "SUPPORT_STARVED"
    ts, cid, mult = _marked_timing(family_id, driver, rng)
    return MetaCell(family_id, float(kappa), nuisance_cell, is_null, s, u, driver, x_ctx, item_feats,
                    classes, covar, ms_id, np.argsort(np.argsort(s, 1), 1),
                    future_timestamps=ts, cluster_ids=cid, multiplicity=mult,
                    observable_allowlist=allow, support_status=status)
