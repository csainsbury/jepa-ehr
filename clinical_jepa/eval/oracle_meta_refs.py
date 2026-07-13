"""Corrected exact R0 + E-O1 estimator (Pi keystone corrections #3/#6).

R0 (the content prior available to a content-only predictor) is NOT the null-only class formula. On a
positive cell the residual integrates the random context coupling, and R0 must condition on the content
channel (class) for the actual family/κ. Here R0 is computed by MONTE-CARLO integration over the
family's driver law + item-feature noise + order noise, giving a per-(family, κ) class-pair table
P(a≺b | class_a, class_b). It is validated against a higher-precision MC (``r0_table_mc_error``).

E-O1 is the beyond-R0 Brier skill, aggregated with the PROPER pooled estimator
``1 − Σ Brier_recipe / Σ Brier_R0`` (bootstrap over sequences) — NOT a per-sequence ratio average that
overweights sequences with tiny baseline Brier (Pi #6).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from clinical_jepa.eval.oracle_meta_gen import (
    COUPLING_SCALE, INVARIANT, N_CLASSES, ORDER_NOISE, _driver, _raw_coupling,
)

EO1_TIE_ATOL = 1e-9


def pairwise_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    diff = scores[:, None, :] - scores[:, :, None]
    return 1.0 / (1.0 + np.exp(-diff / max(1e-6, temperature)))


@lru_cache(maxsize=64)
def r0_table(family_id: str, kappa: float, n_mc: int = 60000, seed: int = 12345) -> tuple:
    """P(a≺b | class_a, class_b) for this family/κ by MC over driver-law + item-noise + order-noise.
    Returns an (N_CLASSES, N_CLASSES) tuple-of-tuples (hashable for caching)."""
    inv = INVARIANT
    rng = np.random.default_rng(seed)
    driver, _ = _driver(family_id, n_mc, rng)                    # (n_mc, D_H) family-law driver
    tab = np.full((N_CLASSES, N_CLASSES), 0.5)
    # one item per class per sample; s_c = cm[c] + κ·coupling_c + order noise (shared driver within a draw)
    cm = inv.class_means
    items = {}
    for c in range(N_CLASSES):
        feat = inv.class_embed[c] + 0.3 * rng.standard_normal((n_mc, inv.class_embed.shape[1]))
        coup = COUPLING_SCALE * np.einsum("nd,de,ne->n", driver, inv.M, feat) / inv.coupling_norm
        items[c] = cm[c] + kappa * coup + ORDER_NOISE * rng.standard_normal(n_mc)
    for a in range(N_CLASSES):
        for b in range(N_CLASSES):
            if a != b:
                tab[a, b] = float(np.mean(items[a] < items[b]))
    return tuple(map(tuple, tab))


def r0_pairwise(family_id: str, kappa: float, class_ids: np.ndarray) -> np.ndarray:
    """(N, L, L) content-prior P(a≺b) from the (family,κ) class-pair table indexed by item classes."""
    tab = np.array(r0_table(family_id, float(kappa)))
    ca = class_ids[:, :, None]; cb = class_ids[:, None, :]
    return tab[ca, cb]


def r0_table_mc_error(family_id: str, kappa: float) -> float:
    """Max |coarse-MC − fine-MC| over the class-pair table — the R0-vs-high-precision-MC validation."""
    coarse = np.array(r0_table(family_id, float(kappa), n_mc=60000, seed=12345))
    fine = np.array(r0_table(family_id, float(kappa), n_mc=400000, seed=999))
    return float(np.abs(coarse - fine).max())


def per_sequence_briers(probs: np.ndarray, true_order: np.ndarray, r0_probs: np.ndarray):
    """Per-sequence summed Brier for recipe and R0 over eligible non-tied pairs, + eligible pair count."""
    n, L, _ = probs.shape
    iu, ju = np.triu_indices(L, k=1)
    b_rec = np.zeros(n); b_r0 = np.zeros(n); npair = np.zeros(n, dtype=int)
    for s in range(n):
        to = true_order[s]
        elig = np.abs(to[iu] - to[ju]) > EO1_TIE_ATOL
        if not elig.any():
            continue
        y = (to[iu] < to[ju]).astype(float)[elig]
        b_rec[s] = np.sum((probs[s, iu, ju][elig] - y) ** 2)
        b_r0[s] = np.sum((r0_probs[s, iu, ju][elig] - y) ** 2)
        npair[s] = int(elig.sum())
    return b_rec, b_r0, npair


def pooled_eo1_skill(b_rec: np.ndarray, b_r0: np.ndarray, npair: np.ndarray, *,
                     n_boot: int = 1000, base_seed: int = 0, alpha: float = 0.05):
    """PROPER pooled skill 1 − Σ Brier_recipe / Σ Brier_R0 with a sequence-clustered bootstrap CI.
    Returns (point, lower_ci, upper_ci, n_sequences)."""
    keep = npair > 0
    br, b0 = b_rec[keep], b_r0[keep]
    n = br.shape[0]
    if n == 0 or b0.sum() <= 0:
        return 0.0, 0.0, 0.0, 0
    point = 1.0 - br.sum() / b0.sum()
    rng = np.random.default_rng(base_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = 1.0 - br[idx].sum(1) / np.maximum(1e-9, b0[idx].sum(1))
    return point, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)), n


def per_sequence_eo1(probs: np.ndarray, true_order: np.ndarray, class_ids: np.ndarray, *,
                     family_id: str, kappa: float) -> np.ndarray:
    """Backwards-compatible per-sequence skill vs the corrected R0 (used only for quick diagnostics; the
    gate uses ``pooled_eo1_skill``)."""
    r0 = r0_pairwise(family_id, kappa, class_ids)
    b_rec, b_r0, npair = per_sequence_briers(probs, true_order, r0)
    out = np.full(probs.shape[0], np.nan)
    m = npair > 0
    out[m] = 1.0 - b_rec[m] / np.maximum(1e-9, b_r0[m])
    return out
