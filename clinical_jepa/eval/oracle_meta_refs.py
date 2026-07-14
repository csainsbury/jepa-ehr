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
    COUPLING_SCALE, D_ITEM, INVARIANT, MetaCell, N_CLASSES, ORDER_NOISE, _driver, _raw_coupling,
)

EO1_TIE_ATOL = 1e-9


# ---- reference / control predictors on a MetaCell (all read only context + item content, no labels) ----
def r_bayes_scores(cell: MetaCell) -> np.ndarray:
    """The fair CONTEXT ceiling: recover the driver from context via the analytic posterior map and
    apply the SHARED coupling to the legitimate item content (the oracle may use mechanism knowledge)."""
    inv = INVARIANT
    driver_hat = cell.context_features @ inv.A.T
    legit = cell.item_features[:, :, :D_ITEM]
    coup = COUPLING_SCALE * _raw_coupling(driver_hat, inv.M, legit) / inv.coupling_norm
    return inv.class_means[cell.item_classes] + cell.kappa * coup


def r_nuis_scores(cell: MetaCell) -> np.ndarray:
    """Nuisance-only predictor: rank items by the nuisance channel (eval-only reference, not a recipe)."""
    return cell.nuisance_u.copy()


def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    lo, hi = x.min(), x.max()
    levels = max(2, 2 ** min(int(bits), 16))
    return np.round((x - lo) / (hi - lo + 1e-9) * (levels - 1)) / (levels - 1)


def mean_embed_quantized_scores(cell: MetaCell, bits: int) -> np.ndarray:
    """U6 control 1: order-blind mean-embed of the legitimate item content, quantized to matched bits."""
    return _quantize(cell.item_features[:, :, :D_ITEM].mean(axis=2), bits)


def random_codebook_scores(cell: MetaCell, bits: int, seed: int) -> np.ndarray:
    """U6 control 2: a FROZEN random codebook at matched bits assigns each item a random score."""
    rng = np.random.default_rng(seed)
    legit = cell.item_features[:, :, :D_ITEM]
    codebook = rng.standard_normal((max(2, 2 ** min(int(bits), 12)), D_ITEM))
    idx = np.argmin(((legit[:, :, None, :] - codebook[None, None]) ** 2).sum(-1), axis=2)
    return codebook[idx] @ rng.standard_normal(D_ITEM)


def e_o2_calibration(recipe_probs: np.ndarray, ref_probs: np.ndarray, true_order: np.ndarray):
    """E-O2: calibration of the recipe's predicted pairwise probabilities against the context-Bayes
    reference probabilities. Fit p_recipe ≈ slope·p_ref + intercept on eligible pairs (logit space).
    Returns (slope, intercept)."""
    n, L, _ = recipe_probs.shape
    iu, ju = np.triu_indices(L, k=1)
    xs, ys = [], []
    for s in range(n):
        to = true_order[s]
        elig = np.abs(to[iu] - to[ju]) > EO1_TIE_ATOL
        if elig.any():
            xs.append(ref_probs[s, iu, ju][elig])
            ys.append(recipe_probs[s, iu, ju][elig])
    if not xs:
        return 0.0, 0.5
    x = np.clip(np.concatenate(xs), 1e-4, 1 - 1e-4)
    y = np.clip(np.concatenate(ys), 1e-4, 1 - 1e-4)
    lx, ly = np.log(x / (1 - x)), np.log(y / (1 - y))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)


def pairwise_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    diff = scores[:, None, :] - scores[:, :, None]
    return 1.0 / (1.0 + np.exp(-diff / max(1e-6, temperature)))


def r0_pairwise(family_id: str, kappa: float, class_ids: np.ndarray) -> np.ndarray:
    """(N, L, L) EXACT content-prior P(a≺b | classes) — the conditional Bayes reference (oracle_meta_bayes)."""
    from clinical_jepa.eval.oracle_meta_bayes import r0_pairwise as _exact_r0
    return _exact_r0(family_id, float(kappa), class_ids)


def r_bayes_probs(cell: MetaCell) -> np.ndarray:
    """(N, L, L) EXACT context-Bayes π*(a≺b | context, content) — the fair ceiling and the E-O2 /
    hidden-null reference (a proper pairwise probability, NOT a sigmoid of posterior-mean scores)."""
    from clinical_jepa.eval.oracle_meta_bayes import pi_star_pairwise
    return pi_star_pairwise(cell, cell.kappa)


def r0_table_mc_error(family_id: str, kappa: float) -> float:
    """R0 vs independent high-precision MC (the conditional-estimand validation, oracle_meta_bayes)."""
    from clinical_jepa.eval.oracle_meta_bayes import reference_mc_error
    return reference_mc_error(family_id, float(kappa), which="r0")


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


def paired_skill_contrast(b_target: np.ndarray, b_comparator: np.ndarray, b_r0: np.ndarray,
                          npair: np.ndarray, *, n_boot: int = 1000, base_seed: int = 0,
                          alpha: float = 0.05):
    """PAIRED skill contrast (target − comparator), both relative to R0, over the SAME resampled
    sequences: contrast = (Σ Brier_comparator − Σ Brier_target) / Σ Brier_R0. Returns (point, lower_ci).
    For recipe−R0 pass b_comparator = b_r0 (=> the recipe's own skill)."""
    keep = npair > 0
    bt, bc, b0 = b_target[keep], b_comparator[keep], b_r0[keep]
    n = bt.shape[0]
    if n == 0 or b0.sum() <= 0:
        return 0.0, 0.0
    point = float((bc.sum() - bt.sum()) / b0.sum())
    rng = np.random.default_rng(base_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    denom = np.maximum(1e-9, b0[idx].sum(1))
    boot = (bc[idx].sum(1) - bt[idx].sum(1)) / denom
    return point, float(np.quantile(boot, alpha))       # one-sided lower bound at alpha


def briers_vs_r0(scores: np.ndarray, cell, *, temperature: float = 1.0, positives_only: bool = True) -> tuple:
    """(brier_per_seq, brier_r0_per_seq, npair) for a predictor's scores on a cell."""
    return briers_from_probs(pairwise_probs(scores, temperature), cell, positives_only=positives_only)


def briers_from_probs(probs: np.ndarray, cell, *, positives_only: bool = True) -> tuple:
    """As ``briers_vs_r0`` but from already-decoded pairwise probabilities (e.g. a SAMPLED decode)."""
    r0 = r0_pairwise(cell.family_id, cell.kappa, cell.item_classes)
    b_rec, b_r0, npair = per_sequence_briers(probs, cell.true_order, r0)
    if positives_only:
        npair = np.where(~cell.is_null, npair, 0)
    return b_rec, b_r0, npair


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
