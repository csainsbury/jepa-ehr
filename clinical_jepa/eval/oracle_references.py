"""Reference bracket + paired contrasts + E-O metrics + U6 controls (Pi 2nd-pass Phase 4a).

Everything is scored on CONTEXT-ONLY predicted pairwise order probabilities against the KNOWN true
order, aggregated to ONE value per sequence, then contrasted with a PAIRED same-sequence cluster
bootstrap (within a replicate, resample sequence IDs once and evaluate recipe AND comparator on exactly
those IDs; take the per-replicate difference — never subtract marginal CI endpoints).

  * E-O1  — beyond-content-prior pairwise-probability skill over the exact content prior π0 (=0.5),
            a proper (Brier) skill score, NOT Kendall τ relabelled.
  * R0    — content-prior floor: constant 0.5 pairwise probs (skill 0 by construction).
  * R_bayes — the fair CONTEXT ceiling: a strong context predictor fit on an independent large sample.
  * R_nuis  — nuisance-only predictor (loses skill in Σ-orthogonal cells; partial in leak cells).
  * U6 controls — mean-embed quantized to matched bits, and a frozen random codebook at matched bits.
  * hidden-null — decided from R_bayes − R0 BEFORE the recipe is inspected.

All synthetic / safe-public. Power/MDE and precision simulation are Phase 4b (oracle_power).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.oracle_contracts import DecoderSamplerSpec, SamplerSpec
from clinical_jepa.eval.oracle_literal_gen import LiteralCell, generate_literal_cell
from clinical_jepa.eval.oracle_recipe import (
    CandidateRecipe, GoodContextRecipe, pairwise_probs_from_scores, split_views,
)
from clinical_jepa.eval.oracle_realism_v2 import assert_canonical_certification_cell as _assert_cert

EO1_TIE_ATOL = 1e-9


def restrict_to(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NaN-out sequences outside ``mask`` so the positive gate scores only positive (non-null)
    sequences while the null control scores the nulls. The evaluator knows ``is_null`` (an eval label);
    the recipe never does. Downstream paired_contrast drops NaN sequences."""
    out = np.asarray(vals, float).copy()
    out[~np.asarray(mask, bool)] = np.nan
    return out


def per_sequence_eo1(pairwise_probs: np.ndarray, true_order: np.ndarray) -> np.ndarray:
    """Per-sequence Brier SKILL of predicted P(i precedes j) over the content prior π0=0.5, on eligible
    (non-tied) precedence pairs. skill = 1 - Brier(recipe)/Brier(π0); Brier(π0)=0.25. NaN if a sequence
    has no eligible pairs (dropped downstream)."""
    n, L, _ = pairwise_probs.shape
    iu, ju = np.triu_indices(L, k=1)
    out = np.full(n, np.nan)
    for s in range(n):
        to = true_order[s]
        elig = np.abs(to[iu] - to[ju]) > EO1_TIE_ATOL
        if not elig.any():
            continue
        y = (to[iu] < to[ju]).astype(float)          # 1 == i precedes j
        p = pairwise_probs[s, iu, ju]
        brier = np.mean((p[elig] - y[elig]) ** 2)
        out[s] = 1.0 - brier / 0.25
    return out


# ---- reference / control PREDICTORS -> per-sequence E-O1 ----
def _probs_from_scores(scores: np.ndarray) -> np.ndarray:
    return pairwise_probs_from_scores(scores)


def eo1_recipe(recipe: CandidateRecipe, cell: LiteralCell, *, seed: int = 0) -> np.ndarray:
    _assert_cert(cell, entrypoint="eo1_recipe")
    z = recipe.predict_latent(cell.context_view(), SamplerSpec(), seed)
    probs = recipe.decode_order(z, DecoderSamplerSpec(), seed).pairwise_probs
    return per_sequence_eo1(probs, cell.true_order)


def eo1_r0(cell: LiteralCell) -> np.ndarray:
    _assert_cert(cell, entrypoint="eo1_r0")
    probs = np.full((cell.true_order.shape[0], cell.true_order.shape[1], cell.true_order.shape[1]), 0.5)
    return per_sequence_eo1(probs, cell.true_order)


def eo1_r_nuis(cell: LiteralCell) -> np.ndarray:
    _assert_cert(cell, entrypoint="eo1_r_nuis")
    return per_sequence_eo1(_probs_from_scores(cell.nuisance_u), cell.true_order)


def eo1_r_bayes(cell: LiteralCell, *, ref_n: int = 4000, seed: int = 0) -> np.ndarray:
    """Fair CONTEXT ceiling: a strong context predictor fit on an INDEPENDENT large sample of the same
    (family, kappa, orthogonal) mechanism — the practical E[order | context]."""
    _assert_cert(cell, entrypoint="eo1_r_bayes")
    ref_train = generate_literal_cell(cell.family_id, cell.kappa, "orthogonal", ref_n, seed=seed + 991)
    r = GoodContextRecipe()
    r.fit(split_views(ref_train), split_views(ref_train))
    return eo1_recipe(r, cell, seed=seed)


def eo1_mean_embed_quantized(cell: LiteralCell, *, bits: int = 8) -> np.ndarray:
    """U6 control 1: order-blind mean-embed of item features, quantized to matched bits."""
    _assert_cert(cell, entrypoint="eo1_mean_embed_quantized")
    item = cell.item_features
    scores = item.mean(axis=2)                       # order-blind pooling over item feature dims
    lo, hi = scores.min(), scores.max()
    levels = max(2, 2 ** bits)
    q = np.round((scores - lo) / (hi - lo + 1e-9) * (levels - 1)) / (levels - 1)
    return per_sequence_eo1(_probs_from_scores(q), cell.true_order)


def eo1_random_codebook(cell: LiteralCell, *, bits: int = 8, seed: int = 0) -> np.ndarray:
    """U6 control 2: a FROZEN random codebook at matched bits. Must PASS null and FAIL positive; if it
    spuriously predicts positives the U6 comparison is not interpretable (caller marks not-evaluable)."""
    _assert_cert(cell, entrypoint="eo1_random_codebook")
    rng = np.random.default_rng(seed ^ 0xC0DEB00C)
    n, L, di = cell.item_features.shape
    codebook = rng.standard_normal((max(2, 2 ** min(bits, 12)), di))
    idx = np.argmin(((cell.item_features[:, :, None, :] - codebook[None, None]) ** 2).sum(-1), axis=2)
    scores = codebook[idx] @ rng.standard_normal(di)    # random projection of the assigned codes
    return per_sequence_eo1(_probs_from_scores(scores), cell.true_order)


# ---- paired same-sequence cluster bootstrap ----
@dataclass(frozen=True)
class PairedContrast:
    mean_diff: float
    lower_ci: float
    upper_ci: float
    n_sequences: int


def paired_contrast(a_vals: np.ndarray, b_vals: np.ndarray, *, seed: int = 0, n_boot: int = 1000,
                    alpha: float = 0.05) -> PairedContrast:
    """Per-sequence PAIRED contrast a−b with a sequence-clustered bootstrap. Sequences where EITHER
    value is NaN (no eligible pairs) are dropped together. Within each replicate the SAME resampled
    sequence IDs are used for both arms; the replicate statistic is the difference of means."""
    a = np.asarray(a_vals, float); b = np.asarray(b_vals, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = a.shape[0]
    if n == 0:
        return PairedContrast(0.0, 0.0, 0.0, 0)
    diff = float((a - b).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = a[idx].mean(1) - b[idx].mean(1)           # SAME idx for both arms (paired)
    lo = float(np.quantile(reps, alpha / 2.0)); hi = float(np.quantile(reps, 1.0 - alpha / 2.0))
    return PairedContrast(diff, lo, hi, n)


def hidden_null_excluded(cell: LiteralCell, *, margin: float, seed: int = 0) -> bool:
    """Decide hidden-null from R_bayes − R0 BEFORE the recipe is inspected: if the fair context ceiling
    does not beat the content prior by ``margin`` (lower-CI) on the POSITIVE sequences, the cell is a
    HIDDEN NULL and is excluded from positive certification (Pi ORACLE_HIDDEN_NULL_RULE). Scored on the
    non-null sequences (the evaluator knows is_null) so the camouflage null mixture does not dilute the
    ceiling — a true kappa=0 cell has no positive signal and is correctly excluded."""
    _assert_cert(cell, entrypoint="hidden_null_excluded")
    pos = ~cell.is_null
    pc = paired_contrast(restrict_to(eo1_r_bayes(cell, seed=seed), pos),
                         restrict_to(eo1_r0(cell), pos), seed=seed)
    return pc.lower_ci < margin
