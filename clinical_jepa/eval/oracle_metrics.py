"""Order-skill metrics and the EXACT sequence-level null statistic (Pi consolidated #4).

Order skill is Kendall's tau between a predictor's per-item scores and the true order-scores, computed
over eligible (non-tied) precedence pairs. tau is ``beyond-prior`` by construction: a content-prior
(uniform-random) predictor has expected tau 0, so skill 0 == no context-predictable order signal.

The sequence-level null statistic (Pi #4) collapses a sequence's many precedence pairs into ONE
per-sequence decision, so a long sequence does not get more false-positive opportunities than a short
one. FPR / skill CIs are bootstrapped over SEQUENCES (the cluster unit), never over pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.rung2_contract import (
    ORACLE_N_BOOT, ORACLE_NULL_FIRE_ALPHA, ORACLE_NULL_MIN_PAIRS,
)


def kendall_tau_pairs(pred: np.ndarray, true: np.ndarray, *, tie_atol: float = 1e-9) -> tuple[float, int]:
    """tau over precedence pairs (i<j) whose TRUE order is not tied. Same-class ties carry no order
    information and are excluded (ORACLE_NULL_TIE_HANDLING='exclude_same_class_ties'). Returns
    (tau, n_eligible_pairs); tau is 0.0 when there are no eligible pairs."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = pred.shape[0]
    if n < 2:
        return 0.0, 0
    i, j = np.triu_indices(n, k=1)
    dt = true[i] - true[j]
    eligible = np.abs(dt) > tie_atol            # drop TRUE-tied pairs (no order info)
    if not eligible.any():
        return 0.0, 0
    dp = pred[i] - pred[j]
    dt_e, dp_e = dt[eligible], dp[eligible]
    concordant = np.sign(dp_e) == np.sign(dt_e)
    # predicted ties split as 0.5 (neither concordant nor discordant)
    pred_tie = np.abs(dp_e) <= tie_atol
    score = np.where(pred_tie, 0.5, concordant.astype(float))
    tau = 2.0 * float(score.mean()) - 1.0        # map [0,1] concordant-fraction to [-1,1]
    return tau, int(eligible.sum())


@dataclass(frozen=True)
class SkillResult:
    mean_skill: float
    lower_ci: float
    upper_ci: float
    n_sequences: int
    n_contributing: int          # sequences with >= ORACLE_NULL_MIN_PAIRS eligible pairs
    fires: bool                  # sequence-level: lower_ci > 0 (ORACLE_NULL_FIRE_RULE)


def sequence_skill(per_sequence_pred: list[np.ndarray], per_sequence_true: list[np.ndarray],
                   *, n_boot: int = ORACLE_N_BOOT, base_seed: int = 0,
                   fire_alpha: float = ORACLE_NULL_FIRE_ALPHA) -> SkillResult:
    """Aggregate per-sequence tau into ONE skill estimate with a SEQUENCE-clustered bootstrap CI.

    Each sequence contributes a single tau (Pi #4: one decision per sequence). Sequences with fewer
    than ORACLE_NULL_MIN_PAIRS eligible pairs are dropped (too little order information to decide). The
    study FIRES when the ONE-SIDED lower CI at ``fire_alpha`` exceeds 0
    (ORACLE_NULL_FIRE_RULE='sequence_skill_lower_CI_gt_0'). ``n_boot`` is the bootstrap replicate count
    for THIS study's CI — distinct from the number of independent null studies (Pi #4)."""
    taus = []
    for pred, true in zip(per_sequence_pred, per_sequence_true):
        tau, npairs = kendall_tau_pairs(pred, true)
        if npairs >= ORACLE_NULL_MIN_PAIRS:
            taus.append(tau)
    n_contrib = len(taus)
    if n_contrib == 0:
        return SkillResult(0.0, 0.0, 0.0, len(per_sequence_pred), 0, False)
    arr = np.asarray(taus, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(base_seed)                     # ONE rng, fully vectorized resample
    idx = rng.integers(0, n_contrib, size=(n_boot, n_contrib))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, fire_alpha))                 # one-sided lower bound at the fire level
    hi = float(np.quantile(boot, 1.0 - fire_alpha))
    return SkillResult(mean, lo, hi, len(per_sequence_pred), n_contrib, lo > 0.0)


def clopper_pearson_upper(k: int, n: int, *, conf: float = 0.95) -> float:
    """One-sided (1-`conf` in the tail) Clopper-Pearson UPPER confidence bound on a binomial rate.

    Solves binom_cdf(k; n, p_u) = 1-conf for p_u by bisection (the CDF is monotone decreasing in p).
    Exact and dependency-free — an OC study reports this UPPER bound on the null FPR, not a point rate
    (Pi #4: a point FPR from one pooled sample is not an upper-confidence-bounded OC study)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - conf

    def _log_binom_cdf(kk: int, nn: int, p: float) -> float:
        if p <= 0.0:
            return 0.0                # cdf = 1 -> log 0; but we compare cdf directly below
        if p >= 1.0:
            return float("-inf")
        terms = []
        lp, l1p = math.log(p), math.log1p(-p)
        for i in range(kk + 1):
            lc = math.lgamma(nn + 1) - math.lgamma(i + 1) - math.lgamma(nn - i + 1)
            terms.append(lc + i * lp + (nn - i) * l1p)
        m = max(terms)
        return m + math.log(sum(math.exp(t - m) for t in terms))

    lo, hi = 0.0, 1.0
    for _ in range(80):                # bisection to ~1e-24; plenty for a rate gate
        mid = 0.5 * (lo + hi)
        cdf = math.exp(_log_binom_cdf(k, n, mid))
        if cdf > alpha:                # too much mass at/below k -> p_u must be larger
            lo = mid
        else:
            hi = mid
    return hi


@dataclass(frozen=True)
class NullOCStudy:
    n_studies: int          # independent null studies actually run (evaluable support)
    n_fired: int            # false positives
    point_fpr: float
    upper_ci: float         # one-sided 95% Clopper-Pearson UPPER bound on the null FPR
    passes: bool            # upper_ci <= ORACLE_FPR_UPPER_CI_MAX


def null_oc_from_fires(fires: list[bool]) -> NullOCStudy:
    """Turn a list of INDEPENDENT null-study fire decisions into a point FPR + a one-sided 95% UPPER
    confidence bound (Pi #4). Each element is one study's sequence-level fire decision; studies must be
    independently seeded (NOT deterministic slices of one pooled sample)."""
    from clinical_jepa.eval.rung2_contract import ORACLE_FPR_UPPER_CI_MAX
    n = len(fires)
    k = int(sum(bool(f) for f in fires))
    point = (k / n) if n else 1.0
    upper = clopper_pearson_upper(k, n) if n else 1.0
    return NullOCStudy(n, k, point, upper, upper <= ORACLE_FPR_UPPER_CI_MAX)
