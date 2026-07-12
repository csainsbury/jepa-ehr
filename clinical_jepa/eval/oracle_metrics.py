"""Order-skill metrics and the EXACT sequence-level null statistic (Pi consolidated #4).

Order skill is Kendall's tau between a predictor's per-item scores and the true order-scores, computed
over eligible (non-tied) precedence pairs. tau is ``beyond-prior`` by construction: a content-prior
(uniform-random) predictor has expected tau 0, so skill 0 == no context-predictable order signal.

The sequence-level null statistic (Pi #4) collapses a sequence's many precedence pairs into ONE
per-sequence decision, so a long sequence does not get more false-positive opportunities than a short
one. FPR / skill CIs are bootstrapped over SEQUENCES (the cluster unit), never over pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clinical_jepa.eval.rung2_contract import (
    ORACLE_NULL_MIN_PAIRS, ORACLE_N_NULL_SEEDS,
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


def _seed_sequence(seed: int, n: int) -> np.ndarray:
    """Deterministic per-draw index resample (no global RNG; reproducible from `seed`)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)


def sequence_skill(per_sequence_pred: list[np.ndarray], per_sequence_true: list[np.ndarray],
                   *, n_boot: int = ORACLE_N_NULL_SEEDS, base_seed: int = 0,
                   alpha: float = 0.05) -> SkillResult:
    """Aggregate per-sequence tau into ONE skill estimate with a SEQUENCE-clustered bootstrap CI.

    Each sequence contributes a single tau (Pi #4: one decision per sequence). Sequences with fewer
    than ORACLE_NULL_MIN_PAIRS eligible pairs are dropped (too little order information to decide).
    The lower CI drives the fire rule (ORACLE_NULL_FIRE_RULE='sequence_skill_lower_CI_gt_0')."""
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
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _seed_sequence(base_seed + b, n_contrib)
        boot[b] = arr[idx].mean()
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return SkillResult(mean, lo, hi, len(per_sequence_pred), n_contrib, lo > 0.0)


def null_false_positive_rate(null_results: list[SkillResult]) -> float:
    """Fraction of NULL sequences-groups whose sequence-level statistic FIRED (a false positive).
    Bootstrap/decision unit is the sequence (ORACLE_NULL_BOOTSTRAP_UNIT='sequence')."""
    if not null_results:
        return 0.0
    return float(np.mean([r.fires for r in null_results]))


def realized_alpha(null_pred: list[np.ndarray], null_true: list[np.ndarray],
                   *, n_groups: int = 20, base_seed: int = 0) -> float:
    """Realized false-positive rate on NULL sequences: partition the null population into `n_groups`
    sequence-clustered groups, take the sequence-level FIRE decision per group, and report the
    fraction that fired. Decision/cluster unit is the sequence (Pi #4). Empty -> 0.0."""
    n = len(null_pred)
    if n == 0:
        return 0.0
    groups = min(n_groups, n)
    order = np.arange(n)
    fires = []
    for g in range(groups):
        idx = order[g::groups]
        if idx.size == 0:
            continue
        res = sequence_skill([null_pred[i] for i in idx], [null_true[i] for i in idx],
                             base_seed=base_seed + g)
        fires.append(res.fires)
    return float(np.mean(fires)) if fires else 0.0
