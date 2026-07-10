"""Rung-1 KS precision simulation (Pi R8 #3 — falsifiable, frozen before dev access).

A per-timing-cell pre-run check that the design can actually CERTIFY an upper-95%-CI KS-D
<= gate at a predeclared calibrated alternative population distance ``D*`` (0.025). Draws
PIT samples from a distribution whose sup-distance to Uniform is exactly ``D*`` (a
triangular-perturbation CDF), using the cell's train-only cluster/interval structure, then
checks:
  * nominal one-sided coverage of the bootstrap upper-CI >= PSIM_COVERAGE (0.95);
  * power = P(certify upper-CI <= gate) >= PSIM_POWER (0.80).
On failure the cell is NOT_EVALUABLE or the interval floor is raised — the KS gate is NEVER
weakened. Fixed reps/seeds so the result is reproducible and auditable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_contract import (
    KS_D_GATE, PSIM_COVERAGE, PSIM_D_STAR, PSIM_POWER, PSIM_REPS, PSIM_SEED,
)
from clinical_jepa.eval.rung1_probes import ks_d_upper_ci


def _sample_ks_alternative(n: int, d_star: float, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples in [0,1] whose CDF is the triangular perturbation
    G(x)=x∓2·D*·(x or 1−x), so sup|G−Uniform| = D* exactly (population KS distance)."""
    u = rng.uniform(size=n)
    x = np.where(u <= 0.5 - d_star, u / (1.0 - 2.0 * d_star), (u + 2.0 * d_star) / (1.0 + 2.0 * d_star))
    return np.clip(x, 0.0, 1.0)


def run_precision_sim(
    n_intervals: int,
    n_clusters: int,
    *,
    d_star: float = PSIM_D_STAR,
    gate: float = KS_D_GATE,
    reps: int = PSIM_REPS,
    seed: int = PSIM_SEED,
    n_boot: int = 400,
) -> dict[str, Any]:
    """Return {coverage, power, passes, action} for a timing cell of this size."""
    n_intervals = int(n_intervals)
    n_clusters = max(1, int(n_clusters))
    clusters = np.arange(n_intervals) % n_clusters      # balanced cluster assignment
    rng = np.random.default_rng(seed)
    covered = 0
    certified = 0
    for r in range(int(reps)):
        x = _sample_ks_alternative(n_intervals, d_star, rng)
        res = ks_d_upper_ci(x, clusters, n_boot=n_boot, seed=seed + r + 1)
        if res["ci_hi"] >= d_star:                      # one-sided upper CI covers the truth
            covered += 1
        if res["ci_hi"] <= gate:                        # design certifies the small distance
            certified += 1
    coverage = covered / reps
    power = certified / reps
    passes = bool(coverage >= PSIM_COVERAGE and power >= PSIM_POWER)
    return {
        "n_intervals": n_intervals, "n_clusters": n_clusters, "d_star": d_star, "gate": gate,
        "reps": int(reps), "coverage": coverage, "power": power, "passes": passes,
        "coverage_floor": PSIM_COVERAGE, "power_floor": PSIM_POWER,
        # NEVER weaken the KS gate: an under-powered cell is dropped or its floor raised.
        "action": "certifiable" if passes else "NOT_EVALUABLE_or_raise_interval_floor",
    }
