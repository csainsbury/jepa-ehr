"""Rung-2 sub-gate 2 count-interface scoring (Pi v2: authorized to build).

ONE matched predictive family (Pi #3): interface A (factorized CONTEXT head) and interface B
(concatenated TARGET scalar) are BOTH scored as context→future-count predictors under the SAME
frozen hurdle count-distribution family + the SAME proper score (ranked probability score), so the
comparison is identifiable. The Rung-1 exact-count-from-z+ = 1.000 is a target-side positive
control only (never the decision). Nomination-only: A is the default on a paired practical tie; if B
is restricted to a point estimate it CANNOT win a calibrated-distribution comparison — that is a
structural interface decision (`NOMINATE_FACTORIZED`), not a horse race. numpy-only (the heads are
trained by the driver; this scores their outputs).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval.rung1_probes import ratio_skill_ci
from clinical_jepa.eval.rung2_contract import (
    COUNT_NOMINATE_MARGIN, NEITHER_ADEQUATE, NOMINATE_CONCAT, NOMINATE_FACTORIZED,
)


def ranked_probability_score(pmf: Any, y: Any, k_max: int | None = None) -> np.ndarray:
    """Per-row RPS for a count distribution: Σ_k (F(k) − 1[y≤k])². pmf[i] is the predicted
    probability vector over counts 0..K; y[i] the true count. Lower is better."""
    P = np.asarray(pmf, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    K = P.shape[1] - 1 if k_max is None else k_max
    F = np.cumsum(P, axis=1)                                   # predictive CDF [n, K+1]
    kk = np.arange(K + 1)[None, :]
    ind = (y[:, None] <= kk).astype(np.float64)               # 1[y<=k]
    return np.sum((F - ind) ** 2, axis=1)


def rps_skill_vs_baseline(rps_model_rows: Any, rps_baseline_rows: Any, clusters: Any, **kw) -> dict[str, float]:
    """Cluster-bootstrap RPS SKILL = 1 − E[RPS_model]/E[RPS_baseline] (lower-CI is the gated
    quantity)."""
    return ratio_skill_ci(rps_model_rows, rps_baseline_rows, clusters, **kw)


def count_interface_decision(*, skill_a_lo: float, skill_b_lo: float, paired_b_minus_a_lo: float,
                             b_is_point_estimate: bool, gate: float = 0.0,
                             margin: float = COUNT_NOMINATE_MARGIN) -> dict[str, Any]:
    """Nomination-only decision on the matched proper score. B nominates only if it beats A by the
    paired margin AND clears the gate; A is the default on a paired tie/loss; a point-estimate B is
    a structural NOMINATE_FACTORIZED (cannot win a calibrated comparison)."""
    a_ok = skill_a_lo > gate
    b_ok = skill_b_lo > gate
    if b_is_point_estimate:
        return {"decision": NOMINATE_FACTORIZED, "reason": "B is a point estimate — structural interface limitation, not a calibrated-comparison loss",
                "a_adequate": bool(a_ok)}
    if b_ok and paired_b_minus_a_lo > margin:
        return {"decision": NOMINATE_CONCAT, "reason": "B beats A on the matched proper score by the margin"}
    if a_ok:
        return {"decision": NOMINATE_FACTORIZED, "reason": "A adequate; B not superior by the margin (default on tie)"}
    return {"decision": NEITHER_ADEQUATE, "reason": "context->count predictability is the binding constraint"}
