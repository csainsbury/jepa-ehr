"""Rung-2 sub-gate 1 rollout-export orchestration (Pi v2: authorized to build; no governed data).

Enforces the two-estimand discipline (Pi structural fix): the DIRECT-horizon path (horizon-
conditioned; horizon decay only; NEVER emits exposure-gap/recursive metrics) and the
RECURSIVE-transition path (fixed-width non-overlapping δ states). The recursive path is
`NOT_EVALUABLE` unless the checkpoint metadata proves fixed-width-transition training — no
pseudo-rollouts on horizon-count-1 checkpoints. numpy-only orchestration over precomputed rollout
latents; the torch forward pass is a thin wrapper the governed runbook supplies.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from clinical_jepa.eval import rung2_rollout_diag as RD
from clinical_jepa.eval.rung2_contract import (
    NOT_EVALUABLE, recursive_path_evaluable, validate_direct_path_row,
)


def plan_paths(checkpoint_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Which rollout estimands are evaluable for this checkpoint."""
    return {"direct": True,                                     # direct horizon decay always available
            "recursive": recursive_path_evaluable(checkpoint_meta),
            "recursive_status": ("evaluable" if recursive_path_evaluable(checkpoint_meta) else NOT_EVALUABLE)}


def direct_horizon_metrics(pred_by_W: dict[float, np.ndarray], true_by_W: dict[float, np.ndarray],
                           patients_by_W: dict[float, Any], *, source: str) -> list[dict[str, Any]]:
    """Per-horizon direct-path drift (d_self, ambient-normalised d_NN). Fail-hard: a direct row may
    NEVER carry exposure-gap/recursive metrics (validate_direct_path_row)."""
    rows = []
    for W in sorted(pred_by_W):
        zhat, ztrue, pats = pred_by_W[W], true_by_W[W], np.asarray(patients_by_W[W])
        d = RD.drift_self_over_nn(zhat, ztrue, ztrue, pats, pats)
        ambient = RD.ambient_true_nn_distance(ztrue, pats)
        row = {"path": "direct", "source": source, "window_days": float(W),
               "d_self_mean": float(np.mean(d["d_self"])),
               "d_self_over_ambient_nn": float(np.mean(d["d_self"]) / max(ambient, 1e-9)),
               "n": int(len(zhat))}
        validate_direct_path_row(row)                          # raises if a recursive metric leaked in
        rows.append(row)
    return rows


def recursive_transition_metrics(checkpoint_meta: dict[str, Any] | None, *,
                                 dself_free: np.ndarray | None = None, dself_tf: np.ndarray | None = None,
                                 dself_over_nn_point: float | None = None,
                                 exposure_gap_slope_lo: float = 0.0, dself_slope_hi: float = 0.0,
                                 source: str = "?", window_days: float = 0.0) -> dict[str, Any]:
    """Recursive fixed-width transition diagnostics — NOT_EVALUABLE unless the checkpoint was trained
    on fixed-width non-overlapping transition states (no pseudo-rollout). Emits the exposure gap +
    the frozen-margin signature ONLY on the recursive path."""
    if not recursive_path_evaluable(checkpoint_meta):
        return {"path": "recursive", "source": source, "window_days": float(window_days),
                "status": NOT_EVALUABLE,
                "reason": "checkpoint metadata does not prove fixed-width-transition training"}
    gap = RD.exposure_gap(dself_free, dself_tf) if (dself_free is not None and dself_tf is not None) else None
    sig = RD.classify_signature(dself_over_nn_point=float(dself_over_nn_point or 0.0),
                                exposure_gap_slope_lo=exposure_gap_slope_lo, dself_slope_hi=dself_slope_hi,
                                transition_evaluable=True)
    return {"path": "recursive", "source": source, "window_days": float(window_days), "status": "evaluable",
            "exposure_gap_mean": (float(np.mean(gap)) if gap is not None else None), "signature": sig}
