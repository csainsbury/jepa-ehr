"""Rung-2 sub-gate 1 — DERIVATION of the fixed-width-transition training regime.

Deliberately kept OUT of `rung2_contract`. That module's source bytes are hashed into
`oracle_aggregate_extract.extraction_code_identity()`, the frozen executable-closure identity for the
oracle certification boundary, and `test_oracle_cert_boundary_guard` pins it. Adding this derivation there
moved that identity — the guard fired correctly — so the logic lives here instead and the governed closure
stays byte-stable. Nothing here is imported by `rung2_contract`.

The regime this derives is what makes the recursive-transition path of sub-gate 1 meaningful: states must be
a contiguous, equal-width, non-overlapping tiling produced by a genuinely recursive predictor. Anything else
is a different estimand, and the exposure gap / drift signature would not mean what the contract says.
"""
from __future__ import annotations

from typing import Any

# The checkpoint-manifest flag consumed by `rung2_contract.recursive_path_evaluable`. Re-exported here so
# writers have one import site, but OWNED by the contract (which is frozen and must not move).
from clinical_jepa.eval.rung2_contract import TRANSITION_META_KEY

__all__ = ["TRANSITION_META_KEY", "is_fixed_width_transition_training"]


def is_fixed_width_transition_training(*, autoregression_mode: str | None, horizon_count: Any,
                                       horizon_stride_tokens: Any, max_target_tokens: Any,
                                       encode_empty: bool = False) -> bool:
    """DERIVE whether a training configuration produces fixed-width NON-OVERLAPPING transition states.

    Single source of truth for the `fixed_width_transition_trained` flag: DERIVED from the training config,
    never caller-asserted. Requires ALL of:

      * `autoregression_mode == "recursive"` — horizon-conditioned heads consume the context latent, not the
        previous step, so they are a different estimand entirely;
      * `horizon_count >= 2` — one window is a single prediction, not a transition sequence, so there is
        nothing recursive to diagnose. This is why every existing v0B / encode-empty checkpoint fails: they
        train at the default `horizon_count=1`, and `--encode-empty` pins it to 1;
      * `stride == max_target_tokens` — equal-width windows tiled contiguously. A stride BELOW the width
        overlaps (states share events, so drift is not attributable to the transition); a stride ABOVE it
        leaves gaps (the states are not a contiguous delta-tiling).
    """
    if encode_empty:
        return False                                   # encode-empty pins horizon_count to 1 by construction
    if autoregression_mode != "recursive":
        return False
    if isinstance(horizon_count, bool) or isinstance(horizon_stride_tokens, bool):
        return False
    try:
        k = int(horizon_count)
        stride = int(horizon_stride_tokens)
        width = int(max_target_tokens)
    except (TypeError, ValueError):
        return False
    return k >= 2 and width > 0 and stride == width
