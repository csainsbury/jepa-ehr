"""Sub-window partitioning for rung 0 (coarse-vs-fine horizon-decay).

Carve an admitted wall-clock T0 target interval ``[t_query, t_query+W)`` into K
equal sub-windows so the FINE granularity (per-sub-window latents) can be compared
to the COARSE granularity (the full-W pool). Reuses the production span logic
(``_wall_clock_target_span``), so sub-window membership uses the identical half-open
rule as the parent block.

Partition modes (Pi R5 C6/Q6):
  - ``fixed_width_delta`` (PRIMARY): K = round(W/δ), w = W/K — near-fixed-width, no
    remainder cell; the small deviation of w from δ is recorded. Holding fine target
    duration ~fixed is necessary for a per-unit-wall-clock lag claim.
  - ``k2_proportional`` (SENSITIVITY): K=2, w=W/2 — holds rollout cardinality fixed.
  - ``k1_null`` (HARNESS NULL): K=1, w=W — coarse ≡ fine, the gap must be ~0.
  - ``k4`` (granularity SENSITIVITY): K=4.

The parent block is required non-censored / fully-observed by _t0_wall_clock_block,
so every sub-window inside it is fully observed — no per-sub-window censoring.
Aggregate/structural only; no raw tokens leave here.
"""
from __future__ import annotations

from typing import Any

from clinical_jepa.targets.extract_blocks import _wall_clock_target_span

PARTITION_MODES = ("fixed_width_delta", "k2_proportional", "k1_null", "k4")


def resolve_partition(window_days: float, delta_days: float, mode: str = "fixed_width_delta") -> tuple[int, float]:
    """Return (K, w) for a horizon W and source resolution δ. w = W/K exactly."""
    W = float(window_days)
    if mode == "k1_null":
        return 1, W
    if mode == "k2_proportional":
        return 2, W / 2.0
    if mode == "k4":
        return 4, W / 4.0
    if mode == "fixed_width_delta":
        K = max(1, int(round(W / float(delta_days)))) if delta_days > 0 else 1
        return K, W / float(K)
    raise ValueError(f"unknown partition mode {mode!r}; expected one of {PARTITION_MODES}")


def carve_subwindows(
    cumulative_days: Any,
    seq_len: int,
    context_end: int,
    t_query: float,
    K: int,
    w: float,
    *,
    segment_ids: Any = None,
) -> list[dict[str, Any]]:
    """K contiguous sub-windows tiling [t_query, t_query + K*w); each via the same
    half-open span rule as the parent block. Fully-observed parent => fully-observed
    sub-windows (empty here means genuine silence, never censoring)."""
    subs: list[dict[str, Any]] = []
    for k in range(int(K)):
        tq = float(t_query) + k * float(w)
        ts, te, n = _wall_clock_target_span(
            cumulative_days, seq_len, context_end, tq, float(w), segment_ids=segment_ids
        )
        subs.append({
            "subwindow_k": k,
            "t_query_sub": tq,
            "target_start_ref": int(ts),
            "target_end_ref": int(te),
            "n_target_events": int(n),
            "empty_target": bool(n == 0),
        })
    return subs


def annotate_block_subwindows(
    block: dict[str, Any],
    cumulative_days: Any,
    *,
    delta_days: float,
    mode: str = "fixed_width_delta",
    seq_len: int | None = None,
    segment_ids: Any = None,
) -> dict[str, Any]:
    """Carve a wall-clock T0 ``block`` (carrying context_end_ref, t_query, window_days)
    into sub-windows. Returns {partition_mode, K, w_days, delta_days, w_minus_delta,
    subwindows}. The event-count identity across sub-windows is a target-side quantity
    (Pi R5 C1): it may audit targets but must NEVER weight a predicted query."""
    W = float(block["window_days"])
    context_end = int(block["context_end_ref"])
    t_query = float(block["t_query"])
    n = int(seq_len if seq_len is not None else block.get("n_seq", len(cumulative_days)))
    K, w = resolve_partition(W, delta_days, mode)
    subs = carve_subwindows(cumulative_days, n, context_end, t_query, K, w, segment_ids=segment_ids)
    total_sub_events = sum(s["n_target_events"] for s in subs)
    return {
        "partition_mode": mode,
        "K": int(K),
        "w_days": float(w),
        "delta_days": float(delta_days),
        "w_minus_delta": float(w - delta_days),   # C6: record the deviation from δ
        "window_days": W,
        "subwindows": subs,
        "n_events_full": int(block.get("n_target_events", total_sub_events)),
        "n_events_subwindows_sum": int(total_sub_events),
    }
