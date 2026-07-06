"""Shared, empty-target-aware readers for target blocks (single source of truth).

Wall-clock T0 blocks encode a zero-event target as the sentinel
``target_start_ref = target_end_ref = EMPTY_TARGET_REF = -1`` with
``empty_target: true`` (Pi item 3: encode, don't drop). The event-index
consumers were written before wall-clock mode and read a target span with
``t0 = max(0, int(target_start_ref))`` → ``max(0, -1) = 0``, silently misreading
an empty future as the START of the sequence (DATASET / [BOS] tokens).

Every consumer (arms, eval rollouts, leakage audit, embedding-cache feeder, tte)
MUST route target reads through the helpers here so ``-1`` is never turned into a
span again. This module is deliberately dependency-light (numpy only) so it can be
imported everywhere without pulling the heavy extractor; a unit test asserts
``EMPTY_TARGET_REF`` stays in sync with ``extract_blocks.EMPTY_TARGET_REF``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Mirrors clinical_jepa.targets.extract_blocks.EMPTY_TARGET_REF (kept in sync by a test).
EMPTY_TARGET_REF = -1


def is_empty_target(block: dict[str, Any]) -> bool:
    """True iff the block's target is genuine silence (a zero-event, fully-observed
    wall-clock window).

    Recognised via the explicit ``empty_target`` flag, or (defensively) target refs
    equal to the ``EMPTY_TARGET_REF`` sentinel. A ``censored`` block also carries
    ``-1`` refs (zero events) but is NOT silence — the absence is unverifiable
    (window past observed time; Pi R4 Q7) — so it is never treated as empty.
    Populated event-index blocks (no flag, non-negative refs) are never empty.
    """
    if block.get("censored"):
        return False
    if bool(block.get("empty_target")):
        return True
    ts = block.get("target_start_ref")
    if ts is None:
        return False
    te = block.get("target_end_ref")
    return int(ts) == EMPTY_TARGET_REF and (te is None or int(te) == EMPTY_TARGET_REF)


def is_censored(block: dict[str, Any]) -> bool:
    """True iff the block is a censored zero-event window (absence unverifiable,
    ineligible for empty encoding; Pi R4 Q7). Distinct from silence."""
    return bool(block.get("censored"))


def empty_target_len(block: dict[str, Any]) -> int:
    """Inclusive target-span length in events; 0 for an empty target.

    Replaces the buggy ``target_end_ref - target_start_ref + 1`` (which yields
    ``-1 - (-1) + 1 = 1`` for an empty block) at every call site.
    """
    if is_empty_target(block) or is_censored(block):
        return 0
    ts = block.get("target_start_ref")
    te = block.get("target_end_ref")
    if ts is None or te is None:
        return 0
    if int(ts) < 0 or int(te) < 0:   # any sentinel ref carries no readable span
        return 0
    return max(0, int(te) - int(ts) + 1)


def read_target_span(block: dict[str, Any], arr: Any) -> tuple[np.ndarray, bool]:
    """Return ``(target_token_ids_slice, is_empty)`` — never misread ``-1`` as ``arr[0:]``.

    - empty target → ``(empty int64 array, True)``.
    - populated → the inclusive ``[target_start_ref, target_end_ref]`` slice,
      clamped to the array, with ``is_empty=False``. A populated-but-out-of-range
      block returns an empty array with ``is_empty=False`` (it is not silence — it
      is an unreadable populated ref, which the caller can count separately).
    """
    if is_empty_target(block):
        return np.asarray([], dtype=np.int64), True
    ts = block.get("target_start_ref")
    te = block.get("target_end_ref")
    if ts is None or te is None:
        return np.asarray([], dtype=np.int64), False
    n = len(arr)
    s = max(0, int(ts))
    e = min(n - 1, int(te))
    if e < s:
        return np.asarray([], dtype=np.int64), False
    return np.asarray(arr[s : e + 1], dtype=np.int64), False


def target_occupancy(block: dict[str, Any]) -> tuple[int, int]:
    """Return ``(occupancy in {0,1}, event_count)`` for the target.

    Occupancy 0 == empty. Count is taken from ``n_target_events`` when present
    (wall-clock blocks carry it), else the inclusive span length.
    """
    if is_empty_target(block):
        return 0, 0
    n = block.get("n_target_events")
    count = int(n) if n is not None else empty_target_len(block)
    return (1 if count > 0 else 0), int(count)
