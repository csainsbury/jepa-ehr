#!/usr/bin/env python3
"""Order-line reopening probe — is the Rung-1 "order failure is STRUCTURAL" closure target-specific?

Rung 1 recorded order as `NOT_EVALUATED` and described the direct-order failure as **structural**
(permutation invariance). That is exactly true of the incumbent `mean_embed` target: a mean over token
embeddings is invariant to any reordering of the same multiset, so its order-decode ceiling is EXACTLY
chance by construction — no decoder, no predictor and no amount of training can move it.

But "structural" was then carried forward as if it closed the ORDER LINE. It does not. It closes ONE TARGET.
The Rung-2 frozen order targets T1/T2/T3 (`clinical_jepa.targets.order_targets`, already built and frozen)
exist precisely to break that invariance. This probe measures, on synthetic ordered sequences and with NO
governed data, the property that Rung 1's structural claim rests on:

    SEPARABILITY — given a sequence and a PERMUTATION OF THE SAME MULTISET, does the target distinguish them?

Separability is a strict UPPER BOUND on order decodability: if a target maps two different orderings of the
same events to the same vector, nothing downstream can ever tell them apart (ceiling = chance). If it maps
them apart, the order question becomes an empirical one about predictors and decoders — i.e. a Rung-2
`PREDICTOR_BOTTLENECK` vs `TARGET_BOTTLENECK` question, not a closed line.

This probe deliberately does NOT claim order is recoverable in practice, or that any of T1-T3 is the right
target. It establishes only that the structural closure is target-specific — the minimum needed to stop a
whole line of investigation being written off.

Synthetic only: random ids against a random frozen embedding table. No substrate, no checkpoint, no governed
read, no TEST. Run: PYTHONPATH=<repo> /usr/bin/python3 scripts/order_separability_probe.py
"""
from __future__ import annotations

import json

import numpy as np

from clinical_jepa.targets.order_targets import ORDER_TARGET_NAMES, build_order_target

SEED = 20260725
VOCAB, D = 1050, 64          # vocab matches the real substrate's 1050; D is a stand-in width
N_PAIRS = 400
SEQ_LENS = (4, 8, 16)
ATOL = 1e-6                  # float32 target vectors; separation must clear numerical noise


def mean_embed(ids, E, z_empty):
    """The Rung-1 incumbent target: mean of token embeddings — permutation-invariant BY CONSTRUCTION."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.asarray(z_empty, dtype=np.float32)
    return np.asarray(E)[ids].mean(axis=0).astype(np.float32)


def derangement(rng, n):
    """A permutation with NO fixed point, so the reordering is genuine at every position."""
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def probe():
    rng = np.random.default_rng(SEED)
    E = rng.normal(size=(VOCAB, D)).astype(np.float32)
    z_empty = np.zeros(D, dtype=np.float32)

    targets = {"mean_embed (Rung-1 incumbent)": lambda ids: mean_embed(ids, E, z_empty)}
    for name in ORDER_TARGET_NAMES:
        targets[name] = (lambda nm: lambda ids: build_order_target(nm, ids, E=E, z_empty=z_empty)[0])(name)

    rows = {}
    for label, fn in targets.items():
        per_len = {}
        for L in SEQ_LENS:
            sep, dists = 0, []
            for _ in range(N_PAIRS):
                # DISTINCT ids => a reordering is a genuinely different sequence of the SAME multiset
                ids = rng.choice(VOCAB, size=L, replace=False)
                perm = ids[derangement(rng, L)]
                a, b = fn(ids), fn(perm)
                d = float(np.max(np.abs(a - b)))
                dists.append(d)
                sep += int(d > ATOL)
            per_len[L] = {"separated_frac": sep / N_PAIRS,
                          "median_max_abs_diff": float(np.median(dists))}
        rows[label] = per_len
    return rows


def main():
    rows = probe()
    print(f"Order SEPARABILITY under permutation of the same multiset "
          f"({N_PAIRS} deranged pairs per length, atol={ATOL})\n")
    print(f"{'target':<34} " + "  ".join(f"L={L:<2} sep   med|Δ|" for L in SEQ_LENS))
    for label, per_len in rows.items():
        cells = []
        for L in SEQ_LENS:
            r = per_len[L]
            cells.append(f"{r['separated_frac']:>6.3f} {r['median_max_abs_diff']:>8.4f}")
        print(f"{label:<34} " + "  ".join(cells))

    inc = rows["mean_embed (Rung-1 incumbent)"]
    invariant = all(inc[L]["separated_frac"] == 0.0 for L in SEQ_LENS)
    separating = [n for n in ORDER_TARGET_NAMES
                  if all(rows[n][L]["separated_frac"] == 1.0 for L in SEQ_LENS)]

    print(f"\nmean_embed permutation-invariant at every length : {invariant}")
    print(f"frozen order targets separating at every length   : {separating}")
    verdict = invariant and len(separating) > 0
    print("\nCONCLUSION:", (
        "the Rung-1 structural order closure is TARGET-SPECIFIC. mean_embed's order ceiling is exactly "
        "chance by construction, but the frozen T1-T3 targets separate permutations at every tested "
        "length, so their order ceiling is NOT structurally zero. Whether order is RECOVERABLE remains "
        "an open empirical question for a Rung-2 predictor/decoder — it is not closed."
        if verdict else
        "INCONCLUSIVE — separation did not behave as the target definitions imply; investigate before "
        "drawing any conclusion about the order line."))
    print("\nSCOPE: synthetic ids + random frozen E; separability is an UPPER BOUND on order decodability, "
          "not a demonstration that order is recoverable in practice. No governed data, no checkpoint, "
          "no TEST. This reopens a line; it does not settle it.")
    print("\n" + json.dumps({"verdict_order_line_reopened": bool(verdict),
                            "mean_embed_permutation_invariant": bool(invariant),
                            "separating_frozen_targets": separating}, indent=2))
    return verdict


if __name__ == "__main__":
    main()
