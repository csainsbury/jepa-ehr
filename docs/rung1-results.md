---
title: Rung 1 — frozen-decode ceiling RESULTS (dev; Pi R7/R8 + result-gate-corrected)
created: 2026-07-11
status: governed run complete + Pi Rung-1 result-gate REVISE folded in (order relabelled, real slot metric, timing M2 swap, symmetric CRPS); amended aggregate + independent recompute; DEV-ONLY ceiling (Pi Q8); TEST untouched
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); latent = per-source encode-empty v0B (reused from Rung 0)
reporting: aggregate-only (per source×horizon-cell metrics, floor-adjusted excess, CIs, verdicts; no seq ids / tokens / embeddings / real paths)
---

# Rung 1 — results (amended)

The frozen-decode ceiling of `z⁺ = mean_embed(ids)` — an upper bound on what any generator on
it could reconstruct — run on the real substrate through the Pi-R7/R8 harness, with the
**Rung-1 result-gate REVISE** corrections folded in (contract-hashed `aa823835…`). Rung 1a =
incumbent `mean_embed`; Rung 1b = parameter-free target panel (NOMINATE-only). Fit on TRAIN,
evaluated on DEV once, **TEST sealed**.

> **Interpretation frame (Pi):** "not generation-capable" is too strong — a pooled latent + a
> decoder prior may still render marginally plausible sequences. This rung rejects **frozen
> per-instance exact count / order / timing fidelity**, not all generation.

## Verdict
| Arm | count | order | timing |
|-----|-------|-------|--------|
| **mean_embed (1a)** | `NOT_DECODABLE` | `STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY` | `NOT_DECODABLE` (mostly `MARGINAL_ONLY`) |
| tap_concat (1b) | `NOT_DECODABLE` | content-prior-only | `NOT_DECODABLE` — **direct-timing ceiling at SCID 30 d only** |
| count_concat (1b) | **`DECODABLE_SIMPLE`** → nominate | content-prior-only | `NOT_DECODABLE` |
| temporal_slot (1b) | `NOT_DECODABLE` | `NOT_DECODABLE` (real slot metric) | `NOT_DECODABLE` |

**Sole nomination: `count_concat` → count target (an intentionally-trivial representation-
sufficiency control).** No arm meets the per-instance order gate; timing has a short-horizon
exception (below).

## Count — cardinality normalized out as a direct channel
`mean_embed` exact-count sits at the NN-copy floor (0.038 SCID / 0.129 MIMIC vs NN 0.02/0.06);
the `1/N` mean **normalizes cardinality out as a direct channel** (proxy information may remain
at short horizons, but the exact-count gate fails). `count_concat` = `[mean E ⊕ log1p N]` is
trivially read by the M1 ridge — **independent recompute: exact-count from the appended
log-count dim = 1.000**, clearing every primary cell → `DECODABLE_SIMPLE`. Per Pi this nominates
an **explicit count channel** (representational sufficiency), *not* that context can predict
future count; Rung 2 must compare an appended scalar against the existing factorized context
occupancy/count head before adopting either.

## Order — unconditional NOT_EVALUATED; structural + labelled oracle probe
The frozen **unconditional, untruncated** exact-order decoder is **not implemented** →
`order = NOT_EVALUATED` for the order-blind arms (the earlier pairwise metric was
oracle-multiset-conditioned, N≤16, and subsampled — it cannot serve as the unconditional
ceiling). The **structural arm-A finding stands analytically**: `mean_embed` carries no
instance order beyond its multiset (permutation-pair invariance, bit-exact).

A clearly-**labelled oracle-assisted** N≤16 pairwise probe (tie-aware token-sequence match,
`information_scope=oracle_assisted`, never gates/nominates) confirms the residual is a
content→order prior — SCID exact 0.226 (W30) → 0.070 (W730), swap-excess 0.04 → −0.01. **Order
coverage is strongly selected toward short N**: post-cap denominators + excluded-large fractions
are 130 (W30) → 5175 (W730) excluded, published per cell.

## Temporal-slot fidelity — the real metric (Pi result-gate #2)
Exact all-slot token-multiset reconstruction decoded (non-oracle) from the per-slot means via
the analytic pseudo-inverse, with the slot readout's own wrong-instance swap excess:

| cell | M=4 exact / F1 / swap-excess | M=8 (sensitivity) |
|------|-----------------------------:|------------------:|
| SCID W30 | 0.357 / 0.51 / +0.34 | 0.374 / 0.54 |
| SCID W365 | 0.058 / 0.17 / +0.05 | 0.069 / 0.24 |
| MIMIC W0.5 | 0.086 / 0.19 / +0.08 | 0.097 / 0.23 |
| MIMIC W2 | 0.003 / 0.11 / +0.00 | 0.007 / 0.10 |

Slot fidelity is **genuinely present and latent-dependent** (positive swap excess) but **decays
sharply with event density** and **never clears the 0.70 slot gate** → `NOT_DECODABLE`; **M=8 is
non-rescuing** (marginally higher, still ≪0.70). So temporal-slot pooling does **not** nominate
a coarse-temporal target. (Independent recompute: SCID W30 M=4 exact = 0.346 vs manifest 0.357.)

## Timing — symmetric CRPS + M2's own swap; MARGINAL_ONLY + a short-horizon exception
The conditional hurdle (Δt=0 point mass + randomized PIT) gives calibrated KS (upper-CI
0.009–0.054). CRPS is now **sample-matched** (same hurdle estimator + 64 draws for conditional
and marginal). Crucially, timing now runs **M2's own window-level wrong-instance swap**:

- The swap excess is **positive** (SCID +1.4 to +3.2 days; MIMIC +0.005 to +0.016) → the timing
  heads **do read the latent** — so the earlier `PRIOR_MASKED` was unwarranted. With calibrated
  KS but sub-gate conditional skill and a *passing* swap, most cells are **`MARGINAL_ONLY`**
  (uses the latent, reproduces ≈ the marginal, no gate-clearing per-instance skill).
- **Short-horizon exception (Pi):** `tap_concat` **clears BOTH timing gates at SCID 30 d**
  (KS 0.015, CRPS-skill lower-CI **+0.101**) → a genuine **`DIRECT_TIMING_CEILING` at the
  shortest horizon** — but it fails the conjunctive all-cell rule (SCID 365 d KS upper-CI 0.054
  > 0.05), so the combined verdict is `NOT_DECODABLE`. This is **"short-horizon direct-timing
  ceiling only," not "no arm carries any direct timing."**

Timing decodability across all primary cells is not met; time-augmented pooling reaches a direct
ceiling only at the shortest horizon. Timing belongs in a dedicated continuous-time head (P3);
the ~70 % (SCID) / ~98 % (MIMIC) Δt=0 mass means occupancy/simultaneity dominates.

## What Rung 2 inherits
- **count** → explicit channel needed; compare an appended scalar vs the factorized context
  occupancy/count head (nomination ≠ superiority).
- **order** → unconditional ceiling to be built (per-event / seq-of-latents / VQ) and the
  predictor trained; pooled targets (incl. temporal-slot) do not clear it.
- **timing** → dedicated continuous-time head with zero/simultaneity-aware calibration; a
  short-horizon direct ceiling exists for time-augmented pooling but not across horizons.

## Harness integrity / honesty
- **Per-readout floor-adjusted swap now applied to timing M2 too** — `PRIOR_MASKED` requires the
  readout's own swap to fail; `MARGINAL_ONLY` = uses the latent but sub-gate skill.
- **Order relabelled** NOT_EVALUATED (unconditional) + labelled oracle probe; **real slot metric**
  with own-swap + M=8 sensitivity; **symmetric CRPS**; coverage/exclusion fractions published.
- Information scope in every verdict; cluster-based adequacy; timing cells pass the falsifiable
  KS precision sim; MIMIC 2 d sensitivity-only; aggregate-only; DEV-only; TEST sealed.
- **Process note:** the first governed pass proceeded before Pi answered the separate
  implementation gate (a process deviation; aggregate data unaffected). This amended pass uses
  the result-gate-corrected contract.

## Local artifacts (gitignored)
`run-workspace/local-governed/rung1/verdict/rung1-ceiling-manifest.json` (config_hash `aa823835…`,
`slot_m8_sensitivity`, `process_notes`).

## Next
Route to Pi as the amended result gate (Pi: if the corrected timing/slot decisions are
unchanged, this can be the next result gate, not a new blueprint round). Then a **separated
Rung-2 blueprint** (Pi ruling): (1) incumbent no-training `d_t/v_t` diagnosis; (2) count
interface choice; (3) order-target candidate + prediction-achieved gate; (4) continuous-time
head + zero/simultaneity calibration gate. Nomination remains non-adoption.
