---
title: Rung 1 — frozen-decode ceiling RESULTS (dev; Pi R7+R8-corrected)
created: 2026-07-11
status: governed run complete (8 source×horizon cells, 4 arms, 96 cell-metrics); DEV-ONLY architecture-selection ceiling (inside the circle, Pi Q8); TEST untouched
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); latent = per-source encode-empty v0B (reused from Rung 0)
reporting: aggregate-only (per source×horizon-cell metrics, floor-adjusted excess, CIs, verdicts; no seq ids / tokens / embeddings / real paths)
---

# Rung 1 — results

The frozen-decode ceiling of the pooled target latent `z⁺ = mean_embed(ids)` — the upper
bound on what any generator built on it could reconstruct — run on the real substrate through
the full Pi-R7/R8 harness (contract-hashed `aa823835…`). **Rung 1a** = the incumbent
mean_embed ceiling (independently verdictable); **Rung 1b** = a parameter-free target-contrast
panel that may only *nominate* Rung-2 targets. Fit on TRAIN, evaluated on DEV, **TEST sealed**
(no `--confirm-test` path exists). Caps: 12k train / 6k dev blocks per cell (≫ the 500-cluster
adequacy floor), logged.

## Verdict
| Arm | count | order | timing |
|-----|-------|-------|--------|
| **mean_embed (1a)** | `NOT_DECODABLE` | `STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY` | `PRIOR_MASKED` |
| tap_concat (1b) | `NOT_DECODABLE` | content-prior-only | `PRIOR_MASKED` |
| count_concat (1b) | **`DECODABLE_SIMPLE`** → nominate | content-prior-only | `NOT_DECODABLE` |
| temporal_slot (1b) | `NOT_DECODABLE` | `NOT_DECODABLE` (coarse-slot) | `NOT_DECODABLE` (coarse-slot) |

**Sole nomination: `count_concat` → `NOMINATE_TARGET_FOR_RUNG2` (count).** No order or timing
target is nominated by any cheap pooled arm.

**Headline:** the incumbent `mean_embed` pooled latent is **not generation-capable** for
count, order, or timing at the decode ceiling. The only parameter-free rescue that clears a
gate is appending the log-count (`count_concat`); order and timing require targets that Rung 2
must build and *train* (no pooled alternative reaches them).

## Count — divided out by the mean; recovered only by carrying it explicitly
Worst primary cell, exact-count vs the nearest-neighbour copy floor:

| arm | SCID (worst) | MIMIC (worst) |
|-----|-------------:|--------------:|
| mean_embed | 0.038 (NN 0.020) | 0.129 (NN 0.061) |
| **count_concat** | simply decodable (ridge reads the appended log-count) | simply decodable |
| tap_concat | 0.047 | 0.139 |
| temporal_slot | 0.026 | 0.116 |

`mean_embed` exact-count sits essentially at the NN-copy floor — the `1/N` normalization
divides cardinality out, so a generator on `mean_embed` needs an **explicit occupancy/count
head** (which the model already has *from context*). `count_concat` — `[mean E ⊕ log1p N]` — is
trivially read by the simple (M1 ridge) readout, so it is the cheapest count-capable target.

## Order — content-prior only, never latent instance-order
Order is evaluable only where windows are short enough to admit exact reconstruction (SCID
short horizons; MIMIC's ICU density excludes it). Exact ordered-seq recon and its floor-adjusted
(wrong-instance-swap) excess:

| arm | SCID W30 exact / excess | SCID W90 exact / excess |
|-----|------------------------:|------------------------:|
| mean_embed | 0.223 / +0.029 | 0.095 / −0.005 |
| temporal_slot | 0.303 / +0.080 | 0.125 / +0.018 |

The raw exact-order score is entirely a **content→order prior** (its swap excess is ≤0.08,
below the 0.10 margin) and decays as windows lengthen. `temporal_slot` is marginally better
(coarse wall-clock slots) but still sub-gate and is scoped `COARSE_SLOT` — it can never emit an
exact-order verdict. Arm-A order is forced to `STRUCTURALLY_INVARIANT_CONTENT_PRIOR_ONLY` by
construction (permutation-pair invariance). **Exact order needs a per-event / order-preserving
target** (sequence-of-latents / VQ), which Rung 2 must build.

## Timing — a marginal (zero-inflation) envelope, not per-instance timing
The striking substrate fact: **inter-event Δt is ~70% zeros in SCID and ~98% zeros in MIMIC**
(simultaneous events). This is why a bare quantile head fails; the first governed pass showed
the exact evaluator-artifact Pi warned about (KS-D near-identical ~0.6–0.9 across all arms).
Fixing the conditional model to a **hurdle (Δt=0 point mass + positive tail) with randomized
PIT** (Pi R8 #7) makes KS-D well-calibrated (upper-CI 0.009–0.056) and shifts the discriminating
signal to the **conditional-vs-marginal CRPS skill**:

| arm | SCID W90 ks / skill_lo | MIMIC W0.5 ks / skill_lo |
|-----|-----------------------:|-------------------------:|
| mean_embed | 0.043 / −0.047 | 0.009 / −0.165 |
| tap_concat | 0.035 / +0.008 | 0.010 / −0.036 |

`mean_embed` and `tap_concat` **reproduce the marginal Δt distribution** (KS passes) but carry
**no per-instance conditional skill** (CRPS-skill lower-CI < the 0.05 gate; max across all cells
+0.075) → **PRIOR_MASKED**. The time-augmented pooling (`tap_concat`) adds only ~+0.008 skill,
nowhere near the gate — **pooled targets cannot carry direct timing**; timing belongs in a
dedicated continuous-time (marked-TPP / occupancy-hurdle) head, exactly the P3 spine.

## What Rung 2 inherits (the actionable output)
- **count** → `mean_embed` loses it; the cheapest fix is to carry log-count in the target
  (`count_concat` nominated). The context occupancy/count head already covers this.
- **order** → no pooled target reaches exact order; build a per-event / sequence-of-latents /
  VQ order-preserving target and *train the predictor to reach it* (nomination ≠ adoption).
- **timing** → route to a dedicated continuous-time head; the ~70–98% Δt=0 mass means
  occupancy/simultaneity dominates and a pooled Δt target only reproduces the marginal.

## Harness integrity / honesty
- **Per-readout floor-adjusted attribution (Pi R7 #1):** every verdict rests on the readout's
  own wrong-instance-swap excess; `PRIOR_MASKED` = raw gate passes but the decoder's own excess
  fails (marginal reproduction), never a weak-M1 veto.
- **Evaluator-artifact caught + fixed:** the first pass' uniform KS-D failure was a missing
  conditional zero-mass model, not the latent — fixed to a hurdle, unit-verified (calibrated
  hurdle KS<0.03 at 30% zeros; wrong zero-rate KS>0.10).
- **Information scope in every verdict** (direct / content_proxy / coarse_slot); oracle-assisted
  companions and content-proxy findings never nominate; incumbent 1a never nominates.
- **Adequacy:** cluster-based floors; timing cells pass the falsifiable KS precision simulation
  (power ≥0.80 to certify KS-upper-CI ≤0.05 at D*=0.025). MIMIC 2 d is sensitivity-only (never
  gates). Aggregate-only; DEV-only; TEST sealed.

## Local artifacts (gitignored)
`run-workspace/local-governed/rung1/verdict/rung1-ceiling-manifest.json` — per-cell metrics +
verdicts (config_hash `aa823835…`).

## Next
1. Route to Pi as the **implementation/result gate** (no R9 blueprint round per R8). 2. **Rung 2**
   (rollout `d_t`/`v_t`) with the nominated targets — train a predictor to reach an
   order-preserving target + a continuous-time head, and test *prediction-achieved* fidelity
   before adopting any target (nomination ≠ adoption).
