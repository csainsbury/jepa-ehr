---
title: Within-source wall-clock RUNG 0 — design blueprint (design-panel synthesis)
created: 2026-07-09
status: BLUEPRINT — awaiting Pi blueprint gate before the governed run
produced_by: 3-proposal design panel + adversarial critique + synthesis (workflow rung0-design)
scope: does a within-source timescale separation exist (coarse decays slower per unit wall-clock time) → build a hierarchical JEPA? Per source at frozen horizons.
---

# Rung 0 blueprint — cardinality-normalized sub-window horizon-decay

## Chosen operationalization
Coarse and fine describe the SAME interval `[t_query, t_query+W)` over the SAME events,
differing ONLY in temporal resolution:

- **COARSE** `z_coarse` = `mean_embed` over ALL events in W (the §3.3 rate/intensity
  envelope; genuine silence → frozen `z_empty`). Predicted `ẑ_coarse` = event-weighted
  mean of the K predicted sub-window latents (one rollout, one encoder frame).
- **FINE** = the ordered tuple `(z_1..z_K)`, `z_k = target_latent(events in sub-window k)`,
  `w = W/K`, empty sub-windows → `z_empty`. Predicted `ẑ_k = predict_rollout_from_latent(ctx, K)[:,k,:]`.
- **Pooling identity** (the fairness guarantee): the event-weighted mean of the K
  sub-window means **equals** the full-W mean — coarse is a strict, order-forgetting
  *reduction* of the fine tuple, with zero extra events/information. So a coarse
  retrieval win **is** the abstraction edge, not a data/pooling advantage.
- Decompositions: **K=2 proportional = PRIMARY**; fixed-width `w=δ` (SCID 30d, MIMIC
  0.25d) = per-unit-time sensitivity; **K=1 = null** (coarse≡fine ⇒ gap must be ~0, a
  harness check); K=4 = granularity sensitivity.

## Retrieval + candidate matching
Query = predicted latent; candidates/targets = true observed latents; cosine rank via
`retrieval.compute_retrieval_metrics` (true target always inserted). **New policy
`same_source_split_target_type_len_occ_rate_bin`** = source + split + target_type +
occupancy + length + **context-event-rate** bin (SCID/MIMIC never distract each other;
empties never distract populated; equal candidate N across coarse/coarse_k/fine and all
W ⇒ identical chance rate). **Fine is cardinality-normalized**: a SEPARATE retrieval per
sub-position k (add `subwindow_k` to the group key), same N; `R@10_fine = mean_k R@10(k)`.
The joint K-tuple recall is reported (quantifies the cardinality penalty) but NOT gated.

## The BUILD gate (per source, independent — ALL required)
1. **Literal pre-registered** (reported): `R@10_coarse(W*) − R@10_fine(W*) ≥ 0.10` with
   non-overlapping patient-bootstrap 95% CIs, on the **populated** stratum (headline).
2. **De-confounded** (recommended decision quantity): same ≥0.10 non-overlap test with
   `coarse_k` (mean_embed over a fixed event budget = per-source median non-empty
   sub-window occupancy, uniform fixed-seed sub-sample of the full-W span) — strips the
   pooling-variance/SNR shortcut.
3. **Per-unit-time SLOPE** (§3.3's literal criterion): fit R@10 vs log W per channel over
   the frozen horizons; require `β_fine − β_coarse > 0`, patient-bootstrap CI excluding 0.
4. **Out-of-circle raw-count co-gate** (the only external anchor): score
   `predict_occupancy_from_latent` occupancy+`log1p`-count against the OBSERVED, never-
   encoded `n_target_events` per W and per sub-window — BUILD only if latent-retrieval
   AND raw-count both show coarse better-preserved per unit time (a scissors ⇒ inside-
   circle circularity ⇒ no hierarchy).
5. **Vetoes**: within-sequence time-shuffle null reproduces ≥½ the gap ⇒ not temporal;
   K=1 null gap ≠ 0 ⇒ harness bug.
6. **Sufficiency ablation** (Pi necessary-not-sufficient): flat (recursive) vs 2-level
   (coarse-anchored) matched-compute — BUILD requires 2-level to LOWER long-horizon
   fine-drift slope (`Δslope>0`); else the separation merely relocates drift ⇒ single-scale.

Any failure ⇒ **single-scale for that source** (NOT programme failure). **Bootstrap unit =
patient/sequence (2000×), never window.** Per source×horizon×granularity cell adequacy
(matched ≥500, empty ≤0.5, non-empty median occ ≥2, MIMIC saturated ≤0.5) checked BEFORE
trusting any non-overlap result — an underpowered cell is reported INCONCLUSIVE, not gated.

## Empty handling
Censored excluded everywhere (`is_censored`); admitted W-blocks are fully-observed so no
sub-window can be mis-labeled empty. Genuine empties encoded (→`z_empty`), occupancy-
class-separated candidate pools, R@k reported by-occupancy; the **gate is on the
populated stratum** (silence self-retrieving to the single `z_empty` attractor would
trivially inflate both channels).

## P5 (retrieval circularity) — the honest limit
`mean_embed` is order/timing-blind, so "fine" is already coarsened at the target →
structurally biased toward coarse. **Every rung-0 verdict is a DEV-ONLY architecture-
selection signal**; full fine-fidelity validation (order/Δt hard negatives, frozen-decode
ceiling) is deferred to rung 1/3. Defenses: pooling-identity non-degeneracy, the out-of-
circle raw-count co-gate, `coarse_k`, time-shuffle null, source/length/rate/occupancy-
matched candidates + dataloader source mask, and never optimizing target+metric+reward in
one loop (predictor frozen; metric = retrieval rank + external counts).

## Steps (largely training-free; reuse encode-empty)
1. Pre-register + **Pi blueprint gate** (open questions below) before any GPU run.
2. Composite rung −1 readiness gate on real data (green before compute).
3. Train per-source encode-empty v0B (SCID, MIMIC × recursive + horizon_conditioned) —
   4 tiny checkpoints, <1 GPU-hr, source-masked.
4. `extract_blocks --unit wall_clock` per source × frozen W (reuse).
5. **NEW `subwindow_blocks.py`** — carve K sub-windows via repeated `_wall_clock_target_span`.
6. **NEW `export_coarse_fine_latents.py`** — wall-clock latent export (the existing export
   is event-index + skips empties): `z_coarse`/`coarse_k`/`{z_k}`/`ẑ_coarse`/`{ẑ_k}` +
   predicted AND observed raw counts + sidecar (governed-local, fp16).
7. **retrieval.py extensions** — context-rate bin + `subwindow_k` key + option to return
   local per-query (rank, patient_hash, occ, source, W, subwindow_k, granularity) records.
8. **NEW `rung0_horizon_decay.py`** — bootstrap CIs + the full multi-part gate + controls
   + raw-count scissors co-gate + adequacy; aggregate-only per-source verdict.
9. **NEW `drift_ablation.py`** — recursive vs horizon_conditioned + coarse-anchoring
   drift-slope (sufficiency). Optional: train one horizon_conditioned fine-render head.
10. **Adversarially verify the headline**, then Pi result gate. No hierarchy claim without
    the full gate. Any failure ⇒ single-scale.

## Open questions for Pi (blueprint gate)
1. **Gate-quantity amendment**: make de-confounded `coarse_k` the DECISION quantity
   (literal `coarse_full` reported secondary), or keep `coarse_full` as the decision?
2. **Slope co-required**: confirm the per-unit-time slope separation is co-required with
   the ≥0.10 level (not merely reported).
3. **Raw-count co-gate**: approve promoting it from veto to CO-PRIMARY; acceptable that
   its features are in-frame (only the target count external)?
4. **SCID median horizon**: W*=90d headline + 365d co-reported (4-element set has no
   single median), pre-frozen on train/dev, test held out?
5. **Sufficiency**: require the flat-vs-2-level `Δslope>0` as a HARD co-gate; is the
   training-free coarse-anchoring proxy acceptable, or must a horizon_conditioned
   fine-render head be trained (relax the `horizon_count=1` pin)?
6. **Decomposition primary**: K=2 proportional vs fixed-width δ as the pre-registered primary?
7. **Power vs effect**: a cell meeting adequacy but failing non-overlap — valid single-scale
   verdict, or reported UNDERPOWERED/INCONCLUSIVE?
8. **Dev-only status**: confirm any green gate is a dev-only architecture-selection signal
   (fine handicapped at the target), with full fine-fidelity validation deferred to rung 1/3.
