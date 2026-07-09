---
title: Within-source wall-clock RUNG 0 — experiment scoping (next arc)
created: 2026-07-09
status: scoping — needs a design blueprint + Pi gate before the governed GPU run
depends_on: encode-empty implementation COMPLETE (blueprint done, 155 tests); frozen horizons confirmed
---

# Rung 0 (within-source horizon-decay) — scoping

**Objective (rung0_1_run_specs.md rung 0):** does a timescale separation exist
*within each source* → justify a hierarchical JEPA (fable5 R2.1)? Re-scoped to
within-source (Pi R3): SCID and MIMIC each at their own frozen band; MIMIC↔SCI-D is a
transportability diagnostic only.

**Pre-registered gate (Pi R2):** build the hierarchy iff coarse Recall@10 at the median
wall-clock horizon exceeds fine by **≥ 0.10 with non-overlapping 95% CIs (patient/
sequence bootstrap)**, WITHIN each source, length/context-rate-matched + source-
stratified negatives; else single-scale (failure ⇒ single-scale, NOT programme failure).

**Frozen horizons (confirmed on real dev, censored=0):** SCID {30, 90, 365, 730} d;
MIMIC {0.25, 0.5, 1} d.

## Building blocks now in place (encode-empty implementation)
- Wall-clock encode-empty target blocks (`extract_blocks --unit wall_clock`, censored
  excluded, empties = silence).
- v0B encode-empty latent (`train_minimal_jepa --encode-empty`: frozen z_empty +
  occupancy/count heads + calibration + collapse diagnostics).
- Embedding export (`export_mean_token_rollouts` / `extract_flatascend_embeddings`,
  helper-routed, empty-aware).
- Retrieval with occupancy split + source-matched candidates (`retrieval.py`,
  `same_source_split_target_type_occ`, `by_occupancy`).
- Rung-1 empty-decode falsifier; wall-clock readiness + composite rung −1 gate.

## Open design questions (for the blueprint → Pi gate)
1. **Coarse vs fine granularity** in the wall-clock encode-empty setting: how is
   "coarse (block)" vs "fine (event)" retrieval operationalized? (e.g. block-pooled z⁺
   at horizon W vs event-level next-step prediction.)
2. **One model across horizons vs per-horizon heads** (horizon_conditioned) — the
   frozen horizons span 0.25–730 d; recursive rollout vs per-horizon occupancy heads.
3. **Retrieval target** at each horizon: predicted-latent → true target-block latent
   (occupancy-matched candidates); empties as their own class (already supported).
4. **Bootstrap unit** = patient/sequence (NOT window) for the CIs.
5. **Compute plan** on titan 3090 Ti: per-source training + per-horizon embedding
   export + retrieval; governed, aggregate-only outputs.

## Process (per project discipline — this is a headline experiment)
design blueprint → **Pi gate** → composite rung −1 gate on real data → governed GPU
run → adversarial verify the horizon-decay headline before it becomes a hierarchy
conclusion. Do NOT promote a hierarchy claim without the pre-registered CI gate.
