---
title: Clinical-JEPA rung −1 / 0 / 1 run specs (post-gate)
created: 2026-07-05
status: incorporates Pi GO-WITH-CHANGES required changes (2026-07-05); not yet executed
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050) — see clinical-jepa-native-generation-design.md §1a/§4a
latent_space: fresh minimal v0B flat-token JEPA on the new substrate (Chris decision 2026-07-05)
Pi-reviewed: GO-WITH-CHANGES — required changes folded in below
---

# Rung −1 / 0 / 1 run specs

Operationalises the readiness gate and the two cheapest diagnostic rungs, with Pi's required changes folded in. Runs on the joint MIMIC+SCI-D corrected substrate (vocab 1050, vigintile-factored). Latent space = a freshly-trained minimal flat-token JEPA (v0B); frozen 350M states are a comparator only. Aggregate-only reporting. **Language discipline:** "overlap-gated associational operator" / "transportability diagnostic" / "semi-synthetic known-effect accuracy" — never "treatment effect" or "causal accuracy".

## Cross-cutting guards (Pi — mandatory in every rung)

- **Source-shortcut mask:** `DATASET:SCID/MIMIC` and any source-only anchor are stripped from the encoder/predictor/operator input (seq index 0–1), retained only in the evaluator for stratification. Report a **source-prediction probe from latents** (should sit near source base-rate if masking works).
- **`is_outcome` leakage audit (real, not stub):** exclude `is_outcome==1` and endpoint-proximal positions from context/target/eval for any endpoint-facing task; unit tests + a manifest; the indexer refuses to emit a violating block.
- **Candidate normalisation:** distractors/retrieval matched on **source + wall-clock horizon + length + event rate** (else recall/decay is inflated by trivial cohort structure).
- **MIMIC windowing (decided 2026-07-05):** source-specific shorter wall-clock windows for MIMIC (per-admission); **MIMIC↔SCI-D is a supporting transportability diagnostic only**; the semi-synthetic oracle is the primary counterfactual yardstick.
- **Rank-scale semantics:** per-source vigintile bins are source-specific → cross-source "same token" comparisons are rank-scale, not raw-scale.

## Rung −1 — substrate / eval-readiness gate (fail-closed; gates everything)

- **Do:** build the JSONL index carrying per-sequence `source_dataset`; emit a **per-source manifest** — patients/sequences/windows, token-count median/quantiles, wall-clock span, candidate-action frequency, **block yield under each proposed (source-specific) horizon**, split counts.
- **Fail-closed rule:** if either source contributes fewer than **[FREEZE: N_min]** valid matched windows per split, the indexer errors — no silent near-all-SCID result.
- **Also:** the two leakage guards above pass their unit tests.
- **Gate:** both sources yield adequate matched windows + audits green ⇒ proceed to rung 0.

## Rung 0 — horizon-decay pre-test (decides hierarchy only)

- **Objective:** does a timescale separation exist → build a hierarchical JEPA (fable5 R2.1)?
- **Method:** predicted-latent retrieval accuracy vs **wall-clock** horizon at coarse (block) vs fine (event) granularity, **source-stratified** (within MIMIC, within SCI-D); add count/rate probes and **length-matched hard negatives** (Pi: horizon-decay alone is necessary-not-sufficient); the round-1 `d_t` drift sweep at each level; a 2-level-vs-flat matched-compute drift ablation.
- **Pre-registered gate [FREEZE before running]:** build the hierarchy iff coarse per-unit-wall-clock decay is slower than fine by ≥ **[Δ, with CI]** in **both** sources; else single-scale.
- **Scope:** gates **hierarchy only**, not the programme.

## Rung 1 — frozen-decode ceiling (is the latent decodable?)

- **Objective:** can pooled `z⁺` decode to exact `(token, Δt)` order/count/timing (P1/P5)? Upper-bounds any generator.
- **Method:** freeze the JEPA target encoder; train only a read-out decoder `D` on `(z⁺ → future)`; contrast `D(mean z⁺)` vs `D(sampled z⁺)`.
- **Metrics + pre-registered "adequate decode" criteria [FREEZE]:** exact-order recon ≥ **[x]**; exact-count recon ≥ **[x]**; timing via time-rescaling QQ within **[band]**; mean-vs-sample crispness gap ≥ **[x]**.
- **Decoder-free cross-check:** predict raw summary statistics (counts, type marginals, inter-event-time quantiles) straight from `z⁺` vs the data — bounds the decoder's generative-prior masking (falsifier ladder).
- **Attribution:** `D(z⁺)` fails ⇒ representation bottleneck (change targets: VQ / seq-of-latents); `D(z⁺)` fine but `D(mean)` blurs while `D(sample)` crisp ⇒ predictor bottleneck (go distributional).

## Pre-registered before rung 5 / arm training (Pi #7, #8 — not in these rungs)

- The **three arms** (flat AR / pure-latent / **strengthened hybrid** — with plan-faithfulness losses + re-encode/contrast probes, per Pi's §4.2 correction) + metrics, pre-registered now.
- The **semi-synthetic oracle** spec (known effects, confounded policy, overlap failures, no-effect controls, source-like rate/length) — before any arm training; it is the **primary** counterfactual yardstick.
- **Abstention coverage** reported as a headline metric for any operator result.

## Governance
Aggregate-only; no raw tokens / per-patient rows / embeddings / checkpoints / governed paths in committed reports. Results route back to the design record and to Pi before any downstream rung is promoted.
