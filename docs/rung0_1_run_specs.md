---
title: Clinical-JEPA rung −1 / 0 / 1 run specs (post-gate)
created: 2026-07-05
status: incorporates Pi GO-WITH-CHANGES required changes (2026-07-05); not yet executed
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050) — see clinical-jepa-native-generation-design.md §1a/§4a
latent_space: fresh minimal v0B flat-token JEPA on the new substrate (Chris decision 2026-07-05)
Pi-reviewed: GO-WITH-CHANGES round 2 (2026-07-05) — gates confirmed with fixes; guard-hardening + wall-clock definition folded in
---

# Rung −1 / 0 / 1 run specs

Operationalises the readiness gate and the two cheapest diagnostic rungs, with Pi's required changes folded in. Runs on the joint MIMIC+SCI-D corrected substrate (vocab 1050, vigintile-factored). Latent space = a freshly-trained minimal flat-token JEPA (v0B); frozen 350M states are a comparator only. Aggregate-only reporting. **Language discipline:** "overlap-gated associational operator" / "transportability diagnostic" / "semi-synthetic known-effect accuracy" — never "treatment effect" or "causal accuracy".

## Cross-cutting guards (Pi — mandatory in every rung)

- **Source-shortcut mask:** `DATASET:SCID/MIMIC` and any source-only anchor are stripped from the encoder/predictor/operator input (seq index 0–1), retained only in the evaluator for stratification. Report a **source-prediction probe from latents** (should sit near source base-rate if masking works).
- **`is_outcome` leakage audit (real, not stub):** exclude `is_outcome==1` and endpoint-proximal positions from context/target/eval for any endpoint-facing task; unit tests + a manifest; the indexer refuses to emit a violating block.
- **Candidate normalisation:** distractors/retrieval matched on **source + wall-clock horizon + length + event rate** (else recall/decay is inflated by trivial cohort structure).
- **MIMIC windowing (decided 2026-07-05):** source-specific shorter wall-clock windows for MIMIC (per-admission); **MIMIC↔SCI-D is a supporting transportability diagnostic only**; the semi-synthetic oracle is the primary counterfactual yardstick.
- **Rank-scale semantics:** per-source vigintile bins are source-specific → cross-source "same token" comparisons are rank-scale, not raw-scale.
- **Guard fail-hard (Pi round 2):** the DATASET mask requires `source_prefix_len ≥ 2` and treats `not_configured` as a **failure** (tested on the actual dataloader tensor slice, not just block refs); the `is_outcome` audit **fails** on `blocks_checked==0` when a channel is configured (and scans target/eval spans for endpoint-facing tasks); the source-prediction probe is an **alarm/report only** — near-base-rate is NOT required as proof (MIMIC/SCID differ distributionally; rely on source-matched candidates for the hard control).

## Rung −1 — substrate / eval-readiness gate (fail-closed; gates everything)

- **Do:** build the JSONL index carrying per-sequence `source_dataset`; emit a **per-source manifest** — patients/sequences/windows, token-count median/quantiles, wall-clock span, **candidate-action frequency**, **matched-candidate yield**, **block yield under each proposed (source-specific) horizon**, split counts.
- **Fail-closed rule (Pi-confirmed 2026-07-05):** floor = **500 valid matched windows / source / split**, counted **after** source/horizon/length/context-rate eligibility (NOT merely "a T0-feasible sequence exists") — a minimum sanity floor, not a power guarantee (per-action/per-overlap floors come later). The gate **also fails** if any expected source `{MIMIC, SCID}` or required split is absent (absence must fail, not silently drop from `per_source`).
- **Also:** the leakage guards pass their unit tests (fail-hard config checks per the cross-cutting guards).
- **Gate:** both sources meet the floor on post-eligibility matched windows + audits green ⇒ proceed to rung 0.

## Rung 0 — horizon-decay pre-test (decides hierarchy only)

- **Objective:** does a timescale separation exist → build a hierarchical JEPA (fable5 R2.1)?
- **Method:** predicted-latent retrieval accuracy vs **wall-clock** horizon at coarse (block) vs fine (event) granularity, **source-stratified** (within MIMIC, within SCI-D); add count/rate probes and **length-matched hard negatives** (Pi: horizon-decay alone is necessary-not-sufficient); the round-1 `d_t` drift sweep at each level; a 2-level-vs-flat matched-compute drift ablation.
- **Pre-registered gate (Pi-confirmed 2026-07-05):** build the hierarchy iff coarse Recall@10 at the median wall-clock horizon exceeds fine by **≥ 0.10 with non-overlapping 95% CIs (by patient/sequence bootstrap, NOT window bootstrap), in both MIMIC and SCI-D**, at **common wall-clock horizons**, with **same candidate-set size + source-stratified, length/context-rate-matched negatives**; else single-scale (failure ⇒ single-scale, NOT programme failure).
- **Scope:** gates **hierarchy only**, not the programme.

## Rung 1 — frozen-decode ceiling (is the latent decodable?)

- **Objective:** can pooled `z⁺` decode to exact `(token, Δt)` order/count/timing (P1/P5)? Upper-bounds any generator.
- **Method:** freeze the JEPA target encoder; train only a read-out decoder `D` on `(z⁺ → future)`; contrast `D(mean z⁺)` vs `D(sampled z⁺)`.
- **Metrics + pre-registered "adequate decode" criteria (Pi round-2 modification):** exact-order recon **≥ 0.70** and exact-count recon **≥ 0.80**, both **pre-fixed and compared to chance / nearest-neighbour baselines**; timing = **KS-D below a fixed tolerance [FREEZE with Pi] + calibration/coverage plots** (NOT "KS not rejected at α=0.05" — sample-size dependent). The **mean-vs-sample crispness gap is a mean-collapse *diagnostic*, not a required pass gate** (a genuinely unimodal future may show little gap).
- **Decoder-free cross-check:** predict raw summary statistics (counts, type marginals, inter-event-time quantiles) straight from `z⁺` vs the data — bounds the decoder's generative-prior masking (falsifier ladder).
- **Attribution:** `D(z⁺)` fails ⇒ representation bottleneck (change targets: VQ / seq-of-latents); `D(z⁺)` fine but `D(mean)` blurs while `D(sample)` crisp ⇒ predictor bottleneck (go distributional).

## Wall-clock target-block definition (Pi-specified 2026-07-05)

The deferred seam. Build blocks on **absolute cumulative time**, not event count:
- Target interval = half-open **`[t_query, t_query + W)`** — all events whose `cumulative_days` fall inside.
- **`t_query` = context-end + a fixed wall-clock gap** (scheduled), NOT "the next event after context."
- **Empty-interval handling is predefined:** do NOT drop zero-event intervals (biases rate/horizon tests) — encode them; if empty targets cannot yet be encoded, **report their rate and treat the wall-clock rung as conditional/incomplete**.
- **MIMIC:** blocks must NOT cross admission/discharge boundaries unless patient-level concatenation with explicit gap/admission anchors is adopted (censored/discharged time ≠ "no events").
- Require **monotone nondecreasing `cumulative_days`** (reject negative resets); handle simultaneous events deterministically.
- Compare sources only on **overlapping wall-clock horizons**; source-specific windows are allowed for yield, but the **hierarchy claim (rung 0) is made at common horizons**.

## Pre-registered before rung 5 / arm training (Pi #7, #8 — not in these rungs)

- The **three arms** (flat AR / pure-latent / **strengthened hybrid** — with plan-faithfulness losses + re-encode/contrast probes, per Pi's §4.2 correction) + metrics, pre-registered now.
- The **semi-synthetic oracle** spec (known effects, confounded policy, overlap failures, no-effect controls, source-like rate/length) — before any arm training; it is the **primary** counterfactual yardstick.
- **Abstention coverage** reported as a headline metric for any operator result.

## Governance
Aggregate-only; no raw tokens / per-patient rows / embeddings / checkpoints / governed paths in committed reports. Results route back to the design record and to Pi before any downstream rung is promoted.
