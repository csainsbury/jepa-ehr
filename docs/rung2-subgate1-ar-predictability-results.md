---
title: Rung-2 sub-gate 1 — autoregressive predictability on the joint substrate (DEV)
created: 2026-07-26
status: DEV-only, NOMINATE-only, aggregate-only; TEST sealed; not certified against the semi-synthetic oracle
substrate: joint MIMIC+SCI-D corrected (vocab 1050); target blocks T0, event-count windows
reporting: aggregate-only (per source × arm ratios; no sequence ids, tokens or patient-level records)
---

# Rung-2 sub-gate 1 — is the future predictable from context?

The first measured answer to the local Clinical-JEPA question — *can latent-state prediction support
accurate autoregressive futures?* — on the real substrate rather than on synthetic fixtures.

**Headline.** The future **is** predictable from context, for **adjacent** and **sufficiently coarse**
windows on sequences with enough future remaining, and the trained model sits **0.016–0.029 from the best
achievable on this representation**. The design levers are **horizon distance** and **window granularity** —
not predictor size, not target order-awareness, not encoder richness.

Every number below is `d_self / ambient_NN`: the prediction's cosine distance to its own target, divided by
the distance to the nearest *other* sequence's target. **< 1.0 clears the bar** — the prediction is closer to
its own future than to the nearest wrong one. It is a demanding criterion, not a threshold for usefulness.

## The result

Train-fitted ceiling (families fitted on TRAIN with more rows than the model saw, scored on DEV; matched
eligibility of ≥256 future tokens):

| | MIMIC | SCID |
|---|---|---|
| n train / dev | 16,981 / 2,207 | 23,019 / 3,793 |
| trained model | 0.796 | 0.817 |
| **ceiling (RFF, train-fitted)** | **0.780** | **0.788** |
| headroom (model − ceiling) | +0.016 | +0.029 |
| persistence (no fit) | 0.879 | 0.842 |
| chance (shuffled) | 1.615 | 3.994 |

All guards pass simultaneously: the ceiling **bounds the model**, is **not interpolating**
(features/train-rows 0.045–0.060), and the target geometry is **healthy** (shared-component norm 0.58–0.60).

**Not target-limited** — the ceiling is well below 1.0, so instance-specific information is present and
recoverable out of sample. **Not meaningfully capacity-limited either** — 0.02–0.03 of headroom is worth a
few percent, not a category change.

## What actually drives predictability

Measured with all arms on the same rows, so the comparisons are internally valid:

| target definition | ratio |
|---|---|
| `latent_next8` | 1.308 |
| **`latent_next32`** | **0.824** |
| `latent_next128` | 0.855 |
| `latent_far32` (offset 128) | 1.282 |
| `token_histogram32` | 0.812 |

Three effects: an **8-event window is too fine** to predict; predictability **decays sharply with horizon
distance** (adjacent 0.824 vs offset-128 1.282); and target **type barely matters** (histogram 0.812 ≈
latent 0.824).

## What was ruled out

| hypothesis | test | outcome |
|---|---|---|
| predictor capacity | width/steps ladder + train-fitted ceiling | only 0.02–0.03 of headroom |
| target invariance | frozen order targets T1/T2/T3 | **worse**: 0.963 / 1.221 / 1.199 vs 0.824 mean-pooled |
| context encoding | multiscale, mean+std, full token histogram | improvement 0.028; a raw 1050-dim count vector adds nothing |

The order-target result is the sharpest negative. `mean_embed` is permutation-invariant and provably cannot
distinguish orderings that T1/T2/T3 separate at 1.000 — yet making the target order-aware makes it *harder*
to predict. The bottleneck is not the target's invariance.

## The confounder that reframed everything

Row eligibility — how much future a row must have — is a **selection on how much sequence remains, and it
dominates the result**. Same target, model and metric; only the rule changed:

| eligibility | SCID n | `latent_next32` |
|---|---|---|
| ≥ 32 future tokens | 1,722 | **1.134** (fails) |
| ≥ 256 future tokens | 3,775 | **0.824** (clears) |

The ambient normaliser *fell* (0.212 → 0.165), so the denominator got harder and the ratio still improved —
a real gain, not a normalisation artefact. **Any predictability claim on this substrate must state its
eligibility rule.** Three earlier probes were computed under the ≥32 rule; re-run at ≥256, one conclusion
(capacity) was overturned and two (order targets, encoder) survived.

## Rollout behaviour

From the sub-gate 1 rollout diagnosis (4-step recursive, dev, n≈5.1k/source): the step-0 exposure gap is
**exactly 0.0** on real data, as construction requires. SCID shows a CI-separated compounding gap
(0.049 → 0.099, CI_lo 0.046 → 0.096) with free-running degrading while teacher-forced stays flat — the
classic self-conditioning signature. MIMIC shows a negative gap and no drift.

**Instrument finding:** `classify_signature` tests collapse *first* and short-circuits, so SCID's real,
CI-separated drift never reaches the categorical label. The continuous diagnostics carry it; the label
under-reports.

## Method: six failures, and the guards they bought

Six methodological problems arose; five were caught by controls, one by re-running on instruction. Each guard
is now **in code**, not in judgement.

1. **A "ceiling" the model beat.** 1-NN borrows a single neighbour's target (high variance) while a predictor
   smooths. → a ceiling must be *checked* to bound every trained arm.
2. **An in-sample ceiling.** Claimed 0.708 with 0.44 headroom; held out it vanished. → fits must be
   out-of-sample.
3. **Cross-space comparability.** The ratio is *not* comparable across target spaces when one has a large
   shared component: T3 on random embeddings is 90 % shared rank-code, ambient collapses 0.65 → 0.13, and a
   spurious 0.775 appears — the same value on two unrelated datasets, which is what exposed it. →
   degeneracy flag.
4. **A data-starved ceiling.** Fitted on ~2,200 dev rows against a model trained on ~60,000 blocks; the model
   beat it. → fit on TRAIN with more rows than the model saw.
5. **The eligibility confounder** (above). → matched `--min-future-tokens` across probes.
6. **Undefined persistence across spaces** (context 256-dim vs order targets 4,224-dim). → persistence built
   *in* the target space.

The degeneracy guard also caught `family_mix32` and `presence_binary32` returning ratios of 132 and 13,991
against near-zero ambient — unguarded, they would have read as spectacular results.

## Scope and limits

- **DEV only; TEST sealed.** NOMINATE-only: on real dev the ceiling of any decision is nomination, never
  adoption. Not certified against the semi-synthetic oracle, so this is a **diagnosis, not a qualification**.
- The ceiling is linear/RFF over **mean-pooled latents from one checkpoint's embedding table**. It bounds any
  smooth map *on this representation* — **not any model**. A learned encoder could move the representation
  itself, which is outside what this construction can bound.
- Ratio > 1.0 does not mean "no information": SCID's chance arm is 3.994 against a model at 0.817.
- One target family (T0 event-count windows). Wall-clock windows and clinically-derived endpoints are
  untested.

## Reproduce

```
scripts/rung2_train_fitted_ceiling.py        # the headline ceiling (all guards pass)
scripts/rung2_target_definition_ceiling.py   # horizon / granularity effects + the confounder control
scripts/rung2_order_target_ceiling.py        # order targets (carries a confounder banner)
scripts/rung2_context_encoder_ceiling.py     # encoder arms (carries a confounder banner)
scripts/rung2_honest_ceiling.py              # held-out ceiling + controls (carries a confounder banner)
scripts/rung2_collapse_capacity_vs_target.py # SUPERSEDED — retraction banner; ladder ratios only
```

Aggregate artifacts (gitignored, local-governed): `run-workspace/local-governed/rung2_subgate1/`.
