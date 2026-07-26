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

**Headline.** Predictability is real but it lives in **event space**, not in wall-clock time, and the route
between them is **autoregressive**.

1. **Event-count windows are predictable.** For adjacent, sufficiently coarse windows on sequences with
   enough future remaining, the trained model sits **0.016–0.029 from the best achievable on this
   representation**. Levers: horizon distance and window granularity — not predictor size, target
   order-awareness, or encoder richness.
2. **Direct wall-clock prediction fails outright** — 0/12 cells for either source (SCID best 1.284, MIMIC
   1.064). Since the clinical question is always temporal, this matters more than (1).
3. **The AR decomposition rescues it.** Predicting the next K events and then cutting at the horizon — rather
   than predicting a time window in one shot — recovers the target for **both** sources (MIMIC 0.385, SCID
   0.944 from the true event block, versus 0.907 / 2.077 from context alone). Coverage is not the obstacle:
   the events needed are already inside the block that is predictable. **The entire remaining difficulty is
   the cut point**, which is a timing problem — sub-gate 4's continuous-time head, not a bigger encoder.

The single most important mechanical finding: **mean pooling cannot express a temporal prefix.** Retaining
per-event identity and relative time moves SCID from 1.433 (fails) to 0.944 (clears) on identical
information.

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

## The operating envelope

Train-fitted linear ridge across a horizon × granularity grid, one eligibility rule (≥384 future tokens) for
every cell so the grid is internally comparable. Zero degenerate cells.

**SCID** (n_train 15,060 / n_dev 2,267) — ceiling by offset × window:

| offset ↓ / window → | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| 0 | 2.149 | 1.170 | 0.846 | 0.675 | **0.609** | 0.634 |
| 32 | 2.206 | 1.262 | 0.943 | 0.775 | 0.724 | 0.759 |
| 128 | 2.381 | 1.407 | 1.153 | 1.032 | 1.026 | 1.074 |
| 256 | 2.474 | 1.581 | 1.362 | 1.318 | 1.325 | 1.354 |

**MIMIC** (n_train 10,082 / n_dev 1,167):

| offset ↓ / window → | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| 0 | 1.723 | 1.244 | 0.942 | 0.760 | 0.691 | **0.674** |
| 32 | 1.734 | 1.268 | 0.972 | 0.797 | 0.740 | 0.717 |
| 128 | 1.778 | 1.304 | 1.007 | 0.860 | 0.819 | 0.798 |
| 256 | 1.834 | 1.329 | 1.044 | 0.895 | 0.855 | 0.844 |

- **Granularity has a hard floor.** A 4-event window is catastrophically unpredictable (2.15 / 1.72); 8 is
  still unpredictable; **16 events is where it crosses 1.0**. Below ~16 events there is essentially no
  recoverable instance-specific signal.
- **Granularity saturates.** 32 → 64 → 128 flattens, and SCID *reverses* (0.675 → 0.609 → 0.634). Sweet spot
  ≈ **64 events (SCID)**, **128 (MIMIC)**. More averaging stops helping.
- **Horizon decay differs sharply by source.** MIMIC clears in 14/24 cells, SCID in 8/24. Counter-intuitively
  MIMIC — the *short*, per-admission source — holds predictability further out than long-trajectory SCID.
  Plausibly a bounded homogeneous episode versus years of heterogeneous care, but that is an interpretation,
  not something measured here.

## The usable horizon is COHORT-CONDITIONAL, not a single number

A fine offset sweep (32→128, step 16) at windows 32 and 64, run under two eligibility rules:

| eligibility | window 32 | window 64 |
|---|---|---|
| natural (≥192 tok), n_train 21,109 | crosses **~33** | crosses **~33** |
| matched (≥384 tok), n_train 15,060 | crosses **~110** | crosses **~118** |

Same model, same metric, same offsets — a **3–4× difference in usable horizon** from which rows qualify.
Requiring more future tokens selects sequences with more future remaining, and those stay predictable
further out: **eligibility and horizon are entangled.** The crossing *is* stable across window size within
each rule (33/33, 110/118), so it is genuinely an offset effect rather than window noise.

**MIMIC never crosses under either rule** (0.740–0.876 across all cells) — robust and
eligibility-insensitive.

Design consequence, deliberately stated conditionally:

- **MIMIC** — predictable across the tested horizon range, eligibility-insensitive; safe to design against.
- **SCID** — usable horizon between **~33 and ~118 events depending on the cohort conditioned on**.
  Restricting to patients with substantial remaining record gives the longer figure; an unrestricted cohort
  the shorter. That is a **cohort-definition question, not a modelling one**, and this data does not settle
  it. Quoting either figure alone would be confidently misleading.

## Wall-clock targets: direct prediction fails, decomposition works

Everything above uses **event-count** windows. The clinical question is temporal, so the translation matters.

**Direct wall-clock prediction: 0/12 cells clear, both sources.** Per-source grids from the frozen Rung-0
horizons, fully-observed windows only (censoring excluded 25,290 MIMIC and 3,057 SCID train rows; MIMIC kept
sub-day because it saturates at ≥63 % by W=3 d).

| source | grid | best cell |
|---|---|---|
| SCID | 30/90/365 d windows × 0/30/90/365 d offsets | **1.284** |
| MIMIC | 0.25/0.5/1 d × 0/0.25/0.5/1 d | **1.064** |

This is *consistent* with the event-count envelope, not contradictory: SCID's 30-day window holds a median of
**4 events** and its 365-day window only **40**, against an established floor of ~16. The temporal framing
runs into the same granularity wall from the other side.

**The AR decomposition.** The direct probe demanded content *and* timing simultaneously. Splitting them:

| | MIMIC (1 d) | SCID (30 d) |
|---|---|---|
| next-64 span, median | 1.96 d | 419 d |
| coverage of window by next-32 | 0.889 | **1.000** |
| context only | 0.907 | 2.077 |
| oracle next-32, mean-pooled | 0.316 | 1.433 ✗ |
| **oracle next-32, timing-preserved** | **0.385** | **0.944 ✓** |

Coverage is not the obstacle — 32 events contain SCID's entire 30-day window in 99.6 % of cases. What failed
was the *representation*: a mean-pooled oracle cannot express "the first 14 % of these events". Preserving
per-event identity and relative time clears the bar for both sources on identical information.

Note the asymmetry in how the arms scale: the timing-preserved arm is **flat across K** (0.385 / 0.944 at
K = 32, 64, 128) because it can select the relevant events, whereas the mean-pooled arm **degrades** with K
(0.316 → 0.580, 1.433 → 1.862) as irrelevant events dilute the average.

**Limits of this arm.** The oracle uses the **true** future block, so it bounds the decomposition rather than
describing a system. Real performance is event-block prediction (~0.609–0.675) *composed with* the cut, and
excludes multi-step compounding (SCID's exposure gap grew 0.049 → 0.099 over four steps). The
timing-preserved arm caps at 32 events, so K = 64/128 rows are truncated and the flatness partly reflects that
cap. SCID's 0.944 only just clears.

## Scope and limits

- **DEV only; TEST sealed.** NOMINATE-only: on real dev the ceiling of any decision is nomination, never
  adoption. Not certified against the semi-synthetic oracle, so this is a **diagnosis, not a qualification**.
- The ceiling is linear/RFF over **mean-pooled latents from one checkpoint's embedding table**. It bounds any
  smooth map *on this representation* — **not any model**. A learned encoder could move the representation
  itself, which is outside what this construction can bound.
- Ratio > 1.0 does not mean "no information": SCID's chance arm is 3.994 against a model at 0.817.
- Wall-clock results are for **T0 event-count-extracted blocks re-cut by time**; clinically-derived endpoints
  (MACE and similar) remain untested.
- The AR decomposition is validated as a **route**, not as a working 30-day predictor: no probe here composes
  event-block prediction with a learned cut, and none includes multi-step compounding.
- The envelope grid uses a ≥384-token eligibility rule, which re-selects the longest sequences; its absolute
  values are comparable *within* the grid only, not against the ≥256 runs above.
- The sweep uses linear ridge, justified because RFF beat it by only 0.001–0.004 in the validated
  train-fitted run — a margin that is untested at the grid extremes.

## Reproduce

```
scripts/rung2_train_fitted_ceiling.py        # the headline ceiling (all guards pass)
scripts/rung2_horizon_granularity_sweep.py   # the envelope grid + the fine offset sweep (--windows/--offsets)
scripts/rung2_wallclock_target_ceiling.py    # direct wall-clock targets (per-source Rung-0 horizons)
scripts/rung2_ar_decomposition_probe.py      # span / coverage / oracle-content, incl. the timing-preserved arm
scripts/rung2_target_definition_ceiling.py   # horizon / granularity effects + the confounder control
scripts/rung2_order_target_ceiling.py        # order targets (carries a confounder banner)
scripts/rung2_context_encoder_ceiling.py     # encoder arms (carries a confounder banner)
scripts/rung2_honest_ceiling.py              # held-out ceiling + controls (carries a confounder banner)
scripts/rung2_collapse_capacity_vs_target.py # SUPERSEDED — retraction banner; ladder ratios only
```

Aggregate artifacts (gitignored, local-governed): `run-workspace/local-governed/rung2_subgate1/`.
