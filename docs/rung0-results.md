---
title: Within-source wall-clock RUNG 0 — RESULTS (dev; Pi R5+R6-corrected)
created: 2026-07-10
status: governed run complete + adversarially verified (4 lenses CONFIRMED); Pi R6 ACCEPT-WITH-CHANGES folded in + aggregate-only recompute confirms both verdicts — no hierarchy built (SCID inconclusive, MIMIC single-scale); DEV-ONLY architecture-selection signal (Pi Q8)
substrate: joint MIMIC+SCI-D corrected 350M (vocab 1050); eval split = dev (test held out)
reporting: aggregate-only (per source×horizon R@10 gaps + CIs; no seq ids / tokens / real paths)
---

# Rung 0 — results

Within-source coarse-vs-fine wall-clock horizon-decay (does a timescale separation justify
a hierarchical JEPA?), run on the real substrate through the Pi-R5+R6-corrected harness:
composite rung −1 gate (PASS) → per-source encode-empty v0B checkpoints → dev-only
coarse/fine latent export → patient-bootstrap paired verdict. Frozen horizons (test held
out): SCID {30,90,365,730} d (co-primary 90 + 365); MIMIC {0.25,0.5,1,2} d (co-primary 0.5).
The **decision slope** is fit over the frozen **primary band** only — SCID {30,90,365,730},
MIMIC {0.25,0.5,1} — with MIMIC's **2 d treated as a saturation-boundary sensitivity horizon**
(reported apart, not gated; Pi R6 #1). The slope gate is applied in **registered range-scale
(implied-widening) units** (Pi R6 #2).

## Verdict
| Source | Decision |
|--------|----------|
| **SCID** | `NO-BUILD_INCONCLUSIVE` |
| **MIMIC** | `NO-BUILD_EFFECT-RULED-OUT` (single-scale) |

Both verdicts are **unchanged and reconfirmed** under the Pi-R6 aggregate-only recompute
(primary-band slope, widening-unit gate, coarse_B/fine_B hard-gated adequacy).

**No hierarchy is built.** Per Pi Q8 this is a *dev-only architecture-selection* signal —
`mean_embed` is order/timing-blind, so fine is handicapped at the target; full
fine-fidelity validation (order/Δt hard negatives, frozen-decode ceiling) is deferred to
rung 1/3, and MIMIC↔SCI-D remains transportability-only.

## SCID — the pooling confound the correction caught
| horizon | raw coarse−fine R@10 (CI) | **budget-matched `coarse_B`** (CI) |
|--------:|--------------------------:|-----------------------------------:|
| 90 d | **+0.195** [0.188, 0.202] | **+0.055** [0.047, 0.063] |
| 365 d | **+0.487** [0.480, 0.496] | **+0.140** [0.132, 0.148] |

- Slope (decay rate): β_fine − β_coarse = **+0.176** [0.173, 0.179] → coarse decays slower
  per unit time; implied **range widening +0.562, widening CI [0.553, 0.570]** (registered
  range-scale units) ≫ the 0.05 practical widening → `slope_ok = True`.
- **The literal level gate passes hugely, but the de-confounded `coarse_B` gate FAILS at
  the co-primary 90 d cell** (+0.055 < the 0.10 practical level; CI upper 0.063). The
  bilateral budget control removes **~71–72 % of the raw gap at both central horizons**
  (71.8 % at 90 d: +0.195→+0.055; 71.3 % at 365 d: +0.487→+0.140). So most of SCID's raw
  coarse advantage is **consistent with / attributable under this control to pooling-SNR**
  rather than a real abstraction edge — the confound Pi's C1/`coarse_B` corrections were
  built to expose. This control does **not** prove the residual signal is unreal; it shows
  the residual does not satisfy the frozen hierarchy gate. A naive level-only gate would
  have declared a slam-dunk hierarchy (+0.49 at 365 d).
- The de-confounded evidence is **horizon-split** (fails 90 d, holds 365 d), and the
  sufficiency/raw-count co-gates were **not run** (recorded `not_run`, not fail) →
  **INCONCLUSIVE**, not a hierarchy claim.
- Well-powered (n_patients ~12k); K=1 harness null passed; adequacy met.
- **Caveat (pool-conditional magnitude, from verification):** the *absolute* gap is
  pool-conditional — the harness's stratified+capped pool gives +0.195 @ 90 d; an
  independent whole-channel patient-disjoint pool gives +0.156. **Sign, direction, and
  gate outcomes are robust to pool definition**; read the exact number as pool-conditional.

## MIMIC — a clean, well-powered null (primary band {0.25, 0.5, 1} d)
| horizon | raw coarse−fine R@10 (CI) | `coarse_B` (CI) |
|--------:|--------------------------:|-----------------:|
| 0.5 d | −0.004 [−0.006, −0.002] | +0.017 [0.014, 0.020] |

- Coarse ≈ fine (tiny negative level gap). **Primary-band decision slope** (over {0.25, 0.5,
  1} d, Pi R6 #1): β_fine − β_coarse = **+0.0051** [0.0035, 0.0067], implied **range widening
  +0.0071, widening CI [0.0049, 0.0093]**. Excluding the 2 d saturation point flips the point
  sign slightly positive (with 2 d it was −0.0006), **but the entire widening CI sits an order
  of magnitude below the 0.05 practical threshold** — the practical effect is ruled out *from
  above*. The `coarse_B` CI upper (0.020) likewise excludes the practical level.
- **Sensitivity slope** (all four horizons incl. 2 d, non-decisional): β-diff −0.0006,
  widening CI [−0.0034, +0.0009] — also below threshold. Both bands agree: no timescale
  separation.
- → **EFFECT-RULED-OUT**, not merely underpowered (n_patients 42,208). **Single-scale is the
  answer for MIMIC.** Per Pi R6: the corrected primary-band slope CI still rules out the
  practical effect, so MIMIC's verdict **stands** (no downgrade to INCONCLUSIVE).

## Why the co-gates can't flip this
A BUILD additionally requires the drift-sufficiency, raw-count, and time-shuffle co-gates.
They were **not run this pass** and are recorded as `not_run` (distinct from `fail`; Pi R6
#4) — a skipped control can never satisfy BUILD. They are moot here regardless: SCID already
fails the de-confounded `coarse_B` level at 90 d and MIMIC's effect is ruled out — neither
reaches BUILD.

## Pi R6 corrections applied (ACCEPT-WITH-CHANGES)
1. **Decision slope over the frozen primary band only** — MIMIC's 2 d saturation horizon is
   excluded from the decision slope and reported as a separate sensitivity slope. The
   primary-band recompute confirms MIMIC EFFECT-RULED-OUT (widening CI [0.0049, 0.0093] ≪ 0.05).
2. **Slope gate in registered range-scale units** — gated on the bootstrapped implied-widening
   CI `(β_fine−β_coarse)·log(W_max/W_min)`, not the raw per-log-time β-diff (unit test with two
   horizon ranges).
3. **SCID causal wording qualified** — "~71–72 % of the raw gap removed under the bilateral
   budget control, consistent with pooling-SNR; residual does not satisfy the frozen gate."
4. **Skipped controls explicit** — `not_run` recorded separately from `pass`/`fail`.
5. **Adequacy failed closed on the decision quantities** — every co-primary cell must be
   adequate for `coarse`+`fine` **and** `coarse_B`+`fine_B` (all met here).

## Harness integrity (real data)
- **K=1 harness null passed** both sources (coarse ≡ fine ⇒ ~0 gap) — the harness is sound.
- Queries are **context-only** (C1); candidates **patient-disjoint** + source/occupancy/
  length/rate-matched; **censored excluded**; **dev only** (test untouched).
- Composite rung −1 gate PASS (readiness + leakage + wall-clock readiness + governance).

## Local artifacts (gitignored)
`run-workspace/local-governed/rung0/` — checkpoints, sidecars, `verdict/rung0-verdict-manifest.json`.

## Adversarial verification — 4 lenses, all CONFIRMED
- **numeric-recompute:** an independent from-scratch recompute off the sidecar latents
  reproduces SCID W90 gap = **0.1951** (manifest 0.19509, exact to 4 dp; populated counts
  11993/27411 match exactly). The **K=1 harness null = 0.0000 exactly** under two methods.
- **coarse_B-confound:** the budget is applied **bilaterally** to `coarse_B` *and* `fine_B`
  (same B, same fixed-seed rule, <B dropped both sides); queries stay context-only. The
  pooling-SNR reading is correctly-signed and fairly constructed — not a budgeting artifact.
- **decision-logic:** both 3-way verdicts re-derive correctly from the raw CIs; adequacy +
  K1-null guards were on the critical path and passed. It also **found a latent gate-logic
  bug** — EFFECT-RULED-OUT used the *worst* co-primary cell's `ci_hi`; **fixed** to require
  ALL co-primary cells to exclude the effect (does not change these two verdicts).
- **governance-leak:** doc aggregate-only, C1 context-only (test-enforced), dev-only (test
  held out), nothing governed tracked.

## Next
1. Pi Round 6 = **ACCEPT-WITH-CHANGES**; all five corrections folded in and the aggregate-only
   recompute confirms both verdicts (Rung-0 record frozen). 2. **Rung 1 (frozen-decode ceiling)**
   is the next rung — the order/Δt fidelity `mean_embed` cannot see, before any generation claim.
